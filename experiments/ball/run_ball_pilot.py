"""Reproduce the local ball detector pilot and its fixed SoccerNet comparison."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'outputs/ball_training'
SOURCE = ROOT / 'outputs/ball_dataset/reviewed_001'
HOLDOUT = ROOT / 'outputs/ball_dataset/soccernet_holdout_001'
HOLDOUT_SHA = '604e1cb00a2870516cabde083444ddd8a6d2350b2ef9bedb0bb30760f88071fd'
RUN = WORK / 'pilot_001'

sys.path.insert(0, str(WORK / 'cuda_packages'))
os.environ['YOLO_CONFIG_DIR'] = str(WORK / 'ultralytics_config')
os.environ['YOLO_AUTOINSTALL'] = 'false'

import cv2
import torch
import yaml
from ultralytics import YOLO

PRETRAINED = ROOT / 'models/yolov8n.pt'
BASELINE = ROOT / 'models/yolov8m.pt'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def prepare():
    root = RUN / 'development'
    if RUN.exists():
        raise FileExistsError(f'Experiment already exists: {RUN}')
    original = read_json(SOURCE / 'manifest.json')
    images = sorted((SOURCE / 'images').glob('*.jpg'))
    if len(images) != original['exported_images']:
        raise ValueError('Incomplete development set')
    labels = {path.stem: path for path in (SOURCE / 'labels').glob('*.txt')}
    if set(labels) != {image.stem for image in images}:
        raise ValueError('Development images and labels do not match')
    splits = {'train': [], 'val': []}
    for image in images:
        frame = int(image.stem.rsplit('_', 1)[1])
        split = 'val' if 7200 <= frame <= 9700 else 'train'
        splits[split].append({'id': image.stem, 'frame': frame,
                             'image_sha256': digest(image), 'label_sha256': digest(labels[image.stem]),
                             'positive': bool(labels[image.stem].read_text().strip())})
    for split, rows in splits.items():
        if not any(row['positive'] for row in rows) or not any(not row['positive'] for row in rows):
            raise ValueError(f'{split} needs visible and negative examples')
        for kind in ('images', 'labels'):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
        for row in rows:
            stem = row['id']
            shutil.copyfile(SOURCE / 'images' / f'{stem}.jpg', root / 'images' / split / f'{stem}.jpg')
            shutil.copyfile(labels[stem], root / 'labels' / split / f'{stem}.txt')
    data_yaml = root / 'data.yaml'
    data_yaml.write_text(yaml.safe_dump({'path': str(root), 'train': 'images/train',
                                          'val': 'images/val', 'names': {0: 'ball'}}), encoding='utf-8')
    write_json(RUN / 'split.json', {
        'purpose': 'Local development split. Contiguous frame range reserved for model selection; both splits use the same match.',
        'validation_frames_inclusive': [7200, 9700], 'source_manifest_sha256': digest(SOURCE / 'manifest.json'),
        'source_group': original['source_family'], 'splits': splits,
    })
    print({k: {'images': len(v), 'positive': sum(r['positive'] for r in v)} for k,v in splits.items()}, flush=True)
    return data_yaml


def choose_device():
    if not torch.cuda.is_available():
        raise RuntimeError('This pilot needs the project-local CUDA-enabled PyTorch package')
    print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
    return 0


def train():
    device = choose_device()
    data_yaml = prepare()
    args = {
        'data': str(data_yaml), 'epochs': 15, 'imgsz': 960, 'batch': 4, 'device': device,
        'workers': 0, 'amp': False, 'patience': 0, 'plots': False, 'seed': 42,
        'optimizer': 'AdamW', 'lr0': 0.001, 'mosaic': 0.0, 'scale': 0.20,
        'project': str(RUN), 'name': 'training', 'verbose': False,
    }
    write_json(RUN / 'experiment.json', {
        'initial_weights_sha256': digest(PRETRAINED), 'training': args,
        'checkpoint_policy': 'best.pt chosen on the local validation block only',
        'holdout_manifest_sha256': HOLDOUT_SHA, 'comparison': {
            'confidence': 0.05, 'center_tolerance_pixels': 20,
            'input_sizes': {'pretrained_n': 960, 'fine_tuned_n': 960, 'current_full_m': 1280},
            'scope': 'Full-image detector only, without tracking or hand-written filtering',
        },
    })
    started = time.perf_counter()
    YOLO(str(PRETRAINED)).train(**args)
    best = RUN / 'training/weights/best.pt'
    write_json(RUN / 'trained_checkpoint.json', {
        'checkpoint': str(best.relative_to(RUN)), 'sha256': digest(best),
        'training_seconds': round(time.perf_counter() - started, 2),
    })


def center(box):
    x, y, X, Y = box
    return (x + X) / 2, (y + Y) / 2


def iou(a, b):
    x, y, X, Y = a
    l, t, r, d = b
    intersection = max(0, min(X, r) - max(x, l)) * max(0, min(Y, d) - max(y, t))
    return intersection / ((X-x)*(Y-y) + (r-l)*(d-t) - intersection)


def score(rows):
    errors = [r['center_error'] for r in rows if r['center_error'] is not None]
    correct = sum(error <= 20 for error in errors)
    return {'visible_images': len(rows), 'correct_within_20px': correct,
            'wrong_center': len(errors) - correct, 'no_detection': len(rows) - len(errors),
            'center_recall': correct / len(rows),
            'top_box_iou_at_least_0_5': sum((r['iou'] or 0) >= 0.5 for r in rows)}


def evaluate():
    device = choose_device()
    path = HOLDOUT / 'manifest.json'
    if digest(path) != HOLDOUT_SHA:
        raise ValueError('Evaluation manifest changed')
    trained = read_json(RUN / 'trained_checkpoint.json')
    best = RUN / trained['checkpoint']
    if digest(best) != trained['sha256']:
        raise ValueError('Checkpoint changed after training')
    split = read_json(RUN / 'split.json')
    development_hashes = {r['image_sha256'] for rows in split['splits'].values() for r in rows}
    manifest = read_json(path)
    frames = []
    for row in manifest['frames']:
        image = HOLDOUT / row['image']
        sha = digest(image)
        if sha != row['sha256'] or sha in development_hashes:
            raise ValueError(f'Image mismatch or training overlap: {image}')
        if row['status'] == 'visible':
            frames.append((row, cv2.imread(str(image))))
    report = {'scope': 'Top-confidence full-image detection on visible images. No absent-ball samples.',
              'excluded_uncertain_images': len(manifest['frames'])-len(frames), 'models': {}}
    for name, weights, class_id, size in [
        ('pretrained_n', PRETRAINED, 32, 960),
        ('fine_tuned_n', best, 0, 960),
        ('current_full_m', BASELINE, 32, 1280),
    ]:
        model = YOLO(str(weights))
        if model.names[class_id] not in ('sports ball', 'ball'):
            raise ValueError(f'Incorrect class mapping for {name}')
        options = dict(imgsz=size, conf=0.05, classes=[class_id], device=device, verbose=False)
        for _, image in frames[:5]:
            model.predict(image, **options)
        rows, latencies = [], []
        for annotation, image in frames:
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = model.predict(image, **options)[0]
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000)
            coords, conf = result.boxes.xyxy.cpu(), result.boxes.conf.cpu()
            box = coords[int(conf.argmax())].tolist() if len(coords) else None
            targets = annotation['bboxes']
            error = min(math.dist(center(box), center(b)) for b in targets) if box else None
            overlap = max(iou(box,b) for b in targets) if box else None
            rows.append({'id': annotation['id'], 'game': annotation['source_group'], 'box': box,
                         'confidence': float(conf.max()) if len(conf) else None,
                         'center_error': error, 'iou': overlap})
        groups = sorted({r['game'] for r in rows})
        median = statistics.median(latencies)
        report['models'][name] = {
            **score(rows), 'weights_sha256': digest(weights), 'imgsz': size,
            'median_ms': round(median,2), 'mean_ms': round(statistics.mean(latencies),2),
            'p95_ms': round(sorted(latencies)[math.ceil(.95*len(latencies))-1],2),
            'per_game': {key: score([r for r in rows if r['game']==key]) for key in groups},
            'details': rows,
        }
        print(name, score(rows), 'median ms', round(median,2), flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(RUN / 'comparison.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train', 'evaluate'])
    args = parser.parse_args()
    {'train': train, 'evaluate': evaluate}[args.stage]()
