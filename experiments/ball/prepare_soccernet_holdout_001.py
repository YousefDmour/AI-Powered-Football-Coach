"""Rebuild the original 60-image sparse SoccerNet evaluation subset."""
import argparse
import concurrent.futures
import csv
import hashlib
import math
from pathlib import Path
import zipfile

import cv2
import numpy as np

from project_paths import PROJECT_ROOT
from vision.ball_dataset import write_json
from .soccernet_source import DATASET, REVISION, SIZES, URL_ROOT, RemoteZip, load_entry, parse_ini

URL = f'{URL_ROOT}/test.zip'
SEQUENCES = ('SNMOT-116', 'SNMOT-132')
OUTPUT = PROJECT_ROOT / 'outputs/ball_dataset/soccernet_holdout_001'


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(RemoteZip(URL, SIZES['test'])) as archive:
        index = {entry.filename: entry for entry in archive.infolist()}
    plans, samples = [], []
    for sequence in SEQUENCES:
        prefix = f'test/{sequence}'
        info = parse_ini(load_entry('test', index[f'{prefix}/gameinfo.ini']))
        seq = parse_ini(load_entry('test', index[f'{prefix}/seqinfo.ini']))
        width, height, count = (int(seq[key]) for key in ('imWidth', 'imHeight', 'seqLength'))
        ball_ids = {
            int(key.removeprefix('trackletid_')) for key, value in info.items()
            if key.startswith('trackletid_') and value.split(';')[0].strip().lower() == 'ball'
        }
        plan = {'sequence': sequence, 'game_id': info['gameID'], 'action': info.get('actionClass'),
                'width': width, 'height': height, 'fps': int(seq['frameRate'])}
        plans.append(plan)
        boxes = {}
        ground_truth = load_entry('test', index[f'{prefix}/gt/gt.txt']).decode('utf-8-sig')
        for frame, identity, left, top, w, h, *_ in csv.reader(ground_truth.splitlines()):
            if int(identity) not in ball_ids:
                continue
            left, top, w, h = map(float, (left, top, w, h))
            if not all(map(math.isfinite, (left, top, w, h))) or min(w, h) <= 0:
                raise ValueError('Invalid SoccerNet ground truth')
            box = [max(0, left), max(0, top), min(width, left+w), min(height, top+h)]
            if box[0] < box[2] and box[1] < box[3]:
                boxes.setdefault(int(frame), []).append(box)
        picks = [1+int((i+.5)*count/30) for i in range(30)]
        for frame in picks:
            samples.append((plan, frame, boxes.get(frame, []), index[f'{prefix}/img1/{frame:06}.jpg']))
    if sum(entry.compress_size + 1024 for _, _, _, entry in samples) > 32_000_000:
        raise ValueError('Image download budget exceeded')

    def fetch(sample):
        plan, frame, boxes, entry = sample
        data = load_entry('test', entry)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (plan['height'], plan['width']):
            raise ValueError('Image size mismatch')
        stem = f"{plan['sequence']}_{frame:06}"
        row = {'id': stem, 'image': f'images/{stem}.jpg', 'sequence': plan['sequence'],
               'source_group': 'soccernet_game_'+plan['game_id'], 'source_frame': frame,
               'width': plan['width'], 'height': plan['height'], 'bboxes': boxes,
               'status': 'visible' if boxes else 'uncertain', 'sha256': hashlib.sha256(data).hexdigest()}
        return row, data

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch, samples))
    (output/'images').mkdir(parents=True)
    (output/'labels').mkdir()
    for row, data in results:
        (output/row['image']).write_bytes(data)
        if row['bboxes']:
            w, h = row['width'], row['height']
            lines = [f'0 {(x+X)/2/w:.8f} {(y+Y)/2/h:.8f} {(X-x)/w:.8f} {(Y-y)/h:.8f}\n'
                     for x, y, X, Y in row['bboxes']]
            (output/'labels'/f"{row['id']}.txt").write_text(''.join(lines), encoding='utf-8')
    write_json(output/'manifest.json', {
        'purpose': 'Frozen evaluation only; do not use for training, model selection, or tuning.',
        'dataset': DATASET, 'official_split': 'test', 'revision': REVISION,
        'archive_url': URL, 'class_names': ['ball'],
        'sampling': '30 equally spaced bin centers per sequence, fixed before model predictions. Two distinct game IDs. Eligible sequences have at least 12 ball-annotated sampled frames.',
        'visibility_policy': 'Missing ball annotations are uncertain and excluded from scoring unless reviewed separately.',
        'sequences': plans, 'frames': [row for row, _ in results],
    })
    print(f'Saved {len(results)} images to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    prepare(parser.parse_args().output)
