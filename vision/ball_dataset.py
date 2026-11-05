"""Prepare a local ball-labeling batch, serve its review page, and import YOLO labels.

Usage: python -m vision.ball_dataset prepare|serve|import-review --help
"""
import argparse
import hashlib
import json
import math
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from project_paths import PROJECT_ROOT

DEFAULT_BATCH = PROJECT_ROOT / 'outputs' / 'ball_dataset' / 'batch_001'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(video, output, count=200):
    import cv2
    import numpy as np
    video, output = Path(video).resolve(), Path(output).resolve()
    if count < 1:
        raise ValueError('Count must be positive')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Batch directory must be empty; existing reviews are never overwritten')
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    cap = cv2.VideoCapture(str(video))
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        raise ValueError('Unreadable video')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir()
    candidates = []
    previous = None
    try:
        # Sample every half-second; compare compact thumbnails to the last
        # retained candidate, then spread the final selection over the clip.
        step = max(1, round(fps / 2))
        for frame in range(total):
            if not cap.grab():
                raise ValueError(f'Cannot decode source frame {frame}')
            if frame < step or frame % step:
                continue
            ok, image = cap.retrieve()
            if not ok:
                raise ValueError(f'Cannot retrieve source frame {frame}')
            thumb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 36))
            if previous is not None and np.abs(thumb.astype(float) - previous.astype(float)).mean() < 3:
                continue
            candidates.append(frame)
            previous = thumb
        if len(candidates) < count:
            raise ValueError(f'Only {len(candidates)} distinct candidates; request fewer images or add another clip')
        selected = [candidates[i] for i in np.linspace(0, len(candidates) - 1, count, dtype=int)]
        records = []
        cap.release()
        cap = cv2.VideoCapture(str(video))
        selected_set = set(selected)
        number = 0
        for frame in range(total):
            if not cap.grab():
                raise ValueError(f'Cannot decode source frame {frame}')
            if frame not in selected_set:
                continue
            number += 1
            ok, image = cap.retrieve()
            if not ok:
                raise ValueError(f'Could not decode selected frame {frame}')
            name = f'match_{frame:06d}'
            filename = f'images/{name}.jpg'
            if not cv2.imwrite(str(output / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError('Could not write image')
            records.append({'id': name, 'image': filename, 'source_frame': frame,
                            'time_seconds': round(frame / fps, 3), 'width': image.shape[1],
                            'height': image.shape[0], 'source_group': digest})
            if number % 50 == 0:
                print(f'Extracted {number}/{count}', flush=True)
        batch_id = hashlib.sha256((digest + json.dumps(selected)).encode()).hexdigest()[:20]
        manifest = {'schema_version': 1, 'batch_id': batch_id, 'source': str(video),
                    'source_sha256': digest, 'fps': fps, 'source_frames': total,
                    'sampling': 'Half-second candidates; adjacent retained thumbnail MAE >=3; evenly distributed final selection. Similarity reduction is heuristic, not scene deduplication.',
                    'split_policy': 'Unsplit staging batch. Keep this entire source group in one future split; no training YAML until independent validation data is available.',
                    'frames': records}
        write_json(output / 'manifest.json', manifest)
        web = Path(__file__).with_name('ball_labeler')
        for name in ('index.html', 'app.js', 'style.css'):
            shutil.copyfile(web / name, output / name)
        print(f'Batch ready: {output}', flush=True)
        return manifest
    finally:
        cap.release()


def validate_review(manifest, review):
    if review.get('schema_version') != 1 or review.get('batch_id') != manifest['batch_id']:
        raise ValueError('Review belongs to a different batch or schema')
    known = {frame['id']: frame for frame in manifest['frames']}
    if not isinstance(review.get('annotations'), dict):
        raise ValueError('annotations must be an object')
    labels, skipped = {}, []
    for frame_id, annotation in review['annotations'].items():
        if frame_id not in known or not isinstance(annotation, dict):
            raise ValueError(f'Unknown image or invalid annotation: {frame_id}')
        frame = known[frame_id]
        status, box = annotation.get('status'), annotation.get('bbox')
        if status in ('pending', 'uncertain'):
            skipped.append(frame_id)
            continue
        if status == 'not_visible':
            if box is not None:
                raise ValueError('A negative image cannot have a box')
            labels[frame_id] = ''
        elif status == 'visible':
            if not isinstance(box, list) or len(box) != 4 or any(type(n) not in (int, float) or not math.isfinite(n) for n in box):
                raise ValueError(f'Invalid box: {frame_id}')
            x1, y1, x2, y2 = box
            w, h = frame['width'], frame['height']
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f'Box outside image or empty: {frame_id}')
            labels[frame_id] = f'0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}\n'
        else:
            raise ValueError(f'Unknown review status: {status}')
    return labels, skipped


def import_review(batch, review_path, output):
    batch, output = Path(batch).resolve(), Path(output).resolve()
    manifest = json.loads((batch / 'manifest.json').read_text(encoding='utf-8'))
    review = json.loads(Path(review_path).read_text(encoding='utf-8-sig'))
    labels, skipped = validate_review(manifest, review)
    if not labels:
        raise ValueError('No reviewed visible or confidently negative images to import')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory; prior exports are never overwritten')
    # Reconstruct image paths from trusted manifest; never use paths from review.
    sources = {}
    for frame in manifest['frames']:
        if frame['id'] in labels:
            source = (batch / frame['image']).resolve()
            if batch not in source.parents or not source.is_file():
                raise ValueError('Image missing or outside batch directory')
            sources[frame['id']] = source
    (output / 'images').mkdir(parents=True, exist_ok=True)
    (output / 'labels').mkdir()
    for frame_id, label in labels.items():
        # Image stem comes from prepared image, not an external review key.
        stem = sources[frame_id].stem
        shutil.copyfile(sources[frame_id], output / 'images' / f'{stem}.jpg')
        (output / 'labels' / f'{stem}.txt').write_text(label, encoding='utf-8')
    write_json(output / 'review.json', review)
    write_json(output / 'manifest.json', {'batch_id': manifest['batch_id'],
               'source_sha256': manifest['source_sha256'],
               'source_family': manifest.get('source_family', manifest['source_sha256']),
               'source_groups': sorted({f.get('source_group', manifest['source_sha256']) for f in manifest['frames'] if f['id'] in labels}),
               'evaluation_overlap_warning': manifest.get('evaluation_overlap_warning'), 'class_names': ['ball'],
               'exported_images': len(labels), 'negative_images': sum(not v for v in labels.values()),
               'skipped_uncertain_or_pending': skipped,
               'unreviewed_images': len(manifest['frames']) - len(review['annotations']),
               'split_policy': manifest['split_policy']})
    print(f'Imported {len(labels)} reviewed images; output is unsplit staging data, not a training configuration.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--video', type=Path, default=PROJECT_ROOT / 'data/videos/match.mp4')
    p.add_argument('--output', type=Path, default=DEFAULT_BATCH)
    p.add_argument('--count', type=int, default=200)
    s = sub.add_parser('serve')
    s.add_argument('--batch', type=Path, default=DEFAULT_BATCH)
    s.add_argument('--port', type=int, default=8781)
    i = sub.add_parser('import-review')
    i.add_argument('--batch', type=Path, default=DEFAULT_BATCH)
    i.add_argument('--review', type=Path, required=True)
    i.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.video, args.output, args.count)
    elif args.command == 'import-review':
        import_review(args.batch, args.review, args.output)
    else:
        batch = args.batch.resolve()
        if not (batch / 'manifest.json').is_file():
            parser.error('Batch does not exist; run prepare first')
        handler = partial(SimpleHTTPRequestHandler, directory=str(batch))
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
        print(f'Ball labeler: http://127.0.0.1:{args.port}/ (Ctrl+C to stop)', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
