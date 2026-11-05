"""Restore manually reviewed negative intervals for a matching local manifest."""
import argparse
import hashlib
import json
from pathlib import Path

from project_paths import PROJECT_ROOT

BATCHES = ('continuous_001', 'continuous_002', 'cone_holdout_001')


def restore(batch):
    local = PROJECT_ROOT / 'outputs/ball_training' / batch
    manifest = local / 'source_manifest.json'
    review = Path(__file__).with_name('reviews') / f'{batch}.json'
    data = review.read_bytes()
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != json.loads(data)['source_manifest_sha256']:
        raise ValueError('The reviewed source manifest does not match this batch')
    target = local / 'negative_review.json'
    with target.open('xb') as file:
        file.write(data)
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', choices=BATCHES)
    restore(parser.parse_args().batch)
