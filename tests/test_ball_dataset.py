import json

import pytest

from vision.ball_dataset import import_review, validate_review


def manifest():
    return {'batch_id': 'batch-a', 'source_sha256': 'source-a', 'split_policy': 'Keep source together',
            'frames': [{'id': str(i), 'image': f'images/{i}.jpg', 'width': 100, 'height': 50} for i in range(4)]}


def review(annotations):
    return {'schema_version': 1, 'batch_id': 'batch-a', 'annotations': annotations}


def test_only_reviewed_examples_become_labels():
    data = review({'0': {'status': 'visible', 'bbox': [10, 10, 30, 20]},
                   '1': {'status': 'not_visible', 'bbox': None},
                   '2': {'status': 'uncertain', 'bbox': None}})
    labels, skipped = validate_review(manifest(), data)
    assert labels == {'0': '0 0.20000000 0.30000000 0.20000000 0.20000000\n', '1': ''}
    assert skipped == ['2']
    assert '3' not in labels


@pytest.mark.parametrize('box', [[-1, 0, 10, 10], [0, 0, 101, 10], [10, 0, 10, 10],
                                 [0, float('nan'), 10, 10], [0, True, 10, 10]])
def test_invalid_boxes_rejected(box):
    with pytest.raises(ValueError):
        validate_review(manifest(), review({'0': {'status': 'visible', 'bbox': box}}))


def test_foreign_batches_unknown_images_and_conflicting_negatives_rejected():
    data = review({})
    data['batch_id'] = 'another-batch'
    with pytest.raises(ValueError):
        validate_review(manifest(), data)
    with pytest.raises(ValueError):
        validate_review(manifest(), review({'../bad': {'status': 'not_visible'}}))
    with pytest.raises(ValueError):
        validate_review(manifest(), review({'0': {'status': 'not_visible', 'bbox': [1, 1, 5, 5]}}))


def test_import_creates_matching_images_and_labels_without_training_split(tmp_path):
    batch = tmp_path / 'batch'
    (batch / 'images').mkdir(parents=True)
    (batch / 'manifest.json').write_text(json.dumps(manifest()), encoding='utf-8')
    for i in range(4):
        (batch / 'images' / f'{i}.jpg').write_bytes(b'test-image')
    source = tmp_path / 'review.json'
    source.write_text(json.dumps(review({'0': {'status': 'visible', 'bbox': [10, 10, 30, 20]},
                                         '1': {'status': 'not_visible'},
                                         '2': {'status': 'uncertain'}})), encoding='utf-8')
    output = tmp_path / 'export'
    import_review(batch, source, output)
    assert sorted(p.name for p in (output / 'images').iterdir()) == ['0.jpg', '1.jpg']
    assert (output / 'labels' / '1.txt').read_text() == ''
    assert not (output / 'labels' / '2.txt').exists()
    assert not (output / 'data.yaml').exists()
    audit = json.loads((output / 'manifest.json').read_text())
    assert audit['unreviewed_images'] == 1 and audit['negative_images'] == 1
    assert audit['source_groups'] == ['source-a']
    with pytest.raises(ValueError):
        import_review(batch, source, output)

def test_prepare_extracts_exact_source_indices_sequentially(tmp_path):
    import cv2
    import numpy as np
    from vision.ball_dataset import prepare
    source = tmp_path / 'clip.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
    assert writer.isOpened()
    for frame in range(30):
        writer.write(np.full((48, 64, 3), frame * 6, dtype=np.uint8))
    writer.release()
    output = tmp_path / 'prepared'
    data = prepare(source, output, count=3)
    assert [row['source_frame'] for row in data['frames']] == [5, 15, 25]
    assert len({row['source_group'] for row in data['frames']}) == 1
    for row in data['frames']:
        image = cv2.imread(str(output / row['image']))
        assert image is not None
        assert abs(float(image.mean()) - row['source_frame'] * 6) < 10
    with pytest.raises(ValueError):
        prepare(source, output, count=3)
