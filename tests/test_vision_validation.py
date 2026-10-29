import pytest

from vision.validation import validate_cache, validation_gate


def test_cache_requires_the_same_original_frame_window(tmp_path):
    source=tmp_path/'source.mp4'
    metadata={'source':str(source),'source_start_frame':300,'processed_frames':240}
    validate_cache(metadata,source,300,240)
    validate_cache(metadata,source,320,100)
    with pytest.raises(ValueError):validate_cache(metadata,source,0,240)
    with pytest.raises(ValueError):validate_cache(metadata,source,400,240)
    with pytest.raises(ValueError):validate_cache(metadata,tmp_path/'other.mp4',300,240)
    validate_cache({'source':str(source),'processed_frames':240},source,0,240)


def test_validation_gate_cannot_pass_without_negative_examples_or_identity_evidence():
    ball={'visible_annotated_frames':10,'reviewed_not_visible_frames':5,'correct_centers':9,
          'wrong_centers_on_visible_frames':0,'false_visible_on_not_visible_frames':0}
    people={'player_identity_transitions_checked':10,'observed_player_id_changes':0}
    assert validation_gate(ball,people)['passed']
    assert not validation_gate({**ball,'reviewed_not_visible_frames':0},people)['passed']
    assert not validation_gate({**ball,'correct_centers':7},people)['passed']
    assert not validation_gate({**ball,'false_visible_on_not_visible_frames':1},people)['passed']
    assert not validation_gate(ball,{**people,'observed_player_id_changes':2})['passed']
    assert not validation_gate(ball,{**people,'player_identity_transitions_checked':0})['passed']
