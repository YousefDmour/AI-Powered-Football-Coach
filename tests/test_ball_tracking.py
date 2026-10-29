import numpy as np
import pytest

from vision.ball_tracking import BallTrack, outside_people, valid_ball


def candidate(x,y,confidence=.8):
    return {'bbox':[x-5,y-5,x+5,y+5],'confidence':confidence}


def test_confirmation_missing_and_recovery_never_interpolate():
    track=BallTrack()
    assert track.update(0,[candidate(100,100)])['state']=='pending'
    assert track.update(1,[candidate(110,100)])['x']==110
    assert track.update(2,[candidate(120,100)])['state']=='observed'
    assert len(track.trail)==2
    missing=track.update(3,[])
    assert missing['state']=='unknown' and missing['x'] is None and missing['y'] is None
    assert not track.trail
    assert track.update(4,[candidate(130,100)])['state']=='pending'
    assert track.update(5,[candidate(140,100)])['state']=='observed'
    assert len(track.trail)==1


def test_outliers_ambiguity_and_low_confidence_do_not_become_positions():
    track=BallTrack()
    assert track.update(0,[candidate(100,100,.05)])['x'] is None
    with pytest.raises(ValueError):track.update(0,[])
    assert track.update(1,[candidate(100,100),candidate(200,200)])['state']=='unknown'
    track.update(2,[candidate(100,100)])
    track.update(3,[candidate(101,101)])
    assert track.update(4,[candidate(900,900)])['state']=='unknown'
    with pytest.raises(ValueError):track.update(3,[])
    track.reset()
    assert track.last_position is None and not track.trail


def test_pitch_and_size_gate_rejects_background_and_implausible_boxes():
    grass=np.full((100,100,3),(30,130,30),dtype=np.uint8)
    assert not valid_ball(grass,[45,45,55,55])
    grass[47:53,47:53]=(230,230,230)
    assert valid_ball(grass,[45,45,55,55])
    assert not valid_ball(grass,[0,0,80,80])
    assert not valid_ball(np.zeros((100,100,3),dtype=np.uint8),[45,45,55,55])
    assert not outside_people([45,85,55,95],[[40,30,60,90]])
    assert outside_people([80,80,90,90],[[40,30,60,90]])


def test_global_recovery_requires_a_second_observation():
    track=BallTrack()
    track.update(0,[candidate(100,100)])
    track.update(1,[candidate(101,100)])
    assert track.update(2,[candidate(500,500,.8)],global_search=True)['state']=='pending'
    assert track.update(3,[candidate(501,500,.8)])['x']==501
