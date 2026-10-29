import numpy as np

from vision.team_colors import TeamSmoother, classify_shirt, shirt_features


def classify_color(bgr):
    frame = np.full((100,50,3),bgr,dtype=np.uint8)
    return classify_shirt(shirt_features(frame,(0,0,50,100)))[0]


def test_shirt_profiles_and_out_of_frame_boxes():
    assert classify_color((160,30,30)) == "team_a"
    assert classify_color((230,230,230)) == "team_b"
    assert classify_color((0,230,230)) == "official_candidate"
    assert classify_color((20,20,230)) == "other_kit"
    assert classify_color((20,120,20)) == "unknown"
    frame=np.zeros((100,50,3),dtype=np.uint8)
    assert shirt_features(frame,(70,150,100,180)) is None
    assert classify_shirt(None) == ("unknown",0.0)


def test_team_smoothing_does_not_merge_ids_or_force_unknown_roles():
    smoother=TeamSmoother()
    assert smoother.update("0:1","team_a") == "unknown"
    assert smoother.update("0:1","team_a") == "team_a"
    assert smoother.update("0:1","team_b") == "team_a"
    assert smoother.update("0:2","team_b") == "unknown"
    assert smoother.update("0:1","official_candidate") == "official_candidate"
    assert smoother.update("0:1","unknown") == "unknown"
    smoother.reset()
    assert smoother.update("0:1","team_b") == "unknown"
