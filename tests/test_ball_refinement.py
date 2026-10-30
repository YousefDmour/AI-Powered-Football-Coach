import numpy as np

from vision.ball_refinement import filter_candidates,guard_foot_candidates,merge_views,tile_windows
from vision.ball_tracking import BallTrack


def test_tiles_cover_edges_and_small_images():
    windows=tile_windows(1920,1080)
    assert len(windows)==6
    assert windows[0]==(0,0,960,720)
    assert windows[-1]==(960,360,1920,1080)
    assert tile_windows(100,80)==[(0,0,100,80)]
    for y in range(0,1080,100):
        for x in range(0,1920,100):
            assert any(left<=x<right and top<=y<bottom for left,top,right,bottom in windows)


def test_view_merge_does_not_count_same_tile_twice():
    rows=[{'bbox':[50,50,60,60],'confidence':.8,'views':[1]},
          {'bbox':[51,50,61,60],'confidence':.7,'views':[1]},
          {'bbox':[50,51,60,61],'confidence':.6,'views':[2]},
          {'bbox':[100,100,110,110],'confidence':.9,'views':[3]}]
    merged=merge_views(rows)
    assert len(merged)==2
    assert sorted(merged[1]['views'])==[1,2]
    assert merged[1]['confidence']==.8


def test_near_foot_exception_needs_color_two_views_and_strong_confidence():
    frame=np.full((120,120,3),(30,130,30),dtype=np.uint8)
    frame[52:58,52:58]=230
    person=[[45,20,65,60]]
    row={'bbox':[50,50,60,60],'confidence':.8,'views':[0,1]}
    assert filter_candidates(frame,[row],person)==[]
    accepted=filter_candidates(frame,[row],person,True)
    assert len(accepted)==1 and accepted[0]['near_feet_exception']
    assert filter_candidates(frame,[{**row,'views':[0,0]}],person,True)==[]
    assert filter_candidates(frame,[{**row,'confidence':.2}],person,True)==[]
    assert filter_candidates(frame,[row],[[45,50,65,100]],True)==[]


def test_foot_exception_cannot_bootstrap_or_jump_to_unrelated_footwear():
    tracker=BallTrack()
    near={'bbox':[55,50,65,60],'confidence':.8,'near_feet_exception':True}
    normal={**near,'near_feet_exception':False}
    assert guard_foot_candidates([near,normal],tracker,0)==[normal]
    tracker.update(0,[{'bbox':[45,50,55,60],'confidence':.8}])
    tracker.update(1,[{'bbox':[46,50,56,60],'confidence':.8}])
    assert guard_foot_candidates([near],tracker,2)==[near]
    assert guard_foot_candidates([{**near,'bbox':[500,500,510,510]}],tracker,2)==[]
    assert guard_foot_candidates([near],tracker,5)==[]
