import pandas as pd
import pytest

from vision.evaluate_tracking import evaluate, iou


def test_iou_and_one_to_one_matching():
    assert iou([0,0,10,10],[0,0,10,10]) == 1
    assert iou([0,0,10,10],[20,20,30,30]) == 0
    assert iou([0,0,0,0],[0,0,0,0]) == 0
    tracks=pd.DataFrame([{'source_frame':0,'x1':0,'y1':0,'x2':10,'y2':10,
                          'team':'team_a','track_id':'0:1'}])
    labels={'0':[{'bbox':[0,0,10,10],'team':'team_a','person_id':'p'},
                 {'bbox':[0,0,10,10],'team':'team_a','person_id':'q'}]}
    result=evaluate(tracks,labels)
    assert result['matched_boxes']==1
    assert result['unmatched_boxes']==1


def test_unknowns_and_sparse_identity_changes_are_not_hidden():
    tracks=pd.DataFrame([
        {'source_frame':frame,'x1':0,'y1':0,'x2':10,'y2':10,'team':team,'track_id':identity}
        for frame,team,identity in [(0,'team_a','0:1'),(1,'unknown','0:1'),(2,'team_b','0:9')]])
    labels={str(frame):[{'bbox':[0,0,10,10],'team':'team_a','person_id':'p'}] for frame in range(3)}
    result=evaluate(tracks,labels)
    assert result['kit_accuracy_on_matched']==pytest.approx(1/3)
    assert result['assigned_team_accuracy']==.5
    assert result['observed_id_changes']==1
    assert result['identity_transitions_checked']==2
    assert result['player_identity_transitions_checked']==2
    assert result['observed_player_id_changes']==1
