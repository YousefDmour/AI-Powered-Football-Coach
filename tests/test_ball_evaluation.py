import pandas as pd

from vision.evaluate_ball import evaluate


def test_ball_evaluation_reports_misses_wrong_positions_and_false_visibility():
    table=pd.DataFrame([
        {'source_frame':0,'state':'observed','x':10,'y':10},
        {'source_frame':1,'state':'observed','x':200,'y':200},
        {'source_frame':2,'state':'unknown','x':None,'y':None},
        {'source_frame':3,'state':'observed','x':20,'y':20},
        {'source_frame':4,'state':'unknown','x':None,'y':None},
    ])
    labels=[{'source_frame':i,'visibility':'visible','center':[10,10]} for i in range(3)]
    labels.extend([{'source_frame':3,'visibility':'not_visible'},
                   {'source_frame':4,'visibility':'uncertain'}])
    result=evaluate(table,labels)
    assert result['correct_centers']==1
    assert result['wrong_centers_on_visible_frames']==1
    assert result['missed_visible_frames']==1
    assert result['false_visible_on_not_visible_frames']==1
    assert result['uncertain_frames_excluded']==1
