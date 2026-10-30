import numpy as np

from vision.diagnose_ball_misses import explain_filters, replay_tracker


def test_filter_explanation_keeps_frozen_gate_order():
    frame=np.full((120,120,3),(30,130,30),dtype=np.uint8)
    frame[52:58,52:58]=(230,230,230)
    box=[50,50,60,60]
    assert explain_filters(frame,box,[])['first_rejection'] is None
    assert explain_filters(frame,box,[[45,30,65,65]])['first_rejection']=='person_exclusion'
    assert explain_filters(frame,[0,0,80,80],[])['first_rejection']=='size_or_shape'
    assert explain_filters(frame,[10,10,20,20],[])['first_rejection']=='white_fraction'
    white=np.full((120,120,3),230,dtype=np.uint8)
    assert explain_filters(white,box,[])['first_rejection']=='pitch_fraction'


def test_sparse_review_replays_all_intermediate_tracker_frames(tmp_path):
    import json
    import pandas as pd
    from vision.ball_tracking import BallTrack

    candidates=[{'source_frame':i,'candidates':[{'bbox':[100+i,100,110+i,110],'confidence':.8}]} for i in range(4)]
    tracker=BallTrack();rows=[]
    for item in candidates:
        observation=tracker.update(item['source_frame'],item['candidates'],global_search=True)
        rows.append({'source_frame':item['source_frame'],**{k:v for k,v in observation.items() if k!='bbox'}})
    (tmp_path/'summary.json').write_text(json.dumps({'search_mode':'full','scene_cuts':[]}))
    (tmp_path/'candidates.json').write_text(json.dumps(candidates))
    pd.DataFrame(rows).to_csv(tmp_path/'ball_tracks.csv',index=False)
    states,count=replay_tracker(tmp_path,{0,3})
    assert count==4
    assert states[0]['after']['state']=='pending'
    assert states[3]['before']['hits']==3
    assert states[3]['after']['state']=='observed'
