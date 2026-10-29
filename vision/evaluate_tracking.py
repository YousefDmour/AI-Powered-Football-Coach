"""Evaluate manually annotated boxes, kit labels and sparse identity continuity."""

import argparse
import json
from pathlib import Path

import pandas as pd


def iou(a,b):
    left,top=max(a[0],b[0]),max(a[1],b[1])
    right,bottom=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,right-left)*max(0,bottom-top)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    union=area_a+area_b-intersection
    return intersection/union if union else 0.0


def evaluate(tracks, annotations, threshold=.4):
    matched,correct,assigned,correct_assigned,switches,transitions=0,0,0,0,0,0
    last_ids={}
    player_changes,player_transitions=0,0
    details=[]
    for frame, labels in sorted(annotations.items(),key=lambda item:int(item[0])):
        predictions=tracks.loc[tracks.source_frame.eq(int(frame))]
        # Global greedy IoU matching prevents duplicate assignment to one detection.
        pairs=[]
        for index,label in enumerate(labels):
            for prediction_index,row in predictions.iterrows():
                overlap=iou(label['bbox'],[row.x1,row.y1,row.x2,row.y2])
                if overlap >= threshold:pairs.append((overlap,index,prediction_index))
        used_labels,used_predictions=set(),set()
        for overlap,index,prediction_index in sorted(pairs,reverse=True):
            if index in used_labels or prediction_index in used_predictions:continue
            used_labels.add(index);used_predictions.add(prediction_index)
            label=labels[index];row=predictions.loc[prediction_index]
            matched+=1
            team_correct=row.team==label['team']
            correct+=int(team_correct)
            is_assigned=row.team in {'team_a','team_b'}
            if label['team'] in {'team_a','team_b'}:
                assigned+=int(is_assigned)
                correct_assigned+=int(is_assigned and team_correct)
            person=label.get('person_id')
            valid_id=pd.notna(row.track_id)
            if person and valid_id:
                if person in last_ids:
                    transitions+=1
                    switches+=int(last_ids[person]!=row.track_id)
                    if label['team'] in {'team_a','team_b'}:
                        player_transitions+=1
                        player_changes+=int(last_ids[person]!=row.track_id)
                last_ids[person]=row.track_id
            details.append({'frame':int(frame),'person_id':person,'expected_team':label['team'],
                'predicted_team':row.team,'track_id':row.track_id if valid_id else None,'iou':round(overlap,3)})
    total=sum(len(labels) for labels in annotations.values())
    return {'annotated_boxes':total,'matched_boxes':matched,'unmatched_boxes':total-matched,
        'correct_kit_labels':correct,'kit_accuracy_on_matched':correct/matched if matched else None,
        'assigned_team_labels':assigned,'correct_assigned_team_labels':correct_assigned,
        'assigned_team_accuracy':correct_assigned/assigned if assigned else None,
        'identity_transitions_checked':transitions,'observed_id_changes':switches,
        'player_identity_transitions_checked':player_transitions,
        'observed_player_id_changes':player_changes,
        'matching_iou_threshold':threshold,
        'limits':'Small curated visual annotation, not an independent human benchmark. Sparse ID changes omit switches between sampled frames. Not IDF1, HOTA or full detection recall.',
        'details':details}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    labels=json.loads(args.annotations.read_text(encoding='utf-8'))
    result=evaluate(pd.read_csv(args.tracks),labels['frames'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key!='details'},indent=2))


if __name__=='__main__':main()
