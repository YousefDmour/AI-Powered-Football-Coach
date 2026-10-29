"""Evaluate sampled ball centers without treating uncertain labels as ground truth."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def evaluate(table, annotations, tolerance=20):
    visible=correct=wrong=missed=false_visible=not_visible=uncertain=0
    errors=[];details=[]
    for sample in annotations:
        frame=sample['source_frame']
        selected=table.loc[table.source_frame.eq(frame)]
        if len(selected)!=1:raise ValueError(f'Expected one observation for frame {frame}')
        row=selected.iloc[0]
        observed=row.state=='observed'
        error=None
        if sample['visibility']=='visible':
            visible+=1
            if observed:
                if not math.isfinite(float(row.x)) or not math.isfinite(float(row.y)):
                    raise ValueError('Observed ball positions must be finite')
                error=math.dist((float(row.x),float(row.y)),sample['center'])
                errors.append(error)
                if error<=tolerance:correct+=1
                else:wrong+=1
            else:missed+=1
        elif sample['visibility']=='not_visible':
            not_visible+=1;false_visible+=int(observed)
        else:
            uncertain+=1
        details.append({'source_frame':frame,'visibility':sample['visibility'],
            'prediction_state':row.state,'center_error_pixels':round(error,2) if error is not None else None})
    return {'visible_annotated_frames':visible,'correct_centers':correct,
        'wrong_centers_on_visible_frames':wrong,'missed_visible_frames':missed,
        'visible_frame_recall':correct/visible if visible else None,
        'reviewed_not_visible_frames':not_visible,'false_visible_on_not_visible_frames':false_visible,
        'uncertain_frames_excluded':uncertain,'center_tolerance_pixels':tolerance,
        'mean_error_on_observed_visible_frames':sum(errors)/len(errors) if errors else None,
        'details':details,
        'limits':'Small assistant-reviewed calibration sample, not independent or held out. Uncertain frames are excluded. No global precision or possession accuracy is established.'}


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
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
