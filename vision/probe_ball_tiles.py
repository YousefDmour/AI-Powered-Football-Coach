"""Measure tiled candidates on development labels, without fitting on future clips."""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from project_paths import PROJECT_ROOT,SAMPLE_VIDEO
from vision.ball_refinement import detect_tiles,filter_candidates
from vision.ball_tracking import center


def run(model_path,output):
    labels=json.loads((PROJECT_ROOT/'data/annotations/heldout_ball_300_539_v2_user_reviewed.json').read_text(encoding='utf-8'))
    players=pd.read_csv(PROJECT_ROOT/'outputs/vision/validation_300_539/players/tracks.csv')
    detector=YOLO(str(model_path));cap=cv2.VideoCapture(str(SAMPLE_VIDEO));records=[]
    started=time.perf_counter()
    try:
        for sample in labels['frames']:
            index=sample['source_frame'];cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
            if not ok:raise ValueError(index)
            raw=detect_tiles(detector,frame)
            people=players.loc[players.source_frame.eq(index),['x1','y1','x2','y2']].values.tolist()
            strict=filter_candidates(frame,raw,people)
            relaxed=filter_candidates(frame,raw,people,allow_feet=True)
            def hit(rows):
                return any(math.dist(center(row['bbox']),sample['center'])<=20 for row in rows) if sample['center'] else None
            row={'source_frame':index,'visibility':sample['visibility'],'raw':raw,
                 'strict':strict,'feet':relaxed,'raw_hit':hit(raw),'strict_hit':hit(strict),'feet_hit':hit(relaxed)}
            records.append(row)
            print(Path(model_path).name,index,'hits:',row['raw_hit'],row['strict_hit'],row['feet_hit'],flush=True)
    finally:cap.release()
    summary={'model':str(Path(model_path).resolve()),'seconds':round(time.perf_counter()-started,2),
        'visible_frames':sum(row['visibility']=='visible' for row in records),
        **{f'{mode}_visible_candidate_hits':sum(row['visibility']=='visible' and bool(row[f'{mode}_hit']) for row in records) for mode in ('raw','strict','feet')},
        'warning':'Candidate availability on development frames, not tracking recall or precision; future validation remains untouched.',
        'frames':records}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='frames'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.model,args.output)
