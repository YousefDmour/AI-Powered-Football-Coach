"""Detect and track the ball on the sample clip; optionally overlay cached people."""

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from project_paths import SAMPLE_VIDEO, VISION_OUTPUT_DIR
from vision.ball_tracking import BallTrack, outside_people, valid_ball
from vision.team_colors import PALETTE
from vision.team_tracking import QUALITY_MODEL
from vision.validation import validate_cache


def detect(detector, frame, crop=None, imgsz=1280):
    if crop is None:
        image=frame;offset_x,offset_y=0,0
    else:
        offset_x,offset_y,right,bottom=crop
        image=frame[offset_y:bottom,offset_x:right]
    result=detector.predict(image,classes=[32],conf=.05,imgsz=imgsz,verbose=False)[0]
    candidates=[]
    for box,confidence in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
        mapped=[box[0]+offset_x,box[1]+offset_y,box[2]+offset_x,box[3]+offset_y]
        if valid_ball(frame,mapped):candidates.append({'bbox':mapped,'confidence':confidence})
    return candidates


def run(video=SAMPLE_VIDEO, output=VISION_OUTPUT_DIR/'ball_tracking'/'quality',
        model=QUALITY_MODEL, max_frames=240, players_csv=None, search='full', start_frame=0):
    video,output,model=Path(video),Path(output),Path(model)
    if max_frames<1:raise ValueError('max_frames must be positive')
    if start_frame<0:raise ValueError('start_frame cannot be negative')
    if search not in {'full','adaptive'}:raise ValueError('Unknown search mode')
    if not video.is_file() or not model.is_file():raise FileNotFoundError('Local video/model is missing')
    players={}
    if players_csv:
        players_csv=Path(players_csv)
        metadata=json.loads((players_csv.parent/'summary.json').read_text(encoding='utf-8'))
        validate_cache(metadata,video,start_frame,max_frames)
        players={int(key):value for key,value in pd.read_csv(players_csv).groupby('source_frame')}
    output.mkdir(parents=True,exist_ok=True)
    detector=YOLO(str(model))
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Cannot open video')
    fps=cap.get(cv2.CAP_PROP_FPS);width,height=int(cap.get(3)),int(cap.get(4))
    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    if fps<=0:
        cap.release();raise ValueError('Source FPS is unavailable')
    writer=cv2.VideoWriter(str(output/'ball_and_teams.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():
        cap.release();raise OSError('Cannot open output video')
    tracker=BallTrack();rows=[];candidate_log=[];previous_histogram=None;cuts=[]
    started=time.perf_counter()
    try:
        for offset in range(max_frames):
            index=start_frame+offset
            ok,frame=cap.read()
            if not ok:break
            thumb=cv2.cvtColor(cv2.resize(frame,(160,90)),cv2.COLOR_BGR2HSV)
            histogram=cv2.calcHist([thumb],[0,1],None,[24,16],[0,180,0,256]);cv2.normalize(histogram,histogram)
            if previous_histogram is not None and cv2.compareHist(previous_histogram,histogram,cv2.HISTCMP_BHATTACHARYYA)>.65:
                tracker.reset();cuts.append(index)
            previous_histogram=histogram
            crop=None
            # Recovery search uses only previous detections, never annotations.
            if search=='adaptive' and tracker.last_position is not None and index%10!=0 and tracker.last_frame is not None and index-tracker.last_frame<=4:
                x,y=map(int,tracker.last_position)
                crop=(max(0,x-200),max(0,y-200),min(width,x+200),min(height,y+200))
            candidates=detect(detector,frame,crop,imgsz=640 if crop else 1280)
            if index in players:
                boxes=players[index][['x1','y1','x2','y2']].values.tolist()
                candidates=[item for item in candidates if outside_people(item['bbox'],boxes)]
            observation=tracker.update(index,candidates,global_search=crop is None)
            rows.append({'source_frame':index,'time_seconds':index/fps,**{key:value for key,value in observation.items() if key!='bbox'},'candidate_count':len(candidates),'search':'crop' if crop else 'full'})
            candidate_log.append({'source_frame':index,'candidates':candidates})
            annotated=frame.copy()
            if index in players:
                for player in players[index].itertuples():
                    color=PALETTE.get(player.team,PALETTE['unknown'])
                    cv2.rectangle(annotated,(player.x1,player.y1),(player.x2,player.y2),color,2)
                    cv2.putText(annotated,f'{player.track_id} {player.team}',(player.x1,max(58,player.y1-5)),0,.4,color,1,cv2.LINE_AA)
            if observation['state']=='observed':
                trail=list(tracker.trail)
                for a,b in zip(trail,trail[1:]):cv2.line(annotated,a,b,(0,180,255),2,cv2.LINE_AA)
                point=(int(observation['x']),int(observation['y']))
                cv2.circle(annotated,point,12,(0,180,255),2)
                cv2.putText(annotated,f"BALL {observation['confidence']:.2f}",(point[0]+16,max(60,point[1]-10)),0,.6,(0,180,255),2)
            cv2.rectangle(annotated,(0,0),(width,44),(25,30,25),-1)
            status='OBSERVED' if observation['state']=='observed' else 'NOT RELIABLY VISIBLE'
            cv2.putText(annotated,f'Ball: {status} | frame {index} | no interpolated positions | A: dark, B: white',(15,28),0,.65,(240,240,235),1,cv2.LINE_AA)
            writer.write(annotated)
            if offset in {0,15,30,60,90,120,150,180,210,max_frames-1}:
                cv2.imwrite(str(output/f'frame_{index:04}.jpg'),annotated)
            if (offset+1)%30==0:print(f'ball: {offset+1}/{max_frames}',flush=True)
    finally:
        cap.release();writer.release()
    elapsed=time.perf_counter()-started
    table=pd.DataFrame(rows);table.to_csv(output/'ball_tracks.csv',index=False)
    (output/'candidates.json').write_text(json.dumps(candidate_log,indent=2),encoding='utf-8')
    summary={'source':str(video.resolve()),'model':str(model.resolve()),'processed_frames':len(rows),'source_fps':fps,'source_start_frame':start_frame,
        'wall_seconds':round(elapsed,3),'processing_fps':round(len(rows)/elapsed,2),
        'state_counts':table.state.value_counts().to_dict(),'scene_cuts':cuts,'search_mode':search,
        'cached_players_csv':str(Path(players_csv).resolve()) if players_csv else None,
        'limits':'Generic sports-ball detector and heuristic gates; unknown has no position. FPS includes ball inference and cached-person overlay, not person inference. No possession or event statistics.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return table,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,default=SAMPLE_VIDEO)
    parser.add_argument('--model',type=Path,default=QUALITY_MODEL)
    parser.add_argument('--output',type=Path,default=VISION_OUTPUT_DIR/'ball_tracking'/'quality')
    parser.add_argument('--max-frames',type=int,default=240)
    parser.add_argument('--start-frame',type=int,default=0)
    parser.add_argument('--players-csv',type=Path)
    parser.add_argument('--search',choices=['full','adaptive'],default='full')
    args=parser.parse_args();run(args.video,args.output,args.model,args.max_frames,args.players_csv,args.search,args.start_frame)


if __name__=='__main__':main()
