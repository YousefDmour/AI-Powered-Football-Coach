"""Compare causal baseline/tiled strategies using cached same-frame baseline detections."""

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from project_paths import DEFAULT_MODEL,PROJECT_ROOT
from vision.ball_refinement import detect_tiles,filter_candidates,guard_foot_candidates,merge_views
from vision.ball_tracking import BallTrack
from vision.team_colors import PALETTE
from vision.validation import validate_cache


def run(baseline_dir,players_csv,output,raw_cache=None,strategy='tiled_strict'):
    baseline_dir,players_csv,output=map(Path,(baseline_dir,players_csv,output))
    metadata=json.loads((baseline_dir/'summary.json').read_text(encoding='utf-8'))
    if strategy not in {'tiled_strict','tiled_feet'}:raise ValueError('Unknown output strategy')
    if metadata['search_mode']!='full':raise ValueError('Full-frame baseline cache is required')
    video=Path(metadata['source']);start=metadata['source_start_frame'];count=metadata['processed_frames']
    validate_cache(json.loads((players_csv.parent/'summary.json').read_text(encoding='utf-8')),video,start,count)
    people={int(key):rows for key,rows in pd.read_csv(players_csv).groupby('source_frame')}
    cached={row['source_frame']:row['candidates'] for row in json.loads((baseline_dir/'candidates.json').read_text())}
    if sorted(cached)!=list(range(start,start+count)):raise ValueError('Incomplete baseline cache')
    output.mkdir(parents=True,exist_ok=True)
    cached_tiles=({item['source_frame']:item['raw_tiled_candidates']
                  for item in json.loads(Path(raw_cache).read_text())} if raw_cache else None)
    if raw_cache:
        cached_metadata=json.loads((Path(raw_cache).parent/'summary.json').read_text(encoding='utf-8'))
        if (Path(cached_metadata['source']).resolve()!=video.resolve()
                or cached_metadata['source_start_frame']!=start
                or cached_metadata['processed_frames']!=count):
            raise ValueError('Tiled cache provenance does not match the baseline')
    if cached_tiles is not None and sorted(cached_tiles)!=list(range(start,start+count)):
        raise ValueError('Tiled detection cache must cover exactly this source interval')
    detector=None if raw_cache else YOLO(str(DEFAULT_MODEL))
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    fps=metadata['source_fps'];width,height=int(cap.get(3)),int(cap.get(4))
    writer=cv2.VideoWriter(str(output/'ball_refined.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():cap.release();raise OSError('Cannot write video')
    trackers={name:BallTrack() for name in ('baseline','tiled_strict','tiled_feet')}
    rows={name:[] for name in trackers};audit=[];started=time.perf_counter()
    try:
        for frame_number in range(start,start+count):
            ok,frame=cap.read()
            if not ok:raise ValueError(f'Cannot read {frame_number}')
            if frame_number in metadata['scene_cuts']:
                for tracker in trackers.values():tracker.reset()
            persons=people.get(frame_number)
            boxes=persons[['x1','y1','x2','y2']].values.tolist() if persons is not None else []
            raw=cached_tiles[frame_number] if cached_tiles is not None else detect_tiles(detector,frame)
            original=[{**item,'views':[-1],'near_feet_exception':False} for item in cached[frame_number]]
            sets={'baseline':original}
            for name,allow in [('tiled_strict',False),('tiled_feet',True)]:
                additions=filter_candidates(frame,raw,boxes,allow)
                if allow:additions=guard_foot_candidates(additions,trackers['tiled_strict'],frame_number)
                # Merge only after filtering; a rejected high-confidence view
                # must not suppress an accepted baseline candidate.
                sets[name]=merge_views(original+additions)
            for name,tracker in trackers.items():
                observation=tracker.update(frame_number,sets[name],global_search=True)
                rows[name].append({'source_frame':frame_number,'time_seconds':frame_number/fps,
                    **{key:value for key,value in observation.items() if key!='bbox'},'candidate_count':len(sets[name])})
            audit.append({'source_frame':frame_number,'raw_tiled_candidates':raw,
                          'strict':sets['tiled_strict'],'feet':sets['tiled_feet']})
            annotated=frame.copy()
            if persons is not None:
                for person in persons.itertuples():
                    cv2.rectangle(annotated,(person.x1,person.y1),(person.x2,person.y2),PALETTE.get(person.team,(140,140,140)),1)
            observation=rows[strategy][-1]
            if observation['state']=='observed':
                trail=list(trackers[strategy].trail)
                for a,b in zip(trail,trail[1:]):cv2.line(annotated,a,b,(0,180,255),2)
                point=(int(observation['x']),int(observation['y']));cv2.circle(annotated,point,12,(0,180,255),2)
            cv2.rectangle(annotated,(0,0),(width,44),(25,30,25),-1)
            cv2.putText(annotated,f"TILED + BASELINE | frame {frame_number} | ball: {observation['state']} | no interpolation",(15,28),0,.65,(240,240,235),1)
            writer.write(annotated)
            if (frame_number-start)%30==0 or frame_number==start+count-1:
                cv2.imwrite(str(output/f'frame_{frame_number:04}.jpg'),annotated)
            if (frame_number-start+1)%30==0:print(f'{output.name}: {frame_number-start+1}/{count}',flush=True)
    finally:cap.release();writer.release()
    elapsed=time.perf_counter()-started
    for name,records in rows.items():pd.DataFrame(records).to_csv(output/f'{name}.csv',index=False)
    original=pd.read_csv(baseline_dir/'ball_tracks.csv')
    replay=pd.DataFrame(rows['baseline'])
    if original.state.tolist()!=replay.state.tolist():raise AssertionError('Baseline replay changed')
    for key in ('x','y'):
        if not original[key].fillna(-1).equals(replay[key].fillna(-1)):
            # CSV float serialization may round the last bit.
            if (original[key].fillna(-1)-replay[key].fillna(-1)).abs().max()>1e-6:
                raise AssertionError('Baseline positions changed')
    (output/'candidates.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    summary={'source':str(video),'source_start_frame':start,'processed_frames':count,
        'tile_model':str(DEFAULT_MODEL),'tile_size':[960,720],'tile_imgsz':960,'tile_conf':.05,
        'foot_exception':{'minimum_confidence':.30,'minimum_distinct_views':2,'lower_body_fraction':.65,
                          'requires_confirmed_strict_track':True,'maximum_gap':2},
        'raw_tile_cache_used':str(raw_cache) if raw_cache else None,
        'selected_strategy':strategy,
        'baseline_replay_verified':True,'refinement_seconds':round(elapsed,3),
        'refinement_fps_with_cached_baseline_and_people':None if raw_cache else round(count/elapsed,2),
        'baseline_seconds':metadata['wall_seconds'],
        'serial_ball_stage_fps_excluding_people':None if raw_cache else round(count/(elapsed+metadata['wall_seconds']),2),
        'limits':'Development strategy with two detectors. Timings exclude person inference. Uses no future detections or annotation locations. New evaluation required.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--players-csv',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--raw-cache',type=Path,help='Replay raw tiles from a previous run on the identical interval')
    parser.add_argument('--strategy',choices=['tiled_strict','tiled_feet'],default='tiled_strict')
    args=parser.parse_args();run(args.baseline_dir,args.players_csv,args.output,args.raw_cache,args.strategy)
