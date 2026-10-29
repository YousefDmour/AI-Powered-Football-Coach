"""Score a frozen same-video segment and record the possession eligibility gate."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from project_paths import PROJECT_ROOT
from vision.evaluate_ball import evaluate as evaluate_ball
from vision.evaluate_tracking import evaluate as evaluate_players
from vision.validation import validation_gate


def build_report(manifest_path, ball_dir, player_dir, output):
    manifest_path,ball_dir,player_dir,output=map(Path,(manifest_path,ball_dir,player_dir,output))
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    fingerprints={**manifest['sha256'],**manifest['annotation_sha256']}
    mismatches=[name for name,digest in fingerprints.items()
                if hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest()!=digest]
    if mismatches:raise ValueError(f'Frozen files changed: {mismatches}')
    window=manifest['window'];start=window['start_frame'];count=window['frame_count']
    table=pd.read_csv(ball_dir/'ball_tracks.csv')
    if table.source_frame.tolist()!=list(range(start,start+count)):
        raise ValueError('Ball output does not match the frozen original-frame interval')
    for directory in (ball_dir,player_dir):
        summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        if summary['source_start_frame']!=start or summary['processed_frames']!=count:
            raise ValueError('Summary interval does not match the frozen window')
        if Path(summary['source']).resolve()!=(PROJECT_ROOT/window['source']).resolve():
            raise ValueError('Summary source does not match the frozen source')
    parameters=manifest['parameters']
    ball_summary=json.loads((ball_dir/'summary.json').read_text(encoding='utf-8'))
    player_summary=json.loads((player_dir/'summary.json').read_text(encoding='utf-8'))
    if ball_summary['search_mode']!=parameters['search']:
        raise ValueError('Ball search mode differs from frozen settings')
    if (player_summary['imgsz']!=parameters['person_imgsz']
            or player_summary['conf']!=parameters['person_conf']
            or player_summary['track_buffer']!=parameters['buffer']):
        raise ValueError('Player parameters differ from frozen settings')
    ball_labels=json.loads((PROJECT_ROOT/'data/annotations/heldout_ball_300_539.json').read_text(encoding='utf-8'))
    player_labels=json.loads((PROJECT_ROOT/'data/annotations/heldout_players_300_539.json').read_text(encoding='utf-8'))
    ball=evaluate_ball(table,ball_labels['frames'],parameters['ball_center_tolerance'])
    players=evaluate_players(pd.read_csv(player_dir/'tracks.csv'),player_labels['frames'])
    ball['limits']='Frozen same-video temporal holdout; assistant-reviewed labels, not independently human-labeled or an independent-match benchmark. Uncertain samples are excluded.'
    players['limits']='Frozen same-video temporal holdout with approximate assistant-reviewed boxes. Sparse identity checks do not detect all switches and are not IDF1/HOTA.'
    result={'window':window,'frozen_file_hashes_verified':True,'ball':ball,'players':players,
            'gate':validation_gate(ball,players),
            'limits':'Same video, assistant-reviewed labels, approximate boxes/centers. Five negative samples come from one short occlusion episode. No independent-match or independent-human validation.',
            'timing':{'ball_fps_with_cached_people':ball_summary['processing_fps'],
                      'people_fps':player_summary['processing_fps']}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ball_dir/'evaluation.json').write_text(json.dumps(ball,indent=2),encoding='utf-8')
    (player_dir/'evaluation.json').write_text(json.dumps(players,indent=2),encoding='utf-8')
    print(json.dumps({'ball':{k:v for k,v in ball.items() if k!='details'},
                      'players':{k:v for k,v in players.items() if k!='details'},
                      'gate':result['gate']},indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=PROJECT_ROOT/'data/validation/heldout_300_539.json')
    parser.add_argument('--ball-dir',type=Path,default=PROJECT_ROOT/'outputs/vision/validation_300_539/ball')
    parser.add_argument('--player-dir',type=Path,default=PROJECT_ROOT/'outputs/vision/validation_300_539/players')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/vision/validation_300_539/report.json')
    args=parser.parse_args();build_report(args.manifest,args.ball_dir,args.player_dir,args.output)


if __name__=='__main__':main()
