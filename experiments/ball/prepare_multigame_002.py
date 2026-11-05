import concurrent.futures
import csv
import hashlib
import json
import math
import zipfile

import cv2
import numpy as np

from project_paths import PROJECT_ROOT
from .soccernet_source import DATASET, REVISION, SIZES, URL_ROOT, RemoteZip, catalog, load_entry, parse_ini

ROOT = PROJECT_ROOT / 'outputs/ball_dataset/soccernet_multigame_002'
PREVIOUS_GAMES = {'7', '8'}


def make_plan(game_rows, index, split, sample_count, seq_count):
    # Prefer two distinct labeled actions within each game.
    candidates, action_seen = [], set()
    for row in game_rows:
        if row['action'] not in action_seen:
            candidates.append(row)
            action_seen.add(row['action'])
    candidates.extend(r for r in game_rows if r not in candidates)
    plans = []
    for row in candidates:
        official, p = row['official_split'], row['prefix']
        entries = index[official]
        ini = parse_ini(load_entry(official, entries[p + '/seqinfo.ini']))
        width, height, length = (int(ini[k]) for k in ('imWidth', 'imHeight', 'seqLength'))
        gt = load_entry(official, entries[p + '/gt/gt.txt']).decode('utf-8-sig')
        boxes = {}
        for f, tid, x, y, w, h, *_ in csv.reader(gt.splitlines()):
            if int(tid) not in row['ball_ids']:
                continue
            x, y, w, h = map(float, (x, y, w, h))
            if not all(map(math.isfinite, (x, y, w, h))) or min(w,h) <= 0:
                raise ValueError('Invalid annotated box')
            box = [max(0,x), max(0,y), min(width,x+w), min(height,y+h)]
            if box[0] < box[2] and box[1] < box[3]:
                boxes.setdefault(int(f), []).append(box)
        picks = [1 + int((j+0.5)*length/sample_count) for j in range(sample_count)]
        if sum(f in boxes for f in picks) < sample_count // 2:
            continue
        images = sorted(k for k in entries if k.startswith(p + '/img1/') and k.endswith('.jpg'))
        assert len(images) == length
        plan = []
        for f in picks:
            if f not in boxes and split != 'test':
                continue
            name = images[f-1]
            assert int(Path(name).stem) == f
            identifier = f"{row['name']}_{f:06}"
            plan.append((entries[name], {'id':identifier, 'image':f'images/{split}/{identifier}.jpg',
                       'split':split,'official_split':official,'sequence':row['name'],
                       'action':row['action'], 'source_group':'soccernet_game_'+row['game'],
                       'source_frame':f,'width':width,'height':height,'bboxes':boxes.get(f, []),
                       'status':'visible' if f in boxes else 'uncertain'}))
        plans.extend(plan)
        seq_count -= 1
        if seq_count == 0:
            break
    if not plans:
        raise ValueError(f'No eligible clips for {game_rows[0]}')
    return plans


def download():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {s:pool.submit(catalog,s) for s in ('train','test')}
        catalogs = {s:f.result() for s,f in pending.items()}
    index = {s:v[0] for s,v in catalogs.items()}
    by_split = {}
    for s, (_, rows) in catalogs.items():
        groups = {}
        for row in rows:
            if row['game'] not in PREVIOUS_GAMES:
                groups.setdefault(row['game'], []).append(row)
        by_split[s] = groups
    test_games = [('test', g) for g in sorted(by_split['test'], key=int)]
    test_ids = {g for _,g in test_games}
    development = [g for g in sorted(by_split['train'], key=int) if g not in test_ids]
    if not test_games or len(development)<3:
        raise ValueError(f'Not enough distinct source games: {test_games}, {development}')
    val_game = development[-1]
    train_games = development[:-1]
    print('Assignment:', {'train':train_games,'val':[val_game],'test':test_games}, flush=True)
    tasks = [('train','train',g,30,4) for g in train_games]
    tasks += [('val','train',val_game,24,3)]
    tasks += [('test',official,g,20,4) for official,g in test_games]
    def plan(args):
        split, official, game, samples, sequences = args
        return make_plan(by_split[official][game], index, split, samples, sequences)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        records = [r for group in pool.map(plan,tasks) for r in group]
    total = sum(i.compress_size + 1024 for i,r in records)
    if total > 150_000_000:
        raise ValueError('Image download exceeds cap')
    print('Fetching',len(records),'images;',round(total/1e6,1),'MB',flush=True)
    def fetch(pair):
        entry, row = pair
        data = load_entry(row['official_split'],entry)
        image = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (row['height'],row['width']):
            raise ValueError(f"Invalid image dimensions: {row['id']}")
        row['sha256'] = hashlib.sha256(data).hexdigest()
        return row,data
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as pool:
        results = []
        for result in pool.map(fetch,records):
            results.append(result)
            if len(results)%50==0:
                print('Downloaded',len(results),'/',len(records),flush=True)
    if len({r['sha256'] for r,_ in results}) != len(results):
        raise ValueError('Exact duplicate source frames')
    for split in ('train','val','test'):
        (ROOT/'images'/split).mkdir(parents=True)
        (ROOT/'labels'/split).mkdir(parents=True)
    for row,data in results:
        (ROOT/row['image']).write_bytes(data)
        if not row['bboxes']:
            continue
        w,h = row['width'],row['height']
        lines = [f'0 {(x+X)/2/w:.8f} {(y+Y)/2/h:.8f} {(X-x)/w:.8f} {(Y-y)/h:.8f}\n'
                 for x,y,X,Y in row['bboxes']]
        (ROOT/'labels'/row['split']/f"{row['id']}.txt").write_text(''.join(lines),encoding='utf-8')
    manifest={'dataset':DATASET,'revision':REVISION,'class_names':['ball'],
              'sampling':'Fixed evenly spaced sample per clip, up to two clips per game with different actions preferred. No detector predictions used.',
              'split_policy':'Match IDs are disjoint. Games 7 and 8 were used for pilot evaluation and are excluded. Missing ball boxes are not used as negatives.',
              'frames':[r for r,_ in results]}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print({s:sum(r['split']==s for r,_ in results) for s in ('train','val','test')},flush=True)


if __name__ == '__main__':
    download()
