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

ROOT = PROJECT_ROOT / 'outputs/ball_training/hard_negatives_001'
USED = {'SNMOT-060','SNMOT-061','SNMOT-062','SNMOT-063','SNMOT-097','SNMOT-098','SNMOT-099','SNMOT-100',
        'SNMOT-116','SNMOT-120','SNMOT-132','SNMOT-150','SNMOT-187','SNMOT-188','SNMOT-189','SNMOT-190','SNMOT-193','SNMOT-200'}
ACTIONS = ('substitution', 'throw-in', 'throw in', 'foul', 'direct free-kick', 'corner')

def clip_plan(row,index,role):
    p,split = row['prefix'], row['official_split']
    info = parse_ini(load_entry(split,index[p+'/seqinfo.ini']))
    length = int(info['seqLength'])
    gt = load_entry(split,index[p+'/gt/gt.txt']).decode('utf-8-sig')
    boxes = {}
    for frame,tid,x,y,w,h,*_ in csv.reader(gt.splitlines()):
        if int(tid) in row['ball_ids']:
            x,y,w,h=map(float,(x,y,w,h))
            boxes.setdefault(int(frame),[]).append([x,y,x+w,y+h])
    n=10 if role=='train' else 12
    step=length/n
    # Pick positions at evenly spaced times without using detector output.
    frames=sorted({1+int((i+.5)*step) for i in range(n)})
    return [(index[f'{p}/img1/{f:06}.jpg'],
             {'id':f"{row['name']}_{f:06}", 'role':role,'game':row['game'],'sequence':row['name'],
              'frame':f,'action':row['action'],'bboxes':boxes.get(f,[]),
              'path':f"candidates/{role}/{row['name']}_{f:06}.jpg"}) for f in frames]


def build():
    if ROOT.exists():raise FileExistsError(ROOT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        refs={s:pool.submit(catalog,s) for s in ('train','test')}
        refs={s:f.result() for s,f in refs.items()}
    selected=[]
    for split,game,n in [('train','4',5),('train','6',5),('test','8',4),('test','11',3)]:
        rows=sorted((r for r in refs[split][1] if r['game']==game and r['name'] not in USED),
                    key=lambda r:(next((i for i,k in enumerate(ACTIONS) if k in r['action'].lower()),len(ACTIONS)),r['name']))
        for row in rows[:n]:
            selected.append((row,refs[split][0],split))
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        plans=[p for batch in pool.map(lambda job:clip_plan(*job),selected) for p in batch]
    if sum(i.compress_size+1024 for i,r in plans)>65_000_000:
        raise ValueError('Image budget exceeded')
    def fetch(pair):
        i,r=pair
        data=load_entry(r['role'],i)
        im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if im is None or im.shape[:2]!=(1080,1920):
            raise ValueError('Unexpected source resolution')
        r['sha256']=hashlib.sha256(data).hexdigest()
        return r,data,im
    for role in ('train','test'):
        (ROOT/'candidates'/role).mkdir(parents=True,exist_ok=True)
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for row,data,im in pool.map(fetch,plans):
            (ROOT/row['path']).write_bytes(data)
            records.append(row)
    (ROOT/'candidate_manifest.json').write_text(json.dumps({'dataset':DATASET,'revision':REVISION,'frames':records},indent=2),encoding='utf-8')
    print([(row['name'],row['game'],row['action']) for row,_,_ in selected],flush=True)
    print('Downloaded',len(records),'frames',flush=True)

if __name__=='__main__':build()
