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

ROOT = PROJECT_ROOT / 'outputs/ball_training/cone_holdout_001'
USED={'SNMOT-116','SNMOT-120','SNMOT-132','SNMOT-135','SNMOT-138','SNMOT-140','SNMOT-141','SNMOT-150',
      'SNMOT-187','SNMOT-188','SNMOT-189','SNMOT-190','SNMOT-193','SNMOT-197','SNMOT-198','SNMOT-199','SNMOT-200'}


def clip_info(row,index):
    p=row['prefix']; seq=parse_ini(load_entry('test',index[p+'/seqinfo.ini']))
    length=int(seq['seqLength'])
    gt=load_entry('test',index[p+'/gt/gt.txt']).decode('utf-8-sig')
    boxes={}
    for n,tid,x,y,w,h,*_ in csv.reader(gt.splitlines()):
        if int(tid) in row['ball_ids']:
            x,y,w,h=map(float,(x,y,w,h))
            boxes.setdefault(int(n),[]).append([x,y,x+w,y+h])
    return {'sequence':row['name'],'game':row['game'],'action':row['action'],'prefix':p,
            'fps':int(seq['frameRate']),'width':int(seq['imWidth']),'height':int(seq['imHeight']),
            'length':length,'boxes':boxes}


def scout():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    index,rows=catalog('test')
    rows=[r for r in rows if r['game'] in ('7','11') and r['name'] not in USED]
    print([(r['name'],r['game'],r['action']) for r in rows],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        clips=list(pool.map(lambda r:clip_info(r,index),rows))
    jobs=[]
    for clip in clips:
        n=clip['length']; boxes=clip['boxes']; gaps=[]; i=1
        while i<=n:
            if i in boxes:
                i+=1;continue
            a=i
            while i<=n and i not in boxes:i+=1
            if i-a>=10:gaps.append((i-a,a,i-1))
        gaps.sort(reverse=True)
        frames={int((j+.5)*n/6)+1 for j in range(6)}
        for _,a,b in gaps[:2]:frames.update((max(1,a-3),(a+b)//2,min(n,b+3)))
        clip['gaps']=gaps
        for frame in sorted(frames):
            path=f"{clip['prefix']}/img1/{frame:06}.jpg"
            jobs.append((clip,frame,index[path]))
    if sum(i.compress_size+1024 for _,_,i in jobs)>100_000_000:
        raise ValueError('Scout download cap exceeded')
    def fetch(job):
        clip,n,entry=job
        data=load_entry('test',entry)
        im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if im is None:raise ValueError('Bad scout image')
        return clip,n,data,im
    (ROOT/'scout').mkdir(parents=True)
    tiles={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as pool:
        for clip,n,data,im in pool.map(fetch,jobs):
            name=clip['sequence']; (ROOT/'scout'/f'{name}_{n:06}.jpg').write_bytes(data)
            tile=cv2.resize(im,(384,216)); cv2.rectangle(tile,(0,0),(383,21),(20,20,20),-1)
            cv2.putText(tile,f"{name} f{n} {clip['action']} {'ball box' if n in clip['boxes'] else 'unlabeled'}",
                        (3,15),0,.43,(255,255,255),1)
            tiles.setdefault(name,[]).append(tile)
    for name,ts in tiles.items():
        ts += [np.zeros_like(ts[0])]*(-len(ts)%4)
        cv2.imwrite(str(ROOT/f'{name}_scout.jpg'),np.vstack([np.hstack(ts[i:i+4]) for i in range(0,len(ts),4)]))
    (ROOT/'scout.json').write_text(json.dumps({'dataset':DATASET,'revision':REVISION,'clips':clips},indent=2),encoding='utf-8')
    print('Scout images',len(jobs),'from',len(clips),'clips',flush=True)


def fetch_window(sequence,start,count):
    data=json.loads((ROOT/'scout.json').read_text())
    clip=next(c for c in data['clips'] if c['sequence']==sequence)
    if not 1<=start<=start+count-1<=clip['length']:
        raise ValueError('Window outside clip')
    with zipfile.ZipFile(RemoteZip(f'{URL_ROOT}/test.zip',SIZES['test'])) as archive:
        indices={f:archive.getinfo(f"{clip['prefix']}/img1/{f:06}.jpg") for f in range(start,start+count)}
    if sum(i.compress_size+1024 for i in indices.values())>50_000_000:
        raise ValueError('Window download cap exceeded')
    def fetch(pair):
        f,entry=pair
        sample=load_entry('test',entry)
        im=cv2.imdecode(np.frombuffer(sample,np.uint8),cv2.IMREAD_COLOR)
        if im is None or im.shape[:2]!=(clip['height'],clip['width']):
            raise ValueError('Frame dimensions mismatch')
        known=clip['boxes'].get(str(f),[])
        name=f'source/{sequence}/{f:06}.jpg'
        return sample,{'source_frame':f,'visibility':'visible' if len(known)==1 else 'uncertain',
                       'bbox':known[0] if len(known)==1 else None,'image':name,
                       'sha256':hashlib.sha256(sample).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as pool:
        frames=list(pool.map(fetch,indices.items()))
    output=ROOT/'source_manifest.json'
    if output.exists():raise FileExistsError(output)
    (ROOT/'source'/sequence).mkdir(parents=True)
    for sample,r in frames:(ROOT/r['image']).write_bytes(sample)
    fields={k:v for k,v in clip.items() if k not in ('boxes','gaps')}
    fields['frames']=[r for _,r in frames]
    output.write_text(json.dumps({'dataset':DATASET,'revision':REVISION,
          'selection':'Different venue from cone regression clip. Chosen via visual review and provider labels before running either detector.',
          'clips':[fields]},indent=2),encoding='utf-8')
    print(sequence,len(frames),'consecutive frames',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--window',nargs=3,metavar=('SEQUENCE','START','COUNT'))
    args=p.parse_args()
    if args.window:
        s,a,n=args.window
        fetch_window(s,int(a),int(n))
    else:scout()
