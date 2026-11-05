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

USED = {'SNMOT-116','SNMOT-132','SNMOT-187','SNMOT-188','SNMOT-189','SNMOT-190'}
COUNT = 125

def candidate(row, index):
    prefix = row['prefix']
    seq = parse_ini(load_entry('test',index[prefix+'/seqinfo.ini']))
    w,h,n,fps = (int(seq[k]) for k in ('imWidth','imHeight','seqLength','frameRate'))
    gt = load_entry('test', index[prefix+'/gt/gt.txt']).decode('utf-8-sig')
    boxes = {}
    for f,tid,x,y,bw,bh,*_ in csv.reader(gt.splitlines()):
        if int(tid) in row['ball_ids']:
            x,y,bw,bh=map(float,(x,y,bw,bh))
            if bw>0 and bh>0:
                boxes.setdefault(int(f),[]).append([x,y,x+bw,y+bh])
    # Require a sizable missing-label episode with labeled footage nearby.
    gaps=[]
    i=1
    while i<=n:
        if i in boxes:
            i+=1
            continue
        first=i
        while i<=n and i not in boxes:
            i+=1
        if 16<=i-first<=75:
            start=max(1,min(n-COUNT+1,first-30))
            keys=range(start,start+COUNT)
            if sum(len(boxes.get(f,[]))==1 for f in keys)>=42 and start<=first and i-1<start+COUNT:
                gaps.append((i-first,first,i-1,start))
    if not gaps:
        return None
    _,first,last,start=max(gaps)
    result={'sequence':row['name'],'game':row['game'],'action':row['action'],'width':w,'height':h,'fps':fps,
            'selected_gap':[first,last],'frames':[]}
    for f in range(start,start+COUNT):
        name=f'{prefix}/img1/{f:06}.jpg'
        assert name in index,name
        candidates=boxes.get(f,[])
        status='visible' if len(candidates)==1 else 'uncertain'
        result['frames'].append({'source_frame':f,'visibility':status,'bbox':candidates[0] if status=='visible' else None,
                                 'image':f"source/{row['name']}/{f:06}.jpg",'member':name})
    return result

def prepare(batch):
    root = PROJECT_ROOT / 'outputs/ball_training' / f'continuous_{batch}'
    used = USED if batch == '001' else USED | {'SNMOT-120', 'SNMOT-193'}
    games = ('11', '7', '8') if batch == '001' else ('8', '11', '7')
    if root.exists():
        raise FileExistsError(root)
    index,rows=catalog('test')
    rows=[r for r in rows if r['name'] not in used]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        options=[c for c in pool.map(lambda r:candidate(r,index), rows) if c]
    picks=[]
    for game in games:
        choices=[r for r in options if r['game']==game]
        if choices:
            picks.append(max(choices,key=lambda c:c['selected_gap'][1]-c['selected_gap'][0]))
        if len(picks)==2:
            break
    if len(picks)<2:
        raise ValueError('Two qualifying clips required')
    print([(r['sequence'],r['game'],r['action'],r['selected_gap'],r['frames'][0]['source_frame']) for r in picks],flush=True)
    jobs=[f for p in picks for f in p['frames']]
    if sum(index[f['member']].compress_size+1024 for f in jobs)>100_000_000:
        raise ValueError('Download cap exceeded')
    def fetch(f):
        data=load_entry('test',index[f['member']])
        image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Invalid JPEG')
        f['sha256']=hashlib.sha256(data).hexdigest()
        return data
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as pool:
        images=list(pool.map(fetch,jobs))
    root.mkdir(parents=True)
    for f,data in zip(jobs,images):
        path=root/f['image']
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    for p in picks:
        # Review only frames where ground truth has no single ball.
        preview=[]
        for f in p['frames']:
            if f['visibility']!='uncertain':
                continue
            im=cv2.imread(str(root/f['image']))
            tile=cv2.resize(im,(384,216))
            cv2.rectangle(tile,(0,0),(383,19),(20,20,20),-1)
            cv2.putText(tile,str(f['source_frame']),(6,14),0,.47,(255,255,255),1)
            preview.append(tile)
        for start in range(0,len(preview),24):
            tiles=preview[start:start+24]
            tiles += [np.zeros_like(tiles[0])]*(-len(tiles)%4)
            sheet=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
            cv2.imwrite(str(root/f"review_{p['sequence']}_{start//24+1}.jpg"),sheet)
    (root/'source_manifest.json').write_text(json.dumps({'dataset':DATASET,'revision':REVISION,
          'selection':'Two contiguous 125-frame windows from unused test clips. Deliberately includes missing-label intervals; test games previously appeared in sparse evaluation.',
          'clips':picks},indent=2),encoding='utf-8')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare a recorded continuous SoccerNet evaluation batch.')
    parser.add_argument('batch', choices=('001', '002'))
    prepare(parser.parse_args().batch)
