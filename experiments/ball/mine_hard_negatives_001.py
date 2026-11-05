"""Mine suspicious detections from training matches for human-visible review."""
from pathlib import Path
import sys
from . import run_ball_pilot as pilot

from vision.ball_tracking import valid_ball
import numpy as np
cv2=pilot.cv2
ROOT=pilot.WORK/'hard_negatives_001'

def region(center, width, height):
    x,y=center
    left=max(0,min(1920-width,round(x-width/2)))
    top=max(0,min(1080-height,round(y-height/2)))
    return [left,top,left+width,top+height]


def main():
    manifest=pilot.read_json(ROOT/'candidate_manifest.json')
    ck=pilot.WORK/'pilot_002_multigame/training/weights/best.pt'
    pilot.choose_device()
    model=pilot.YOLO(str(ck))
    candidates=[]

    for row in manifest['frames']:
        if row['role']!='train': continue
        path=ROOT/row['path']
        if pilot.digest(path)!=row['sha256']:raise ValueError('Source changed')
        im=cv2.imread(str(path))
        boxes=model.predict(im,classes=[0],imgsz=960,conf=.05,device=0,verbose=False)[0].boxes
        for box,conf in sorted(zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist()),key=lambda v:-v[1]):
            if not valid_ball(im,box): continue
            center=pilot.center(box)
            if any(np.linalg.norm(np.subtract(center,pilot.center(target)))<=40 for target in row['bboxes']):continue
            for w,h in [(960,540),(768,432)]:
                x,y,X,Y=region(center,w,h)
                if not any(min(X,b[2]+15)>max(x,b[0]-15) and min(Y,b[3]+15)>max(y,b[1]-15) for b in row['bboxes']):
                    candidates.append({**row,'bbox':box,'confidence':conf,'region':[x,y,X,Y]})
                    break
            else:
                continue
            break

    dest=ROOT/'review'
    dest.mkdir(exist_ok=False)
    for start in range(0,len(candidates),36):
        tiles=[]
        for i,r in enumerate(candidates[start:start+36],start+1):
            im=cv2.imread(str(ROOT/r['path']))
            cx,cy=pilot.center(r['bbox'])
            a,b,A,B=region((cx,cy),180,130)
            cv2.circle(im,(round(cx),round(cy)),14,(0,210,255),1)
            tile=im[b:B,a:A].copy()
            cv2.rectangle(tile,(0,0),(179,19),(25,25,25),-1)
            cv2.putText(tile,f"{i} {r['id'][6:]} {r['confidence']:.2f}",(3,14),0,.39,(255,255,255),1)
            tiles.append(tile)
        tiles.extend([np.zeros_like(tiles[0])]*(-len(tiles)%6))
        cv2.imwrite(str(dest/f'train_mined_{start//36+1}.jpg'),np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)]))
    pilot.write_json(dest/'mined.json',{'source_sha256':pilot.digest(ROOT/'candidate_manifest.json'),'candidates':candidates})
    print('Mined',len(candidates),'training candidates',flush=True)

    for role in ('test','train'):
        groups={}
        for row in manifest['frames']:
            if row['role']==role:groups.setdefault(row['sequence'],[]).append(row)
        for name,rows in groups.items():
            tiles=[]
            for r in rows:
                im=cv2.imread(str(ROOT/r['path']))
                for x,y,X,Y in r['bboxes']:
                    cv2.rectangle(im,(round(x),round(y)),(round(X),round(Y)),(45,230,75),2)
                tile=cv2.resize(im,(320,180))
                cv2.rectangle(tile,(0,0),(319,20),(25,25,25),-1)
                cv2.putText(tile,f"{r['id']} {r['action']}",(3,14),0,.43,(255,255,255),1)
                tiles.append(tile)
            tiles.extend([np.zeros_like(tiles[0])]*(-len(tiles)%4))
            cv2.imwrite(str(dest/f'{role}_{name}.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))



if __name__ == '__main__':
    main()
