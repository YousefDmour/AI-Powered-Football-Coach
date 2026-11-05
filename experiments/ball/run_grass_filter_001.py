"""Evaluate a grass-only relaxation without retuning on the second clip set."""
import argparse
import math
from pathlib import Path
from statistics import median
import time

from . import run_continuous_001 as bench

cv2, np = bench.cv2, bench.np
WORK=bench.pilot.WORK
DEV=WORK/'continuous_001'
TEST=WORK/'continuous_002'
OUT=WORK/'grass_filter_001'
MODES={'strict':None,'no_grass':0.05,'off_grass_conf15':0.15,'off_grass_conf30':0.30}
def appearance(frame,box):
    x,y,X,Y=map(float,box)
    w,h=X-x,Y-y
    size=3<=w<=36 and 3<=h<=36 and .4<=w/h<=2.5
    if not size:
        return False,None,None
    cx,cy=int((x+X)/2),int((y+Y)/2)
    radius=max(14,int(max(w,h)*2))
    height,width=frame.shape[:2]
    crop=frame[max(0,cy-radius):min(height,cy+radius),max(0,cx-radius):min(width,cx+radius)]
    patch=frame[max(0,int(y)):min(height,int(Y)+1),max(0,int(x)):min(width,int(X)+1)]
    if not crop.size or not patch.size:
        return True,0.,0.
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    white=((hsv[:,:,1]<=100)&(hsv[:,:,2]>=155)).mean()
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    grass=((hsv[:,:,0]>=35)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>=45)&(hsv[:,:,2]>=35)).mean()
    return True,float(white),float(grass)


def load_data(root):
    manifest=root/'source_manifest.json'
    review_path=root/'negative_review.json'
    source,review=bench.read_json(manifest),bench.read_json(review_path)
    if bench.digest(manifest)!=review['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    for clip in source['clips']:
        frames=clip['frames']
        expected=list(range(frames[0]['source_frame'],frames[0]['source_frame']+len(frames)))
        if [r['source_frame'] for r in frames]!=expected:
            raise ValueError('Discontinuous source frames')
        absent=set(review['clips'][clip['sequence']]['not_visible_frames'])
        if not absent<={r['source_frame'] for r in frames if r['visibility']=='uncertain'}:
            raise ValueError('Negative review conflicts with source')
        for r in frames:
            if bench.digest(root/r['image'])!=r['sha256']:
                raise ValueError('Image hash mismatch')
            if r['source_frame'] in absent:
                r['visibility']='not_visible'
    return source


def permitted(white,grass,conf,off_grass_min):
    if white < .20:
        return False
    return grass >= .35 or (off_grass_min is not None and conf >= off_grass_min)


def evaluate(root,modes,dest,render=False):
    source=load_data(root)
    checkpoint=bench.read_json(bench.TRAINED/'trained_checkpoint.json')
    weight=bench.TRAINED/checkpoint['checkpoint']
    if bench.digest(weight)!=checkpoint['sha256']:
        raise ValueError('Checkpoint changed')
    person_model=bench.pilot.YOLO(str(bench.pilot.BASELINE))
    model=bench.pilot.YOLO(str(weight))
    report={'source_sha256':bench.digest(root/'source_manifest.json'),
            'review_sha256':bench.digest(root/'negative_review.json'),'trained_sha256':checkpoint['sha256'],
            'clips':{},'models':{}}
    combined={m:[] for m in modes}
    for clip in source['clips']:
        track={m:bench.BallTrack() for m in modes}
        rows={m:[] for m in modes}
        sequence=clip['sequence']
        prev=None
        video=None
        if render:
            path=dest/f'{sequence}_comparison.mp4'
            video=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),clip['fps'],(1920,574))
            if not video.isOpened(): raise OSError('Video writer failed')
        try:
            for i,r in enumerate(clip['frames']):
                image=cv2.imread(str(root/r['image']))
                if image is None or image.shape[:2]!=(clip['height'],clip['width']):
                    raise ValueError('Image unavailable')
                number=r['source_frame']
                truth=bench.ground_truth(r)
                thumb=cv2.cvtColor(cv2.resize(image,(160,90)),cv2.COLOR_BGR2HSV)
                hist=cv2.calcHist([thumb],[0,1],None,[24,16],[0,180,0,256]); cv2.normalize(hist,hist)
                if prev is not None and cv2.compareHist(prev,hist,cv2.HISTCMP_BHATTACHARYYA)>.65:
                    for t in track.values(): t.reset()
                prev=hist
                if i==0:
                    for _ in range(5):
                        bench.detect(person_model,image,0,1280,.2)
                        bench.detect(model,image,0,960,.05)
                persons,people_ms=bench.elapsed_call(lambda:bench.detect(person_model,image,0,1280,.2))
                raw,ball_ms=bench.elapsed_call(lambda:bench.detect(model,image,0,960,.05))
                people=[b for b,_ in persons]
                top=max(raw,key=lambda candidate:candidate[1],default=None)
                top_error=math.dist(bench.pilot.center(top[0]),truth) if truth is not None and top else None
                began=time.perf_counter()
                candidates=[]
                for box,conf in raw:
                    if not bench.outside_people(box,people):
                        continue
                    size,white,grass=appearance(image,box)
                    if size: candidates.append((box,conf,white,grass))
                features_ms=(time.perf_counter()-began)*1000
                panes=[]
                for key in modes:
                    began=time.perf_counter()
                    kept=[{'bbox':box,'confidence':conf} for box,conf,white,grass in candidates
                          if permitted(white,grass,conf,MODES[key])]
                    obs=track[key].update(number,kept,global_search=True)
                    measured_ms=ball_ms+features_ms+(time.perf_counter()-began)*1000
                    error=math.dist((obs['x'],obs['y']),truth) if truth is not None and obs['state']=='observed' else None
                    rows[key].append({**obs,'source_frame':number,'visibility':r['visibility'],'error_px':error,
                                      'raw_box':top[0] if top else None,'raw_error_px':top_error,
                                      'ball_ms':measured_ms,'people_ms':people_ms})
                    if video: panes.append(bench.render(image,obs,truth,key,r))
                if video: video.write(np.hstack(panes if len(panes)==2 else panes*2))
        finally:
            if video: video.release()
        if video:
            cap=cv2.VideoCapture(str(path)); count=0
            while cap.grab(): count+=1
            cap.release()
            if count!=len(clip['frames']): raise ValueError('Incomplete comparison video')
        report['clips'][sequence]={'fps':clip['fps'],'models':{k:bench.summarize(v) for k,v in rows.items()},'details':rows}
        for key in modes: combined[key].extend(rows[key])
    for key in modes:
        r=bench.summarize(combined[key])
        r['longest_correct_run_frames']=max(c['models'][key]['longest_correct_run_frames'] for c in report['clips'].values())
        r['correct_segments']=sum(c['models'][key]['correct_segments'] for c in report['clips'].values())
        report['models'][key]=r
    bench.write_json(dest/'report.json',report)
    print(report['models'],flush=True)
    return report


def qualifies(candidate,strict):
    return (candidate['correct']>strict['correct'] and candidate['false_marks']<=strict['false_marks']
            and candidate['wrong']<=strict['wrong']+2)


def develop():
    OUT.mkdir()
    report=evaluate(DEV,list(MODES),OUT)
    before=bench.read_json(DEV/'comparison/report.json')
    for seq,clip in report['clips'].items():
        actual=clip['details']['strict']
        expected=before['clips'][seq]['details']['multi_match_n']
        if [(r['state'],r['x'],r['y']) for r in actual]!=[(r['state'],r['x'],r['y']) for r in expected]:
            raise ValueError('Strict replay differs from previous experiment')
    stats=report['models']
    strict=stats['strict']
    passed=[k for k in MODES if k!='strict' and qualifies(stats[k],strict)]
    if passed:
        selected=max(passed,key=lambda k:(stats[k]['correct'],-stats[k]['wrong'],MODES[k]))
    else:
        selected=min((k for k in MODES if k!='strict'),key=lambda k:(stats[k]['false_marks'],stats[k]['wrong'],-stats[k]['correct']))
    selection={'selected':selected,'development_passed':selected in passed,'settings':MODES,
               'gate':'More correct positions; no additional reviewed-negative marks; at most two additional wrong positions on visible frames.',
               'development_report_sha256':bench.digest(OUT/'report.json'),
               'runner_sha256':bench.digest(Path(__file__)),
               'test_manifest_sha256':bench.digest(TEST/'source_manifest.json'),
               'test_review_sha256':bench.digest(TEST/'negative_review.json')}
    bench.write_json(OUT/'selection.json',selection)
    print('Frozen selection:',selected,'development gate:',selected in passed,flush=True)


def verify():
    selection=bench.read_json(OUT/'selection.json')
    for path,key in [(OUT/'report.json','development_report_sha256'),(Path(__file__),'runner_sha256'),
                     (TEST/'source_manifest.json','test_manifest_sha256'),(TEST/'negative_review.json','test_review_sha256')]:
        if bench.digest(path)!=selection[key]: raise ValueError(f'Frozen input changed: {path}')
    dest=OUT/'prospective'
    dest.mkdir()
    selected=selection['selected']
    report=evaluate(TEST,['strict',selected],dest,render=True)
    passed=qualifies(report['models'][selected],report['models']['strict'])
    verdict={'selected':selected,'development_gate_passed':selection['development_passed'],
             'prospective_gate_passed':passed,'candidate_passed_both':selection['development_passed'] and passed}
    bench.write_json(dest/'verdict.json',verdict)
    print(verdict,flush=True)


if __name__=='__main__':
    bench.pilot.choose_device()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['develop','verify'])
    {'develop':develop,'verify':verify}[parser.parse_args().stage]()
