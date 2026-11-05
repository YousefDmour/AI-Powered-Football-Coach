"""Compare two detectors with shared causal BallTrack rules on annotated frame sequences."""
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time

from . import run_ball_pilot as pilot

from vision.ball_tracking import BallTrack, outside_people, valid_ball

cv2, torch = pilot.cv2, pilot.torch
np = __import__('numpy')
ROOT = pilot.WORK/'continuous_001'
SOURCE = ROOT/'source_manifest.json'
REVIEW = ROOT/'negative_review.json'
OUT = ROOT/'comparison'
TRAINED = pilot.WORK/'pilot_002_multigame'
read_json, write_json, digest = pilot.read_json, pilot.write_json, pilot.digest


def detect(model, image, cls, size, conf):
    result=model.predict(image,classes=[cls],imgsz=size,conf=conf,device=0,verbose=False)[0]
    return list(zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()))


def elapsed_call(func):
    torch.cuda.synchronize()
    started=time.perf_counter()
    value=func()
    torch.cuda.synchronize()
    return value,(time.perf_counter()-started)*1000


def full_ball(model, image, people, tracker, frame, cls, size):
    raw=detect(model,image,cls,size,.05)
    candidates=[{'bbox':b,'confidence':conf} for b,conf in raw if valid_ball(image,b) and outside_people(b,people)]
    observation=tracker.update(frame,candidates,global_search=True)
    top=max(raw,key=lambda r:r[1],default=None)
    return observation,top,len(candidates)


def ground_truth(row):
    return pilot.center(row['bbox']) if row['visibility']=='visible' else None


def summarize(rows):
    total=correct=wrong=miss=negative=false=uncertain=0
    run=longest=segments=0
    raw_correct=raw_any=raw_false=0
    for row in rows:
        vis=row['visibility']
        observed=row['state']=='observed'
        found=row['error_px'] is not None and row['error_px']<=20
        if vis=='visible':
            total+=1; correct+=int(found); wrong+=int(observed and not found); miss+=int(not observed)
            raw_any+=int(row['raw_error_px'] is not None)
            raw_correct+=int(row['raw_error_px'] is not None and row['raw_error_px']<=20)
        elif vis=='not_visible':
            negative+=1; false+=int(observed); raw_false+=int(row['raw_box'] is not None)
        else:
            uncertain+=1
        if found:
            segments+=int(run==0); run+=1; longest=max(longest,run)
        else:
            run=0
    return {'visible':total,'correct':correct,'wrong':wrong,'missed':miss,'recall':correct/total if total else None,
            'reviewed_negative':negative,'false_marks':false,'uncertain':uncertain,
            'longest_correct_run_frames':longest,'correct_segments':segments,
            'raw_correct':raw_correct,'raw_wrong':raw_any-raw_correct,'raw_negative_marks':raw_false,
            'median_ball_ms':round(statistics.median(r['ball_ms'] for r in rows),2),
            'median_people_plus_ball_ms':round(statistics.median(r['ball_ms']+r['people_ms'] for r in rows),2)}


def render(frame, observation, truth, model_name, row):
    pane=cv2.copyMakeBorder(cv2.resize(frame,(960,540)),34,0,0,0,cv2.BORDER_CONSTANT,value=(25,30,30))
    point=lambda p:(round(p[0]*.5),round(p[1]*.5)+34)
    if truth is not None:
        cv2.circle(pane,point(truth),10,(45,220,65),2,cv2.LINE_AA)
    if observation['state']=='observed':
        cv2.circle(pane,point((observation['x'],observation['y'])),5,(0,165,255),2,cv2.LINE_AA)
    cv2.rectangle(pane,(0,0),(960,33),(25,30,30),-1)
    msg=f"{model_name} | frame {row['source_frame']} | {observation['state']} | green: label | orange: estimate"
    cv2.putText(pane,msg,(12,22),0,.56,(245,245,245),1,cv2.LINE_AA)
    return pane


def run(source_root=ROOT, output_dir=OUT, configurations=None):
    pilot.choose_device()
    source_path=source_root/'source_manifest.json'
    review_path=source_root/'negative_review.json'
    data,review=read_json(source_path),read_json(review_path)
    if digest(source_path)!=review['source_manifest_sha256']:
        raise ValueError('Source changed after visibility review')
    if configurations is None:
        manifest=read_json(TRAINED/'trained_checkpoint.json')
        tuned=TRAINED/manifest['checkpoint']
        if digest(tuned)!=manifest['sha256']:
            raise ValueError('Trained checkpoint changed')
        configurations=[('current_m',pilot.BASELINE,32,1280),('multi_match_n',tuned,0,960)]
    configs=configurations
    if len(configs)!=2 or len({row[0] for row in configs})!=2:
        raise ValueError('Exactly two uniquely named detector configurations are required')
    negatives={clip['sequence']:set(review['clips'][clip['sequence']]['not_visible_frames']) for clip in data['clips']}
    for clip in data['clips']:
        rows=clip['frames']
        numbers=[r['source_frame'] for r in rows]
        if numbers!=list(range(numbers[0],numbers[0]+len(rows))):
            raise ValueError('Non-contiguous frame sequence')
        if not negatives[clip['sequence']] <= {r['source_frame'] for r in rows if r['visibility']=='uncertain'}:
            raise ValueError('Reviewed negative conflicts with provider label')
        for row in rows:
            if digest(source_root/row['image'])!=row['sha256']:
                raise ValueError('Frame hash mismatch')
            if row['source_frame'] in negatives[clip['sequence']]:
                row['visibility']='not_visible'
    output_dir.mkdir()
    protocol={'source_sha256':digest(source_path),'review_sha256':digest(review_path),'center_tolerance_px':20,
              'person_detection':{'model':'yolov8m','class':0,'imgsz':1280,'conf':.2},
              'detector_conf':.05,'weights':{n:{'sha256':digest(p),'class_id':c,'imgsz':s} for n,p,c,s in configs},
              'tracker_sha256':digest(pilot.ROOT/'vision/ball_tracking.py'),
              'scope':'Same BallTrack, pitch/size/color filter and shared person detections. Original consecutive JPEGs at 25 fps. Timing excludes image I/O, drawing and event analysis. Games were evaluated in earlier sparse comparisons; these clips are new. Gaps were deliberately sampled.'}
    write_json(output_dir/'protocol.json',protocol)
    # Separate predictor instances keep their warmed input shapes.
    people_model=pilot.YOLO(str(pilot.BASELINE))
    models={name:pilot.YOLO(str(weights)) for name,weights,_,_ in configs}
    report={'protocol':protocol,'clips':{},'models':{}}
    all_rows={name:[] for name in models}
    for clip in data['clips']:
        name=clip['sequence']
        track={key:BallTrack() for key in models}
        results={key:[] for key in models}
        cuts=[]
        previous=None
        output=output_dir/f'{name}_comparison.mp4'
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),clip['fps'],(1920,574))
        if not writer.isOpened():
            raise OSError('Video output failed')
        try:
            for i,row in enumerate(clip['frames']):
                image=cv2.imread(str(source_root/row['image']))
                if image is None or image.shape[:2]!=(clip['height'],clip['width']):
                    raise ValueError('Frame size mismatch')
                number=row['source_frame']
                truth=ground_truth(row)
                thumbnail=cv2.cvtColor(cv2.resize(image,(160,90)),cv2.COLOR_BGR2HSV)
                histogram=cv2.calcHist([thumbnail],[0,1],None,[24,16],[0,180,0,256])
                cv2.normalize(histogram,histogram)
                if previous is not None and cv2.compareHist(previous,histogram,cv2.HISTCMP_BHATTACHARYYA)>.65:
                    for tracker in track.values(): tracker.reset()
                    cuts.append(number)
                previous=histogram
                if i==0:
                    for _ in range(5):
                        detect(people_model,image,0,1280,.2)
                        for key,_,cls,size in configs: detect(models[key],image,cls,size,.05)
                persons,people_ms=elapsed_call(lambda:detect(people_model,image,0,1280,.2))
                persons=[b for b,_ in persons]
                annotated={}
                for key,_,cls,size in (configs if i%2==0 else list(reversed(configs))):
                    (obs,raw,count),ms=elapsed_call(lambda:full_ball(models[key],image,persons,track[key],number,cls,size))
                    error=math.dist((obs['x'],obs['y']),truth) if truth is not None and obs['state']=='observed' else None
                    raw_error=math.dist(pilot.center(raw[0]),truth) if raw and truth is not None else None
                    info={**obs,'source_frame':number,'visibility':row['visibility'],'error_px':error,
                          'raw_box':raw[0] if raw else None,'raw_error_px':raw_error,
                          'accepted_candidates':count,'ball_ms':ms,'people_ms':people_ms}
                    results[key].append(info)
                    annotated[key]=render(image,obs,truth,key,row)
                writer.write(np.hstack([annotated[key] for key,_,_,_ in configs]))
        finally:
            writer.release()
        cap=cv2.VideoCapture(str(output))
        decoded=0
        while cap.grab(): decoded+=1
        cap.release()
        if decoded!=len(clip['frames']):
            raise ValueError('Output video frame count mismatch')
        report['clips'][name]={'game':clip['game'],'fps':clip['fps'],'scene_cuts':cuts,
                               'video':output.name,'models':{key:summarize(rows) for key,rows in results.items()},
                               'details':results}
        print(name,report['clips'][name]['models'],flush=True)
        for key,rows in results.items(): all_rows[key].extend(rows)
    for key,rows in all_rows.items():
        combined=summarize(rows)
        combined['longest_correct_run_frames']=max(c['models'][key]['longest_correct_run_frames'] for c in report['clips'].values())
        combined['correct_segments']=sum(c['models'][key]['correct_segments'] for c in report['clips'].values())
        report['models'][key]=combined
    write_json(output_dir/'report.json',report)
    print('Combined:',report['models'],flush=True)


if __name__=='__main__':
    run()
