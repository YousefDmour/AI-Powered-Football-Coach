"""Train a second ball detector with game-disjoint SoccerNet data."""
from pathlib import Path
import math
import shutil
import statistics
import time

from . import run_ball_pilot as pilot

ROOT = pilot.ROOT
RUN = pilot.WORK / 'pilot_002_multigame'
EXTERNAL = ROOT / 'outputs/ball_dataset/soccernet_multigame_002'
LOCAL = pilot.SOURCE
FIRST = pilot.WORK / 'pilot_001'
ORIGINAL = pilot.PRETRAINED
BASELINE = pilot.BASELINE
read_json, write_json, digest = pilot.read_json, pilot.write_json, pilot.digest


def verified_external():
    manifest = read_json(EXTERNAL / 'manifest.json')
    groups = {split: {r['source_group'] for r in manifest['frames'] if r['split'] == split}
              for split in ('train', 'val', 'test')}
    if (any(not group for group in groups.values()) or
            sum(len(group) for group in groups.values()) != len(set.union(*groups.values())) or
            set.union(*groups.values()) & {'soccernet_game_7', 'soccernet_game_8'}):
        raise ValueError('Evaluation game overlap')
    return manifest, groups


def prepare():
    manifest, groups = verified_external()
    local = read_json(LOCAL / 'manifest.json')
    development = RUN / 'development'
    if RUN.exists():
        raise FileExistsError(RUN)
    jobs = []
    for image in sorted((LOCAL/'images').glob('*.jpg')):
        label = LOCAL/'labels'/f'{image.stem}.txt'
        jobs.append(('train', local['source_family'], image, label, digest(image)))
    if len(jobs) != local['exported_images']:
        raise ValueError('Incomplete local dataset')
    for row in manifest['frames']:
        if row['split'] not in ('train','val'):
            continue
        if row['status'] != 'visible':
            raise ValueError('Unverified negative SoccerNet label')
        image = EXTERNAL / row['image']
        label = EXTERNAL/'labels'/row['split']/f"{row['id']}.txt"
        jobs.append((row['split'],row['source_group'],image,label,row['sha256']))
    if len({j[-1] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate development frames')
    holdout_hashes = {r['sha256'] for r in manifest['frames'] if r['split']=='test'}
    previous = read_json(pilot.HOLDOUT/'manifest.json')
    holdout_hashes.update(r['sha256'] for r in previous['frames'])
    if holdout_hashes & {j[-1] for j in jobs}:
        raise ValueError('Duplicate image across development and evaluation')
    records = []
    for split,group,image,label,sha in jobs:
        if digest(image) != sha:
            raise ValueError(f'Image hash mismatch: {image}')
        for kind,source in [('images',image),('labels',label)]:
            folder = development/kind/split
            folder.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,folder/source.name)
        records.append(dict(split=split,source_group=group,image=image.name,image_sha256=sha,
                            label_sha256=digest(label),positive=bool(label.read_text().strip())))
    write_json(RUN/'split.json', {'external_manifest_sha256': digest(EXTERNAL/'manifest.json'),
                                 'previous_holdout_sha256': pilot.HOLDOUT_SHA,
                                 'external_groups': {k:sorted(v) for k,v in groups.items()},
                                 'development':records})
    print({s:{'images':sum(r['split']==s for r in records),
              'positive':sum(r['split']==s and r['positive'] for r in records)}
           for s in ('train','val')},flush=True)
    path = development/'data.yaml'
    path.write_text(pilot.yaml.safe_dump({'path':str(development),'train':'images/train','val':'images/val',
                                         'names':{0:'ball'}}),encoding='utf-8')
    return path


def train():
    device = pilot.choose_device()
    data = prepare()
    args = read_json(FIRST/'experiment.json')['training']
    args.update(data=str(data),device=device,project=str(RUN))
    write_json(RUN/'experiment.json',{'initial_weights_sha256':digest(ORIGINAL),'training':args,
                                      'selection':'best.pt based only on validation game',
                                      'comparison':'Fixed confidence 0.05 and 20-pixel center tolerance. Fresh game first; old benchmark is historical.'})
    started=time.perf_counter()
    pilot.YOLO(str(ORIGINAL)).train(**args)
    weight = RUN/'training/weights/best.pt'
    write_json(RUN/'trained_checkpoint.json',{'checkpoint':str(weight.relative_to(RUN)),
                                             'sha256':digest(weight),'training_seconds':round(time.perf_counter()-started,2)})


def load_test(path, rows, development):
    frames=[]
    for row in rows:
        image = path/row['image']
        if digest(image) != row['sha256'] or row['sha256'] in development:
            raise ValueError(f'Bad holdout image: {image}')
        if row['status']=='visible':
            pixels=pilot.cv2.imread(str(image))
            if pixels is None or pixels.shape[:2] != (row['height'],row['width']):
                raise ValueError(f'Invalid holdout image: {image}')
            frames.append((row,pixels))
    return frames


def compare(frames, models, device):
    report={}
    for name,weights,class_id,size in models:
        model = pilot.YOLO(str(weights))
        if model.names[class_id] not in ('sports ball','ball'):
            raise ValueError(f'Wrong class for {name}')
        options=dict(imgsz=size,conf=0.05,classes=[class_id],device=device,verbose=False)
        for _,image in frames[:5]:
            model.predict(image,**options)
        rows,elapsed=[],[]
        for annotation,image in frames:
            pilot.torch.cuda.synchronize()
            started=time.perf_counter()
            result=model.predict(image,**options)[0]
            pilot.torch.cuda.synchronize()
            elapsed.append((time.perf_counter()-started)*1000)
            coords,conf=result.boxes.xyxy.cpu(),result.boxes.conf.cpu()
            box=coords[int(conf.argmax())].tolist() if len(coords) else None
            targets=annotation['bboxes']
            error=min(math.dist(pilot.center(box),pilot.center(b)) for b in targets) if box else None
            overlap=max(pilot.iou(box,b) for b in targets) if box else None
            rows.append(dict(id=annotation['id'],sequence=annotation['sequence'],box=box,
                             confidence=float(conf.max()) if len(conf) else None,center_error=error,iou=overlap))
        report[name]={**pilot.score(rows),'imgsz':size,'median_ms':round(statistics.median(elapsed),2),
                      'p95_ms':round(sorted(elapsed)[math.ceil(.95*len(elapsed))-1],2),'details':rows}
        print(name,{k:v for k,v in report[name].items() if k!='details'},flush=True)
        del model
        pilot.torch.cuda.empty_cache()
    return report


def evaluate():
    device=pilot.choose_device()
    split=read_json(RUN/'split.json')
    if (digest(EXTERNAL/'manifest.json')!=split['external_manifest_sha256'] or
            digest(pilot.HOLDOUT/'manifest.json')!=pilot.HOLDOUT_SHA):
        raise ValueError('Frozen evaluation manifest changed')
    checkpoints=[]
    for folder in (FIRST,RUN):
        checkpoint=read_json(folder/'trained_checkpoint.json')
        weights=folder/checkpoint['checkpoint']
        if digest(weights)!=checkpoint['sha256']:
            raise ValueError('Checkpoint hash changed')
        checkpoints.append(weights)
    models=[('pretrained_n',ORIGINAL,32,960),('single_match_n',checkpoints[0],0,960),
            ('multi_match_n',checkpoints[1],0,960),('current_full_m',BASELINE,32,1280)]
    manifest,_=verified_external()
    development={r['image_sha256'] for r in split['development']}
    report={'scope':'Top-confidence ball detection on full images. Timing includes inference pre/post-processing on the same GPU. No tracking or absent-ball test samples.',
            'thresholds':{'confidence':0.05,'center_tolerance_pixels':20},
            'model_sha256':{name:digest(weights) for name,weights,_,_ in models},'datasets':{}}
    fresh=[r for r in manifest['frames'] if r['split']=='test']
    older=read_json(pilot.HOLDOUT/'manifest.json')['frames']
    for name,path,rows in [('unseen_game',EXTERNAL,fresh),('historical_two_games',pilot.HOLDOUT,older)]:
        frames=load_test(path,rows,development)
        print(name,len(frames),'visible frames',flush=True)
        report['datasets'][name]={'uncertain_excluded':len(rows)-len(frames),'models':compare(frames,models,device)}
    write_json(RUN/'comparison.json',report)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['train','evaluate'])
    {'train':train,'evaluate':evaluate}[parser.parse_args().stage]()
