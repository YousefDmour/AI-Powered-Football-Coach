"""Retrain the multi-game ball detector with reviewed hard negative crops."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time

from . import run_ball_pilot as pilot
from . import run_ball_multigame_002 as previous

ROOT=pilot.WORK/'hard_negatives_001'
RUN=pilot.WORK/'pilot_003_hard_negatives'
PRIOR=previous.RUN
REVIEWED = Path(__file__).with_name('reviews') / 'hard_negative_crops.json'


def prepare():
    if RUN.exists():
        raise FileExistsError(RUN)
    review=pilot.read_json(REVIEWED)
    manifest_path=ROOT/'candidate_manifest.json'
    if pilot.digest(manifest_path)!=review['source_manifest_sha256']:
        raise ValueError('Source review changed')
    prior=pilot.read_json(PRIOR/'split.json')
    prior_images={r['image_sha256'] for r in prior['development']}
    manifest=pilot.read_json(manifest_path)
    holdout={r['sha256'] for r in manifest['frames'] if r['role']=='test'}
    if prior_images & holdout:
        raise ValueError('Held-out image matches previous training image')
    existing=PRIOR/'development'
    dest=RUN/'development'
    for record in prior['development']:
        split=record['split']; name=record['image']
        paths=[('images',name,'image_sha256'),('labels',Path(name).stem+'.txt','label_sha256')]
        for kind,filename,key in paths:
            source=existing/kind/split/filename
            if pilot.digest(source)!=record[key]:
                raise ValueError('Previous dataset changed')
            folder=dest/kind/split
            folder.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,folder/filename)
    accepted=[]
    sources={row['id']:row for row in manifest['frames']}
    for example in review['examples']:
        number=example['review_index']
        row=sources[example['source']]
        if row['sha256'] != example['source_sha256']:
            raise ValueError('Reviewed image changed')
        source=ROOT/row['path']
        if row['role']!='train' or row['game'] not in ('4','6') or pilot.digest(source)!=row['sha256'] or row['sha256'] in holdout:
            raise ValueError('Invalid negative provenance')
        x,y,X,Y=example['region']
        if any(min(X,b[2]+15)>max(x,b[0]-15) and min(Y,b[3]+15)>max(y,b[1]-15) for b in row['bboxes']):
            raise ValueError('Known ball overlaps negative crop')
        im=pilot.cv2.imread(str(source))
        crop=im[y:Y,x:X]
        stem=f"hardneg_{row['id']}"
        image=dest/'images/train'/f'{stem}.jpg'
        if not pilot.cv2.imwrite(str(image),crop,[pilot.cv2.IMWRITE_JPEG_QUALITY,95]):
            raise OSError('Failed to save negative crop')
        (dest/'labels/train'/f'{stem}.txt').write_text('',encoding='utf-8')
        accepted.append({'review_index':number,'source':row['id'],'source_group':'soccernet_game_'+row['game'],
                         'source_sha256':row['sha256'],'region':example['region'],'image_sha256':pilot.digest(image)})
    pilot.write_json(RUN/'negative_examples.json',{'review':'Assistant visual review of detector-mined crop regions. Visible and ambiguous balls were excluded.',
                                                  'examples':accepted,'source_manifest_sha256':pilot.digest(manifest_path)})
    data=dest/'data.yaml'
    data.write_text(pilot.yaml.safe_dump({'path':str(dest),'train':'images/train','val':'images/val','names':{0:'ball'}}),encoding='utf-8')
    print('Training images:',len(list((dest/'images/train').glob('*.jpg'))),'new negative crops:',len(accepted),flush=True)
    return data


def train():
    device=pilot.choose_device()
    data=prepare()
    old=pilot.read_json(PRIOR/'experiment.json')
    args=old['training']
    args.update(data=str(data),device=device,project=str(RUN))
    if pilot.digest(pilot.PRETRAINED)!=old['initial_weights_sha256']:
        raise ValueError('Initialization changed')
    pilot.write_json(RUN/'experiment.json',{'training':args,'initial_weights_sha256':pilot.digest(pilot.PRETRAINED),
             'design':'Same initial weights, seed, validation game and hyperparameters as pilot 002; add reviewed negative crops only.',
             'fresh_test_manifest_sha256':pilot.digest(ROOT/'candidate_manifest.json')})
    start=time.perf_counter()
    pilot.YOLO(str(pilot.PRETRAINED)).train(**args)
    weights=RUN/'training/weights/best.pt'
    pilot.write_json(RUN/'trained_checkpoint.json',{'checkpoint':str(weights.relative_to(RUN)),
         'sha256':pilot.digest(weights),'training_seconds':round(time.perf_counter()-start,2)})


def evaluate():
    device=pilot.choose_device()
    design=pilot.read_json(RUN/'experiment.json')
    path=ROOT/'candidate_manifest.json'
    if pilot.digest(path)!=design['fresh_test_manifest_sha256']:
        raise ValueError('Fresh test manifest changed')
    tested=[]
    for row in pilot.read_json(path)['frames']:
        if row['role']=='test':
            tested.append({**row,'image':row['path'],'source_group':'soccernet_game_'+row['game'],
                           'status':'visible' if len(row['bboxes'])==1 else 'uncertain','width':1920,'height':1080})
    old=pilot.read_json(PRIOR/'split.json')
    used={r['image_sha256'] for r in old['development']}
    used.update(r['source_sha256'] for r in pilot.read_json(RUN/'negative_examples.json')['examples'])
    if len({r['sha256'] for r in tested})!=len(tested):
        raise ValueError('Duplicate new test images')
    frames=previous.load_test(ROOT,tested,used)
    configs=[]
    for name,folder in [('multigame_baseline',PRIOR),('hard_negative_candidate',RUN)]:
        ck=pilot.read_json(folder/'trained_checkpoint.json'); weights=folder/ck['checkpoint']
        if pilot.digest(weights)!=ck['sha256']:raise ValueError('Checkpoint changed')
        configs.append((name,weights,0,960))
    report={'scope':'New sequences from two previously evaluated test games. Model weights selected on the existing validation game. Center tolerance 20 pixels, confidence .05, full-frame detection only.',
            'uncertain_excluded':len(tested)-len(frames),'models':previous.compare(frames,configs,device)}
    pilot.write_json(RUN/'fresh_comparison.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('train','evaluate'))
    {'train':train,'evaluate':evaluate}[p.parse_args().stage]()
