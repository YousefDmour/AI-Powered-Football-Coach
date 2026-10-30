"""Audit missed ball samples without changing the frozen detector or tracker."""

import argparse
import base64
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from project_paths import PROJECT_ROOT, SAMPLE_VIDEO
from vision.ball_tracking import BallTrack, center, outside_people, valid_ball
from vision.team_tracking import QUALITY_MODEL

REASONS = {
    'detector_no_near_candidate': 'لا يوجد كشف أولي قرب العلامة عند عتبة 0.05',
    'size_or_shape': 'فلتر حجم أو شكل الكرة',
    'empty_crop': 'منطقة الصورة غير صالحة',
    'white_fraction': 'فلتر نسبة اللون الأبيض',
    'pitch_fraction': 'فلتر العشب المحيط بالكرة',
    'person_exclusion': 'فلتر القرب من اللاعب',
    'confirmation': 'انتظار كشف ثانٍ للتأكيد',
    'tracker_confidence': 'درجة الكشف أقل من حد التتبّع 0.06',
    'tracker': 'لم يعتمد التتبّع المرشح؛ راجع الحالة والتفاصيل',
    'observed_elsewhere': 'التتبّع اعتمد موقعًا آخر',
}


def explain_filters(frame, box, people):
    """Mirror gate order for explanation; assert agreement with the frozen function."""
    x1,y1,x2,y2=map(float,box)
    width,height=x2-x1,y2-y1
    size_ok=3<=width<=36 and 3<=height<=36 and .4<=width/height<=2.5
    x,y=center(box)
    radius=max(14,int(max(width,height)*2))
    crop=frame[max(0,int(y)-radius):min(frame.shape[0],int(y)+radius),
               max(0,int(x)-radius):min(frame.shape[1],int(x)+radius)]
    patch=frame[max(0,int(y1)):min(frame.shape[0],int(y2)+1),
                max(0,int(x1)):min(frame.shape[1],int(x2)+1)]
    white=green=None
    if crop.size and patch.size:
        ball_hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
        white=float(np.mean((ball_hsv[:,:,1]<=100)&(ball_hsv[:,:,2]>=155)))
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        green=float(np.mean((hsv[:,:,0]>=35)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>=45)&(hsv[:,:,2]>=35)))
    reason=('size_or_shape' if not size_ok else 'empty_crop' if white is None
            else 'white_fraction' if white<.20 else 'pitch_fraction' if green<.35 else None)
    passes_ball=valid_ball(frame,box)
    if passes_ball != (reason is None):
        raise AssertionError('Diagnostic explanation diverged from the frozen ball filter')
    outside=outside_people(box,people)
    if reason is None and not outside:reason='person_exclusion'
    return {'width':width,'height':height,'white_fraction':white,'pitch_green_fraction':green,
            'passes_ball_filters':passes_ball,'outside_people':outside,'first_rejection':reason}


def replay_tracker(directory, selected):
    """Use every saved candidate frame, not just sparse samples, to preserve history."""
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    if summary['search_mode']!='full':raise ValueError('This audit requires the frozen full-frame search')
    cached=json.loads((directory/'candidates.json').read_text(encoding='utf-8'))
    original=pd.read_csv(directory/'ball_tracks.csv').set_index('source_frame')
    track=BallTrack();states={}
    for item in cached:
        index=item['source_frame']
        if index in summary['scene_cuts']:track.reset()
        before={'last_frame':track.last_frame,'last_position':track.last_position,'hits':track.hits}
        observed=track.update(index,item['candidates'],global_search=True)
        row=original.loc[index]
        if observed['state']!=row.state:raise AssertionError(f'Tracker replay differs on frame {index}')
        if observed['state']=='observed' and math.dist((row.x,row.y),(observed['x'],observed['y']))>1e-6:
            raise AssertionError(f'Tracker location differs on frame {index}')
        if index in selected:states[index]={'before':before,'after':observed,'cached_candidates':item['candidates']}
    return states,len(cached)


def write_review(results, output):
    cards=[]
    for row in results:
        index=row['source_frame'];label=REASONS[row['reason']]
        candidate_rows=[]
        for candidate in row['raw_candidates']:
            def value(number):return '—' if number is None else f'{number:.3f}'
            rejection=candidate['first_rejection']
            candidate_rows.append('<tr>'+''.join(f'<td>{html.escape(str(item))}</td>' for item in [
                candidate['id'],value(candidate['confidence']),f"{candidate['distance_to_annotation']:.1f}",
                value(candidate['white_fraction']),value(candidate['pitch_green_fraction']),
                REASONS.get(rejection,'اجتاز الفلاتر')])+ '</tr>')
        cards.append(f'''<section id="f{index}"><div class="heading"><h2>الإطار {index}</h2><span>{html.escape(label)}</span></div>
<p>حالة النظام: <b>{html.escape(row['original_state'])}</b> · زمن المصدر: {index/30:.2f} ثانية</p>
<details><summary>الصورة الأصلية كاملة</summary><img src="images/source_{index}.jpg" alt="إطار المصدر كاملًا"></details>
<div class="images"><figure><img loading="lazy" src="images/crop_source_{index}.jpg" alt="قصاصة المصدر دون علامات"><figcaption>المصدر دون علامات — هل الكرة ظاهرة هنا؟</figcaption></figure>
<figure><img loading="lazy" src="images/crop_audit_{index}.jpg" alt="العلامة ومرشحات النموذج"><figcaption>سماوي: العلامة الحالية · أحمر: كشف مرفوض · أخضر: كشف اجتاز الفلاتر · أبيض: مربع لاعب</figcaption></figure></div>
<details><summary>تفاصيل كل الكشوف الأولية ({len(row['raw_candidates'])})</summary><div class="table-wrap"><table><thead><tr><th>رقم</th><th>درجة الكشف</th><th>بعده عن العلامة / بكسل</th><th>نسبة الأبيض</th><th>نسبة العشب</th><th>أول سبب رفض</th></tr></thead><tbody>{''.join(candidate_rows)}</tbody></table></div></details>
<div class="review"><label for="decision-{index}">مراجعتك للعلامة</label><select id="decision-{index}" data-frame="{index}"><option value="unreviewed">لم أراجع بعد</option><option value="correct">الكرة واضحة والعلامة صحيحة</option><option value="wrong">العلامة بحاجة لتصحيح</option><option value="uncertain">الكرة غير واضحة لي</option></select><input data-note="{index}" aria-label="ملاحظة على الإطار {index}" placeholder="ملاحظة اختيارية"></div></section>''')
    counts=Counter(row['reason'] for row in results)
    summary=' · '.join(f'{REASONS[reason]}: {count}' for reason,count in counts.items())
    document='''<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>مراجعة الكرة — الإطارات الفائتة</title>
<style>body{font:15px/1.8 Tahoma,Arial,sans-serif;background:#f3f5ef;color:#20362e;margin:0}main{max-width:1250px;margin:auto;padding:30px 22px}h1{font-size:28px;margin-bottom:5px}header,section{background:white;border:1px solid #dde4d7;border-radius:12px;padding:22px;margin-bottom:20px}nav{display:flex;gap:9px;flex-wrap:wrap;margin-top:18px}a{color:#24573f}nav a,button{background:#d9edb4;border:0;padding:7px 15px;border-radius:6px;color:#20362e;text-decoration:none;cursor:pointer;font:inherit}.heading{display:flex;justify-content:space-between;align-items:center;gap:12px}.heading span{background:#fff0d9;padding:5px 12px;border-radius:6px;font-size:12px}.images{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:12px 0}img{width:100%;display:block;background:#edf1e8}figcaption{font-size:12px;color:#67755d}details{margin-top:14px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:8px;border-bottom:1px solid #e4e8df;text-align:right}.table-wrap{overflow:auto}.review{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}select,input{font:inherit;border:1px solid #cbd5c1;border-radius:5px;padding:6px;min-width:180px}input{flex:1}.notice{color:#687258;font-size:13px}button:focus-visible,a:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #648648;outline-offset:3px}@media(max-width:700px){.images{grid-template-columns:1fr}.heading{display:block}main{padding:12px}}</style>
<main><header><h1>نراجع الكرة، قبل ما نغيّر الإعدادات</h1><p>ابدأ بالمصدر دون علامات، ثم قارنه بالعلامة والكشوف. راجع كل إطار وحدّد إذا كانت العلامة صحيحة أو تحتاج تعديلًا.</p>
<p class="notice">العلامات الحالية وضعتها المساعدة بصريًا وليست حقيقة بشرية مؤكدة. الربط بين الكشف والعلامة يستخدم مسافة 20 بكسل؛ قد يربط حذاءً قريبًا، لذلك نحتاج مراجعتك. «لا كشف» تعني عدم وجود مرشح فوق عتبة 0.05، ولا تثبت غياب أي استجابة أضعف.</p>'''
    document+=f'<p>{html.escape(summary)}</p><nav>'+''.join(f'<a href="#f{r["source_frame"]}">{r["source_frame"]}</a>' for r in results)+'</nav><p><button id="export">تنزيل ملاحظات المراجعة JSON</button></p><small id="saved">الملاحظات لا تعدّل ملف العلامات الأصلي.</small></header>'
    document+=''.join(cards)+'''</main><script>
const key='ball-review-300-539-v1';let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch{}
for(const select of document.querySelectorAll('select[data-frame]')){const id=select.dataset.frame,note=document.querySelector(`[data-note="${id}"]`);select.value=saved[id]?.decision||'unreviewed';note.value=saved[id]?.note||'';function store(){saved[id]={decision:select.value,note:note.value};try{localStorage.setItem(key,JSON.stringify(saved));document.getElementById('saved').textContent='حُفظت الملاحظات في هذا المتصفح. نزّل الملف للاحتفاظ بنسخة.'}catch{document.getElementById('saved').textContent='الحفظ المحلي غير متاح؛ نزّل الملاحظات قبل إغلاق الصفحة.'}}select.addEventListener('change',store);note.addEventListener('input',store)}
document.getElementById('export').onclick=()=>{const result={source:'data/annotations/heldout_ball_300_539.json',changes_applied:false,reviews:[]};for(const select of document.querySelectorAll('select[data-frame]')){const id=select.dataset.frame;result.reviews.push({source_frame:Number(id),decision:select.value,note:document.querySelector(`[data-note="${id}"]`).value})}const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='ball-frame-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>'''
    # A self-contained review remains usable after the temporary preview server
    # stops, and can be opened from disk without any network connection.
    for path in (output/'images').glob('*.jpg'):
        reference=f'images/{path.name}'
        if reference in document:
            document=document.replace(reference,'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode('ascii'))
    (output/'index.html').write_text(document,encoding='utf-8')


def run(output):
    output=Path(output);images=output/'images';images.mkdir(parents=True,exist_ok=True)
    base=PROJECT_ROOT/'outputs/vision/validation_300_539'
    manifest=json.loads((PROJECT_ROOT/'data/validation/heldout_300_539.json').read_text(encoding='utf-8'))
    for name,digest in {**manifest['sha256'],**manifest['annotation_sha256']}.items():
        if hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Frozen experiment file changed: {name}')
    evaluation=json.loads((base/'ball/evaluation.json').read_text(encoding='utf-8'))
    misses=[row['source_frame'] for row in evaluation['details']
            if row['visibility']=='visible' and row['prediction_state']!='observed']
    labels=json.loads((PROJECT_ROOT/'data/annotations/heldout_ball_300_539.json').read_text(encoding='utf-8'))
    truth={row['source_frame']:row['center'] for row in labels['frames']}
    states,replayed=replay_tracker(base/'ball',misses)
    players=pd.read_csv(base/'players/tracks.csv')
    model=YOLO(str(QUALITY_MODEL));cap=cv2.VideoCapture(str(SAMPLE_VIDEO));results=[]
    try:
        for index in misses:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
            if not ok:raise ValueError(f'Cannot read source frame {index}')
            detections=model.predict(frame,classes=[32],conf=.05,imgsz=1280,verbose=False)[0]
            people=players.loc[players.source_frame.eq(index),['x1','y1','x2','y2']].values.tolist()
            raw=[]
            for number,(box,confidence) in enumerate(zip(detections.boxes.xyxy.cpu().tolist(),detections.boxes.conf.cpu().tolist()),1):
                raw.append({'id':number,'bbox':box,'confidence':confidence,
                    'distance_to_annotation':math.dist(center(box),truth[index]),**explain_filters(frame,box,people)})
            nearby=[item for item in raw if item['distance_to_annotation']<=20]
            passing=[item for item in nearby if item['first_rejection'] is None]
            state=states[index]
            reason=('tracker_confidence' if passing and all(item['confidence']<BallTrack().minimum_confidence for item in passing)
                    else 'confirmation' if passing and state['after']['state']=='pending'
                    else 'tracker' if passing else min(nearby,key=lambda item:item['distance_to_annotation'])['first_rejection']
                    if nearby else 'detector_no_near_candidate')
            # Compare rerun survivors to the candidates actually consumed in the original run.
            survivors=[item for item in raw if item['first_rejection'] is None]
            cached=state['cached_candidates']
            consistent=len(survivors)==len(cached) and all(any(math.dist(center(s['bbox']),center(c['bbox']))<.1 and abs(s['confidence']-c['confidence'])<1e-4 for c in cached) for s in survivors)
            row={'source_frame':index,'annotation':truth[index],'original_state':state['after']['state'],
                 'reason':reason,'rerun_survivors_match_cached':consistent,'raw_candidates':raw,
                 'tracker':state}
            results.append(row)
            annotated=frame.copy()
            for box in people:
                x1,y1,x2,y2=map(int,box);cv2.rectangle(annotated,(x1,y1),(x2,y2),(230,230,230),1)
            for item in raw:
                x1,y1,x2,y2=map(int,item['bbox']);color=(60,200,60) if item['first_rejection'] is None else (40,40,240)
                cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2);cv2.putText(annotated,str(item['id']),(x1,y1-4),0,.5,color,1)
            x,y=truth[index];cv2.circle(annotated,(x,y),12,(255,255,0),2)
            left,top=max(0,x-110),max(0,y-110);right,bottom=min(frame.shape[1],x+110),min(frame.shape[0],y+110)
            cv2.imwrite(str(images/f'source_{index}.jpg'),frame)
            cv2.imwrite(str(images/f'audit_{index}.jpg'),annotated)
            cv2.imwrite(str(images/f'crop_source_{index}.jpg'),cv2.resize(frame[top:bottom,left:right],(660,660)))
            cv2.imwrite(str(images/f'crop_audit_{index}.jpg'),cv2.resize(annotated[top:bottom,left:right],(660,660)))
            print(index,reason,'rerun matches:',consistent,flush=True)
    finally:cap.release()
    report={'reviewed_misses':len(results),'original_tracker_frames_replayed':replayed,
            'reason_counts':dict(Counter(row['reason'] for row in results)),
            'all_reruns_match_cached':all(row['rerun_survivors_match_cached'] for row in results),
            'limits':'Diagnostic only. No thresholds or frozen annotations modified. Near-candidate matching depends on approximate visual annotations; requires user review.',
            'frames':results}
    (output/'diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    write_review(results,output)
    print(json.dumps({key:value for key,value in report.items() if key!='frames'},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/vision/ball_miss_review')
    run(parser.parse_args().output)


if __name__=='__main__':main()
