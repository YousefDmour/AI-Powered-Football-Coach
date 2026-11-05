'use strict';
const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d'), photo = new Image();
let manifest, annotations = {}, index = 0, drawing = null, ready = false, storageKey;
const names = {pending:'غير مراجعة', visible:'الكرة محددة', not_visible:'الكرة غير ظاهرة بالتأكيد', uncertain:'غير متأكد — مستبعدة'};
function progress() {
  const values = Object.values(annotations);
  $('progress').textContent = `تمت مراجعة ${values.filter(a=>a.status!=='pending').length} / ${manifest.frames.length} — كرة محددة: ${values.filter(a=>a.status==='visible').length}، غير ظاهرة: ${values.filter(a=>a.status==='not_visible').length}، غير متأكد: ${values.filter(a=>a.status==='uncertain').length}`;
}
function save() {
  try {localStorage.setItem(storageKey, JSON.stringify(annotations));}
  catch { $('storage').textContent = 'تعذّر الحفظ التلقائي! نزّل JSON قبل إغلاق الصفحة.'; }
  progress();
}
function resize() {
  const width = Math.max(300, $('viewport').clientWidth - 2);
  canvas.style.width = Math.round(width * Number($('zoom').value)) + 'px';
  canvas.style.height = 'auto';
}
function paint(boxOverride) {
  if (!ready) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(photo,0,0);
  const box = boxOverride || annotations[manifest.frames[index].id]?.bbox;
  if (box) {
    const [x1,y1,x2,y2] = box;
    ctx.strokeStyle = '#00ff97';ctx.lineWidth = 3;
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.fillStyle = '#00ff97';ctx.font = '22px sans-serif';
    ctx.fillText('ball',x1,Math.max(24,y1-8));
  }
}
function updateInfo() {
  const frame = manifest.frames[index];
  $('position').textContent = `${index+1} / ${manifest.frames.length}`;
  $('jump').value = index+1;
  $('info').textContent = `فريم ${frame.source_frame} | ثانية ${frame.time_seconds} | ${names[annotations[frame.id]?.status || 'pending']}`;
  $('prev').disabled = index === 0;$('next').disabled = index === manifest.frames.length-1;
  progress();
}
function show(i) {
  index = Math.max(0,Math.min(manifest.frames.length-1,i));
  ready = false; drawing = null;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  photo.src = manifest.frames[index].image;
  updateInfo();
}
photo.onload = () => {canvas.width=photo.naturalWidth;canvas.height=photo.naturalHeight;ready=true;resize();paint();};
photo.onerror = () => {$('info').textContent='تعذّر تحميل الصورة. تأكد أن الخادم يعمل.';};
function point(event) {
  const rect=canvas.getBoundingClientRect();
  return [Math.max(0,Math.min(canvas.width,(event.clientX-rect.left)*canvas.width/rect.width)), Math.max(0,Math.min(canvas.height,(event.clientY-rect.top)*canvas.height/rect.height))];
}
function boxFrom(a,b) {return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])].map(n=>Math.round(n*100)/100);}
canvas.addEventListener('pointerdown',e=>{if(!ready || e.button!==0)return;e.preventDefault();drawing=point(e);canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(drawing)paint(boxFrom(drawing,point(e)));});
canvas.addEventListener('pointerup',e=>{
  if(!drawing)return;
  const box=boxFrom(drawing,point(e));drawing=null;
  if(box[2]-box[0]>=2 && box[3]-box[1]>=2){annotations[manifest.frames[index].id]={status:'visible',bbox:box};save();}
  paint();updateInfo();
});
canvas.addEventListener('pointercancel',()=>{drawing=null;paint();});
function setStatus(status) {if(!manifest)return;annotations[manifest.frames[index].id]={status,bbox:null};save();paint();updateInfo();}
$('negative').onclick=()=>setStatus('not_visible');$('uncertain').onclick=()=>setStatus('uncertain');
$('clear').onclick=()=>{delete annotations[manifest.frames[index].id];save();paint();updateInfo();};
$('prev').onclick=()=>show(index-1);$('next').onclick=()=>show(index+1);
$('go').onclick=()=>{const n=Number($('jump').value);if(Number.isInteger(n))show(n-1);};
$('unreviewed').onclick=()=>{const i=manifest.frames.findIndex(f=>!annotations[f.id] || annotations[f.id].status==='pending');if(i>=0)show(i);else alert('كل الصور تمت مراجعتها. نزّل العلامات للاحتفاظ بنسخة.');};
$('zoom').onchange=resize;window.addEventListener('resize',resize);
document.addEventListener('keydown',e=>{if(!manifest || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName))return;if(e.key.toLowerCase()==='n')show(index+1);if(e.key.toLowerCase()==='p')show(index-1);if(e.key.toLowerCase()==='u')setStatus('uncertain');});
$('export').onclick=()=>{
  const data={schema_version:1,batch_id:manifest.batch_id,exported_at:new Date().toISOString(),annotations};
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download=`ball-labels-${manifest.batch_id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
function validateLoaded(data) {
  if(data.schema_version!==1 || data.batch_id!==manifest.batch_id || !data.annotations || typeof data.annotations!=='object' || Array.isArray(data.annotations))throw Error('ملف العلامات لا يخص هذه الدفعة');
  const known=new Map(manifest.frames.map(f=>[f.id,f]));
  for(const [id,a] of Object.entries(data.annotations)){
    const f=known.get(id);if(!f || !a || !Object.hasOwn(names,a.status))throw Error('علامة غير صالحة');
    if(a.status==='visible'){
      const b=a.bbox;if(!Array.isArray(b)||b.length!==4||!b.every(Number.isFinite)||!(0<=b[0]&&b[0]<b[2]&&b[2]<=f.width&&0<=b[1]&&b[1]<b[3]&&b[3]<=f.height))throw Error('صندوق غير صالح');
    } else if(a.bbox!=null)throw Error('حالة بدون كرة ومعها صندوق');
  }
  return data.annotations;
}
$('import').onchange=async e=>{
  try{const file=e.target.files[0];if(!file)return;const loaded=validateLoaded(JSON.parse(await file.text()));if(Object.keys(annotations).length && !confirm('استبدال العلامات الحالية بالملف المختار؟ نزّل نسخة أولًا عند الحاجة.'))return;annotations=loaded;save();show(index);}
  catch(error){alert(error.message);}finally{e.target.value='';}
};
(async()=>{
  try{
    const response=await fetch('manifest.json');if(!response.ok)throw Error('manifest');manifest=await response.json();
    storageKey=`ball-labeler:${manifest.batch_id}`;$('jump').max=manifest.frames.length;
    try{const saved=JSON.parse(localStorage.getItem(storageKey)||'{}');annotations=validateLoaded({schema_version:1,batch_id:manifest.batch_id,annotations:saved});}
    catch{$('storage').textContent='تعذّرت استعادة العلامات المحلية. يمكنك استعادة ملف JSON محفوظ.';}
    show(0);
  }catch{$('info').textContent='تعذّر تحميل الدفعة. افتح الصفحة عبر الخادم المحلي بدل فتح HTML مباشرة.';}
})();
