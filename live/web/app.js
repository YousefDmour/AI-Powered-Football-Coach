"use strict";
const el = id => document.getElementById(id);
let state = null, busy = false, view = "new", readUntil = 0, runId = null, controlVersion = 0;
const timeLabel = seconds => `${String(Math.floor(seconds / 60)).padStart(2,"0")}:${String(seconds % 60).padStart(2,"0")}`;
const percent = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const text = (tag, value, cls) => { const node = document.createElement(tag); node.textContent = value; if(cls) node.className = cls; return node; };
const rules = {
  passing: ["مراجعة خيارات التمرير", "راجع الدعم حول حامل الكرة، وفكّر بخيارات تمرير أقصر."],
  shooting: ["مراجعة اختيار التسديدات", "راجع جودة فرص التسديد وإمكانية صناعة فرص أوضح."],
  defending: ["مراجعة التنسيق الدفاعي", "راجع المساندة والتنسيق بين اللاعبين قبل تغيير أسلوب الضغط."]
};
function render(data) {
  state = data;
  if(runId !== data.run_id) {
    runId = data.run_id;
    const saved = sessionStorage.getItem("coach-read");
    try { const record = JSON.parse(saved); readUntil = record?.run === runId ? record.id : 0; } catch { readUntil = 0; }
  }
  const stats = data.statistics, teams = Object.entries(stats.teams), home = teams[0], away = teams[1];
  el("home-name").textContent = el("legend-home").textContent = home?.[0] || "الفريق الأول";
  el("away-name").textContent = el("legend-away").textContent = away?.[0] || "الفريق الثاني";
  el("home-score").textContent = home?.[1].goals || 0;
  el("away-score").textContent = away?.[1].goals || 0;
  el("clock").textContent = timeLabel(stats.match_second);
  el("period").textContent = stats.period ? `الشوط ${stats.period}` : "بانتظار البداية";
  el("status").textContent = data.completed ? "انتهت المحاكاة" : data.paused ? "متوقفة مؤقتًا" : "المحاكاة تعمل";
  el("events").textContent = `${stats.events_processed.toLocaleString("en")} / ${data.total_events.toLocaleString("en")}`;
  el("progress").max = data.total_events;
  el("progress").value = stats.events_processed;
  el("play").textContent = data.completed ? "✓ اكتملت المباراة" : data.paused ? "▶ تشغيل المحاكاة" : "Ⅱ إيقاف مؤقت";
  el("play").disabled = data.completed || busy;
  el("reset").disabled = busy;
  el("speed").disabled = busy;
  el("speed").value = String(data.speed);
  renderStats(home?.[1] || {}, away?.[1] || {});
  renderAlerts();
  const body = el("players"); body.replaceChildren();
  const players = Object.values(stats.players).sort((a,b) => b.passes_attempted-a.passes_attempted);
  for(const player of players) {
    const row = document.createElement("tr");
    for(const value of [player.name,player.team,`${player.passes_completed} / ${player.passes_attempted}`,percent(player.pass_accuracy),player.shots,player.shots_on_target]) row.append(text("td",value));
    body.append(row);
  }
  if(!players.length) { const row = document.createElement("tr"), cell = text("td","تظهر بيانات اللاعبين مع وصول أحداثهم."); cell.colSpan=6; row.append(cell); body.append(row); }
}
function renderStats(home, away) {
  const container = el("stats"); container.replaceChildren();
  for(const [key,label,isPercent] of [["passes_attempted","التمريرات",false],["passes_completed","التمريرات الناجحة",false],["pass_accuracy","دقة التمرير",true],["shots","التسديدات",false],["shots_on_target","التسديدات على المرمى",false],["defensive_actions_successful","التدخلات والاعتراضات الناجحة",false]]) {
    const a=home[key]??0,b=away[key]??0,total=a+b;
    const row=text("div","","stat-row"),heading=text("div","","stat-label");
    heading.append(text("span",isPercent?percent(home[key]):a,"stat-value"),text("span",label),text("span",isPercent?percent(away[key]):b,"stat-value"));
    const bars=text("div","","bars");
    for(const [value,cls] of [[a,"home-color"],[b,"away-color"]]) {const track=text("div","","track"),bar=text("div","",`bar ${cls}`);bar.style.width=`${total ? value/total*100 : 0}%`;track.append(bar);bars.append(track);}
    row.append(heading,bars);container.append(row);
  }
}
function renderAlerts() {
  if(!state) return;
  const alerts=state.alerts, fresh=alerts.filter(a=>a.id>readUntil),old=alerts.filter(a=>a.id<=readUntil);
  el("alert-count").textContent=alerts.length;el("new-count").textContent=fresh.length;el("old-count").textContent=old.length;
  el("mark-read").disabled=!fresh.length;
  for(const [id,tab] of [["new-tab","new"],["history-tab","old"]]) {el(id).classList.toggle("active",view===tab);el(id).setAttribute("aria-pressed",String(view===tab));}
  const container=el("alerts"),scroll=container.scrollTop;container.replaceChildren();
  const visible=[...(view==="new"?fresh:old)].reverse();
  if(!visible.length) {
    const empty=text("div","","empty-state");empty.append(text("span","◎","empty-symbol"),text("strong",view==="new"?"لا توصيات جديدة حتى الآن":"سجل التوصيات فارغ"),text("span",view==="new"?"تظهر التوصيات عند توفر محاولات كافية وتجاوز الحدود المحددة.":"انقل التوصيات التي راجعتها إلى هنا."));container.append(empty);
  }
  for(const alert of visible) {
    const card=text("article","","alert-card"),meta=text("div","","alert-meta");meta.append(text("span",alert.team),text("span",timeLabel(alert.match_second)));
    const rule=rules[alert.rule]||[alert.rule,alert.advice];
    card.append(meta,text("h3",rule[0]),text("p",rule[1]));
    card.append(text("div",`${alert.successful} من ${alert.attempted} (${percent(alert.successful/alert.attempted)}) خلال آخر ${alert.window_seconds/60} دقائق؛ الحد التجريبي ${percent(alert.threshold)}.` ,"evidence"));container.append(card);
  }
  container.scrollTop=scroll;
}
async function request(path, payload) {
  const response=await fetch(path,payload?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}:{});
  const data=await response.json();if(!response.ok) throw new Error(data.error||"تعذر الاتصال");return data;
}
function showError(error) {el("error").hidden=false;el("error").textContent=`تعذر تحديث اللوحة. تأكد من تشغيل الخادم المحلي. ${error.message}`;}
async function control(action,speed) {
  if(busy) return;controlVersion++;busy=true;for(const id of ["play","reset","speed"]) el(id).disabled=true;
  try {const data=await request("/api/control",{action,speed});busy=false;render(data);el("error").hidden=true;} catch(error) {busy=false;showError(error);if(state)render(state);}
}
el("play").onclick=()=>control(state?.paused?"play":"pause");
el("reset").onclick=()=>{if(state?.statistics.events_processed>2 && !confirm("إعادة المباراة من البداية ومسح إحصائيات وتوصيات هذه المحاكاة؟"))return;control("reset");};
el("speed").onchange=event=>control("speed",Number(event.target.value));
el("new-tab").onclick=()=>{view="new";renderAlerts();};el("history-tab").onclick=()=>{view="old";renderAlerts();};
el("mark-read").onclick=()=>{readUntil=state.alerts.at(-1)?.id||0;sessionStorage.setItem("coach-read",JSON.stringify({run:runId,id:readUntil}));view="old";renderAlerts();};
async function poll(){try{if(!busy){const version=controlVersion;const data=await request("/api/state");if(!busy && version===controlVersion){render(data);el("error").hidden=true;}}}catch(error){showError(error);}finally{setTimeout(poll,500);}}
poll();
