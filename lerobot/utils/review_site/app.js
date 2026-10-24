'use strict';
const D = window.REVIEW_DATA;
for(const k of ['offline','online','pairs']){const fields=k==='pairs'?D.pair_fields:D.row_fields;D[k]=D[k].map(row=>Object.fromEntries(fields.map((key,i)=>[key,row[i]])));}
window.REVIEW_EPISODES = {};
const $ = id => document.getElementById(id);
const esc = x => String(x ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const oursText = x => String(x ?? '').replace(/WM-NC(?:\s*\(ours\))?/g,'WM-NC (ours)');
const oursHTML = x => esc(oursText(x)).replace(/WM-NC \(ours\)/g,'<strong class="ours-method">WM-NC (ours)</strong>');
// Match the paper's round-to-even convention for exact halfway percentages.
const pct = x => {if(x==null)return '—';const scaled=1000*x,lo=Math.floor(scaled);const rounded=Math.abs(scaled-lo-.5)<1e-9?lo+(lo%2):Math.round(scaled);return `${(rounded/10).toFixed(1)}%`;};
const num = (x,d=2) => x == null ? '—' : Number(x).toFixed(d);
const sum = (a,k) => a.reduce((s,r)=>s+(Number(r[k])||0),0);
const mean = a => a.length ? a.reduce((s,x)=>s+x,0)/a.length : null;
const viewName = {fix:'Fixed',wrist:'Wrist',both:'Joint input'};
const tasks = Object.fromEntries(D.catalog.map(t=>[t.id,t.title]));
const shortTask = x => (tasks[x]||x).split(' · ')[0];
function fillOptions(e,rows,selected){e.replaceChildren(...rows.map(([v,t])=>{const option=new Option(oursText(t),v);if(String(t).includes('WM-NC'))option.style.fontWeight='700';return option;}));if(rows.some(r=>String(r[0])===String(selected)))e.value=selected;e.classList.toggle('ours-selected',e.selectedOptions[0]?.textContent.includes('WM-NC')||false);}
function options(id, rows, selected) {fillOptions($(id),rows,selected);}
function group(rows,key){const m=new Map(); for(const r of rows){const k=key(r);if(!m.has(k))m.set(k,[]);m.get(k).push(r);}return m;}
const key = r => [r.variant,r.method,r.view,r.calibration].join('|');
const failureKey=(task,mode)=>JSON.stringify([task,mode]);
const failureModes=new Map(D.offline.filter(r=>r.group_type==='failure_mode').map(r=>[failureKey(r.task,r.failure_mode),r]));
let populationKind='task';
const selectedView=()=>document.querySelector('[name=comparison-view]:checked').value;
const selectedFailure=()=>populationKind==='mode'?failureModes.get($('mode-filter').value):null;
const selectedTask=()=>populationKind==='category'?'all':$('task-filter').value;
function categoryOf(r){return r.failure_category||failureModes.get(failureKey(r.task,r.failure_mode))?.failure_category;}
const labels=new Map();
for(const t of D.catalog)for(const g of t.groups)for(const e of g.episodes)labels.set(`${t.id}/${e.id}`,{...g,task:t.id});
const aucIndex=group(D.auc_rows,r=>[r[0],r[2],r[3],r[4]].join('|'));
function auc(pos,neg){if(!pos.length||!neg.length||[...pos,...neg].some(v=>v==null))return null;let wins=0;for(const p of pos)for(const n of neg)wins+=p>n?1:p===n?.5:0;return wins/(pos.length*neg.length);}
function selectedAUC(k){const rows=aucIndex.get(k)||[],out=[];for(const [task,items] of group(rows,r=>r[1])){if(selectedTask()!=='all'&&task!==selectedTask())continue;const positive=[],negative=[];for(const r of items){const g=labels.get(`${r[1]}/${r[5]}`);if(!g||g.split!=='test')continue;if(g.outcome==='success')negative.push(r[6]);else if(originMode(g))positive.push(r[6]);}if(positive.length){const a=auc(positive,negative);if(a==null)return null;out.push(a);}}return mean(out);}
function originMode(r){
 if($('origin-filter').value!=='all'&&r.failure_origin!==$('origin-filter').value)return false;
 if(populationKind==='category'&&$('category-filter').value!=='all')return categoryOf(r)===$('category-filter').value;
 const mode=selectedFailure();return !mode||(r.task===mode.task&&r.failure_mode===mode.failure_mode);
}
function base(r,view=selectedView()){const t=selectedTask();return(t==='all'||r.task===t)&&(view==='all'||r.view===view||r.view==='both')&&(r.calibration===$('calibration-filter').value||r.method==='qwen_v9');}
const isNoAct=r=>r.variant==='wm_k4_l4_action_none'&&r.method==='wm_nc';
function methodName(r){return (isNoAct(r)?'NoACT':D.methods[r.method]||r.method_name||r.method)+' · '+viewName[r.view];}
function aggregate(rows,view=selectedView()){
 const out=[];
 for(const [k,items] of group(rows.filter(r=>base(r,view)),key)){
  const failure=items.filter(r=>r.group_type==='failure_mode'&&originMode(r));
  if(!failure.length)continue;
  const success=items.filter(r=>r.group_type==='success_test');
  const n=sum(failure,'population_n'),hits=sum(failure,'alarm_n'),unknown=sum(failure,'unknown_n');
  const sn=sum(success,'population_n'),fa=sum(success,'alarm_n');
  const matchedN=sum(failure,'matched_detected_n');
  const matchedAvailable=failure.every(r=>r.matched_detected_n!=null);
  const allRows=items.filter(r=>r.group_type==='all_failures');
  const matchedFA=allRows.length&&allRows.every(r=>r.matched_false_alarm_n!=null)?sum(allRows,'matched_false_alarm_n'):null;
  const r=items[0];
  out.push({...r,key:k,n,hits,unknown,sn,fa,fdr:n?hits/n:null,far:sn?fa/sn:null,auc:selectedAUC(k),
    matched_hits:matchedAvailable?matchedN:null,matched_fa:matchedFA,
    matched_fdr:matchedAvailable?matchedN/n:null,matched_far:matchedFA==null||!sn?null:matchedFA/sn,
    early:sum(failure,'early_alarm_n'),streaming:r.method==='qwen_v9'});
 }
 return out;
}
const allRows=[...D.offline,...D.online.filter(r=>r.method==='qwen_v9')];
const mainRows=()=>aggregate(allRows.filter(r=>(r.main_comparison||isNoAct(r))&&!r.supplementary_method&&r.view===selectedView())).sort((a,b)=>{
 const order=Object.keys(D.methods);return order.indexOf(a.method)-order.indexOf(b.method)||Number(isNoAct(a))-Number(isNoAct(b))||a.view.localeCompare(b.view);
});
function barHTML(r,label){const unavailable=r.fdr==null||r.far==null;return `<div class="bar-row ${r.method==='wm_nc'&&r.variant==='main'?'reference':''}" data-method="${esc(r.method)}" data-view="${esc(r.view)}"><div class="bar-name"><strong>${oursHTML(label)}</strong><small>${unavailable?'Matched FAR unavailable':`${r.hits}/${r.n} detected${r.unknown?` · ${r.unknown} unknown`:''}${r.streaming?' · streaming':''}`}</small></div><div class="bar-cell"><div class="bar-track"><span class="bar-fill" style="width:${100*(r.fdr??0)}%"></span></div><span class="bar-value">${pct(r.fdr)}</span></div><div class="bar-cell far"><div class="bar-track"><span class="bar-fill" style="width:${100*(r.far??0)}%"></span></div><span class="bar-value">${pct(r.far)}</span></div>${r.main_comparison&&!r.streaming?`<button data-inspect="${esc(r.method)}" data-view="${esc(r.view)}">Replay ↗</button>`:'<span></span>'}</div>`;}
function bars(id,rows,label=methodName){$(id).innerHTML=rows.length?`<div class="bar-header"><span>Monitor</span><span>Detection · FDR</span><span>False alarms · FAR</span><span></span></div>`+rows.map(r=>barHTML(r,label(r))).join(''):'<p class="empty">No recorded results for this selection.</p>';}
function renderMain(){
 const matched=$('match-far').checked;
 const raw=mainRows(),rows=raw.map(r=>matched?{...r,hits:r.matched_hits,fa:r.matched_fa,fdr:r.matched_fdr,far:r.matched_far,unknown:0}:r);
 bars('main-chart',rows);
 $('comparison-mode').textContent=matched?'Retrospective test thresholds':$('calibration-filter').value==='episode'?'Episode calibration':'Time-varying bands';
 $('comparison-description').textContent=matched?'Selected on successful test episodes; achieved FAR is shown.':'19 calibration successes and 20 held-out successes per task.';
 $('main-title').textContent=matched?'Failure detection at a 5% FAR budget':'Failure detection & false alarms';
 const ref=rows.find(r=>r.variant==='main'&&r.method==='wm_nc'&&r.view===selectedView());
 $('metrics').innerHTML=[['WM-NC detection',pct(ref?.fdr),ref?`${ref.hits} / ${ref.n} failures · ${viewName[ref.view]}`:'No reference for this selection'],['WM-NC false alarms',pct(ref?.far),ref?`${ref.fa} / ${ref.sn} successful test episodes`:'—'],['WM-NC episode AUROC',num(ref?.auc,3),'Whole-episode maximum · task macro mean']].map(([label,v,sub])=>`<div class="metric-card"><span>${oursHTML(label)}</span><strong>${esc(v)}</strong><span>${esc(sub)}</span></div>`).join('');
}
function lineChart(id,family,param){
 const matched=$('sensitivity-matched').checked;
 const selected=aggregate(D.offline.filter(r=>(r.family===family||r.variant==='main')&&r.method==='wm_nc'&&['fix','wrist'].includes(r.view)),'all').map(r=>matched?{...r,fdr:r.matched_fdr,far:r.matched_far,hits:r.matched_hits,fa:r.matched_fa}:r);
 const points=selected.map(r=>({...r,x:r.variant==='main'?(param==='data'?100:4):Number((r.variant+' '+r.variant_title).toLowerCase().match(new RegExp(`(?:^|[_ \\u00b7])${param}(\\d+)`))?.[1])})).filter(r=>Number.isFinite(r.x));
 if(!points.length){$(id).innerHTML='<p class="empty">No comparable results.</p>';return;}
 const xs=[...new Set(points.map(r=>r.x))].sort((a,b)=>a-b),x=v=>46+(v-xs[0])/(xs.at(-1)-xs[0]||1)*336,y=v=>184-v*164;
 const tick=v=>param==='data'?({33:'1/3',66:'2/3',100:'Full'}[v]||`${v}%`):v;
 let svg=`<svg viewBox="0 0 400 212" role="img" aria-label="${param==='data'?'Training data':param.toUpperCase()} sweep: FDR and FAR for fixed and wrist cameras">`;
 for(const v of [0,.25,.5,.75,1])svg+=`<line class="chart-grid" x1="46" x2="382" y1="${y(v)}" y2="${y(v)}"/><text class="chart-label" x="37" y="${y(v)+5}" text-anchor="end">${100*v}%</text>`;
 for(const v of xs)svg+=`<text class="chart-label" x="${x(v)}" y="207" text-anchor="middle">${tick(v)}</text>`;
 for(const view of ['fix','wrist']){
  const items=points.filter(r=>r.view===view).sort((a,b)=>a.x-b.x);
  for(const [field,color] of [['fdr','var(--accent)'],['far','var(--warm)']]){
   const valid=items.filter(r=>r[field]!=null);if(!valid.length)continue;
   svg+=`<polyline data-view="${view}" data-metric="${field}" fill="none" stroke="${color}" stroke-width="2" ${view==='fix'?'stroke-dasharray="7 5"':''} points="${valid.map(r=>`${x(r.x)},${y(r[field])}`).join(' ')}"/>`;
   for(const r of valid)svg+=`<circle cx="${x(r.x)}" cy="${y(r[field])}" r="3.5" stroke="${color}" stroke-width="1.5" fill="${view==='fix'?'white':color}"><title>${viewName[view]} ${param==='data'?'training data':param.toUpperCase()}=${tick(r.x)}: ${field.toUpperCase()} ${pct(r[field])} (${field==='fdr'?`${r.hits}/${r.n}`:`${r.fa}/${r.sn}`})</title></circle>`;
  }
 }
 const legend='<div class="sweep-legend" role="group" aria-label="Curve legend"><span><i class="key-fdr" aria-hidden="true"></i>FDR</span><span><i class="key-far" aria-hidden="true"></i>FAR</span><span><i class="key-fixed" aria-hidden="true"></i>Fixed</span><span><i class="key-wrist" aria-hidden="true"></i>Wrist</span></div>';
 $(id).innerHTML=legend+svg+'</svg>';
}
function renderAblations(){lineChart('k-chart','K','k');lineChart('l-chart','L','l');lineChart('data-chart','data_size','data');}
// Table VI uses the active seed-1 run and the local seed-2/3 realizations.
// Keep explicit variants so archived runs and extra repeats cannot enter the mean.
const paperSeedVariants=['main','wm_k4_l4_seed1','wm_k4_l4_seed2_local_20261018','wm_k4_l4_seed3_local_20261018'];
function seedResults(){
 const result=[];
 for(const view of ['fix','wrist'])for(const [seed,variant] of paperSeedVariants.entries()){
  const rows=D.offline.filter(r=>r.variant===variant&&r.method==='wm_nc'&&r.view===view&&r.calibration==='episode');
  const failures=rows.filter(r=>r.group_type==='all_failures'),successes=rows.filter(r=>r.group_type==='success_test');
  if(!D.catalog.every(t=>failures.filter(r=>r.task===t.id).length===1&&successes.filter(r=>r.task===t.id).length===1))continue;
  const n=sum(failures,'population_n'),sn=sum(successes,'population_n'),hits=sum(failures,'alarm_n'),fa=sum(successes,'alarm_n');
  const matched=failures.every(r=>r.matched_detected_n!=null&&r.matched_false_alarm_n!=null);
  const matchedHits=matched?sum(failures,'matched_detected_n'):null,matchedFa=matched?sum(failures,'matched_false_alarm_n'):null;
  if(n&&sn)result.push({seed,variant,view,n,sn,hits,fa,matchedHits,matchedFa,fdr:hits/n,far:fa/sn,matched_fdr:matched?matchedHits/n:null});
 }
 return result;
}
function renderRobustnessSummary(id,value){
 const fields=[['far','FAR'],['fdr','FDR'],['matched_fdr','FDR at 5% FAR budget']];
 $(id).innerHTML='<div class="robustness-summary-grid">'+['fix','wrist'].map(view=>`<article data-view="${view}"><h4>${viewName[view]}</h4><dl>${fields.map(([field,label])=>`<div><dt>${label}</dt><dd data-metric="${field}">${esc(value(view,field))}</dd></div>`).join('')}</dl></article>`).join('')+'</div>';
}
function renderSeedRobustness(){
 const rows=seedResults();
 const statistic=values=>{if(values.length!==4||values.some(v=>v==null))return 'Unavailable';const m=mean(values),sd=Math.sqrt(values.reduce((s,v)=>s+(v-m)**2,0)/(values.length-1));return `${num(100*m,1)} ± ${num(100*sd,1)}%`;};
 renderRobustnessSummary('seed-summary',(view,field)=>statistic(rows.filter(r=>r.view===view).map(r=>r[field])));
}
function renderSensitivity(){
 const study=D.studies?.find(s=>s.id==='main_calibration_sensitivity_500'),matched=D.studies?.find(s=>s.id==='calibration_matched_far');
 const task=selectedTask()==='all'?'Overall':selectedTask(),calibration=$('calibration-filter').value;
 renderRobustnessSummary('sensitivity-chart',(view,field)=>{
  const source=field==='matched_fdr'?matched:study,metric=field==='far'?'FAR':'FDR';
  const row=source?.rows.find(r=>r.task===task&&r.view===view&&r.calibration===calibration);
  return row?.n_splits===500&&Number.isFinite(row[metric+'_mean'])&&Number.isFinite(row[metric+'_sd'])?`${num(100*row[metric+'_mean'],1)} ± ${num(100*row[metric+'_sd'],1)}%`:'Unavailable';
 });
}
function renderTiming(){
 const family=$('latency-family').value,view=document.querySelector('[name=latency-view]:checked').value;
 const rows=D.pairs.filter(r=>r.on_time_only===true&&(family==='all'?['baseline','K','L'].includes(r.family):r.family===family)&&r.status==='available'&&r.comparator.split('|').at(-1)===view&&r.shared_n>=Math.max(1,Number($('shared-min').value)||1));
 if(!rows.length){$('latency-chart').innerHTML='<p class="empty">No paired comparisons for this view and selection.</p>';return;}
 const value=(r,prefix)=>`<strong>${num(r[prefix+'_mean_s'])}</strong><small>± ${num(r[prefix+'_sd_s'])}</small>`;
 $('latency-chart').innerHTML='<table class="latency-table"><caption>Paired detection latency · seconds</caption><thead><tr><th scope="col">Monitor</th><th scope="col" class="latency-reference">'+oursHTML('WM-NC')+'<small>Wrist</small></th><th scope="col">Comparator<small>'+viewName[view]+'</small></th><th scope="col" class="latency-difference"><abbr title="WM-NC (ours) wrist mean minus comparator mean, in seconds">Δ (s)</abbr></th></tr></thead><tbody>'+rows.map(r=>{
  const difference=r.mean_difference_s,sign=Math.abs(difference)<.005?'':difference<0?'−':'+';
  const delta=`<span class="latency-delta ${difference<-.005?'shorter':difference>.005?'longer':'similar'}" title="WM-NC (ours) wrist minus comparator: ${num(difference,4)} s">${sign}${num(Math.abs(difference))}</span>`;
  const label=r.label.replace(/\s*·\s*(fixed|wrist)$/i,'').replace(/Qwen v9/,'Qwen').replace(/\bpolicy\b/,'Policy');
  return `<tr data-comparator="${esc(r.comparator)}"><th scope="row">${oursHTML(label)}<span class="latency-row-meta">n = ${r.shared_n}<span class="latency-delta-mobile">Δ ${delta}</span></span></th><td class="latency-reference">${value(r,'reference')}</td><td>${value(r,'comparator')}</td><td class="latency-difference">${delta}</td></tr>`;
 }).join('')+'</tbody></table>';
}
function initPopulationFilters(){
 for(const environment of ['Simulation','Real']){const branch=document.createElement('optgroup');branch.label=environment;for(const t of D.catalog.filter(t=>t.title.endsWith(environment)))branch.append(new Option(shortTask(t.id),t.id));$('task-filter').append(branch);}
 options('category-filter',[['all','Full'],...[...new Set([...failureModes.values()].map(r=>r.failure_category))].filter(Boolean).sort().map(c=>[c,c])],'all');
 for(const t of D.catalog){const branch=document.createElement('optgroup');branch.label=shortTask(t.id);for(const g of t.groups.filter(g=>g.outcome==='failure'))branch.append(new Option(g.failure_mode,failureKey(t.id,g.failure_mode)));$('mode-filter').append(branch);}
 document.querySelectorAll('[name=population-kind]').forEach(radio=>{radio.onchange=()=>{populationKind=radio.value;for(const id of ['task-filter','category-filter','mode-filter'])$(id).value='all';syncPopulation();renderResults();};});
 for(const id of ['task-filter','category-filter','mode-filter'])$(id).onchange=()=>{syncPopulation();renderResults();};
 syncPopulation();
}
function syncPopulation(){
 for(const kind of ['task','category','mode'])$(kind+'-filter').disabled=kind!==populationKind;
 if(populationKind==='mode')$('task-filter').value=selectedFailure()?.task||'all';
 if(populationKind==='category'){$('task-filter').value='all';$('mode-filter').value='all';}
 const mode=selectedFailure();
 $('population-note').textContent=mode?`${shortTask(mode.task)} · ${mode.failure_mode}.`:populationKind==='category'&&$('category-filter').value!=='all'?`${$('category-filter').value} · pooled across tasks.`:selectedTask()!=='all'?`${shortTask(selectedTask())} · all failure modes.`:'All four tasks and all failure modes.';
}
function renderResults(){renderMain();renderAblations();renderSensitivity();const mode=selectedFailure(),population=mode?`${shortTask(mode.task)} / ${mode.failure_mode}`:populationKind==='category'&&$('category-filter').value!=='all'?$('category-filter').value:selectedTask()==='all'?'All tasks':shortTask(selectedTask());$('population-summary').textContent=population+($('origin-filter').value==='all'?'':` · ${$('origin-filter').selectedOptions[0].textContent}`)+($('calibration-filter').value==='functional_band'?' · time-varying bands':'');}
function inspect(method,view){
 const matching=g=>g.outcome==='failure'&&originMode(g);
 const task=D.catalog.find(t=>(selectedTask()==='all'||t.id===selectedTask())&&t.groups.some(g=>matching({...g,task:t.id})));
 if(!task)return;
 const replay=replays.get(task.id),index=task.groups.findIndex(g=>matching({...g,task:task.id}));
 replayPanels.get(task.id).selectTask(task.id,{load:false});
 replay.preferredMethod=method;replay.selectGroup(index);replay.selectView(view);replay.activate();
 replay.root.closest('.replay-panel').scrollIntoView({block:'start'});replay.root.focus({preventScroll:true});
}
function revealSection(hash){
 const section=document.getElementById(hash.slice(1)),disclosure=section?.querySelector(':scope > .section-disclosure');
 if(!disclosure)return null;
 disclosure.open=true;
 return section;
}
function init(){
 initPopulationFilters();
 for(const id of ['origin-filter','calibration-filter'])$(id).onchange=renderResults;
 document.querySelectorAll('[name=comparison-view]').forEach(radio=>{radio.onchange=renderResults;});
 $('match-far').onchange=renderMain;
 $('sensitivity-matched').onchange=renderAblations;
 $('shared-min').oninput=renderTiming;
 $('latency-family').onchange=renderTiming;
 document.querySelectorAll('[name=latency-view]').forEach(radio=>{radio.onchange=renderTiming;});
 renderResults();renderSeedRobustness();renderTiming();
 initReplays();
 document.addEventListener('click',e=>{
  const a=e.target.closest('[data-inspect]');if(a)inspect(a.dataset.inspect,a.dataset.view);
  const link=e.target.closest('a[href^="#"]');if(link)revealSection(link.hash);
 });
 window.addEventListener('hashchange',()=>revealSection(location.hash)?.scrollIntoView({block:'start'}));
 revealSection(location.hash)?.scrollIntoView({block:'start'});
 document.addEventListener('keydown',e=>{
  if(e.target.matches('button,select,textarea,input:not([type=range]),summary,a')||e.target.isContentEditable||!activeReplay)return;
  const rect=activeReplay.root.getBoundingClientRect();if(rect.bottom<0||rect.top>innerHeight)return;
  let handled=true;switch(e.code){case 'Space':activeReplay.togglePlay();break;case 'ArrowLeft':activeReplay.seekFrame(activeReplay.frameAt(activeReplay.video.currentTime)-(e.shiftKey?10:1));break;case 'ArrowRight':activeReplay.seekFrame(activeReplay.frameAt(activeReplay.video.currentTime)+(e.shiftKey?10:1));break;case 'KeyN':activeReplay.moveEpisode(1);break;case 'KeyP':activeReplay.moveEpisode(-1);break;default:handled=false;}if(handled)e.preventDefault();
 });
 if('IntersectionObserver'in window){const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){document.querySelectorAll('.topbar nav a').forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id));}},{rootMargin:'-20% 0px -60% 0px'});document.querySelectorAll('section.section').forEach(s=>observer.observe(s));}
 document.documentElement.dataset.ready='true';
}
try{init();}catch(e){console.error(e);const p=document.createElement('p');p.className='error';p.textContent='The report could not be opened: '+e.message;document.querySelector('main').prepend(p);}
