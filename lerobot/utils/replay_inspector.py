"""Inspect labeled PRL recordings in a local browser, without HF conversion or a GPU.

From the repository root, in the lerobot environment:
    python lerobot/utils/replay_inspector.py
    python lerobot/utils/replay_inspector.py --port 8766 --no-browser
    python lerobot/utils/replay_inspector.py --scores main --score-view wrist --port 8766

Reads configs/benchmark/prl_episode_labels.yaml and the preparation.json episode
mapping, then loads one raw NPZ episode at a time. Both camera views share a frame
index and recorded wall-clock timestamps (including simulation). Space: play/pause;
arrows: one frame; shift + arrows: ten frames; N/P: next/previous episode.
Human annotators normally save one first-failure frame for both cameras. Enable
separate views to label fixed/wrist visibility onsets independently. Labels are
per annotator and per view; success episodes cannot have failure onsets.
New annotators inherit the first annotator's existing induced simulation onsets.
Copies are marked with their source and remain editable, independent snapshots.
Recordings are read-only. Label updates are locked and saved atomically.
Optional --scores main (or an archived offline report directory) displays saved
per-view scores and the original episode-calibrated threshold below the images.
No scoring, calibration refit or smoothing is performed. Deep links accept
?task=shunt_insertion&episode=41 (prepared episode IDs).
Dependencies: numpy, PyYAML and Pillow (already present in the lerobot environment).
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import tempfile
from urllib.parse import parse_qs, urlsplit
import webbrowser
from uuid import uuid4

import numpy as np
from PIL import Image
import yaml


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PRL replay inspector</title>
<link rel="icon" href="data:,">
<style>
:root {color-scheme:dark;--bg:#101820;--panel:#192630;--border:#344650;--muted:#a5b9c4;--accent:#5ae0b4}
* {box-sizing:border-box} body {margin:0;background:var(--bg);color:#eef5f7;font:15px system-ui,sans-serif}
[hidden] {display:none!important}
main {max-width:1180px;margin:20px auto;padding:0 22px 24px}
h1 {font-size:26px;margin:0 0 6px} p {margin:0;color:var(--muted);line-height:1.5}
.selectors {display:grid;grid-template-columns:1fr 1.5fr 1fr;gap:16px;margin:18px 0 14px}
label {display:block;color:var(--muted);font-size:13px} select {display:block;width:100%;margin-top:7px}
select,button,input[type=text] {background:var(--panel);color:#eef5f7;border:1px solid var(--border);border-radius:8px;padding:10px;font:inherit}
button {cursor:pointer} button:hover:enabled {border-color:var(--accent)} button:disabled,select:disabled {opacity:.4;cursor:default}
button.primary {background:var(--accent);color:#092c24;font-weight:650;border-color:var(--accent);min-width:110px}
.info {display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.badge {font-size:12px;border:1px solid var(--border);border-radius:20px;padding:5px 10px;color:var(--muted)}
.badge.natural {color:#83d7ff;border-color:#397691}.badge.induced {color:#ffc78c;border-color:#8c6841}
.filename {font-family:ui-monospace,monospace;font-size:12px;color:var(--muted)}
.workspace {display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:18px;align-items:start}
.replay {min-width:0}.cameras {display:grid;grid-template-columns:1fr 1fr;gap:12px}
figure {margin:0;background:#050a0d;border:1px solid var(--border);border-radius:10px;overflow:hidden}
figcaption {padding:8px 12px;background:var(--panel);font-size:13px;color:var(--muted)}
canvas {display:block;width:100%;height:auto;aspect-ratio:1;max-height:min(330px,42vh);object-fit:contain}
#score-panel {margin-top:14px;padding:12px;background:var(--panel);border:1px solid var(--border);border-radius:10px}
#score-chart {display:block;width:100%;height:auto;cursor:crosshair} #score-readout {font-size:12px;margin:6px 0;color:var(--accent)}
.timeline {margin-top:14px;padding:14px;background:var(--panel);border:1px solid var(--border);border-radius:10px}
.readouts {display:flex;justify-content:space-between;gap:10px;font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px}
input[type=range] {display:block;width:100%;height:24px;margin:14px 0;accent-color:var(--accent);cursor:pointer}
.seek-track {position:relative} #onset-markers {position:absolute;inset:0 8px;pointer-events:none}
.onset-marker {position:absolute;top:0;height:24px;width:3px;padding:0;border:0;border-radius:0;transform:translateX(-50%);background:#ffc78c;pointer-events:auto}
.onset-marker.selected {width:4px;background:var(--accent)}
.annotation {position:sticky;top:16px;padding:16px;background:var(--panel);border:1px solid var(--border);border-radius:10px;min-width:0}
.annotation .help {margin-top:8px}
.annotation-controls {display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:end;margin:12px 0}
.annotation-controls .annotator-picker,.annotation-controls .view-mode,#view-picker,#set-onset {grid-column:1/-1}
.view-mode {display:flex;gap:8px;align-items:center}.view-mode input {accent-color:var(--accent)}
figure.annotation-target {border-color:var(--accent)}
.annotation-controls select,.annotation-controls input {width:100%;min-width:0}
.annotation-controls input {display:block;margin-top:7px}
.annotation-controls .view-mode input {margin:0}
.annotation-controls button {min-width:0;padding:10px 8px}
#onset-status,#onset-list {font-size:13px;color:var(--muted);margin:8px 0;overflow-wrap:anywhere}
#annotation-message {min-height:20px;font-size:13px;color:var(--accent)} #annotation-message.error {color:#ffabab}
.controls {display:flex;align-items:center;gap:9px;flex-wrap:wrap}.controls select {width:auto;margin:0;padding:8px}
.controls label {display:flex;align-items:center;gap:6px;margin-left:5px}.controls input {accent-color:var(--accent)}
.group-progress {display:flex;gap:12px;align-items:center;margin:17px 0;font-size:13px;color:var(--muted)}
progress {flex:1;height:8px;accent-color:var(--accent)}
#status {min-height:23px;margin-top:12px;font-size:13px} #status.error {color:#ffabab}
.help {margin-top:14px;font-size:12px} kbd {color:#d7e8ee}
@media(max-width:850px) {.workspace {grid-template-columns:1fr}.annotation {position:static}}
@media(max-width:650px) {main {margin-top:20px;padding:0 12px 20px}.selectors {grid-template-columns:1fr}.cameras {gap:8px}.controls {gap:6px}button {padding:9px}}
</style>
</head>
<body><main>
<h1>PRL replay inspector</h1>
<p>Select a task and label group to inspect the fixed and wrist recordings together.</p>
<div class="selectors">
<label>Task<select id="task" disabled aria-label="Task"></select></label>
<label>Label group<select id="group" disabled aria-label="Label group"></select></label>
<label>Episode<select id="episode" disabled aria-label="Episode"></select></label>
</div>
<div class="info"><strong id="episode-title">Loading labels…</strong><span id="origin" class="badge" hidden></span><span id="filename" class="filename"></span></div>
<div class="workspace">
<div class="replay">
<div class="cameras">
<figure id="fix-panel"><figcaption>Fixed camera</figcaption><canvas id="fix" width="224" height="224" aria-label="Fixed camera replay"></canvas></figure>
<figure id="wrist-panel"><figcaption>Wrist camera</figcaption><canvas id="wrist" width="224" height="224" aria-label="Wrist camera replay"></canvas></figure>
</div>
<section id="score-panel" hidden aria-label="Cached benchmark score">
<strong id="score-title"></strong><p id="score-readout"></p>
<svg id="score-chart" viewBox="0 0 760 250" role="img" aria-label="Score, calibrated threshold, onset and playback cursor"></svg>
<p class="help" id="score-note"></p></section>
<div class="timeline">
<div class="readouts"><span id="frame-label">Frame —</span><span id="time-label">0:00.0 / 0:00.0</span></div>
<div class="seek-track"><input id="seek" type="range" min="0" max="0" value="0" step="1" disabled aria-label="Frame position"><div id="onset-markers" aria-label="Saved failure onset markers"></div></div>
<div class="controls">
<button id="prev" disabled>← Previous episode</button><button id="play" class="primary" disabled>Play</button><button id="next" disabled>Next episode →</button>
<button id="back-frame" disabled title="Previous frame">−1 frame</button><button id="next-frame" disabled title="Next frame">+1 frame</button>
<select id="speed" aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select>
<label><input id="auto-next" type="checkbox" checked> Auto-next episode</label>
</div>
</div>
</div>
<aside class="annotation" aria-label="Failure onset annotation">
<strong>Failure onset · raw frame index (zero-based)</strong>
<p class="help">One click normally saves the same onset for both cameras. Enable separate views when visibility differs. Frame 0 is valid.</p>
<div class="annotation-controls">
<label class="annotator-picker">Human annotator<select id="annotator" aria-label="Human annotator"><option value="">Choose an annotator</option></select></label>
<label>New annotator<input id="annotator-name" type="text" maxlength="80" placeholder="Name" aria-label="New annotator name"></label>
<button id="add-annotator">Add annotator</button>
<p class="help annotator-picker">New annotators inherit the first annotator’s induced simulation onsets. Copied labels remain editable.</p>
<label class="view-mode"><input id="separate-views" type="checkbox"> Label views separately</label>
<label id="view-picker" hidden>Camera to annotate<select id="onset-view" aria-label="Camera to annotate"><option value="fix">Fixed camera</option><option value="wrist">Wrist camera</option></select></label>
<button id="set-onset" disabled>Set onset at current frame</button>
<button id="jump-onset" disabled>Jump to my onset</button>
<button id="clear-onset" disabled>Clear my onset</button>
</div>
<p id="onset-status">Select a failure episode to annotate.</p>
<p id="onset-list"></p>
<p id="annotation-message" role="status" aria-live="polite"></p>
</aside>
</div>
<div class="group-progress"><span id="group-counter">Episode —</span><progress id="group-progress" max="1" value="0" aria-label="Position in label group"></progress></div>
<p id="status" role="status" aria-live="polite"></p>
<p class="help"><kbd>Space</kbd> play/pause · <kbd>← / →</kbd> one frame · <kbd>Shift + ← / →</kbd> ten frames · <kbd>P / N</kbd> previous/next episode. Timeline markers show saved onsets; click a marker to jump to it. Recordings are read-only; annotations are saved to the labels YAML.</p>
</main>
<script>
'use strict';
const $ = id => document.getElementById(id);
let catalog, task, group, episodePosition=0, metadata=null, frame=0, ready=false;
let playing=false, playbackToken=0, selectionToken=0, renderToken=0, baseTime=0, baseClock=0;
let rendering=false, saving=false, addingAnnotator=false;
const message=(text,error=false)=>{$('status').textContent=text;$('status').classList.toggle('error',error);};
const formatTime=seconds=>`${Math.floor(seconds/60)}:${(seconds%60).toFixed(1).padStart(4,'0')}`;
function options(select,entries) {
  select.replaceChildren(...entries.map(([value,text])=>new Option(text,String(value))));
  select.disabled=entries.length===0;
}
function controls() {
  for (const id of ['play','seek','back-frame','next-frame']) $(id).disabled=!ready;
  $('prev').disabled=!ready || episodePosition===0;
  $('next').disabled=!ready || episodePosition>=group.episode_ids.length-1;
  $('back-frame').disabled=!ready || frame===0;
  $('next-frame').disabled=!ready || frame===metadata.frame_count-1;
  $('play').textContent=playing?'Pause':(ready && frame===metadata.frame_count-1?'Replay':'Play');
  annotationControls();
}
function pause() {playing=false;playbackToken++;controls();}
async function getJSON(url) {
  const response=await fetch(url,{cache:'no-store'}), body=await response.json();
  if (!response.ok) throw Error(body.error || `Request failed (${response.status})`);
  return body;
}
async function postJSON(url,data) {
  const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const body=await response.json();
  if(!response.ok) throw Error(body.error || `Request failed (${response.status})`);
  return body;
}
function renderScorePlot() {
  const s=metadata?.score_plot; $('score-panel').hidden=!s;
  if(!s)return;
  const left=64,right=744,top=22,bottom=204,last=Math.max(1,metadata.frame_count-1);
  const low=Math.min(...s.values,s.threshold),high=Math.max(...s.values,s.threshold),pad=Math.max((high-low)*.1,Math.abs(high)*.02,1e-6);
  const ymin=low-pad,ymax=high+pad,x=f=>left+(right-left)*f/last,y=v=>bottom-(bottom-top)*(v-ymin)/(ymax-ymin);
  const aid=$('annotator').value||catalog.annotators[0]?.id;
  const annotator=catalog.annotators.find(a=>a.id===aid)?.name||'unselected';
  const onset=onsetViews(metadata.onset_annotations?.[aid])[s.view]?.frame_index;
  $('score-title').textContent=`${s.method.toUpperCase().replaceAll('_','-')} · ${s.view==='wrist'?'Wrist':'Fixed'} · cached offline scores`;
  $('score-note').textContent=`Threshold ${s.threshold.toFixed(3)} · original ${s.n_calibration} calibration episodes · strict score > threshold. No smoothing. `+
    `Onset: ${annotator}, ${onset===undefined?'unannotated':'frame '+onset}. Click the plot to seek. Full recorded curve shown for inspection; no online computation delay. Checkpoint: ${s.checkpoint}`;
  let svg='';
  for(let i=0;i<=4;i++){
    const v=ymin+(ymax-ymin)*i/4,yy=y(v);
    svg+=`<line x1="${left}" x2="${right}" y1="${yy}" y2="${yy}" stroke="#344650"/><text x="${left-8}" y="${yy+4}" text-anchor="end" fill="#a5b9c4" font-size="12">${v.toFixed(1)}</text>`;
    const f=Math.round(last*i/4),xx=x(f);svg+=`<text x="${xx}" y="225" text-anchor="middle" fill="#a5b9c4" font-size="12">${f}</text>`;
  }
  svg+=`<text x="404" y="246" text-anchor="middle" fill="#a5b9c4" font-size="12">Raw recording frame index</text>`;
  svg+=`<line x1="${left}" x2="${right}" y1="${y(s.threshold)}" y2="${y(s.threshold)}" stroke="#ffdf73" stroke-width="2" stroke-dasharray="7 4"/>`;
  svg+=`<path d="${s.raw_frames.map((f,i)=>(i?'L':'M')+x(f)+','+y(s.values[i])).join(' ')}" fill="none" stroke="#5ae0b4" stroke-width="1.8"/>`;
  for(let i=0;i<s.values.length;i++)if(s.values[i]>s.threshold)svg+=`<circle cx="${x(s.raw_frames[i])}" cy="${y(s.values[i])}" r="2.4" fill="#ff9090"/>`;
  if(onset!==undefined)svg+=`<line x1="${x(onset)}" x2="${x(onset)}" y1="${top}" y2="${bottom}" stroke="#ffc78c" stroke-width="2" stroke-dasharray="4 3"/><text x="${Math.min(right-40,Math.max(left+20,x(onset)))}" y="14" text-anchor="middle" fill="#ffc78c" font-size="12">Onset</text>`;
  svg+=`<line x1="${x(frame)}" x2="${x(frame)}" y1="${top}" y2="${bottom}" stroke="white" stroke-width="1.5"/>`;
  $('score-chart').innerHTML=svg;
  const index=s.raw_frames.indexOf(frame);
  $('score-readout').textContent=`Frame ${frame} · ${metadata.times[frame].toFixed(2)} s · `+(index<0?'No scored endpoint at this frame':`Score ${s.values[index].toFixed(3)} · ${s.values[index]>s.threshold?'ALARM':'below/equal threshold'}`);
}
function annotationMessage(text,error=false) {
  $('annotation-message').textContent=text;$('annotation-message').classList.toggle('error',error);
}
function setAnnotators(entries,preferred=$('annotator').value) {
  catalog.annotators=entries;
  options($('annotator'),[['','Choose an annotator'],...entries.map(item=>[item.id,item.name])]);
  $('annotator').value=entries.some(item=>item.id===preferred)?preferred:'';
  annotationControls();
}
function onsetViews(entry) {
  if(!entry) return {};
  if(entry.views) return entry.views;
  // Older label files are displayed as shared until the next save migrates them.
  return {fix:entry,wrist:entry};
}
function copyNote(label) {
  const source=label?.copied_from_annotator;
  if(!source) return '';
  const name=(catalog?.annotators || []).find(item=>item.id===source)?.name || source;
  return ` (copied from ${name})`;
}
function annotationTarget() {return $('separate-views').checked?$('onset-view').value:'both';}
function ownOnset() {
  const views=onsetViews(metadata?.onset_annotations?.[$('annotator').value]);
  const target=annotationTarget();
  if(target!=='both') return views[target];
  return views.fix && views.wrist && views.fix.frame_index===views.wrist.frame_index?views.fix:null;
}
function resetAnnotationMode() {
  const entry=metadata?.onset_annotations?.[$('annotator').value], views=onsetViews(entry);
  // New annotations always start shared. Resume an existing separate annotation
  // in its saved mode, selecting whichever camera still needs its onset.
  $('separate-views').checked=entry?.mode==='separate';
  $('onset-view').value=!views.fix?'fix':!views.wrist?'wrist':'fix';
  annotationControls();
}
function annotationControls() {
  renderScorePlot();
  const id=$('annotator').value, onsets=metadata?.onset_annotations || {}, own=onsets[id];
  const views=onsetViews(own), target=annotationTarget(), selected=ownOnset();
  const eligible=ready && metadata.outcome==='failure';
  const targetLabel=target==='both'?'both cameras':target==='fix'?'fixed camera':'wrist camera';
  $('view-picker').hidden=target==='both';
  $('separate-views').disabled=!eligible || saving;
  $('onset-view').disabled=!eligible || saving;
  $('annotator').disabled=saving || addingAnnotator;
  $('set-onset').disabled=!eligible || !id || saving || rendering || addingAnnotator;
  $('set-onset').textContent=`Set ${targetLabel} onset at frame ${ready?frame:'—'}`;
  $('clear-onset').disabled=!eligible || !(target==='both'?own:views[target]) || saving || addingAnnotator;
  $('clear-onset').textContent=target==='both'?'Clear both onsets':`Clear ${target} onset`;
  $('jump-onset').disabled=!eligible || !selected || rendering;
  $('jump-onset').textContent=target==='both'?'Jump to shared onset':`Jump to ${target} onset`;
  $('add-annotator').disabled=addingAnnotator || saving;
  for(const view of ['fix','wrist']) $(view+'-panel').classList.toggle('annotation-target',eligible && (target==='both' || target===view));
  if(!ready) $('onset-status').textContent='Select a failure episode to annotate.';
  else if(!eligible) $('onset-status').textContent='Success episode — failure onset labeling is disabled.';
  else if(!id) $('onset-status').textContent='Add or select a human annotator to label this episode.';
  else {
    $('onset-status').textContent=['fix','wrist'].map(view=>{
      const value=views[view], name=view==='fix'?'Fixed':'Wrist';
      return value?`${name}: frame ${value.frame_index} · ${formatTime(metadata.times[value.frame_index])}${copyNote(value)}`:`${name}: unannotated`;
    }).join(' · ');
    if(target==='both' && own?.mode==='separate') $('onset-status').textContent+=' · Saving now replaces both camera labels.';
  }
  const names=new Map((catalog?.annotators || []).map(item=>[item.id,item.name]));
  $('onset-list').textContent=eligible?Object.entries(onsets).map(([key,value])=>{
    const v=onsetViews(value);
    if(v.fix && v.wrist && v.fix.frame_index===v.wrist.frame_index && copyNote(v.fix)===copyNote(v.wrist)) return `${names.get(key) || key}: both ${v.fix.frame_index}${copyNote(v.fix)}`;
    return `${names.get(key) || key}: fixed ${v.fix?.frame_index ?? 'unannotated'}${copyNote(v.fix)}, wrist ${v.wrist?.frame_index ?? 'unannotated'}${copyNote(v.wrist)}`;
  }).join(' · '):'';
  const byFrame=new Map();
  if(eligible) for(const [key,value] of Object.entries(onsets)) {
    for(const [view,label] of Object.entries(onsetViews(value))) {
      if(!byFrame.has(label.frame_index)) byFrame.set(label.frame_index,[]);
      byFrame.get(label.frame_index).push({id:key,view});
    }
  }
  $('onset-markers').replaceChildren(...[...byFrame].map(([index,entries])=>{
    const marker=document.createElement('button');
    const active=entries.some(entry=>entry.id===id && (target==='both' || entry.view===target));
    marker.type='button';marker.className=`onset-marker${active?' selected':''}`;
    marker.style.left=`${metadata.frame_count>1?index/(metadata.frame_count-1)*100:0}%`;
    marker.title=`${entries.map(entry=>`${names.get(entry.id) || entry.id} (${entry.view==='fix'?'fixed':'wrist'})`).join(', ')}: onset at frame ${index}`;
    marker.setAttribute('aria-label',marker.title);marker.dataset.frame=index;
    marker.addEventListener('click',()=>{pause();showFrame(index);});return marker;
  }));
}
async function addAnnotator() {
  if(addingAnnotator) return;
  const name=$('annotator-name').value.trim();
  if(!name) {annotationMessage('Enter a name for the new annotator.',true);return;}
  addingAnnotator=true;annotationControls();
  try {
    const result=await postJSON('/api/annotators',{name});
    setAnnotators(result.annotators,result.annotator_id);resetAnnotationMode();$('annotator-name').value='';
    try {localStorage.setItem('prl-replay-annotator',result.annotator_id);} catch(error) {}
    // The episode metadata was loaded before this annotator's copies existed.
    // Refresh only its labels, preserving the current frame and playback state.
    if(ready && metadata) {
      const selection=selectionToken;
      try {
        const updated=await getJSON(episodeURL('episode'));
        if(selection===selectionToken && metadata) {
          metadata.onset_annotations=updated.onset_annotations;resetAnnotationMode();
        }
      } catch(error) {
        annotationMessage(`Annotator added: ${name}. Reload the episode to show copied labels: ${error.message}`,true);
        return;
      }
    }
    const copied=result.copied_onsets?.episodes || 0;
    annotationMessage(`Annotator added: ${name}. Inherited ${copied} induced simulation episode onset(s).`);
  } catch(error) {annotationMessage(error.message,true);}
  finally {addingAnnotator=false;annotationControls();}
}
async function saveOnset(clear=false) {
  if((clear?$('clear-onset'):$('set-onset')).disabled) return;
  pause();const selection=selectionToken, id=$('annotator').value;
  const target={task:task.id,episode:group.episode_ids[episodePosition],annotator_id:id,frame_index:clear?null:frame,view:annotationTarget()};
  saving=true;annotationControls();annotationMessage('Saving…');
  try {
    const result=await postJSON('/api/onset',target);
    setAnnotators(result.annotators);
    if(selection===selectionToken) {
      metadata.onset_annotations=result.onset_annotations;
      annotationMessage(clear?`Cleared ${target.view==='both'?'both camera':target.view} onset label(s).`:`Saved ${target.view==='both'?'both camera':target.view} onset at frame ${target.frame_index}.`);
    }
  } catch(error) {if(selection===selectionToken) annotationMessage(error.message,true);}
  finally {saving=false;annotationControls();}
}
function episodeURL(endpoint,index=null) {
  const params=new URLSearchParams({task:task.id,episode:group.episode_ids[episodePosition]});
  if(index!==null) params.set('frame',index);
  return `/api/${endpoint}?${params}`;
}
async function showFrame(index) {
  if(!ready) return false;
  index=Math.max(0,Math.min(metadata.frame_count-1,index));
  const request=++renderToken, selection=selectionToken, meta=metadata;
  rendering=true;annotationControls();
  const image=new Image();image.src=episodeURL('frame',index);
  try {await image.decode();} catch(error) {
    if(request===renderToken && selection===selectionToken) {rendering=false;pause();message('Could not load this camera frame. Try selecting the episode again.',true);}
    return false;
  }
  if(request!==renderToken || selection!==selectionToken) return false;
  const [fw,fh]=meta.fix_size,[ww,wh]=meta.wrist_size;
  $('fix').getContext('2d').drawImage(image,0,0,fw,fh,0,0,fw,fh);
  $('wrist').getContext('2d').drawImage(image,fw,0,ww,wh,0,0,ww,wh);
  frame=index;rendering=false;$('seek').value=frame;
  $('frame-label').textContent=`Frame index ${frame} · ${frame+1} / ${meta.frame_count}`;
  $('time-label').textContent=`${formatTime(meta.times[frame])} / ${formatTime(meta.duration)}`;
  controls();return true;
}
async function loadEpisode(position,autoplay=false) {
  pause();const selection=++selectionToken;++renderToken;ready=false;metadata=null;rendering=false;
  episodePosition=position;$('episode').value=position;controls();
  $('episode-title').textContent=`Episode ${group.episode_ids[position]}`;
  $('filename').textContent='';$('origin').hidden=true;
  $('frame-label').textContent='Loading frames…';$('time-label').textContent='—';
  for(const id of ['fix','wrist']) $(id).getContext('2d').clearRect(0,0,$(id).width,$(id).height);
  $('group-counter').textContent=`Episode ${position+1} / ${group.episode_ids.length}`;
  $('group-progress').max=group.episode_ids.length;$('group-progress').value=position+1;
  message('Loading episode…');
  annotationMessage('');
  try {
    const result=await getJSON(episodeURL('episode'));
    if(selection!==selectionToken) return;
    metadata=result;frame=0;ready=true;
    setAnnotators(result.annotators);resetAnnotationMode();
    for(const [id,size] of [['fix',result.fix_size],['wrist',result.wrist_size]]) {
      $(id).width=size[0];$(id).height=size[1];$(id).style.aspectRatio=`${size[0]} / ${size[1]}`;
    }
    $('seek').max=result.frame_count-1;$('seek').value=0;
    $('filename').textContent=result.raw_file;
    $('origin').textContent=group.failure_origin.replaceAll('_',' ');
    $('origin').className=`badge ${group.failure_origin}`;
    $('origin').hidden=group.failure_origin==='not_applicable';
    const displayed=await showFrame(0);
    if(selection!==selectionToken || !displayed) return;
    message(result.timing_note || 'Recorded wall time · 1× follows the original recording speed.');
    if(autoplay) play();
  } catch(error) {
    if(selection!==selectionToken) return;
    ready=false;controls();message(error.message,true);
  }
}
function chooseGroup(index,autoplay=false,episodeId=null) {
  group=task.groups[index];$('group').value=index;
  options($('episode'),group.episode_ids.map((id,i)=>[i,`Episode ${id} (${i+1}/${group.episode_ids.length})`]));
  const position=episodeId===null?0:group.episode_ids.indexOf(episodeId);
  if(position<0)throw Error('Episode is not in the selected label group');
  return loadEpisode(position,autoplay);
}
function chooseTask(id,episodeId=null) {
  task=catalog.tasks.find(item=>item.id===id);$('task').value=id;
  options($('group'),task.groups.map((item,i)=>[i,`${item.title} · ${item.episode_ids.length} episodes`]));
  const index=episodeId===null?0:task.groups.findIndex(g=>g.episode_ids.includes(episodeId));
  if(index<0)throw Error('Unknown episode for this task');
  return chooseGroup(index,false,episodeId);
}
function frameAt(time) {
  let lo=0,hi=metadata.times.length;
  while(lo<hi) {const mid=(lo+hi)>>1;if(metadata.times[mid]<=time) lo=mid+1;else hi=mid;}
  return Math.max(0,lo-1);
}
function play() {
  if(!ready || playing) return;
  if(frame===metadata.frame_count-1 && frame>0) {
    showFrame(0).then(displayed=>{if(displayed) play();});return;
  }
  playing=true;const token=++playbackToken;
  baseTime=metadata.times[frame];baseClock=performance.now();controls();
  const tick=async now=>{
    if(!playing || token!==playbackToken) return;
    // Anchor to elapsed wall time so image-fetch/decode delays do not accumulate.
    const time=baseTime+(now-baseClock)/1000*Number($('speed').value);
    const nextFrame=frameAt(time);
    if(nextFrame!==frame && !(await showFrame(nextFrame))) return;
    if(!playing || token!==playbackToken) return;
    if(time>=metadata.end_time) {
      pause();
      if($('auto-next').checked && episodePosition<group.episode_ids.length-1) loadEpisode(episodePosition+1,true);
      else message(episodePosition===group.episode_ids.length-1?'End of label group.':'End of episode.');
      return;
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
function step(amount) {if(ready) {pause();showFrame(frame+amount);}}
$('score-chart').addEventListener('click',event=>{
  if(!ready)return; const point=$('score-chart').createSVGPoint();point.x=event.clientX;point.y=event.clientY;
  const local=point.matrixTransform($('score-chart').getScreenCTM().inverse());
  pause();showFrame(Math.round((local.x-64)/(744-64)*(metadata.frame_count-1)));
});
$('task').addEventListener('change',()=>chooseTask($('task').value));
$('group').addEventListener('change',()=>chooseGroup(Number($('group').value),true));
$('episode').addEventListener('change',()=>loadEpisode(Number($('episode').value),true));
$('prev').addEventListener('click',()=>loadEpisode(episodePosition-1,true));
$('next').addEventListener('click',()=>loadEpisode(episodePosition+1,true));
$('play').addEventListener('click',()=>playing?pause():play());
$('back-frame').addEventListener('click',()=>step(-1));$('next-frame').addEventListener('click',()=>step(1));
$('seek').addEventListener('input',()=>{pause();showFrame(Number($('seek').value));});
$('speed').addEventListener('change',()=>{if(playing) {pause();play();}});
$('annotator').addEventListener('change',()=>{
  try {localStorage.setItem('prl-replay-annotator',$('annotator').value);} catch(error) {}
  annotationMessage('');resetAnnotationMode();
});
$('add-annotator').addEventListener('click',addAnnotator);
$('annotator-name').addEventListener('keydown',event=>{if(event.key==='Enter') {event.preventDefault();addAnnotator();}});
$('set-onset').addEventListener('click',()=>saveOnset());
$('clear-onset').addEventListener('click',()=>saveOnset(true));
$('separate-views').addEventListener('change',()=>{annotationMessage('');annotationControls();});
$('onset-view').addEventListener('change',()=>{annotationMessage('');annotationControls();});
$('jump-onset').addEventListener('click',()=>{const onset=ownOnset();if(onset) {pause();showFrame(onset.frame_index);}});
document.addEventListener('keydown',event=>{
  const target=event.target;
  const editing=target.isContentEditable || ['SELECT','TEXTAREA'].includes(target.tagName) ||
    (target.tagName==='INPUT' && target.type!=='range');
  if(event.ctrlKey || event.altKey || event.metaKey || editing) return;
  if(event.code==='Space') {event.preventDefault();playing?pause():play();}
  else if(event.key==='ArrowLeft' || event.key==='ArrowRight') {event.preventDefault();step((event.key==='ArrowLeft'?-1:1)*(event.shiftKey?10:1));}
  else if(event.key.toLowerCase()==='n' && !$('next').disabled) loadEpisode(episodePosition+1,true);
  else if(event.key.toLowerCase()==='p' && !$('prev').disabled) loadEpisode(episodePosition-1,true);
});
getJSON('/api/catalog').then(result=>{
  catalog=result;let preferred='';try {preferred=localStorage.getItem('prl-replay-annotator') || '';} catch(error) {}
  setAnnotators(catalog.annotators,preferred);
  options($('task'),catalog.tasks.map(item=>[item.id,item.title]));
  const query=new URLSearchParams(location.search),taskId=query.get('task')||catalog.tasks[0].id;
  if(!catalog.tasks.some(t=>t.id===taskId))throw Error('Unknown task in replay link');
  const episodeId=query.has('episode')?Number(query.get('episode')):null;
  if(episodeId!==null&&(!Number.isInteger(episodeId)||episodeId<0))throw Error('Invalid episode in replay link');
  return chooseTask(taskId,episodeId);
}).catch(error=>message(error.message,true));
</script></body></html>
"""


class LabelsDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def represent_list(dumper, values):
    # Keep the episode ranges compact when annotations are saved.
    flow = len(values) == 2 and all(type(value) is int for value in values)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", values, flow_style=flow)


LabelsDumper.add_representer(list, represent_list)

ONSET_HEADER = """# Failure onsets are per episode, human annotator, and camera view.
# Frame indices are zero-based in the raw recording. Match prepared observations
# using observation.raw_frame_index (preparation may omit a terminal raw frame).
# mode: shared saves one onset into both views; separate edits only the chosen view.
# failure_onsets: {episode_id: {annotator_id: {mode: shared, views: {fix: {frame_index: 42, updated_at: "UTC time"}, wrist: {frame_index: 42, updated_at: "UTC time"}}}}}
# An absent view is unannotated; frame 0 is valid. Success episodes have no onsets.
# Existing Tissue Retraction / Needle Intrusion labels were retained for wrist
# only; the fixed view requires a separate human annotation.
# New annotators copy the FIRST registered annotator's induced simulation onsets.
# Each copied view records copied_from_annotator and copied_at; updated_at retains
# the source timestamp. Manual saves replace that view's copy provenance.
"""


def onset_views(entry: dict) -> dict:
    """Read older shared annotations as two views, without modifying their bytes."""
    return entry.get("views", {"fix": entry, "wrist": entry})


def upgrade_onset_schema(manifest: dict) -> None:
    for node in manifest["tasks"].values():
        for annotations in node.get("failure_onsets", {}).values():
            for annotator, entry in list(annotations.items()):
                if "views" not in entry:
                    annotations[annotator] = {"mode": "shared", "views": {
                        "fix": deepcopy(entry), "wrist": deepcopy(entry)}}
    manifest["schema_version"] = 3


def copy_simulated_induced_onsets(manifest: dict, annotator_id: str, *, overwrite: bool = False) -> dict:
    """Copy existing source labels once; natural/real/success labels are excluded."""
    annotators = manifest.get("annotators", {})
    if annotator_id not in annotators:
        raise ValueError("Choose a registered target annotator.")
    source_id = next(iter(annotators))
    counts = {"source_annotator_id": source_id, "episodes": 0, "views": 0}
    if annotator_id == source_id:
        return counts
    copied_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for task in ("needle_pickup", "ring_over_post"):
        node = manifest.get("tasks", {}).get(task)
        if node is None:
            continue
        onsets = node.get("failure_onsets", {})
        for group in node["groups"]:
            if group.get("outcome") != "failure" or group.get("failure_origin") != "induced":
                continue
            first, last = group["episode_range"]
            for episode in range(first, last + 1):
                annotations = onsets.get(episode, {})
                source = annotations.get(source_id)
                if not source or (annotator_id in annotations and not overwrite):
                    continue
                copied = deepcopy(source)
                for value in copied["views"].values():
                    value["copied_from_annotator"] = source_id
                    value["copied_at"] = copied_at
                annotations[annotator_id] = copied
                counts["episodes"] += 1
                counts["views"] += len(copied["views"])
    return counts


class OnsetLabels:
    """Persist per-annotator labels, including explicitly attributed copies."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.read()

    @staticmethod
    def outcome(manifest: dict, task: str, episode_id: int) -> str:
        node = manifest["tasks"].get(task)
        if node is not None:
            for group in node["groups"]:
                start, end = group["episode_range"]
                if start <= episode_id <= end:
                    return group["outcome"]
        raise ValueError(f"Unknown labeled episode: {task}/{episode_id}")

    @classmethod
    def validate(cls, manifest: dict):
        if not isinstance(manifest, dict) or manifest.get("schema_version", 1) not in {1, 2, 3}:
            raise ValueError("Expected episode labels with schema_version 1, 2 or 3.")
        if type(manifest.get("onset_frame_index_base", 0)) is not int or manifest.get("onset_frame_index_base", 0) != 0:
            raise ValueError("Failure onset frame indices must be zero-based.")
        if manifest.get("onset_frame_id_space", "raw_recording_frame_index") != "raw_recording_frame_index":
            raise ValueError("Failure onset indices must refer to raw recording frames.")
        annotators = manifest.get("annotators", {})
        if not isinstance(annotators, dict):
            raise ValueError("annotators must map annotator IDs to names.")
        for annotator_id, entry in annotators.items():
            if not isinstance(annotator_id, str) or not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"].strip():
                raise ValueError("Each annotator must have a string ID and a nonempty name.")
        for task, node in manifest["tasks"].items():
            onsets = node.get("failure_onsets", {})
            if not isinstance(onsets, dict):
                raise ValueError(f"failure_onsets must be a mapping: {task}")
            for episode_id, annotations in onsets.items():
                if type(episode_id) is not int or cls.outcome(manifest, task, episode_id) != "failure":
                    raise ValueError(f"Only failure episodes may have onset annotations: {task}/{episode_id}")
                if not isinstance(annotations, dict):
                    raise ValueError(f"Expected annotations keyed by annotator ID: {task}/{episode_id}")
                for annotator_id, entry in annotations.items():
                    if annotator_id not in annotators or not isinstance(entry, dict):
                        raise ValueError(f"Unknown or invalid annotator: {task}/{episode_id}/{annotator_id}")
                    if manifest.get("schema_version") == 3 and "views" not in entry:
                        raise ValueError("Schema 3 annotations require a views mapping.")
                    views = onset_views(entry)
                    if not isinstance(views, dict) or not views or not set(views) <= {"fix", "wrist"}:
                        raise ValueError("Onset views must contain fixed and/or wrist camera labels.")
                    if any(not isinstance(value, dict) for value in views.values()):
                        raise ValueError("Each onset view must be a frame annotation mapping.")
                    if "views" in entry:
                        if entry.get("mode") not in {"shared", "separate"}:
                            raise ValueError("Onset mode must be shared or separate.")
                        if entry["mode"] == "shared" and (set(views) != {"fix", "wrist"}
                                or views["fix"].get("frame_index") != views["wrist"].get("frame_index")):
                            raise ValueError("Shared onsets require the same frame in both views.")
                    for value in views.values():
                        index = value.get("frame_index") if isinstance(value, dict) else None
                        if type(index) is not int or index < 0:
                            raise ValueError(f"Onset frame_index must be a nonnegative integer: {task}/{episode_id}")
                        if "copied_from_annotator" in value:
                            source = value["copied_from_annotator"]
                            if (not isinstance(source, str) or source not in annotators or source == annotator_id
                                    or not isinstance(value.get("copied_at"), str) or not value["copied_at"]):
                                raise ValueError("Copied onsets require a different registered source and a copy timestamp.")

    def read(self) -> dict:
        manifest = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.validate(manifest)
        return manifest

    @staticmethod
    def annotators(manifest: dict) -> list[dict]:
        return [{"id": key, "name": entry["name"]} for key, entry in manifest.get("annotators", {}).items()]

    def mutate(self, update) -> dict:
        # Lock the stable directory rather than the YAML inode, which atomic
        # replacement changes. This also serializes separate inspector servers.
        with self.lock:
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            temporary = None
            try:
                fcntl.flock(directory_fd, fcntl.LOCK_EX)
                text = self.path.read_text(encoding="utf-8")
                manifest = yaml.safe_load(text)
                self.validate(manifest)
                upgrade_onset_schema(manifest)
                update(manifest)
                manifest["onset_frame_index_base"] = 0
                manifest["onset_frame_id_space"] = "raw_recording_frame_index"
                self.validate(manifest)
                output = ONSET_HEADER + yaml.dump(manifest, Dumper=LabelsDumper, sort_keys=False, allow_unicode=True)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=f".{self.path.name}-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(output)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, self.path.stat().st_mode & 0o777)
                os.replace(temporary, self.path)
                os.fsync(directory_fd)
                return manifest
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                os.close(directory_fd)

    def add_annotator(self, name: str) -> dict:
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
            raise ValueError("Annotator name must contain 1–80 characters.")
        name = name.strip()
        annotator_id = f"annotator_{uuid4().hex[:12]}"
        copied = {}

        def update(manifest):
            annotators = manifest.setdefault("annotators", {})
            if any(entry["name"].casefold() == name.casefold() for entry in annotators.values()):
                raise ValueError("That annotator name already exists. Select it from the list.")
            annotators[annotator_id] = {"name": name}
            copied.update(copy_simulated_induced_onsets(manifest, annotator_id))

        manifest = self.mutate(update)
        return {"annotator_id": annotator_id, "annotators": self.annotators(manifest), "copied_onsets": copied}

    def set_onset(self, task: str, episode_id: int, annotator_id: str, frame_index: int | None, view: str = "both") -> dict:
        if not isinstance(view, str) or view not in {"both", "fix", "wrist"}:
            raise ValueError("Choose both, fix, or wrist for onset annotation.")
        if frame_index is not None and (type(frame_index) is not int or frame_index < 0):
            raise ValueError("Onset frame_index must be a nonnegative integer.")

        def update(manifest):
            if self.outcome(manifest, task, episode_id) != "failure":
                raise ValueError("Success episodes cannot have failure onset annotations.")
            if annotator_id not in manifest.get("annotators", {}):
                raise ValueError("Select a registered human annotator before saving an onset.")
            onsets = manifest["tasks"][task].setdefault("failure_onsets", {})
            annotations = onsets.setdefault(episode_id, {})
            entry = annotations.get(annotator_id, {"mode": "shared", "views": {}})
            selected = ("fix", "wrist") if view == "both" else (view,)
            for camera in selected:
                if frame_index is None:
                    entry["views"].pop(camera, None)
                else:
                    entry["views"][camera] = {"frame_index": frame_index,
                        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if entry["views"]:
                entry["mode"] = "shared" if view == "both" else "separate"
                annotations[annotator_id] = entry
            else:
                annotations.pop(annotator_id, None)
            if not annotations:
                onsets.pop(episode_id, None)

        manifest = self.mutate(update)
        return {"task": task, "episode_id": episode_id,
                "onset_annotations": manifest["tasks"][task].get("failure_onsets", {}).get(episode_id, {}),
                "annotators": self.annotators(manifest)}


@dataclass
class Episode:
    task: str
    episode_id: int
    raw_file: str
    fix: np.ndarray
    wrist: np.ndarray
    times: np.ndarray
    timing_note: str | None

    def metadata(self) -> dict:
        positive = np.diff(self.times)
        positive = positive[positive > 0]
        last_interval = float(np.median(positive)) if positive.size else 1 / 30
        return {
            "episode_id": self.episode_id,
            "raw_file": self.raw_file,
            "frame_count": len(self.times),
            "fix_size": [int(self.fix.shape[2]), int(self.fix.shape[1])],
            "wrist_size": [int(self.wrist.shape[2]), int(self.wrist.shape[1])],
            "times": self.times.tolist(),
            "duration": float(self.times[-1]),
            "end_time": float(self.times[-1]) + last_interval,
            "timing_source": "estimated_regular_clock" if self.timing_note else "recorded_wall_time",
            "timing_note": self.timing_note,
        }


class CachedScores:
    """Original per-view episode threshold and score endpoints, without inference."""

    def __init__(self, report_dir: Path, method: str = "wm_nc", view: str = "wrist"):
        self.report = json.loads((report_dir / "report.json").read_text())
        self.method, self.view = method, view
        if self.report.get("wm_score_aggregation", "per_view") != "per_view":
            raise ValueError("This replay panel requires a per-view report, not joint/fused scores.")
        protocol = self.report["protocols"]["paper_protocol"]
        if protocol.get("window") != 1:
            raise ValueError("This raw-score replay panel requires an unsmoothed report (window=1).")
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def episode(self, task: str, episode_id: int) -> dict | None:
        from types import SimpleNamespace
        from unified_benchmark import curves
        from results_inspector import resolve_repo_path
        matches = [c for c in self.report["score_caches"]
                   if c["task"] == task and c["method"] == self.method and self.view in c["views"]]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("Ambiguous score cache for the selected task/method/view.")
        with self.lock:
            if task not in self.cache:
                entry = matches[0]
                path = resolve_repo_path(entry["path"])
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError(f"Score cache failed SHA-256 verification: {path}")
                with np.load(path, allow_pickle=False) as source:
                    arrays = {k: source[k] for k in ("episode_index", "frame_index", "raw_frame_index", "wall_timestamp", "scores")}
                settings = entry["scoring_settings"]
                job = SimpleNamespace(task=task, mode=self.method.startswith("wm_"),
                    merged={"data": {"history_len": settings["L"], "pred_horizon": settings["K"]}})
                series = curves(arrays, job, entry["views"].index(self.view), self.report["protocols"]["paper_protocol"])
                records = [r for r in self.report["calibration"] if r["task"] == task and r["method"] == self.method
                           and r["view"] == self.view and r["calibration"] == "episode" and r["protocol"] == "paper"]
                if len(records) != 1 or not np.isfinite(float(records[0]["threshold"])):
                    raise ValueError("Missing or nonfinite saved episode threshold.")
                self.cache[task] = series, records[0], settings["checkpoint"]
                while len(self.cache) > 2:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(task)
            series, record, checkpoint = self.cache[task]
            curve = series[episode_id]
            if not np.isfinite(curve["score"]).all():
                raise ValueError("Nonfinite score curve.")
            return {"method": self.method, "view": self.view, "raw_frames": curve["raw"].tolist(),
                    "values": curve["score"].tolist(), "threshold": float(record["threshold"]),
                    "n_calibration": record["n_calibration"], "checkpoint": Path(checkpoint).parent.parent.name}


def score_report_path(repo: Path, selection: str) -> Path:
    if selection != "main":
        path = Path(selection).resolve()
        return path.parent if path.is_file() else path
    catalog = yaml.safe_load((repo / "lerobot/configs/benchmark/results_catalog.yaml").read_text())
    main = next(r for r in catalog["offline"] if r["id"] == "main")
    matches = []
    for path in (repo / "fomofd/tasks/benchmark_results/offline").glob("*/archive_manifest.json"):
        entry = json.loads(path.read_text())
        if entry["source"] == main["source"] and (not main.get("sha256") or entry["sha256"] == main["sha256"]):
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError(f"Expected one main benchmark archive, found {len(matches)}.")
    return matches[0]


class Recordings:
    """Catalog and a bounded cache; raw image arrays are never all loaded at once."""

    def __init__(self, repo: Path, labels: Path, scores: CachedScores | None = None):
        self.scores = scores
        self.labels = OnsetLabels(labels)
        manifest = self.labels.read()
        if manifest.get("episode_id_space") != "prepared_prl_episode_index":
            raise ValueError("Label episode_id_space must be prepared_prl_episode_index.")
        self.paths: dict[tuple[str, int], Path] = {}
        self.outcomes: dict[tuple[str, int], str] = {}
        self.lock = threading.Lock()
        self.current: Episode | None = None
        self.images: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
        titles = {"needle_pickup": "Needle pickup · Simulation", "ring_over_post": "Ring over post · Simulation",
                  "tissue_retraction": "Tissue retraction · Real", "shunt_insertion": "Shunt insertion · Real"}
        tasks = []
        for task, node in manifest["tasks"].items():
            if not re.fullmatch(r"[A-Za-z0-9_]+", task):
                raise ValueError(f"Invalid task name: {task}")
            base = repo / "fomofd/tasks" / task / "prl_data"
            preparation = base / "processed_le_v21_vfmf_vae_fix_wrist/meta/preparation.json"
            mapping = json.loads(preparation.read_text())["episode_mapping"]
            raw_files = {}
            for row in mapping:
                filename = row["raw_file"]
                if Path(filename).name != filename:
                    raise ValueError(f"Invalid raw filename in {preparation}")
                episode_id = int(row["dataset_episode_index"])
                if episode_id in raw_files:
                    raise ValueError(f"Duplicate prepared episode ID: {task}/{episode_id}")
                raw_files[episode_id] = base / "raw" / filename
            groups, seen = [], set()
            for index, group in enumerate(node["groups"]):
                start, end = group["episode_range"]
                if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end:
                    raise ValueError(f"Invalid episode range: {task}/{index}")
                ids = list(range(start, end + 1))
                if seen.intersection(ids):
                    raise ValueError(f"Overlapping label ranges: {task}/{index}")
                seen.update(ids)
                if group["outcome"] == "success":
                    title = "Success · Calibration" if group["split"] == "calibration" else "Success · Test (FAR)"
                else:
                    title = str(group["failure_mode"])
                groups.append({**group, "title": title, "episode_ids": ids})
                for episode_id in ids:
                    if episode_id not in raw_files or not raw_files[episode_id].is_file():
                        raise ValueError(f"Missing mapped recording: {task}/{episode_id}")
                    self.paths[task, episode_id] = raw_files[episode_id]
                    self.outcomes[task, episode_id] = group["outcome"]
            if node.get("expected_episode_count") is not None:
                if len(seen) != node["expected_episode_count"] or seen != set(raw_files):
                    raise ValueError(f"Label coverage differs from prepared episode mapping: {task}")
            if groups:
                tasks.append({"id": task, "title": titles.get(task, task.replace("_", " ").title()), "groups": groups})
        if not tasks:
            raise ValueError("No labeled episode groups found.")
        self.catalog = {"tasks": tasks}

    def catalog_data(self) -> dict:
        return {**self.catalog, "annotators": self.labels.annotators(self.labels.read())}

    def episode_metadata(self, task: str, episode_id: int) -> dict:
        metadata = self.episode(task, episode_id).metadata()
        manifest = self.labels.read()
        onsets = manifest["tasks"][task].get("failure_onsets", {}).get(episode_id, {})
        if any(value["frame_index"] >= metadata["frame_count"] for entry in onsets.values()
               for value in onset_views(entry).values()):
            raise ValueError(f"Saved onset exceeds the recording's frame count: {task}/{episode_id}")
        return {**metadata, "outcome": self.labels.outcome(manifest, task, episode_id),
                "onset_annotations": onsets, "annotators": self.labels.annotators(manifest),
                "score_plot": self.scores.episode(task, episode_id) if self.scores else None}

    def set_onset(self, task: str, episode_id: int, annotator_id: str, index: int | None, view: str = "both") -> dict:
        if type(episode_id) is not int or not isinstance(task, str) or (task, episode_id) not in self.outcomes:
            raise ValueError("Select a known labeled episode.")
        if self.outcomes[task, episode_id] != "failure":
            raise ValueError("Success episodes cannot have failure onset annotations.")
        if not isinstance(annotator_id, str):
            raise ValueError("Select a registered human annotator.")
        if index is not None:
            if type(index) is not int:
                raise ValueError("Onset frame_index must be an integer.")
            count = len(self.episode(task, episode_id).times)
            if not 0 <= index < count:
                raise ValueError(f"Onset frame_index must be between 0 and {count - 1}.")
        return self.labels.set_onset(task, episode_id, annotator_id, index, view)

    def episode(self, task: str, episode_id: int) -> Episode:
        with self.lock:
            if self.current is not None and (self.current.task, self.current.episode_id) == (task, episode_id):
                return self.current
            if (task, episode_id) not in self.paths:
                raise KeyError(f"Unknown labeled episode: {task}/{episode_id}")
            path = self.paths[task, episode_id]
            # Do not load the object-valued meta member or any policy/model data.
            with np.load(path, allow_pickle=False) as recording:
                fix = recording["images_fix"]
                wrist = recording["images_wrist"]
                # Both eval recorders save time.time() here, including simulation.
                # Preserve actual gaps rather than LeRobot's regular frame clock.
                timestamps = np.asarray(recording["timestamps"], dtype=np.float64).reshape(-1)
            for view, images in [("fixed", fix), ("wrist", wrist)]:
                if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3 or not len(images):
                    raise ValueError(f"{path.name}: invalid {view} RGB array; expected uint8 [T,H,W,3].")
            if len(fix) != len(wrist) or len(fix) != len(timestamps):
                raise ValueError(f"{path.name}: camera and timestamp frame counts differ.")
            note = None
            if not np.isfinite(timestamps).all() or (np.diff(timestamps) < 0).any():
                intervals = np.diff(timestamps)
                valid = intervals[np.isfinite(intervals) & (intervals > 0)]
                interval = float(np.median(valid)) if valid.size else 1 / 30
                times = np.arange(len(fix), dtype=np.float64) * interval
                note = "Recorded timestamps are invalid; replay uses an estimated regular frame clock."
            else:
                times = timestamps - timestamps[0]
            self.current = Episode(task, episode_id, path.name, fix, wrist, times, note)
            return self.current

    def frame(self, task: str, episode_id: int, index: int) -> bytes:
        key = task, episode_id, index
        with self.lock:
            if key in self.images:
                self.images.move_to_end(key)
                return self.images[key]
        episode = self.episode(task, episode_id)
        if not 0 <= index < len(episode.times):
            raise ValueError(f"Frame index must be between 0 and {len(episode.times) - 1}.")
        fix, wrist = Image.fromarray(episode.fix[index]), Image.fromarray(episode.wrist[index])
        combined = Image.new("RGB", (fix.width + wrist.width, max(fix.height, wrist.height)))
        combined.paste(fix, (0, 0))
        combined.paste(wrist, (fix.width, 0))
        buffer = io.BytesIO()
        combined.save(buffer, format="JPEG", quality=90)
        data = buffer.getvalue()
        with self.lock:
            self.images[key] = data
            self.images.move_to_end(key)
            while len(self.images) > 48:
                self.images.popitem(last=False)
        return data


def make_handler(recordings: Recordings, verbose: bool):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if verbose:
                super().log_message(fmt, *args)

        def send(self, status: int, mime: str, data: bytes):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                url = urlsplit(self.path)
                if url.path == "/":
                    self.send(200, "text/html; charset=utf-8", HTML.encode())
                    return
                if url.path == "/api/catalog":
                    self.send(200, "application/json", json.dumps(recordings.catalog_data()).encode())
                    return
                if url.path not in {"/api/episode", "/api/frame"}:
                    raise KeyError("Unknown endpoint")
                query = parse_qs(url.query)
                task, episode_id = query["task"][0], int(query["episode"][0])
                if url.path == "/api/episode":
                    data = json.dumps(recordings.episode_metadata(task, episode_id), allow_nan=False).encode()
                    self.send(200, "application/json", data)
                else:
                    self.send(200, "image/jpeg", recordings.frame(task, episode_id, int(query["frame"][0])))
            except (BrokenPipeError, ConnectionResetError):
                pass  # Browser canceled a request while switching episodes.
            except (KeyError, ValueError, OSError, yaml.YAMLError) as error:
                status = 404 if isinstance(error, KeyError) else 400
                try:
                    self.send(status, "application/json", json.dumps({"error": str(error)}).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    raise ValueError("Annotation requests must come from this inspector page.")
                if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                    raise ValueError("Annotation requests must use application/json.")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid annotation request size.")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object.")
                path = urlsplit(self.path).path
                if path == "/api/annotators":
                    result = recordings.labels.add_annotator(body.get("name"))
                elif path == "/api/onset":
                    required = {"task", "episode", "annotator_id", "frame_index"}
                    if not required.issubset(body):
                        raise ValueError("Onset requests require task, episode, annotator_id and frame_index.")
                    result = recordings.set_onset(body["task"], body["episode"], body["annotator_id"], body["frame_index"], body.get("view", "both"))
                else:
                    raise KeyError("Unknown endpoint")
                self.send(200, "application/json", json.dumps(result).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (KeyError, ValueError, OSError, yaml.YAMLError) as error:
                status = 404 if isinstance(error, KeyError) else 400
                try:
                    self.send(status, "application/json", json.dumps({"error": str(error)}).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--labels", type=Path, help="Defaults to lerobot/configs/benchmark/prl_episode_labels.yaml")
    parser.add_argument("--host", default="127.0.0.1", help="Local interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser")
    parser.add_argument("--scores", help="main, or an offline report directory; read cached scores only")
    parser.add_argument("--score-method", default="wm_nc", help="Per-view method in the selected report")
    parser.add_argument("--score-view", choices=["fix", "wrist"], default="wrist")
    parser.add_argument("--verbose", action="store_true", help="Log individual HTTP requests")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    labels = args.labels.resolve() if args.labels else repo / "lerobot/configs/benchmark/prl_episode_labels.yaml"
    try:
        scores = CachedScores(score_report_path(repo, args.scores), args.score_method, args.score_view) if args.scores else None
        recordings = Recordings(repo, labels, scores)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(recordings, args.verbose))
    except (ValueError, KeyError, OSError, yaml.YAMLError) as error:
        parser.exit(1, f"Could not start replay inspector: {error}\n")
    url = f"http://{args.host}:{server.server_port}"
    print(f"PRL replay inspector: {url}\nLabels: {labels}\nCPU only; recordings are read-only. Human onset annotations are saved to the labels YAML. Ctrl+C stops the server.", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReplay inspector stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
