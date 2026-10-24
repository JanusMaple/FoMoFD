'use strict';

const replayAssetBase = new URL('.', document.currentScript.src);

const replays = new Map();
const replayPanels = new Map();
let activeReplay = null;
const replayScripts = new Map();
const calibrationLoads = new Map();
const replayTaskTitles = {
  needle_pickup: 'Needle Pickup', ring_over_post: 'Ring Over Post',
  tissue_retraction: 'Tissue Retraction', shunt_insertion: 'Shunt Insertion'
};
const defaultReplayModes = {
  needle_pickup: 'Needle Drop', ring_over_post: 'Ring Slippage',
  tissue_retraction: 'Needle Intrusion', shunt_insertion: 'Insertion Misalignment'
};

function loadScript(src) {
  if (replayScripts.has(src)) return replayScripts.get(src);
  const pending = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = new URL(src, replayAssetBase).href;
    script.onload = () => { script.remove(); resolve(); };
    script.onerror = () => { script.remove(); reject(Error('Recording data could not be loaded.')); };
    document.head.append(script);
  }).finally(() => replayScripts.delete(src));
  replayScripts.set(src, pending);
  return pending;
}

async function readReplayEpisode(task, id) {
  const key = `${task}/${String(id).padStart(6, '0')}`;
  if (!window.REVIEW_EPISODES[key]) await loadScript(`data/episodes/${key}.js`);
  const recording = window.REVIEW_EPISODES[key];
  if (!recording) throw Error('Recording data is unavailable.');
  return recording;
}

function loadCalibrationEpisodes(task) {
  if (calibrationLoads.has(task)) return calibrationLoads.get(task);
  const ids = [...new Set(D.catalog.find(t => t.id === task).groups
    .filter(g => g.split === 'calibration' && g.outcome === 'success')
    .flatMap(g => g.episodes.filter(e => e.available).map(e => e.id)))];
  const pending = Promise.all(ids.map(id => readReplayEpisode(task, id)))
    .catch(error => { calibrationLoads.delete(task); throw error; });
  calibrationLoads.set(task, pending);
  return pending;
}

class TaskReplay {
  constructor(task) {
    this.task = task;
    this.recording = null;
    this.group = null;
    this.selection = 0;
    this.calibrationSelection = 0;
    this.preferredMethod = 'wm_nc';
    this.view = 'wrist';
    this.curves = new Map();
    this.calibration = new Map();
    this.drag = null;
    this.root = document.createElement('article');
    this.root.id = `replay-${task.id}`;
    this.root.className = 'task-replay';
    this.root.hidden = true;
    this.root.dataset.task = task.id;
    this.root.dataset.camera = this.view;
    this.root.tabIndex = -1;
    this.root.setAttribute('aria-label', `${replayTaskTitles[task.id] || shortTask(task.id)} recording replay`);
    const chart = (view, title) => `<figure class="camera-score" data-view="${view}" ${view !== this.view ? 'hidden' : ''}>
      <div class="camera-score-heading"><h4>${title}</h4><span data-role="readout-${view}"></span></div>
      <svg data-role="chart-${view}" class="task-score-chart" role="img" aria-label="${esc(task.title)} ${title} score. Click or drag horizontally to seek."></svg>
    </figure>`;
    this.root.innerHTML = `<div class="task-episode-controls">
        <label>Recording group<select data-role="group" aria-label="${esc(task.title)} recording group"></select></label>
        <label>Recording<select data-role="episode" aria-label="${esc(task.title)} recording"></select></label>
      </div>
      <div class="task-camera-controls"><span>Camera</span>
        <div class="view-toggle" role="radiogroup" aria-label="${esc(task.title)} replay camera">
          <label><input type="radio" name="replay-view-${task.id}" data-camera-view="fix" value="fix"><span>Fixed</span></label>
          <label><input type="radio" name="replay-view-${task.id}" data-camera-view="wrist" value="wrist" checked><span>Wrist</span></label>
        </div>
      </div>
      <div class="task-video-viewport"><video data-role="video" preload="metadata" playsinline muted aria-label="${esc(task.title)} Wrist camera recording"></video></div>
      <div class="readouts"><span data-role="frame">Frame —</span><span data-role="time">0.00 s</span></div>
      <div class="seek-wrap"><input data-role="seek" type="range" min="0" max="0" step="1" value="0" aria-label="${esc(task.title)} recording frame"><div class="onset-markers" data-role="markers"></div></div>
      <div class="task-play-controls">
        <button data-action="previous" class="quiet" aria-label="Previous ${esc(task.title)} recording">←</button>
        <button data-action="play" data-role="play" class="primary">Play</button>
        <button data-action="next" class="quiet" aria-label="Next ${esc(task.title)} recording">→</button>
        <button data-action="frame-back" class="quiet" aria-label="Previous frame">−1</button>
        <button data-action="frame-next" class="quiet" aria-label="Next frame">+1</button>
        <label>Speed<select data-role="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option><option value="2" selected>2×</option></select></label>
        <span class="task-onsets" data-role="onsets"></span>
      </div>
      <div class="task-monitor-controls">
        <label>Monitor<select data-role="method" aria-label="${esc(task.title)} monitor"></select></label>
        <label class="far-switch calibration-switch"><input data-role="show-calibration" type="checkbox" role="switch" checked><span class="switch-track" aria-hidden="true"></span><span>Calibration set</span></label>
      </div>
      <div class="task-scores">${chart('fix', 'Fixed score')}${chart('wrist', 'Wrist score')}${chart('both', 'Joint-input score')}</div>
      <div class="score-legend" role="group" aria-label="Score plot legend"><span><i class="legend-score"></i>Score</span><span><i class="legend-threshold"></i>Threshold</span><span><i class="legend-onset"></i>Failure onset</span><span class="calibration-legend" data-role="calibration-legend" hidden><i class="legend-calibration" aria-hidden="true"></i><span data-role="calibration-label"></span></span></div>
      <p class="note calibration-note" data-role="calibration-note" role="status" aria-live="polite" hidden></p>
      <p class="note task-validity" data-role="validity"></p>
      <p class="note task-status" data-role="status" role="status" aria-live="polite"></p>`;
    this.video = this.$('video');
    this.video.muted = true;
    this.video.defaultPlaybackRate = 2;
    this.video.playbackRate = 2;
    this.$('group').onchange = () => this.selectGroup(Number(this.$('group').value));
    this.$('episode').onchange = () => this.loadEpisode();
    this.$('method').onchange = () => this.updateMonitor();
    this.root.querySelectorAll('[data-camera-view]').forEach(radio => {
      radio.onchange = () => this.selectView(radio.value);
    });
    this.$('show-calibration').onchange = () => this.refreshCalibration();
    this.$('seek').oninput = () => this.seekFrame(this.$('seek').value);
    this.$('speed').onchange = () => {
      this.video.defaultPlaybackRate = Number(this.$('speed').value);
      this.video.playbackRate = this.video.defaultPlaybackRate;
    };
    this.root.addEventListener('pointerdown', () => this.activate());
    this.root.addEventListener('focusin', () => this.activate());
    this.root.addEventListener('click', event => {
      const onset = event.target.closest('[data-onset]');
      if (onset) this.seekFrame(onset.dataset.onset);
      const button = event.target.closest('[data-action]');
      if (!button) return;
      switch (button.dataset.action) {
        case 'play': this.togglePlay(); break;
        case 'previous': this.moveEpisode(-1); break;
        case 'next': this.moveEpisode(1); break;
        case 'frame-back': this.seekFrame(this.frameAt(this.video.currentTime) - 1); break;
        case 'frame-next': this.seekFrame(this.frameAt(this.video.currentTime) + 1); break;
      }
    });
    for (const event of ['timeupdate', 'seeked', 'play', 'pause', 'loadedmetadata', 'ended']) {
      this.video.addEventListener(event, () => this.updateFrame());
    }
    if (this.video.requestVideoFrameCallback) {
      const tick = () => { this.updateFrame(); this.video.requestVideoFrameCallback(tick); };
      this.video.requestVideoFrameCallback(tick);
    }
    const resized = () => { this.drawScores(); this.updateFrame(); };
    for (const view of ['fix', 'wrist', 'both']) {
      const svg = this.$(`chart-${view}`);
      this.bindScrubbing(svg);
      if ('ResizeObserver' in window) new ResizeObserver(resized).observe(svg);
    }
    if (!('ResizeObserver' in window)) window.addEventListener('resize', resized);
    fillOptions(this.$('group'), task.groups.map((g, i) => [i, g.title]), 0);
  }

  $(role) { return this.root.querySelector(`[data-role="${role}"]`); }
  activate() {
    if (this.root.hidden) return;
    if (activeReplay === this) return;
    activeReplay?.root.closest('.replay-panel')?.classList.remove('is-active');
    activeReplay = this;
    this.root.closest('.replay-panel')?.classList.add('is-active');
  }
  selectView(view) {
    if (!['fix', 'wrist'].includes(view)) return Promise.resolve();
    this.stopScrub();
    this.view = view;
    this.root.dataset.camera = view;
    this.root.querySelectorAll('[data-camera-view]').forEach(radio => { radio.checked = radio.value === view; });
    this.video.setAttribute('aria-label', `${this.task.title} ${viewName[view]} camera recording`);
    this.showScoreView();
    this.updateOnsets();
    return this.updateMonitor();
  }
  showScoreView() {
    const selected = this.curves.has('both') ? 'both' : this.view;
    for (const view of ['fix', 'wrist', 'both']) this.root.querySelector(`[data-view="${view}"]`).hidden = view !== selected;
  }
  updateOnsets() {
    this.$('onsets').replaceChildren();
    this.$('markers').replaceChildren();
    if (!this.recording) return;
    const view = this.curves.has('both') ? 'both' : this.view;
    const frames = (view === 'both' ? Object.values(this.recording.onsets) : [this.recording.onsets[view]]).filter(Number.isFinite);
    if (!frames.length) return;
    const frame = Math.min(...frames), label = viewName[view];
    this.$('onsets').innerHTML = `<button class="quiet" data-onset="${frame}" title="Jump to ${label} annotated onset">${label} ${num(this.onsetTime(view))} s</button>`;
    this.$('markers').innerHTML = `<button data-onset="${frame}" style="left:${frame / Math.max(1, this.recording.times.length - 1) * 100}%" aria-label="Jump to ${label} onset"></button>`;
  }
  selectGroup(index, wanted) {
    this.$('group').value = index;
    this.group = this.task.groups[index];
    fillOptions(this.$('episode'), this.group.episodes.map((e, index) => [e.id, `${index}${e.available ? '' : ' · unavailable'}`]),
      wanted ?? this.group.episodes.find(e => e.available)?.id ?? this.group.episodes[0]?.id);
    return this.loadEpisode();
  }
  async loadEpisode() {
    const token = ++this.selection;
    ++this.calibrationSelection;
    this.stopScrub();
    this.video.pause();
    this.root.dataset.ready = 'false';
    this.recording = null;
    this.updateOnsets();
    this.curves.clear();
    this.calibration.clear();
    this.drawScores();
    this.$('calibration-note').hidden = true;
    this.$('calibration-legend').hidden = true;
    this.$('validity').textContent = '';
    this.$('status').classList.remove('error');
    this.$('status').textContent = 'Loading recording…';
    const id = Number(this.$('episode').value);
    const position = this.group.episodes.findIndex(e => e.id === id);
    this.root.querySelector('[data-action="previous"]').disabled = position <= 0;
    this.root.querySelector('[data-action="next"]').disabled = position === this.group.episodes.length - 1;
    if (!this.group.episodes[position]?.available) {
      this.video.removeAttribute('src'); this.video.removeAttribute('poster'); this.video.load();
      this.$('status').textContent = 'This recording is not included in this export.';
      return;
    }
    try {
      const recording = await readReplayEpisode(this.task.id, id);
      if (token !== this.selection) return;
      this.recording = recording;
      this.video.src = new URL(recording.video, replayAssetBase).href;
      this.video.poster = new URL(recording.poster, replayAssetBase).href;
      this.video.defaultPlaybackRate = Number(this.$('speed').value);
      this.video.load();
      this.video.playbackRate = this.video.defaultPlaybackRate;
      this.$('seek').max = recording.times.length - 1;
      this.$('seek').value = 0;
      const methods = [...new Map(recording.curves.map(c => [c.method, [c.method, c.label]])).values()];
      fillOptions(this.$('method'), methods, this.preferredMethod);
      this.$('validity').textContent = recording.corrected ? 'Only valid raw frames 0–121 are retained in both views.' : '';
      await this.updateMonitor();
      if (token !== this.selection) return;
      this.root.dataset.ready = 'true';
      this.$('status').textContent = '';
      this.updateFrame();
    } catch (error) {
      if (token !== this.selection) return;
      this.$('status').textContent = error.message;
      this.$('status').classList.add('error');
    }
  }
  updateMonitor() {
    if (!this.recording) return Promise.resolve();
    const method = this.$('method').value;
    this.preferredMethod = method;
    this.$('method').classList.toggle('ours-selected', method === 'wm_nc');
    this.curves = new Map(this.recording.curves.filter(c => c.method === method && (c.view === this.view || c.view === 'both')).map(c => [c.view, c]));
    this.showScoreView();
    this.updateOnsets();
    return this.refreshCalibration();
  }
  async refreshCalibration() {
    const token = ++this.calibrationSelection;
    const note = this.$('calibration-note');
    const legend = this.$('calibration-legend');
    legend.hidden = true;
    this.calibration.clear();
    note.hidden = !this.$('show-calibration').checked || !this.recording;
    this.drawScores(); this.updateFrame();
    if (note.hidden) return;
    note.textContent = 'Loading calibration curves…';
    try {
      const recordings = await loadCalibrationEpisodes(this.task.id);
      if (token !== this.calibrationSelection) return;
      const counts = [];
      for (const [view, curve] of this.curves) {
        const series = recordings.flatMap(recording => recording.curves
          .filter(c => c.method === curve.method && c.view === view)
          .map(c => ({episode: recording.episode, times: c.frames.map(f => recording.times[f]), scores: c.scores})));
        this.calibration.set(view, series);
        counts.push(series.length === curve.n_calibration ? String(series.length) : `${series.length} of ${curve.n_calibration}`);
      }
      const count = [...new Set(counts)].join(' / ');
      if ([...this.calibration.values()].some(series => series.length)) {
        this.$('calibration-label').textContent = `Successful policy rollouts (calibration set, n = ${count})`;
        legend.hidden = false;
        note.textContent = '';
        note.hidden = true;
      } else {
        note.textContent = 'No calibration curves are available for this monitor.';
      }
      this.drawScores(); this.updateFrame();
    } catch (error) {
      if (token !== this.calibrationSelection) return;
      note.textContent = 'Calibration curves could not be loaded. Toggle off and on to retry.';
    }
  }
  onsetTime(view) {
    if (!this.recording) return undefined;
    if (view === 'both') return Math.min(...Object.keys(this.recording.onsets).map(v => this.onsetTime(v)));
    return this.recording.onset_times?.[view] ?? this.recording.times[this.recording.onsets[view]];
  }
  frameAt(time) {
    if (!this.recording) return 0;
    const times = this.recording.times;
    let lo = 0, hi = times.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (times[mid] <= time + 1e-6) lo = mid + 1; else hi = mid; }
    return Math.max(0, lo - 1);
  }
  seekFrame(index) {
    if (!this.recording) return;
    this.video.pause();
    const frame = Math.max(0, Math.min(this.recording.times.length - 1, Number(index)));
    this.video.currentTime = this.recording.times[frame] + 1e-6;
    this.$('seek').value = frame;
    this.updateFrame();
  }
  moveEpisode(delta) {
    const index = this.group.episodes.findIndex(e => e.id === Number(this.$('episode').value));
    const next = this.group.episodes[index + delta];
    if (next) { this.$('episode').value = next.id; return this.loadEpisode(); }
  }
  togglePlay() {
    if (!this.recording || this.root.hidden) return;
    this.activate();
    if (!this.video.paused) { this.video.pause(); return; }
    if (this.video.ended) this.video.currentTime = 0;
    this.video.play().catch(error => {
      if (!this.root.hidden && error.name !== 'AbortError') this.$('status').textContent = error.message;
    });
  }
  geometry(svg) {
    const width = Math.max(140, svg.clientWidth), height = Math.max(70, svg.clientHeight);
    return {width, height, left: 52, right: width - 24, top: 10, bottom: height - 23};
  }
  updateFrame() {
    this.$('play').textContent = this.video.paused ? 'Play' : 'Pause';
    if (!this.recording) return;
    const frame = this.frameAt(this.video.currentTime), times = this.recording.times;
    this.$('seek').value = frame;
    this.$('frame').textContent = `Frame ${frame} / ${times.length - 1}`;
    this.$('time').textContent = `${num(times[frame])} / ${num(times.at(-1))} s`;
    for (const view of ['fix', 'wrist', 'both']) {
      const curve = this.curves.get(view), svg = this.$(`chart-${view}`);
      const cursor = svg.querySelector('.cursor');
      if (cursor) {
        const g = this.geometry(svg), x = g.left + times[frame] / Math.max(.001, times.at(-1)) * (g.right - g.left);
        cursor.setAttribute('x1', x); cursor.setAttribute('x2', x);
      }
      let index = -1;
      if (curve) for (let i = 0; i < curve.frames.length && curve.frames[i] <= frame; i++) index = i;
      const readout = this.$(`readout-${view}`);
      readout.textContent = !curve ? 'No separate score' : index < 0 ? 'Waiting for score' : `${Number(curve.scores[index].toPrecision(4))}${curve.scores[index] > curve.threshold ? ' · alarm' : ''}`;
      readout.classList.toggle('score-alarm', !!curve && index >= 0 && curve.scores[index] > curve.threshold);
    }
  }
  drawScores() {
    if (this.root.hidden) return;
    for (const view of ['fix', 'wrist', 'both']) {
      const svg = this.$(`chart-${view}`), curve = this.curves.get(view);
      if (!this.recording || !curve || svg.closest('figure').hidden) { svg.innerHTML = ''; continue; }
      const g = this.geometry(svg), end = Math.max(.001, this.recording.times.at(-1));
      svg.setAttribute('viewBox', `0 0 ${g.width} ${g.height}`);
      const background = this.calibration.get(view) || [];
      let lo = Math.min(...curve.scores, curve.threshold), hi = Math.max(...curve.scores, curve.threshold);
      for (const series of background) for (let i = 0; i < series.scores.length; i++) {
        if (i === 0 && series.times[i] > end) break;
        lo = Math.min(lo, series.scores[i]); hi = Math.max(hi, series.scores[i]);
        if (series.times[i] > end) break;
      }
      const pad = Math.max((hi - lo) * .1, Math.abs(hi) * .01, 1e-6);
      lo -= pad; hi += pad;
      const x = t => g.left + t / end * (g.right - g.left), y = s => g.bottom - (s - lo) / (hi - lo) * (g.bottom - g.top);
      let drawing = '';
      for (let i = 0; i <= 3; i++) {
        const score = lo + (hi - lo) * i / 3;
        const rounded = String(Number(score.toPrecision(3)));
        const label = rounded.length <= 7 ? rounded : score.toExponential(1).replace('e+', 'e');
        drawing += `<line class="chart-grid" x1="${g.left}" x2="${g.right}" y1="${y(score)}" y2="${y(score)}"/><text class="chart-label" x="${g.left - 5}" y="${y(score) + 3}" text-anchor="end">${label}</text>`;
      }
      for (let i = 0; i <= 2; i++) drawing += `<text class="chart-label" x="${x(end * i / 2)}" y="${g.height - 5}" text-anchor="middle">${num(end * i / 2, 1)} s</text>`;
      const clip = `score-clip-${this.task.id}-${view}`;
      drawing += `<defs><clipPath id="${clip}"><rect x="${g.left}" y="${g.top}" width="${g.right - g.left}" height="${g.bottom - g.top}"/></clipPath></defs><g class="calibration-background" clip-path="url(#${clip})">${background.map(series => `<polyline class="calibration-signal" data-episode="${series.episode}" points="${series.scores.map((score, i) => `${x(series.times[i])},${y(score)}`).join(' ')}"/>`).join('')}</g>`;
      drawing += `<line class="threshold" x1="${g.left}" x2="${g.right}" y1="${y(curve.threshold)}" y2="${y(curve.threshold)}"/>`;
      const onset = this.onsetTime(view);
      if (Number.isFinite(onset)) drawing += `<line class="onset" x1="${x(onset)}" x2="${x(onset)}" y1="${g.top}" y2="${g.bottom}"/>`;
      drawing += `<polyline class="signal" points="${curve.scores.map((score, i) => `${x(this.recording.times[curve.frames[i]])},${y(score)}`).join(' ')}"/><line class="cursor" x1="${g.left}" x2="${g.left}" y1="${g.top}" y2="${g.bottom}"/>`;
      svg.innerHTML = drawing;
    }
  }
  seekScoreAt(event, svg) {
    if (!this.recording) return;
    const point = svg.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
    const local = point.matrixTransform(svg.getScreenCTM().inverse()), g = this.geometry(svg);
    this.seekFrame(this.frameAt(Math.max(0, Math.min(1, (local.x - g.left) / (g.right - g.left))) * this.recording.times.at(-1)));
  }
  stopScrub() {
    const drag = this.drag; this.drag = null;
    if (!drag) return;
    drag.svg.classList.remove('is-scrubbing');
    if (drag.svg.hasPointerCapture(drag.pointer)) drag.svg.releasePointerCapture(drag.pointer);
  }
  bindScrubbing(svg) {
    svg.addEventListener('pointerdown', event => {
      if (!this.recording || !event.isPrimary || event.button !== 0 || this.drag) return;
      event.preventDefault(); this.activate();
      this.drag = {svg, pointer: event.pointerId};
      svg.setPointerCapture(event.pointerId); svg.classList.add('is-scrubbing');
      this.seekScoreAt(event, svg);
    });
    svg.addEventListener('pointermove', event => {
      if (this.drag?.svg !== svg || this.drag.pointer !== event.pointerId) return;
      event.preventDefault(); this.seekScoreAt(event, svg);
    });
    svg.addEventListener('pointerup', event => {
      if (this.drag?.svg !== svg || this.drag.pointer !== event.pointerId) return;
      this.seekScoreAt(event, svg); this.stopScrub();
    });
    for (const name of ['pointercancel', 'lostpointercapture']) svg.addEventListener(name, event => {
      if (this.drag?.svg === svg && this.drag.pointer === event.pointerId) this.stopScrub();
    });
  }
}

class ReplayPanel {
  constructor(id, title, tasks) {
    this.current = null;
    this.tasks = tasks;
    this.root = document.createElement('section');
    this.root.id = `replay-panel-${id}`;
    this.root.className = 'panel replay-panel';
    this.root.setAttribute('aria-labelledby', `replay-heading-${id}`);
    this.root.innerHTML = `<div class="replay-panel-heading"><h3 id="replay-heading-${id}">${title}</h3></div>
      <div class="view-toggle replay-task-toggle" role="radiogroup" aria-label="${title} task">${tasks.map(task => `<label><input type="radio" name="replay-task-${id}" value="${task.id}" aria-controls="replay-${task.id}"><span>${esc(replayTaskTitles[task.id] || shortTask(task.id))}</span></label>`).join('')}</div>`;
    for (const task of tasks) {
      const replay = new TaskReplay(task);
      replays.set(task.id, replay);
      replayPanels.set(task.id, this);
      this.root.append(replay.root);
    }
    this.root.querySelectorAll(`[name="replay-task-${id}"]`).forEach(radio => {
      radio.onchange = () => this.selectTask(radio.value);
    });
  }
  selectTask(taskId, {activate = true, load = true} = {}) {
    const replay = replays.get(taskId);
    if (!replay || !this.tasks.some(task => task.id === taskId)) return;
    if (this.current !== replay) {
      if (this.current) {
        this.current.stopScrub();
        this.current.video.pause();
        this.current.root.hidden = true;
      }
      this.current = replay;
      replay.root.hidden = false;
      this.root.dataset.task = taskId;
      this.root.querySelectorAll('.replay-task-toggle input').forEach(radio => { radio.checked = radio.value === taskId; });
    }
    if (activate) replay.activate();
    if (load && !replay.group) {
      let index = replay.task.groups.findIndex(g => g.failure_mode === defaultReplayModes[taskId] && g.episodes.some(e => e.available));
      if (index < 0) index = replay.task.groups.findIndex(g => g.outcome === 'failure' && g.episodes.some(e => e.available));
      replay.selectGroup(Math.max(0, index));
    }
    replay.drawScores(); replay.updateFrame();
    return replay;
  }
}

function initReplays() {
  const query = new URLSearchParams(location.search), wantedTask = query.get('task');
  for (const [id, title, taskIds, defaultTask] of [
    ['simulation', 'Simulation', ['needle_pickup', 'ring_over_post'], 'needle_pickup'],
    ['real', 'Real', ['tissue_retraction', 'shunt_insertion'], 'shunt_insertion']
  ]) {
    const tasks = taskIds.map(id => D.catalog.find(task => task.id === id)).filter(Boolean);
    if (!tasks.length) continue;
    const panel = new ReplayPanel(id, title, tasks);
    $('replay-grid').append(panel.root);
    const task = tasks.find(task => task.id === wantedTask) || tasks.find(task => task.id === defaultTask) || tasks[0];
    const episode = task.id === wantedTask && query.has('episode') ? Number(query.get('episode')) : null;
    const requested = episode === null ? -1 : task.groups.findIndex(g => g.episodes.some(e => e.id === episode));
    const replay = panel.selectTask(task.id, {activate: false, load: requested < 0});
    if (requested >= 0) {
      replay.selectGroup(requested, episode);
    }
  }
  (replays.get(wantedTask) || [...replays.values()].find(replay => !replay.root.hidden))?.activate();
}
