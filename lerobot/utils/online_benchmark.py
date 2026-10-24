"""Online benchmark: raw-image inference with one latest eligible input at a time.

Planning never imports Torch or launches inference. --run is explicit, checks
offline dependencies/GPU availability, and then runs end-to-end model adapters.
The experiment YAML supports multiple checkpoint variants without conflating
their scores, calibration thresholds, or onset results. Schema 1 retains the
conditional latency experiment. Schema 2 evaluates ALL test episodes and shares
one scored stream between constant and functional-band thresholds, with separate
first alarms and full-test outcome/onset/deadline denominators.

    python lerobot/utils/online_benchmark.py \
      --offline-report logs/benchmarks/2026-10-07/batched_unified_run/results/all_tasks_complete \
      --methods wm_nc wm_pe logpzo rnd --output logs/benchmarks/2026-10-07/online_plan

Protocol:
1. Restore and warm up one method/task model outside the episode timer. Decode
   ONE raw episode into host RAM, with state/actions from prepared scalar columns.
   Exclude file I/O and model startup from detection latency, report them separately.
2. Reset episode feature caches; start a monotonic clock at recording time zero.
   Recorded wall timestamps (including simulation) determine frame availability.
   A searchsorted lookup provides the newest available prepared observation.
3. At each iteration process one eligible frame/window. Never queue every frame,
   batch multiple windows, use future observations, or replay faster than 1x.
   For a WM endpoint e, anchor=e-K; history=[anchor-L+1,...,anchor]; executed
   actions=[anchor,...,anchor+K-1]. Recorded action indices are checked against
   observation indices: shift 0 (real tasks) and shift 1 (simulation) both make
   the final action available by endpoint e. Observation detectors do not read
   actions or require an action alignment.
4. Include image transforms, CPU->GPU transfer, DINOv2, deterministic VAE or the
   exact saved ACT token compressor, normalization, score, and alarm comparison.
   On-demand encoding may cover several past images within ONE selected window.
   Causal feature reuse is permitted; skipped frames needed for history are
   backfilled on demand, with that cost charged to this window. Never preload
   visual features. Adapters return only after their GPU work is synchronized.
5. Attach a score/threshold to its source endpoint time, but timestamp the alarm
   when computation actually finishes. After finishing, select the latest input
   again. If faster than arrivals, sleep until the next eligible frame. At source
   end, drain the newest eligible input once; no unlimited backlog.
6. Stop a scalar detector after its FIRST alarm, without consulting onset labels.
   A WM emits both camera scores in one forward pass: record each first alarm,
   and continue until all selected camera detectors have alarmed or source ends.
   Report per-camera/annotator delay, pre-onset evidence, online misses, source
   age, processing time, evaluated/skipped endpoints, and effective throughput.

Selection `offline-detected` skips successes and offline misses, using the OFFLINE
all-endpoints latency protocol. This estimates latency/retention CONDITIONAL on
that selection; it cannot estimate overall online FDR or FAR. `failures` selects
all failures for a future online FDR pass; `all-test` also includes successes for
online FAR. All modes reuse success-calibrated offline thresholds. Online protocol
FAR is not guaranteed to match dense offline FAR when windows are skipped.

Combined monitors use the existing component thresholds at the SAME raw endpoint:
OR=max / AND=min of signed scaled margins, alarm iff >0. View fusion shares one
WM inference; WM-NC/native-logpZO fusion runs both encoders/scorers sequentially,
including both costs, before selecting the next latest window. It never latches
alarms from different frames. OR has no unchanged-alpha familywise guarantee.

Latency = (source endpoint time - onset time) +
          (score completion time - source endpoint time).
Evidence preceding onset is kept as an early alarm even if slow computation
finishes after onset; it is never credited as successful post-onset detection.

Encoder paths:
* Restore encoder weights/preprocessing exactly from preparation provenance.
  VAE path: DINO intermediate layers 2,5,8,11 -> L2-normalized 3072 patch features
  -> camera VAE posterior mean -> saved expert latent statistics.
  Native path: saved ACT DINO last-layer patches -> its 16x768 camera compressor
  -> exact saved baseline normalization; logpZO includes saved robot state.
* Reuse existing scorer functions/solvers at window batch 1, including WM-PU/ACE
  sample counts and reproducible per-endpoint random inputs. WM-PE uses the
  deterministic x0=0 ODE prediction, matching offline scoring. No cached scores
  substitute for measured inference. Feature/score parity checks run before each
  model group, outside episode timing.

Skipping offline misses is a monotonic shortcut only with identical per-input
scores/thresholds. Batch-dependent numerics can change scores. This implementation
therefore keeps conditional denominators and NEVER reports a measured overall
online FAR/FDR from a selected-only run, even if sample parity checks pass.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable, Protocol

import numpy as np
import yaml

from unified_benchmark import METHODS, load_timeline
from model_paths import resolve_model_path

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Window:
    endpoint: int
    raw_endpoint: int
    source_time_s: float
    anchor: int | None
    history: tuple[int, ...]
    actions: tuple[int, ...]


def input_window(endpoint, raw, elapsed, method, horizon=0, history_len=1):
    if method.startswith('wm_'):
        if horizon < 1 or history_len < 1:
            raise ValueError('WM requires positive K/L')
        anchor = endpoint - horizon
        history = tuple(range(anchor - history_len + 1, anchor + 1))
        actions = tuple(range(anchor, anchor + horizon))
        if history[0] < 0 or max((*history, *actions)) >= endpoint:
            raise ValueError('WM window is unavailable or noncausal')
    else:
        anchor, history, actions = None, (), ()
    return Window(endpoint, int(raw[endpoint]), float(elapsed[endpoint]), anchor, history, actions)


class SynchronousAdapter(Protocol):
    """Adapter: own resident models, no file I/O inside score().

    reset_episode clears previous-episode feature caches. score consumes only
    window.history/window.endpoint images and causal scalar inputs in payload;
    no multi-window batching. Return finite scalar scores AFTER GPU completion.
    Exact input normalization/weights must match the offline cache provenance.
    """
    def reset_episode(self) -> None: ...
    def score(self, payload: dict, window: Window) -> dict[str, float]: ...


class OnlineODEMeter:
    """One process-wide integrator wrapper, installed before importing scorers."""
    def __init__(self):
        import torchdiffeq
        from torchdiffeq._impl.odeint import SOLVERS
        original = torchdiffeq.odeint
        self.calls = []
        def measured(func, y0, t, *args, **kwargs):
            method = kwargs.get('method') or 'dopri5'
            counts = defaultdict(int)
            class Counted:
                def __call__(self, *a, **kw):
                    counts['evaluations'] += 1
                    return func(*a, **kw)
            counted = Counted()
            callbacks = SOLVERS[method].valid_callbacks()
            for name in callbacks:
                def callback(*a, _name=name, **kw):
                    counts[_name] += 1
                    if hasattr(func, _name):
                        return getattr(func, _name)(*a, **kw)
                setattr(counted, name, callback)
            result = original(counted, y0, t, *args, **kwargs)
            self.calls.append({'solver': method,
                               'steps': counts['callback_accept_step'] if 'callback_accept_step' in callbacks else counts['callback_step'],
                               'evaluations': counts['evaluations'], 'attempts': counts['callback_step'],
                               'rejections': counts['callback_reject_step']})
            return result
        torchdiffeq.odeint = measured


ODE_METER = None


def frozen_model_package(identity, directory):
    """Verify a complete historical WM package against the original score cache.

    Loading under a private package name keeps historical separate-branch and
    current fused models independent. Never waive a provenance hash mismatch.
    """
    folder = Path(directory).resolve()
    original = REPO / 'lerobot/models/flow_matching_wm'
    expected = {Path(name).name: sha for name, sha in identity['sha256'].items()
                if Path(name).parent == original}
    required = {'__init__.py', 'config.py', 'model.py', 'dataset.py', 'score.py'}
    if set(expected) != required or {p.name for p in folder.glob('*.py')} != required:
        raise ValueError('Frozen WM package must contain exactly the recorded scorer modules')
    for filename, sha in expected.items():
        if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Frozen WM source differs from score provenance: {filename}')
    return {str(original / name): folder / name for name in expected}


class EndToEndAdapter:
    """Resident visual encoder + saved detector; actual inference is batch-one.

    Construction is reachable only through explicit --run. Normalization calls
    reuse existing dataset helpers on CPU, including the saved gripper transform.
    The only per-episode feature cache contains features computed causally here.
    """
    def __init__(self, trial, device):
        global ODE_METER
        import importlib
        import importlib.util
        import torch
        import unified_benchmark as offline
        self.torch, self.offline = torch, offline
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('This hardware benchmark requires CUDA; no CPU fallback')
        torch.set_num_threads(2)
        sys.path.insert(0, str(REPO / 'lerobot'))
        # The environment can retain an editable ACT installation from EMILS.
        # Resolve the encoder from this repository, as dataset preparation does.
        sys.path.insert(0, str(REPO / 'lerobot/policies/act_dinov2/src'))
        if ODE_METER is None:
            ODE_METER = OnlineODEMeter()
        self.meter = ODE_METER
        with np.load(trial['offline_score_cache'], allow_pickle=False) as cache:
            self.identity = json.loads(str(cache['provenance_json'].item()))['identity']
        if (self.identity['task'], self.identity['method']) != (trial['task'], trial['method']):
            raise ValueError('Offline cache belongs to a different method/task')
        # Sampled PRL caches fingerprint the whole offline script. Restore that
        # exact committed helper implementation, rather than waiving its hash
        # after unrelated reporting/calibration changes to the current script.
        frozen = trial.get('frozen_offline_scorer')
        if frozen:
            original = str(REPO / 'lerobot/utils/unified_benchmark.py')
            expected = self.identity['sha256'].get(original)
            if not expected or offline.digest(Path(frozen)) != expected:
                raise ValueError('Frozen offline scorer does not match PRL provenance')
            spec = importlib.util.spec_from_file_location('frozen_offline_'+expected, frozen)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            module.REPO = REPO
            offline = self.offline = module
        torch.manual_seed(self.identity['seed'])
        torch.cuda.manual_seed_all(self.identity['seed'])
        model_sources = frozen_model_package(self.identity, trial['frozen_model_source']) if trial.get('frozen_model_source') else {}
        for name, expected in self.identity['sha256'].items():
            source = Path(frozen) if frozen and name == str(REPO / 'lerobot/utils/unified_benchmark.py') else model_sources.get(name, Path(name))
            if offline.digest(resolve_model_path(source)) != expected:
                raise ValueError(f'Offline scoring input changed: {name}')
        self.method, self.task = trial['method'], trial['task']
        self.views = tuple(self.identity['views'])
        self.score_aggregation = trial.get('score_aggregation', 'per_view')
        self.output_views = ('both',) if self.score_aggregation == 'joint' else self.views
        self.spec = {'task': self.task, 'method': self.method, 'seed': self.identity['seed']}
        self.merged = json.loads(json.dumps(self.identity['merged_config']))
        if self.score_aggregation == 'joint' and (self.views != ('fix', 'wrist') or
                self.merged.get('backbone', {}).get('view_fusion') != 'concat_channels'):
            raise ValueError('Joint score requires a two-view concat_channels WM')
        for key in ['dataset_root', 'vfmf_vae_latent_stats_path', 'action_stats_path', 'state_stats_path']:
            if self.merged['data'].get(key):
                self.merged['data'][key] = str(offline.rooted(self.merged['data'][key], REPO / 'lerobot'))
        self.prepare = json.loads((Path(trial['dataset']) / 'meta/preparation.json').read_text())
        from utils.npz_to_le_v21 import (_load_dinov2_model, _load_frozen_visual_tokenizer,
                                         _load_frozen_vfmf_vae)
        self.cache, self.last_stage_times = {}, {}
        self.vaes, self.tokenizer, self.backbone = {}, None, None
        if self.method.endswith('_native') or self.method == 'ace':
            act = self.recorded_checkpoint('model.safetensors')
            self.tokenizer = _load_frozen_visual_tokenizer(str(act.parent), str(self.device))
            if self.tokenizer.config.use_cls_token:
                raise ValueError('Current baseline checkpoints require exactly 16 compressed tokens, without CLS')
        else:
            self.backbone = _load_dinov2_model(str(self.device)).eval()
            for view in self.views:
                checkpoint = self.recorded_checkpoint(f'vfmf_vae_{view}_L8_3072/checkpoints/best.pt')
                self.vaes[view] = _load_frozen_vfmf_vae(str(checkpoint), str(self.device))[0]
        if self.method == 'ace':
            self.model = offline.load_act_decoder(resolve_model_path(trial['checkpoint']), self.device)
            self.pre, self.eps = offline.processor_stats(resolve_model_path(trial['checkpoint']), 'policy_preprocessor.json', 'normalizer_processor')
            self.post, _ = offline.processor_stats(resolve_model_path(trial['checkpoint']), 'policy_postprocessor.json', 'unnormalizer_processor')
            sampling = self.merged['sampling']
            from sampled_score_protocol import ace_cell_widths
            self.cell_width = ace_cell_widths(sampling, REPO)
            self.normalizer = None
        else:
            family = 'flow_matching_wm' if self.method.startswith('wm_') else ('faildetect_logpzo' if self.method.startswith('logpzo') else 'fiper_rnd')
            package = f'models.{family}'
            if model_sources:
                if family != 'flow_matching_wm':
                    raise ValueError('Frozen model packages currently support FM-WM only')
                directory = Path(trial['frozen_model_source']).resolve()
                package = '_frozen_wm_' + hashlib.sha256(str(directory).encode()).hexdigest()
                if package not in sys.modules:
                    spec = importlib.util.spec_from_file_location(package, directory / '__init__.py',
                                                                 submodule_search_locations=[str(directory)])
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[package] = module
                    spec.loader.exec_module(module)
            self.scorer = importlib.import_module(package + '.score')
            self.cfg = self.scorer.build_config(self.merged)
            self.model = self.scorer.load_model_from_checkpoint(self.cfg, resolve_model_path(trial['checkpoint']), self.device)
            datasets = importlib.import_module(package + '.dataset')
            normalizer = datasets.EndpointWindowDataset if self.method.startswith('wm_') else datasets.SingleStepObservationDataset
            self.normalizer = normalizer(None, {}, [], self.cfg)
            if self.method.startswith('wm_') and self.cfg.data.action_chunk_len != self.cfg.data.pred_horizon:
                raise ValueError('Validate action availability for unequal action length/K before benchmarking')
        self.reset_episode()

    def recorded_checkpoint(self, suffix):
        matches = [(name, sha) for name, sha in self.prepare['checkpoint_sha256'].items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f'Encoder checkpoint provenance missing/ambiguous: {suffix}')
        name, expected = matches[0]
        path = resolve_model_path(name)
        if not path.is_file() and '/fomofd/tasks/' in name:
            path = REPO / ('fomofd/tasks/' + name.split('/fomofd/tasks/', 1)[1])
        if self.offline.digest(path) != expected:
            raise ValueError(f'Encoder weights differ from prepared features: {path}')
        return path

    def reset_episode(self):
        self.cache.clear()

    def encode(self, payload, indices):
        """FP32 visual preprocessing; optional detector AMP is separate."""
        torch = self.torch
        torch.set_float32_matmul_precision('highest')
        views = ('fix', 'wrist') if self.views == ('both',) else self.views
        with torch.inference_mode():
            for view in views:
                missing = sorted({int(i) for i in indices if (view, int(i)) not in self.cache})
                if not missing:
                    continue
                images = payload['images'][view][payload['raw'][missing]]
                tensor = torch.from_numpy(images.copy()).to(self.device).float().permute(0, 3, 1, 2) / 255.0
                if self.tokenizer is not None:
                    key = f'observation.images.{view}_camera'
                    patches, cls = self.tokenizer._extract_dino(tensor)
                    safe = key.replace('.', '_')
                    features = self.tokenizer.camera_tokenizers[safe](
                        patch_tokens=patches, camera_embed=self.tokenizer.camera_embeds[safe],
                        patch_pos=self.tokenizer._make_patch_pos(patches))
                else:
                    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    layers = self.backbone.get_intermediate_layers((tensor - mean) / std, n=[2, 5, 8, 11], reshape=False)
                    features = torch.nn.functional.normalize(torch.cat(layers, dim=-1), p=2, dim=2)
                    cfg = self.vaes[view].config.data
                    features = features.reshape(len(missing), cfg.patch_grid_height, cfg.patch_grid_width, -1)
                    features = self.vaes[view].encode(features, deterministic=True)
                # Match the preparation's FP32 CPU representation and existing
                # dataset normalization, instead of silently changing arithmetic.
                for index, feature in zip(missing, features.float().cpu()):
                    self.cache[view, index] = feature

    def feature(self, view, index):
        return self.cache[view, index]

    def score(self, payload, window):
        torch, offline = self.torch, self.offline
        started = time.perf_counter()
        self.meter.calls.clear()
        needed = (*window.history, window.endpoint)
        self.encode(payload, needed)
        encoded = time.perf_counter()
        if self.method == 'ace':
            sampling = self.merged['sampling']
            row = {'observation.state': payload['prepared']['observation.state'][window.endpoint]}
            for view in ['fix', 'wrist']:
                row[f'observation.tokens.{view}_camera'] = self.feature(view, window.endpoint).numpy()
            inputs = offline.ace_inputs([row], self.pre, self.eps, self.device)
        elif self.method.startswith('wm_'):
            data = self.cfg.data
            inputs = {'fix_hist': None, 'wrist_hist': None, 'future_fix': None, 'future_wrist': None,
                      'state_hist': None, 'future_state': None}
            for view in self.views:
                normalize = self.normalizer._fix_latent_map if view == 'fix' else self.normalizer._wrist_latent_map
                key = data.fix_key if view == 'fix' else data.wrist_key
                inputs[f'{view}_hist'] = torch.stack([normalize({key: self.feature(view, index)})
                                                    for index in window.history]).unsqueeze(0).to(self.device)
                inputs[f'future_{view}'] = normalize({key: self.feature(view, window.endpoint)}).unsqueeze(0).to(self.device)
            inputs['action_chunk'] = torch.stack([self.normalizer._action(
                {data.action_key: payload['prepared']['action'][index]}) for index in window.actions]).unsqueeze(0).to(self.device)
            if data.use_state:
                inputs['state_hist'] = torch.as_tensor(payload['prepared']['observation.state'][list(window.history)],
                                                     dtype=torch.float32, device=self.device).unsqueeze(0)
                inputs['future_state'] = torch.as_tensor(payload['prepared']['observation.state'][window.endpoint],
                                                       dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            data = self.cfg.data
            inputs = {'fix_obs': None, 'wrist_obs': None, 'state_obs': None}
            for view in self.views:
                normalize = self.normalizer._fix_obs if view == 'fix' else self.normalizer._wrist_obs
                key = data.fix_key if view == 'fix' else data.wrist_key
                inputs[f'{view}_obs'] = normalize({key: self.feature(view, window.endpoint)}).unsqueeze(0).to(self.device)
            if data.use_state:
                inputs['state_obs'] = torch.as_tensor(payload['prepared']['observation.state'][window.endpoint],
                                                     dtype=torch.float32, device=self.device).unsqueeze(0)
        prepared = time.perf_counter()
        if self.method == 'wm_pu':
            torch.set_float32_matmul_precision(self.merged['sampling']['matmul_precision'])
        else:
            torch.set_float32_matmul_precision('highest')
        amp = self.merged.get('runtime', {})
        with torch.inference_mode():
            if self.method == 'wm_pu':
                sampling = self.merged['sampling']
                noise = offline.frame_noise(self.spec, [payload['episode_index']], [window.endpoint], sampling['num_samples'],
                                             (len(self.views), self.cfg.data.latent_grid_height,
                                              self.cfg.data.latent_grid_width, self.cfg.data.latent_channels))
                step_options = {'ode_steps': sampling['ode_steps']} if 'ode_steps' in sampling else {}
                values = offline.score_wm_pu_batch(self.model, inputs, noise,
                    sampling['sample_batch_size'], sampling['precision'], **step_options)
            elif self.method == 'ace':
                noise = offline.frame_noise(self.spec, [payload['episode_index']], [window.endpoint], sampling['num_samples'],
                                             (self.model.config.latent_dim,))
                actions = offline.sample_ace_batch(self.model, inputs, noise, sampling['sample_batch_size'],
                                                   sampling.get('horizon') or self.model.config.chunk_size)
                actions = actions * self.post['action.std'] + self.post['action.mean']
                values = np.asarray([[offline.ace_histogram_entropy(actions[0, :, :, :3].cumsum(axis=1), self.cell_width)]])
            else:
                dtype = self.scorer.get_amp_dtype(amp.get('amp_dtype', 'bfloat16'))
                with torch.autocast(device_type='cuda', dtype=dtype, enabled=amp.get('use_amp', False)):
                    if self.method == 'wm_nc':
                        out = self.model.compute_reactive_nonconformity_scores(**inputs)
                        values = self.scorer._compute_reactive_scores_from_eps_hat(out['eps_hat'],
                            self.cfg.fm_head.score_kind, self.cfg.fm_head.base_std)['score_view']
                    elif self.method == 'wm_pe':
                        context = {key: inputs[key] for key in ['fix_hist', 'wrist_hist', 'state_hist', 'action_chunk']}
                        predicted = self.scorer._predict_endpoint_latent_center(self.model, **context)
                        values = self.scorer._compute_predictive_scores_from_prediction(
                            predicted, self.scorer._stack_future_views_from_inputs(inputs))['score_view']
                    elif self.method.startswith('logpzo'):
                        values = self.model.compute_nonconformity_scores(**inputs)['scores'].reshape(1, 1)
                    else:
                        values = self.model.compute_scores(**inputs)['scores'].reshape(1, 1)
        if isinstance(values, torch.Tensor):
            values = values.detach().float().cpu().numpy()
        torch.cuda.synchronize(self.device)
        finished = time.perf_counter()
        floor = min(window.history) if window.history else window.endpoint
        self.cache = {key: value for key, value in self.cache.items() if key[1] >= floor}
        self.last_stage_times = {'encoder_s': encoded - started, 'input_preparation_s': prepared - encoded,
                                 'detector_s': finished - prepared, 'ode_calls': list(self.meter.calls)}
        if self.score_aggregation == 'joint':
            from unified_benchmark import aggregate_wm_scores
            values = aggregate_wm_scores(self.method, values, self.score_aggregation)
        return dict(zip(self.output_views, map(float, np.asarray(values).reshape(-1))))


def replay_episode(payload, adapter: SynchronousAdapter, thresholds: Callable,
                   requested_views, method, horizon=0, history_len=1,
                   *, clock=time.perf_counter, sleep=time.sleep, stop_at_first_alarm=True):
    """One synchronous worker, one latest eligible input at a time; no CUDA import.

    Payload must already be decoded to host RAM. `elapsed` starts at RAW recording
    zero, not the first eligible WM window. All timing includes adapter work and
    threshold comparison. Labels/onsets are intentionally absent from this loop.
    """
    elapsed = np.asarray(payload['elapsed'], dtype=np.float64)
    raw = np.asarray(payload['raw'], dtype=np.int64)
    if (elapsed.ndim != 1 or not len(elapsed) or raw.shape != elapsed.shape
            or not np.isfinite(elapsed).all() or elapsed[0] < 0
            or (np.diff(elapsed) < 0).any() or raw[0] < 0 or (np.diff(raw) <= 0).any()):
        raise ValueError('Invalid recorded wall clock/raw frame mapping')
    first = horizon + history_len - 1 if method.startswith('wm_') else int(method.startswith('act_ensemble_'))
    if first >= len(elapsed) or not requested_views:
        raise ValueError('No eligible inputs or selected detector views')
    adapter.reset_episode()
    pending = set(requested_views)
    alarms, events = {}, []
    last = first - 1
    started = clock()
    while pending or not stop_at_first_alarm:
        available_at = clock() - started
        endpoint = min(len(elapsed) - 1, int(np.searchsorted(elapsed, available_at, side='right')) - 1)
        if endpoint < first:
            sleep(max(0.0, float(elapsed[first]) - available_at))
            continue
        if endpoint <= last:
            if last == len(elapsed) - 1:
                break
            sleep(max(0.0, float(elapsed[last + 1]) - available_at))
            continue
        window = input_window(endpoint, raw, elapsed, method, horizon, history_len)
        processing_start = clock() - started
        scores = adapter.score(payload, window)
        score_ready = clock() - started
        if score_ready < processing_start:
            raise ValueError('Clock must be monotonic')
        if not set(requested_views) <= set(scores):
            raise ValueError('Adapter omitted a detector view')
        fired = []
        used_thresholds = {}
        for view in sorted(pending if stop_at_first_alarm else requested_views):
            value = float(scores[view])
            threshold = float(thresholds(view, window.source_time_s))
            if not math.isfinite(value) or math.isnan(threshold):
                raise ValueError('Invalid score/threshold')
            used_thresholds[view] = threshold
            if value > threshold:
                fired.append(view)
        # Comparison/CPU dispatch time is part of the alarm timestamp too.
        alarm_ready = clock() - started
        first_fired = [view for view in fired if view not in alarms]
        for view in fired:
            if view not in alarms:
                alarms[view] = {'raw_frame': window.raw_endpoint, 'endpoint': endpoint,
                            'source_time_s': window.source_time_s, 'alarm_time_s': alarm_ready,
                            'score': float(scores[view]), 'threshold': used_thresholds[view]}
            pending.discard(view)
        events.append({'endpoint': endpoint, 'raw_frame': window.raw_endpoint,
                       'source_time_s': window.source_time_s, 'started_s': processing_start,
                       'score_ready_s': score_ready, 'alarm_ready_s': alarm_ready,
                       'processing_s': score_ready - processing_start,
                       'source_age_s': alarm_ready - window.source_time_s,
                       'skipped_since_previous': endpoint - last - 1,
                       'scores': {key: float(value) for key, value in scores.items()},
                       'thresholds': used_thresholds, 'first_alarms': first_fired,
                       'stages': getattr(adapter, 'last_stage_times', {})})
        last = endpoint
        if endpoint == len(elapsed) - 1:
            break
    seconds = clock() - started
    eligible_seen = max(0, last - first + 1)
    return {'alarms': alarms, 'missed_views': sorted(pending), 'events': events,
            'elapsed_s': seconds, 'processed_endpoints': len(events),
            'skipped_endpoints': eligible_seen - len(events),
            'eligible_endpoints_through_stop': eligible_seen,
            'total_eligible_endpoints': len(elapsed) - first,
            'throughput_hz': len(events) / seconds if seconds else None,
            'stop_reason': 'all_selected_views_alarmed' if stop_at_first_alarm and not pending else 'source_exhausted',
            'stop_at_first_alarm': stop_at_first_alarm}


def fusion_margin(scores, components, operation):
    """Same-endpoint OR/AND of signed, scaled component threshold margins."""
    if operation not in {'and', 'or'} or len(scores) != 2 or len(components) != 2:
        raise ValueError('Fusion requires two component scores and AND/OR')
    margins = []
    for score, component in zip(scores, components):
        scale, threshold = float(component['scale']), float(component['threshold'])
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(threshold) or not math.isfinite(score):
            raise ValueError('Fusion requires finite scores/thresholds and positive scales')
        margins.append((score - threshold) / scale)
    return (min if operation == 'and' else max)(margins)


class FusionAdapter:
    """Two synchronous scorers on one endpoint, including BOTH encoder costs.

    View fusion shares the one WM forward pass. Method fusion loads the WM and
    native observation model and evaluates them sequentially on the same raw
    image index. There is no stale alarm latch, short circuit or cached-score use.
    """
    def __init__(self, trial, device, factory=EndToEndAdapter):
        self.fusion = trial['fusion']
        self.adapters = {}
        for component in self.fusion['components']:
            key = component['offline_score_cache']
            if key not in self.adapters:
                self.adapters[key] = factory(dict(trial, **component), device)
        self.last_stage_times = {}

    def reset_episode(self):
        for adapter in self.adapters.values():
            adapter.reset_episode()

    def score(self, payload, window):
        values, stages = {}, []
        for component in self.fusion['components']:
            key = component['offline_score_cache']
            if key in values:
                continue
            child = payload['components'][component['dataset']]
            endpoint = int(np.searchsorted(child['raw'], window.raw_endpoint))
            if endpoint >= len(child['raw']) or child['raw'][endpoint] != window.raw_endpoint:
                raise ValueError('Fusion component has no matching raw endpoint')
            if not math.isclose(float(child['elapsed'][endpoint]), window.source_time_s, rel_tol=0, abs_tol=1e-6):
                raise ValueError('Fusion component wall time differs')
            selected = input_window(endpoint, child['raw'], child['elapsed'], component['method'],
                                    component['K'] or 0, component['L'] or 1)
            adapter = self.adapters[key]
            values[key] = adapter.score(child, selected)
            # The ODE meter is shared process-wide; copy before the next scorer.
            stages.append(json.loads(json.dumps(adapter.last_stage_times)))
        components = self.fusion['components']
        scores = [float(values[c['offline_score_cache']][c['view']]) for c in components]
        margin = fusion_margin(scores, components, self.fusion['operation'])
        self.last_stage_times = {name: sum(s.get(name, 0.) for s in stages)
                                 for name in ['encoder_s', 'input_preparation_s', 'detector_s']}
        self.last_stage_times.update(ode_calls=[call for s in stages for call in s.get('ode_calls', [])],
                                     component_scores=scores, component_stages=stages,
                                     operation=self.fusion['operation'])
        return {self.fusion['output_view']: margin}


def threshold_lookup(rows, task, method, calibration):
    if method.startswith('wm_nc_'):
        if calibration != 'episode':
            raise ValueError('Online fusion currently requires constant episode thresholds')
        return lambda view, source_time_s: 0.0  # FusionAdapter returns signed margins.
    selected = [row for row in rows if row['task'] == task and row['method'] == method
                and row['calibration'] == calibration and row['protocol'] == 'latency']
    def threshold(view, source_time_s):
        candidates = [row for row in selected if row['view'] == view and
                      (calibration == 'episode' or
                       float(row['bin_start_s']) <= source_time_s < float(row['bin_end_s']))]
        if not candidates:
            if calibration == 'time_varying':
                return math.inf
            raise ValueError(f'Calibration missing: {task}/{method}/{view}')
        if len(candidates) != 1:
            raise ValueError('Ambiguous calibration threshold')
        return float(candidates[0]['threshold'])
    return threshold


def latency_record(alarm, onset_raw_frame, raw_wall):
    """Label AFTER replay; never let an onset change stopping or frame selection."""
    if onset_raw_frame is None:
        return {'status': 'unannotated', 'delay_s': None}
    if type(onset_raw_frame) is not int or not 0 <= onset_raw_frame < len(raw_wall):
        raise ValueError('Onset outside raw recording')
    onset_s = float(raw_wall[onset_raw_frame] - raw_wall[0])
    if alarm is None:
        return {'status': 'missed', 'delay_s': None, 'onset_time_s': onset_s}
    signed = alarm['alarm_time_s'] - onset_s
    early = alarm['raw_frame'] < onset_raw_frame
    return {'status': 'pre_onset_evidence' if early else 'detected',
            'delay_s': None if early else signed, 'signed_alarm_delay_s': signed,
            'source_evidence_delay_s': alarm['source_time_s'] - onset_s,
            'onset_time_s': onset_s}


def frozen_latencies(result, trial, payload, labels, spec):
    """Apply labels only AFTER full replay; combined detectors use earlier onset."""
    from benchmark_frozen_protocol import alarm_run, allows_anticipation, recorded_curve, uses_post_onset
    from benchmark_onset_results import onset_frame
    from unified_benchmark import episode_labels
    from benchmark_consensus_onsets import elapsed_onset, select_annotation
    if spec.get('detection', {}).get('onset_reference') and spec['detection']['primary_annotator'] not in labels['annotators']:
        raise ValueError('Derive mean onset labels before interpreting a v3 replay')
    task, ep = trial['task'], trial['episode_index']
    group = episode_labels(labels['tasks'][task])[ep]
    annotations = labels['tasks'][task].get('failure_onsets', {}).get(ep, {})
    wall = payload['raw_wall']
    records = []
    for view in trial['requested_detector_views']:
        for aid, annotator in labels['annotators'].items():
            onset = onset_frame(annotations.get(aid), view)
            onset_s = elapsed_onset(select_annotation(annotations.get(aid), view), wall)
            interpreted = alarm_run(recorded_curve(result['events'], view, 'episode'), onset, onset_s,
                allow_anticipation=allows_anticipation(task, group['failure_origin']),
                complete=result['stop_reason'] == 'source_exhausted',
                first_post_onset=uses_post_onset(spec))
            records.append({**interpreted, 'variant': trial['variant'], 'task': task,
                'method': trial['method'], 'view': view, 'onset_view': 'earlier_view' if view == 'both' else view,
                'episode_index': ep, 'annotator_id': aid, 'annotator': annotator['name'],
                'alarm': interpreted['post_alarm'], 'any_time_alarm': view in result['alarms'],
                'delay_s': interpreted['detection_delay_s'], 'onset_time_s': onset_s,
                'late_alarm': bool(interpreted['detection_time'] is not None and
                                   interpreted['detection_time'] > wall[-1]-wall[0]),
                'failure_origin': group['failure_origin'],
                'threshold_protocol': spec['protocol_id'], 'evaluation_protocol': spec['protocol_id']})
    return records


def validate_action_alignment(prepared, observation_raw, raw_wall, action_shift):
    """Verify WM action availability from recorded indices, before replay timing.

    Actions in a K-step window end at prepared row e-1. Checking every adjacent
    row establishes availability by endpoint e for every positive K and L.
    Shift 0 and shift 1 preserve the existing offline/training convention; this
    check never shifts or recomputes actions.
    """
    if type(action_shift) is not int or action_shift not in (0, 1):
        raise ValueError(f'Unsupported WM action_shift={action_shift!r}; expected recorded shift 0 or 1')
    action_raw = np.asarray(prepared['action.raw_frame_index']).reshape(-1)
    if (not np.issubdtype(action_raw.dtype, np.integer)
            or action_raw.shape != observation_raw.shape
            or not np.array_equal(action_raw, observation_raw + action_shift)
            or (action_raw < 0).any() or (action_raw >= len(raw_wall)).any()):
        raise ValueError('Recorded action indices disagree with preparation alignment')
    if ((action_raw[:-1] > observation_raw[1:]).any()
            or (raw_wall[action_raw[:-1]] > raw_wall[observation_raw[1:]]).any()):
        raise ValueError('A WM action is unavailable at its scoring endpoint')
    actions = np.asarray(prepared['action'])
    if actions.ndim != 2 or len(actions) != len(observation_raw) or not np.isfinite(actions).all():
        raise ValueError('Missing or invalid prepared WM actions')
    return action_raw


def preload_episode(dataset: Path, episode_index: int, input_views, *, method=None, raw_payload=None):
    """Decode one raw episode and prepared scalar inputs BEFORE starting the clock.

    This helper is not called by the planner. No latent/token column is read.
    WM inputs retain their recorded action alignment for exact normalization.
    Other methods read observations/state only. method=None retains the older
    helper API, loading and validating actions for callers without a method.
    """
    import pyarrow.parquet as pq
    start = time.perf_counter()
    prep = json.loads((dataset / 'meta/preparation.json').read_text())
    needs_actions = method is None or method.startswith('wm_')
    matches = [row for row in prep['episode_mapping'] if row['dataset_episode_index'] == episode_index]
    if len(matches) != 1:
        raise ValueError('Episode mapping missing or ambiguous')
    filename = matches[0]['raw_file']
    if Path(filename).name != filename:
        raise ValueError('Unsafe raw recording filename')
    raw_path = dataset.parent / 'raw' / filename
    if raw_payload is None:
        with np.load(raw_path, allow_pickle=False) as recording:
            wall = np.asarray(recording['timestamps'], dtype=np.float64).reshape(-1)
            images = {view: recording[f'images_{view}'] for view in input_views}
    else:
        if raw_path.resolve() != Path(raw_payload['raw_path']).resolve():
            raise ValueError('Fusion components refer to different raw recordings')
        wall = raw_payload['raw_wall']
        images = {view: raw_payload['images'][view] for view in input_views}
    if not np.isfinite(wall).all() or not len(wall) or (np.diff(wall) < 0).any():
        raise ValueError('Online latency requires valid recorded wall timestamps')
    for array in images.values():
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[0] != len(wall) or array.shape[-1] != 3:
            raise ValueError('Invalid raw RGB arrays')
    columns = ['episode_index', 'frame_index', 'observation.raw_frame_index',
               'observation.wall_timestamp', 'observation.state']
    if needs_actions:
        columns.extend(['action', 'action.raw_frame_index'])
    tables = [pq.read_table(path, columns=columns, filters=[('episode_index', '=', episode_index)])
              for path in sorted((dataset / 'data').rglob('*.parquet'))]
    prepared = {key: np.asarray([item for table in tables for item in table[key].to_pylist()])
                for key in columns}
    order = np.argsort(prepared['frame_index'])
    prepared = {key: value[order] for key, value in prepared.items()}
    indices = prepared['observation.raw_frame_index'].astype(np.int64).reshape(-1)
    if (not len(indices) or not np.array_equal(prepared['frame_index'], np.arange(len(indices)))
            or indices[0] < 0 or indices[-1] >= len(wall) or (np.diff(indices) <= 0).any()
            or not np.allclose(prepared['observation.wall_timestamp'], wall[indices], atol=1e-5, rtol=0)):
        raise ValueError('Prepared/raw observation alignment differs')
    if needs_actions:
        validate_action_alignment(prepared, indices, wall, prep['job'].get('action_shift'))
    return {'images': images, 'raw_wall': wall, 'raw': indices,
            'elapsed': wall[indices] - wall[0], 'prepared': prepared,
            'raw_path': str(raw_path), 'preload_s': time.perf_counter() - start}


def preload_trial(trial):
    payload = preload_episode(Path(trial['dataset']), trial['episode_index'], trial['input_views'], method=trial['method'])
    payload['episode_index'] = trial['episode_index']
    if trial.get('fusion'):
        children = {trial['dataset']: dict(payload)}
        for component in trial['fusion']['components']:
            path = component['dataset']
            if path not in children:
                child = preload_episode(Path(path), trial['episode_index'], component['input_views'],
                                        method=component['method'], raw_payload=payload)
                child['episode_index'] = trial['episode_index']
                children[path] = child
                payload['preload_s'] += child['preload_s']
            child = children[path]
            # Restrict this implementation to coverage of EVERY eligible WM
            # endpoint. Reject missing inputs rather than silently altering time.
            raw = payload['raw'][trial['first_eligible_endpoint']:]
            if not np.isin(raw, child['raw']).all():
                raise ValueError('Fusion component is missing eligible raw frames')
        payload['components'] = children
    return payload


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def fusion_inventories(report, methods, calibration):
    """Virtual inventories bind existing component caches and calibrated rules."""
    from unified_benchmark import detector_definitions, digest
    definitions = detector_definitions(['wm_nc', 'logpzo_native'], {'fusion': {
        'wm_views': True, 'conditional_marginal': True, 'operations': ['and', 'or']}})
    requested = [d for d in definitions if d[2] is not None and d[0] in methods]
    if requested and calibration != 'episode':
        raise ValueError('Online fusion currently requires constant episode thresholds')
    result = []
    for task in report['tasks']:
        for method, view, (left, right, operation) in requested:
            components, primary = [], None
            for name, camera in [left, right]:
                matches = [c for c in report['score_caches']
                           if c['task'] == task and c['method'] == name and camera in c['views']]
                if len(matches) != 1:
                    raise ValueError(f'Missing/ambiguous fusion component cache: {task}/{name}/{camera}')
                inventory = matches[0]
                if digest(Path(inventory['path'])) != inventory['sha256']:
                    raise ValueError('Fusion component score cache differs from the offline report')
                if primary is None:
                    primary = inventory
                thresholds = [r for r in report['calibration']
                              if (r['task'], r['method'], r['view'], r['protocol'], r['calibration'])
                              == (task, name, camera, 'latency', calibration)]
                if len(thresholds) != 1:
                    raise ValueError('Missing/ambiguous fusion component calibration')
                settings = inventory['scoring_settings']
                components.append({**{k: settings[k] for k in ['dataset', 'checkpoint', 'K', 'L']},
                                   'method': name, 'view': camera, 'input_views': inventory['views'],
                                   'offline_score_cache': inventory['path'],
                                   'threshold': thresholds[0]['threshold'], 'scale': thresholds[0]['scale']})
            # Validate signs/scales before any model is loaded.
            fusion_margin([0., 0.], components, operation)
            result.append({**primary, 'method': method, 'views': [view],
                           'input_views': primary['views'], 'fusion': {
                               'operation': operation, 'output_view': view, 'components': components}})
    return result


def build_plan(report_dir, methods, tasks, calibration, selection, *, frozen_protocol=None):
    report = json.loads((report_dir / 'report.json').read_text())
    if frozen_protocol:
        if report.get('evaluation_protocol', {}).get('protocol_id') != frozen_protocol['protocol_id']:
            raise ValueError(f'Offline report does not use the frozen protocol: {report_dir}')
        if report['labels_sha256'] != frozen_protocol['frozen_inputs']['onsets']['sha256']:
            raise ValueError('Offline report onset labels differ from frozen protocol')
    inventories = report['score_caches'] + fusion_inventories(report, methods, calibration)
    if not set(methods) <= {c['method'] for c in inventories} or not set(tasks) <= set(report['tasks']):
        raise ValueError('Requested task/method is missing from the offline report')
    if calibration not in report['calibration_schemes']:
        raise ValueError('Calibration scheme not present in report')
    episode_rows = csv_rows(report_dir / 'episode_results.csv')
    latency_rows = csv_rows(report_dir / 'latency_episodes.csv')
    thresholds = csv_rows(report_dir / 'calibration.csv')
    candidates, alarmed = defaultdict(set), defaultdict(set)
    alarm_times = {}
    for row in episode_rows:
        if row['calibration'] != calibration or row['split'] != 'test':
            continue
        if selection != 'all-test' and row['outcome'] != 'failure':
            continue
        candidates[row['task'], row['method'], row['view']].add(int(row['episode_index']))
    # Latency rows retain every WM endpoint, unlike historical paper trimming.
    # Deduplication across annotators prevents counting/replaying an episode twice.
    for row in latency_rows:
        if row['calibration'] == calibration and row['first_alarm_raw_frame']:
            episode = int(row['episode_index'])
            alarmed[row['task'], row['method'], row['view']].add(episode)
            key = (row['task'], row['method'], row['view'], episode)
            value = float(row['first_alarm_wall_timestamp'])
            if key in alarm_times and alarm_times[key] != value:
                raise ValueError('Annotator rows disagree on the detector first alarm')
            alarm_times[key] = value
    if frozen_protocol:
        # Select using PRIMARY post-onset detections, not historical first alarms.
        alarmed.clear()
        for row in episode_rows:
            if (row['calibration'] == calibration and row['split'] == 'test'
                    and row['outcome'] == 'failure' and row['alarm'].lower() == 'true'):
                alarmed[row['task'], row['method'], row['view']].add(int(row['episode_index']))
    trials, summary, seen = [], [], set()
    for inventory in inventories:
        task, method = inventory['task'], inventory['method']
        if task not in tasks or method not in methods:
            continue
        views = tuple(inventory.get('output_views', inventory['views']))
        key = (task, method, views)
        if key in seen:
            raise ValueError('Ambiguous checkpoint/cache: choose one run per task/method')
        seen.add(key)
        settings = inventory['scoring_settings']
        dataset = Path(settings['dataset'])
        prep = json.loads((dataset / 'meta/preparation.json').read_text())
        mapping = {row['dataset_episode_index']: row for row in prep['episode_mapping']}
        required = {}
        all_candidates = set()
        for view in views:
            eps = candidates[task, method, view]
            if inventory.get('fusion') and not eps:
                raise ValueError(f'Offline fusion results missing: {task}/{method}/{view}')
            all_candidates.update(eps)
            selected = eps & alarmed[task, method, view] if selection == 'offline-detected' else eps
            for ep in selected:
                required.setdefault(ep, []).append(view)
        timeline = load_timeline(dataset) if required else {}
        duration, alarm_proxy = 0.0, 0.0
        for ep, selected_views in sorted(required.items()):
            if ep not in mapping or ep not in timeline:
                raise ValueError(f'Mapped episode missing: {task}/{ep}')
            seconds = float(mapping[ep]['wall_duration_seconds'])
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Invalid recorded duration')
            first = ((settings['K'] + settings['L'] - 1) if method.startswith('wm_')
                     else int(method.startswith('act_ensemble_')))
            if len(timeline[ep]['frame']) <= first:
                raise ValueError('Episode has no eligible windows')
            lookup = threshold_lookup(thresholds, task, method, calibration)
            for view in selected_views:
                lookup(view, float(timeline[ep]['wall'][first] - timeline[ep]['wall'][0]))
            duration += seconds
            dense_stop = None
            if all((task, method, view, ep) in alarm_times for view in selected_views):
                dense_stop = float(max(alarm_times[task, method, view, ep] for view in selected_views)
                                   - timeline[ep]['wall'][0])
                if dense_stop < 0 or dense_stop > seconds + 1e-4:
                    raise ValueError('Offline alarm outside recorded episode')
            # For full-failure/all-test selection, dense misses consume full replay.
            alarm_proxy += seconds if dense_stop is None else dense_stop
            trials.append({'task': task, 'method': method, 'episode_index': ep,
                           'input_views': inventory.get('input_views', ['fix', 'wrist'] if views == ('both',) else list(views)),
                           'requested_detector_views': selected_views,
                           'dataset': str(dataset), 'checkpoint': settings['checkpoint'],
                           'K': settings['K'], 'L': settings['L'],
                           'raw_file': mapping[ep]['raw_file'], 'recorded_duration_s': seconds,
                           'offline_first_alarm_stop_proxy_s': dense_stop,
                           'first_eligible_endpoint': first, 'total_prepared_frames': len(timeline[ep]['frame']),
                           'offline_ODE_settings': settings, 'offline_score_cache': inventory['path'],
                           **({'score_aggregation': inventory['score_aggregation']} if inventory.get('score_aggregation') else {}),
                           **({'fusion': inventory['fusion']} if inventory.get('fusion') else {}),
                           **({'evaluation_protocol': frozen_protocol['protocol_id'],
                               'stop_at_first_alarm': False} if frozen_protocol else {})})
        summary.append({'task': task, 'method': method, 'views': ','.join(views),
                        'candidate_episodes': len(all_candidates), 'selected_episodes': len(required),
                        'excluded_episodes': len(all_candidates) - len(required),
                        'full_replay_seconds': duration, 'offline_alarm_proxy_seconds': alarm_proxy})
    return {'draft_status': 'CPU plan only; end-to-end inference requires explicit --run',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'offline_report': str(report_dir),
            'offline_report_sha256': hashlib.sha256((report_dir / 'report.json').read_bytes()).hexdigest(),
            'source_file_sha256': {name: hashlib.sha256((report_dir / name).read_bytes()).hexdigest()
                                   for name in ['episode_results.csv', 'latency_episodes.csv',
                                                'calibration.csv', 'labels_snapshot.yaml']},
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'methods': methods, 'tasks': tasks, 'calibration': calibration, 'selection': selection,
            'clock': 'recorded wall time at 1x; monotonic single-worker replay',
            'online_window_batch_size': 1, 'stop': ('source end; retain all selected decisions' if frozen_protocol
                else 'each selected view at its first alarm; otherwise source end'),
            'full_replay_seconds': sum(row['full_replay_seconds'] for row in summary),
            'offline_alarm_proxy_seconds': sum(row['offline_alarm_proxy_seconds'] for row in summary),
            'estimate_note': 'Full selected-episode durations, before early stopping. Excludes preload, '
                             'model loading/warmup and final in-flight computation. Not a measured online runtime.',
            'proxy_note': 'Dense first-alarm source times provide an optimistic timing proxy, not a runtime '
                          'guarantee or an online lower bound. Skipped alarms, encoder/scorer costs, and '
                          'batch/RNG differences can alter both scores and stopping times.',
            'selection_note': 'Offline-detected selection estimates conditional latency/retention only; '
                              'it cannot measure overall online FAR/FDR. Online misses are retained.',
            'report_todos': ['per-annotator/camera onset-to-alarm delay and pre-onset evidence',
                             'online misses and conditional retention with explicit selection denominators',
                             'processed/skipped endpoints, source age, processing stage timings and throughput',
                             'ODE work, samples, precision, hardware and preload/warmup times'],
            'summary': summary, 'trials': trials}


def repo_path(value):
    return resolve_model_path(value, REPO)


def experiment_plan(config_path, chosen_variants=None):
    """Resolve completed reports; pending variants get explicit proxy budgets only."""
    config = yaml.safe_load(config_path.read_text())
    frozen_protocol = None
    if config.get('evaluation_protocol'):
        from benchmark_frozen_protocol import load_protocol
        frozen_protocol = load_protocol(repo_path(config['evaluation_protocol']))
        if config.get('stop_at_first_alarm') is not False:
            raise ValueError('Frozen onset protocol requires complete selected-episode traces')
    if config.get('schema_version') != 1 or config.get('selection') != 'offline-detected':
        raise ValueError('This experiment requires schema 1 and offline-detected failure selection')
    if config.get('window_batch_size') != 1 or config.get('replay_speed') != 1:
        raise ValueError('Online replay must use one window and recorded time at 1x')
    names = [target['id'] for target in config['targets']]
    if len(names) != len(set(names)) or any(not name.replace('_', '').isalnum() for name in names):
        raise ValueError('Choose unique simple variant IDs')
    if chosen_variants and not set(chosen_variants) <= set(names):
        raise ValueError('Unknown requested variant')
    calibration = config['calibration']
    tasks = config['tasks']
    reference_dir = repo_path(config['estimate_reference_report'])
    reference = build_plan(reference_dir, ['wm_nc'], tasks, calibration, 'offline-detected', frozen_protocol=frozen_protocol)
    estimates = {row['task']: row for row in reference['summary']}
    combined = dict(reference, methods=[], trials=[], summary=[], pending=[], sources=[],
                    targets=config['targets'], experiment_config=str(config_path.resolve()),
                    experiment_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    draft_status='Prepared experiment; no online inference has been launched')
    used = set()
    for target in config['targets']:
        variant = target['id']
        if chosen_variants and variant not in chosen_variants:
            continue
        for task in target.get('tasks', tasks):
            report_dir = repo_path(target.get('reports_by_task', {}).get(task, target.get('offline_report', '')))
            expected = target.get('checkpoints', {}).get(task)
            if expected and not repo_path(expected).is_file():
                raise FileNotFoundError(f'Expected checkpoint missing: {expected}')
            if not (report_dir / 'report.json').is_file():
                if target['methods'] != ['wm_nc']:
                    raise ValueError(f'No main method report: {report_dir}')
                row = dict(estimates[task], variant=variant, estimate_status='reference_projection')
                combined['summary'].append(row)
                combined['pending'].append({'variant': variant, 'task': task, 'offline_report': str(report_dir),
                                            'reason': 'Offline scores/calibration report not complete',
                                            'offline_config': target.get('offline_config')})
                continue
            sub = build_plan(report_dir, target['methods'], [task], calibration, 'offline-detected', frozen_protocol=frozen_protocol)
            inventories = json.loads((report_dir / 'report.json').read_text())['score_caches']
            for inventory in inventories:
                if inventory['task'] != task or inventory['method'] not in target['methods']:
                    continue
                settings = inventory['scoring_settings']
                if expected and not resolve_model_path(settings['checkpoint']).samefile(repo_path(expected)):
                    raise ValueError(f'{variant}/{task}: report uses a different checkpoint')
                shape = target.get('shape')
                if shape and (settings['K'], settings['L']) != (shape['K'], shape['L']):
                    raise ValueError(f'{variant}/{task}: report K/L differs from experiment')
                key = (task, inventory['method'], tuple(inventory['views']),
                       resolve_model_path(settings['checkpoint']).stat().st_ino,
                       resolve_model_path(settings['checkpoint']).stat().st_dev)
                if key in used:
                    raise ValueError('A shared checkpoint/method appears twice; keep one K=4/L=4 reference')
                used.add(key)
            for trial in sub['trials']:
                trial.update(variant=variant, offline_report=str(report_dir))
                sources = target.get('frozen_model_sources', {})
                if trial.get('fusion'):
                    # Each trial needs its own component descriptors (subplans
                    # may share inventory dictionaries during construction).
                    trial['fusion'] = json.loads(json.dumps(trial['fusion']))
                    for component in trial['fusion']['components']:
                        if component['method'] in sources:
                            component['frozen_model_source'] = str(repo_path(sources[component['method']]))
                elif trial['method'] in sources:
                    trial['frozen_model_source'] = str(repo_path(sources[trial['method']]))
                scorer_sources = target.get('frozen_offline_scorers', {})
                if trial['method'] in scorer_sources:
                    trial['frozen_offline_scorer'] = str(repo_path(scorer_sources[trial['method']]))
                for component in trial.get('fusion', {}).get('components', [trial]):
                    overrides = config.get('cache_source_overrides', {}).get(component['offline_score_cache'], {})
                    for key in ('frozen_model_source', 'frozen_offline_scorer'):
                        if overrides.get(key):
                            component[key] = str(repo_path(overrides[key]))
                combined['trials'].append(trial)
            combined['summary'].extend(dict(row, variant=variant, estimate_status='measured_offline_selection')
                                       for row in sub['summary'])
            combined['sources'].append({'variant': variant, 'task': task, 'offline_report': str(report_dir),
                                        'report_sha256': sub['offline_report_sha256'],
                                        'source_file_sha256': sub['source_file_sha256']})
            combined['methods'] += [method for method in target['methods'] if method not in combined['methods']]
    combined['full_replay_seconds'] = sum(row['full_replay_seconds'] for row in combined['summary'])
    combined['offline_alarm_proxy_seconds'] = sum(row['offline_alarm_proxy_seconds'] for row in combined['summary'])
    combined['readiness'] = 'pending_offline_reports' if combined['pending'] else 'ready_for_gpu_validation'
    combined['ready_trial_count'] = len(combined['trials'])
    combined['selection_note'] += ' Pending targets have NO episode selections or thresholds yet; '
    combined['selection_note'] += 'their budgets are explicitly projected from the main WM-NC and never used as calibration.'
    if frozen_protocol:
        combined['evaluation_protocol'] = frozen_protocol
        combined['evaluation_protocol_sha256'] = hashlib.sha256(repo_path(config['evaluation_protocol']).read_bytes()).hexdigest()
        combined['stop_at_first_alarm'] = False
        combined['selection_note'] = ('Only primary-annotator OFFLINE POST-ONSET detections are selected. '
            'Every selected episode is replayed to source end, without onset-dependent scheduling. '
            'Conditional latency/retention only; no measured full-test online FAR/FDR.')
    return combined, config


def write_plan(plan, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    rows = plan['summary']
    with (output / 'replay_budget.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else
                                ['task', 'method', 'views', 'candidate_episodes', 'selected_episodes',
                                 'excluded_episodes', 'full_replay_seconds', 'offline_alarm_proxy_seconds'])
        writer.writeheader(); writer.writerows(rows)
    totals = defaultdict(lambda: {'episodes': 0, 'seconds': 0.0, 'proxy': 0.0, 'bases': set()})
    for row in rows:
        key = row.get('variant', 'main') + '/' + row['method']
        totals[key]['episodes'] += row['selected_episodes']
        totals[key]['seconds'] += row['full_replay_seconds']
        totals[key]['proxy'] += row['offline_alarm_proxy_seconds']
        totals[key]['bases'].add(row.get('estimate_status', 'measured_offline_selection'))
    headers = ['Method', 'Selected trials', 'Dense-alarm proxy minutes', 'Full replay minutes', 'Budget basis']
    cells = [[key, value['episodes'], f"{value['proxy']/60:.1f}", f"{value['seconds']/60:.1f}",
              'mixed' if len(value['bases']) > 1 else 'projected' if 'reference_projection' in value['bases'] else 'offline measured']
             for key, value in totals.items()]
    table = '\n'.join(['| ' + ' | '.join(headers) + ' |', '| --- | --- | --- | --- | --- |'] +
                      ['| ' + ' | '.join(map(str, row)) + ' |' for row in cells])
    text = (f"# Online benchmark draft plan\n\n{plan['draft_status']}\n\n"
            f"Selection: `{plan['selection']}`; calibration: `{plan['calibration']}`.\n\n"
            f"{plan['selection_note']}\n\n{plan['estimate_note']}\n\n{plan['proxy_note']}\n\n" + table +
            f"\n\nTotal selected replay budget: **{plan['full_replay_seconds']/3600:.2f} hours**. "
            f"Dense-alarm source-time proxy: **{plan['offline_alarm_proxy_seconds']/3600:.2f} hours**. "
            + ("Complete selected traces are required. " if plan.get('stop_at_first_alarm') is False else
             "Stopping at first alarms can shorten it. ") + "Preload/startup and final computation add overhead.\n")
    (output / 'report.md').write_text(text)
    page = ('<!doctype html><html><head><meta charset="utf-8"><title>Online benchmark draft</title>'
            '<style>body{font:16px/1.5 system-ui;max-width:1000px;margin:40px auto;padding:0 20px}'
            'td,th{padding:8px 16px;text-align:left;border-bottom:1px solid #ddd}</style></head><body>'
            '<h1>Online benchmark draft plan</h1><p>' + html.escape(plan['draft_status']) + '</p><p>' +
            html.escape(plan['selection_note']) + '</p><p>' + html.escape(plan['estimate_note']) + '</p><p>' +
            html.escape(plan['proxy_note']) + '</p>'
            '<table><thead><tr>' + ''.join('<th>' + html.escape(x) + '</th>' for x in headers) +
            '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>'
            for x in row) + '</tr>' for row in cells) + '</tbody></table><p>Total selected replay budget: ' +
            f"{plan['full_replay_seconds']/3600:.2f} hours.</p>" +
            '<p><a href="plan.json">Exact trial list</a> · <a href="replay_budget.csv">Per-task budget</a> · '
            '<a href="report.md">Markdown</a></p></body></html>')
    (output / 'report.html').write_text(page)
    print(table)
    print(f"Total selected replay budget: {plan['full_replay_seconds']/3600:.2f} hours; no GPU inference performed.")
    if plan.get('pending'):
        print(f"Pending offline task/variant reports: {len(plan['pending'])}. These rows use provisional reference budgets.")
    print('Plan:', output)


def assert_exclusive_gpu(config):
    """No model import before pending/offline/GPU checks have passed."""
    import subprocess
    for path in config.get('blocked_by_queues', []):
        state_path = repo_path(path)
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text())
        if state.get('status') == 'running':
            try:
                os.kill(state['pid'], 0)
            except ProcessLookupError:
                continue
            raise RuntimeError(f'Offline queue is still running: {state_path}')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                            check=True, text=True, capture_output=True)
    # Desktop display/remote-desktop processes are allowed; another compute job
    # would invalidate the intended dedicated-desktop latency measurement.
    busy = [line for line in result.stdout.splitlines() if line.strip() and
            not any(name in line for name in ['gnome-remote-desktop', 'snapd-desktop-integration', 'Xorg'])]
    if busy:
        raise RuntimeError('GPU has another compute process; do not benchmark concurrently: ' + '; '.join(busy))


def validate_adapter(adapter, payload, trial, options):
    """Compare one fresh raw-image input with prepared features and offline score.

    This is an explicit GPU preflight during --run, not called by the CPU planner.
    Passing this sampled check does not prove monotonicity on skipped episodes.
    """
    if trial['method'].startswith('act_ensemble_'):
        from act_ensemble_score import validate_adapter as validate_ensemble
        return validate_ensemble(adapter, payload, trial, options)
    if trial.get('fusion'):
        return validate_fusion_adapter(adapter, payload, trial, options)
    import pyarrow.parquet as pq
    first = trial['first_eligible_endpoint']
    window = input_window(first, payload['raw'], payload['elapsed'], trial['method'], trial['K'] or 0, trial['L'] or 1)
    adapter.reset_episode()
    measured = adapter.score(payload, window)
    data = adapter.merged['data']
    views = ['fix', 'wrist'] if adapter.views == ('both',) else adapter.views
    columns = ['frame_index'] + [f'observation.tokens.{v}_camera' if adapter.tokenizer is not None
                                else data['fix_key' if v == 'fix' else 'wrist_key'] for v in views]
    rows = [row for path in sorted((Path(trial['dataset']) / 'data').rglob('*.parquet'))
            for row in pq.read_table(path, columns=columns,
                                     filters=[('episode_index', '=', trial['episode_index']), ('frame_index', '=', first)]).to_pylist()]
    if len(rows) != 1:
        raise ValueError('Prepared feature parity input missing/ambiguous')
    feature_errors, feature_relative_l2 = {}, {}
    for view, column in zip(views, columns[1:]):
        fresh = adapter.feature(view, first).numpy()
        saved = np.asarray(rows[0][column], dtype=np.float32).reshape(fresh.shape)
        feature_errors[view] = float(np.max(np.abs(fresh - saved)))
        feature_relative_l2[view] = float(np.linalg.norm(fresh - saved) / max(np.linalg.norm(saved), 1e-12))
        if not np.allclose(fresh, saved, rtol=options['feature_rtol'], atol=options['feature_atol']):
            raise ValueError(f'Raw/prepared feature parity failed: {trial["variant"]}/{trial["task"]}/{view}; '
                             f'max_abs={feature_errors[view]:.6g}, relative_L2={feature_relative_l2[view]:.6g}')
    with np.load(trial['offline_score_cache'], allow_pickle=False) as cache:
        mask = (cache['episode_index'] == trial['episode_index']) & (cache['frame_index'] == first)
        expected = cache['scores'][mask]
    if expected.shape != (1, len(adapter.views)):
        raise ValueError('Offline score parity input missing/ambiguous')
    aggregation = trial.get('score_aggregation', 'per_view')
    if aggregation == 'joint':
        from unified_benchmark import aggregate_wm_scores
        expected = aggregate_wm_scores(trial['method'], expected, aggregation)
    current = np.asarray([measured[view] for view in (('both',) if aggregation == 'joint' else adapter.views)])
    if not np.allclose(current, expected[0], rtol=options['score_rtol'], atol=options['score_atol']):
        raise ValueError(f'Online/offline score parity failed: {trial["variant"]}/{trial["task"]}/{trial["method"]}; '
                         f'online={current.tolist()}, offline={expected[0].tolist()}')
    adapter.reset_episode()
    return {'feature_max_abs_error': feature_errors, 'feature_relative_l2': feature_relative_l2,
            'online_scores': measured,
            'offline_scores': expected[0].tolist(), 'status': 'sampled_parity_passed',
            'score_relative_errors': (np.abs(current-expected[0])/np.maximum(np.abs(expected[0]),1e-12)).tolist(),
            'score_rtol': options['score_rtol'], 'score_atol': options['score_atol'],
            'note': 'Sampled numerical parity, not a proof that online cannot catch an offline miss'}


def validate_fusion_adapter(adapter, payload, trial, options):
    checks = {}
    first = trial['first_eligible_endpoint']
    raw = int(payload['raw'][first])
    for component in trial['fusion']['components']:
        key = component['offline_score_cache']
        if key in checks:
            continue
        child = payload['components'][component['dataset']]
        endpoint = int(np.searchsorted(child['raw'], raw))
        component_trial = dict(trial, **component, first_eligible_endpoint=endpoint)
        component_trial.pop('fusion')
        checks[key] = validate_adapter(adapter.adapters[key], child, component_trial, options)
    expected = []
    for component in trial['fusion']['components']:
        key = component['offline_score_cache']
        views = adapter.adapters[key].views
        expected.append(checks[key]['offline_scores'][views.index(component['view'])])
    expected_margin = fusion_margin(expected, trial['fusion']['components'], trial['fusion']['operation'])
    adapter.reset_episode()
    window = input_window(first, payload['raw'], payload['elapsed'], trial['method'], trial['K'], trial['L'])
    measured = adapter.score(payload, window)[trial['fusion']['output_view']]
    # Components are checked in their original score units above. Propagate that
    # error bound to calibrated margins, which can have very small scales.
    tolerance = max((options['score_atol'] + options['score_rtol'] * abs(score)) / float(c['scale'])
                    for c, score in zip(trial['fusion']['components'], expected))
    if abs(measured - expected_margin) > tolerance:
        raise ValueError('Online/offline same-frame fusion margin parity failed')
    adapter.reset_episode()
    return {'status': 'sampled_parity_passed', 'components': checks,
            'offline_margin': expected_margin, 'online_margin': measured,
            'margin_error_bound': tolerance, 'operation': trial['fusion']['operation']}


def trial_viewtag(trial):
    tag = '_'.join(trial['input_views'])
    if trial.get('fusion'):
        tag += '_out_' + trial['fusion']['output_view']
    elif trial.get('score_aggregation') == 'joint':
        tag += '_out_joint'
    return tag


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return '+inf' if value > 0 else '-inf' if value < 0 else 'nan'
    return value


def atomic_json(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def online_dashboard(output, state, completed):
    rows = []
    aggregates = defaultdict(list)
    for result in completed:
        for row in result['latencies']:
            aggregates[row['variant'], row['task'], row['method'], row['view'], row['onset_view'], row['annotator']].append(row)
    for key, items in sorted(aggregates.items()):
        delays = [row['delay_s'] for row in items if row['status'] == 'detected' and not row.get('late_alarm')]
        rows.append([*key, len(items), sum(row['alarm'] is True for row in items),
                     sum(row['alarm'] is False for row in items),
                     sum(bool(row.get('early_alarm', row['status'] == 'pre_onset_evidence')) for row in items),
                     sum(bool(row.get('late_alarm')) for row in items), sum(row['alarm'] is None for row in items),
                     '—' if not delays else f'{np.median(delays):.3f}',
                     '—' if not delays else f'{np.quantile(delays, .95):.3f}'])
    headers = ['Variant', 'Task', 'Method', 'View', 'Onset view', 'Annotator', 'Selected', 'Alarms', 'Misses',
               'Early evidence', 'Late detections', 'Unknown', 'On-time median delay (s)', 'On-time p95 delay (s)']
    work = defaultdict(list)
    for result in completed:
        trial = result['trial']
        work[trial['variant'], trial['task'], trial['method'], ','.join(trial['input_views'])].append(result)
    timing_headers = ['Variant', 'Task', 'Method', 'Input views', 'Trials', 'Processed', 'Skipped',
                      'Median compute (s)', 'p95 source age (s)', 'ODE solvers',
                      'Mean ODE steps / score', 'Mean ODE evaluations / score']
    timing_rows = []
    for key, results in sorted(work.items()):
        events = [event for result in results for event in result['events']]
        solves = [call for event in events for call in event['stages'].get('ode_calls', [])]
        compute = [event['processing_s'] for event in events]
        ages = [event['source_age_s'] for event in events]
        timing_rows.append([*key, len(results), sum(r['processed_endpoints'] for r in results),
                            sum(r['skipped_endpoints'] for r in results),
                            '—' if not compute else f'{np.median(compute):.3f}',
                            '—' if not ages else f'{np.quantile(ages, .95):.3f}',
                            ','.join(sorted({call['solver'] for call in solves})) or '—',
                            '—' if not solves else f"{sum(call['steps'] for call in solves)/len(events):.2f}",
                            '—' if not solves else f"{sum(call['evaluations'] for call in solves)/len(events):.2f}"])
    status = f"Status: {state['status']}; completed {state['completed_trials']}/{state['total_trials']} selected trials; current {state.get('current', '—')}"
    note = 'Successful episodes and offline misses are excluded. These are conditional online latency/retention results, not measured overall online FAR/FDR. Missing onset labels have no latency; their alarm/miss flags remain counted. Latency includes raw-image encoding and detector computation. See per-trial traces for skipped windows, source age, ODE work, and stage timings.'
    table = '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                      ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])
    timing_table = '\n'.join(['| ' + ' | '.join(timing_headers) + ' |',
                              '| ' + ' | '.join(['---'] * len(timing_headers)) + ' |'] +
                             ['| ' + ' | '.join(map(str, row)) + ' |' for row in timing_rows])
    timing_note = 'ODE steps/evaluations are measured totals per scored window, including all Monte Carlo solves for WM-PU; per-solve counts and configured sampling settings are preserved in the trial traces.'
    (output / 'report.md').write_text('# Online benchmark\n\n' + status + '\n\n' + note + '\n\n' + table +
                                      '\n\n' + timing_note + '\n\n' + timing_table + '\n')
    refresh = '<meta http-equiv="refresh" content="30">' if state['status'] == 'running' else ''
    page = '<!doctype html><html><head><meta charset="utf-8">' + refresh + '<title>Online benchmark</title><style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}</style></head><body><h1>Online benchmark</h1><p>' + html.escape(status) + '</p><p>' + note + '</p><p>' + html.escape(state.get('error', '')) + '</p><table><tr>' + ''.join('<th>' + html.escape(x) + '</th>' for x in headers) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</table><p>' + timing_note + '</p><table><tr>' + ''.join('<th>' + html.escape(x) + '</th>' for x in timing_headers) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in timing_rows) + '</table></body></html>'
    (output / 'report.html').write_text(page)
    with (output / 'latency_summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(headers); writer.writerows(rows)
    with (output / 'timing_summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(timing_headers); writer.writerows(timing_rows)


def run_experiment(plan, config, output, resume=False):
    if plan.get('pending'):
        raise RuntimeError(f'{len(plan["pending"])} offline reports are pending. Finish calibration/scoring before --run.')
    assert_exclusive_gpu(config)
    import fcntl
    import gc
    # Importing Torch/loading models begins only after the guards above.
    import torch
    if output.exists() and not resume:
        raise FileExistsError('Online output exists; use --resume after inspecting its state')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if resume:
        old = json.loads((output / 'plan.json').read_text())
        for key in ['experiment_config_sha256', 'sources', 'script_sha256']:
            if old[key] != plan[key]:
                raise ValueError('Online experiment inputs changed; choose a new output directory')
    else:
        atomic_json(output / 'plan.json', plan)
    (output / 'trials').mkdir(exist_ok=True)
    groups = defaultdict(list)
    for trial in plan['trials']:
        key = (trial['variant'], trial['task'], trial['method'], trial['checkpoint'],
               trial.get('fusion', {}).get('output_view', ''))
        groups[key].append(trial)
    completed = [json.loads(path.read_text()) for path in sorted((output / 'trials').glob('*.json'))]
    done = {row['trial_id'] for row in completed}
    state = {'status': 'running', 'pid': os.getpid(), 'started': datetime.now(timezone.utc).isoformat(),
             'completed_trials': len(done), 'total_trials': len(plan['trials'])}
    atomic_json(output / 'state.json', state)
    environment = {'python': sys.version, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(torch.device(config['device'])),
                   'window_batch_size': 1, 'replay_speed': 1, 'selection': plan['selection']}
    atomic_json(output / 'environment.json', environment)
    try:
        for (variant, task, method, checkpoint, output_view), trials in groups.items():
            remaining = []
            for trial in trials:
                viewtag = trial_viewtag(trial)
                trial['trial_id'] = f'{variant}_{task}_{method}_{viewtag}_{trial["episode_index"]:06d}'
                if trial['trial_id'] not in done:
                    remaining.append(trial)
            if not remaining:
                continue
            print(f'Loading {variant}/{task}/{method} ({len(remaining)} selected episodes)', flush=True)
            loading = time.perf_counter()
            factory = FusionAdapter if remaining[0].get('fusion') else EndToEndAdapter
            if method.startswith('act_ensemble_'):
                from act_ensemble_score import EnsembleAdapter
                factory = EnsembleAdapter
            adapter = factory(remaining[0], config['device'])
            load_s = time.perf_counter() - loading
            report_dir = Path(remaining[0]['offline_report'])
            thresholds = threshold_lookup(csv_rows(report_dir / 'calibration.csv'), task, method, plan['calibration'])
            labels = yaml.safe_load((report_dir / 'labels_snapshot.yaml').read_text())
            if plan.get('evaluation_protocol'):
                from benchmark_consensus_onsets import derive_labels
                labels = derive_labels(labels, plan['evaluation_protocol'], repo_path)
            first = remaining[0]
            payload = preload_trial(first)
            parity = validate_adapter(adapter, payload, first, config['validation'])
            window = input_window(first['first_eligible_endpoint'], payload['raw'], payload['elapsed'], method,
                                  first['K'] or 0, first['L'] or 1)
            warm_start = time.perf_counter()
            for _ in range(config['warmup_windows']):
                adapter.reset_episode(); adapter.score(payload, window)
            warm_s = time.perf_counter() - warm_start
            adapter.reset_episode()
            atomic_json(output / f'validation_{variant}_{task}_{method}_{trial_viewtag(first)}.json',
                        {**parity, 'model_load_s': load_s, 'warmup_s': warm_s})
            for trial in remaining:
                state['current'] = trial['trial_id']; atomic_json(output / 'state.json', state)
                if len(done) % config.get('dashboard_every_trials', 1) == 0:
                    online_dashboard(output, state, completed)
                if trial is not first:
                    payload = preload_trial(trial)
                result = replay_episode(payload, adapter, thresholds, trial['requested_detector_views'], method,
                                        trial['K'] or 0, trial['L'] or 1,
                                        stop_at_first_alarm=config.get('stop_at_first_alarm', True))
                latencies = []
                for view in trial['requested_detector_views']:
                    onset_views = ['fix', 'wrist'] if view == 'both' else [view]
                    for annotator_id, annotator in labels.get('annotators', {}).items():
                        annotations = labels['tasks'][task].get('failure_onsets', {}).get(trial['episode_index'], {})
                        for onset_view in onset_views:
                            onset = annotations.get(annotator_id, {}).get('views', {}).get(onset_view, {})
                            record = latency_record(result['alarms'].get(view), onset.get('frame_index'), payload['raw_wall'])
                            latencies.append({**record, 'variant': variant, 'task': task, 'method': method,
                                              'view': view, 'onset_view': onset_view, 'episode_index': trial['episode_index'],
                                              'annotator_id': annotator_id, 'annotator': annotator['name'],
                                              'copied_from_annotator': onset.get('copied_from_annotator'),
                                              'alarm': view in result['alarms']})
                if plan.get('evaluation_protocol'):
                    latencies = frozen_latencies(result, trial, payload, labels, plan['evaluation_protocol'])
                result.update(trial_id=trial['trial_id'], trial=trial, latencies=latencies,
                              preload_s=payload['preload_s'], hardware=environment,
                              offline_ODE_settings=trial['offline_ODE_settings'])
                atomic_json(output / 'trials' / f'{trial["trial_id"]}.json', result)
                completed.append(result); done.add(trial['trial_id'])
                state['completed_trials'] = len(done); atomic_json(output / 'state.json', state)
                print(f'{len(done)}/{len(plan["trials"])} {trial["trial_id"]}: {result["stop_reason"]}; '
                      f'{result["elapsed_s"]:.2f}s, {result["processed_endpoints"]} processed, '
                      f'{result["skipped_endpoints"]} skipped', flush=True)
            del adapter, payload
            gc.collect(); torch.cuda.empty_cache()
        state.update(status='complete', finished=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(status='failed', error=str(exc), finished=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        atomic_json(output / 'state.json', state)
        online_dashboard(output, state, completed)
        from results_inspector import archive_result
        try:
            archived = archive_result(output, 'online')
            print(f"Online run {state['status']}: {len(done)}/{len(plan['trials'])} trials. "
                  f"Preserved result archive: {archived}", flush=True)
        except (OSError, ValueError) as archive_error:
            # Preserve the original training/inference exception if this run failed.
            if state['status'] == 'complete':
                raise
            print(f"Result archival failed: {archive_error}. Original results remain at {output}.", flush=True)


def paired_thresholds(bindings, prepared_origin_s):
    """Threshold at source time; functional templates use prepared-frame zero."""
    from functional_conformal import template_values
    def lookup(detector, source_time_s):
        scheme, view = detector.split('/', 1)
        spec = bindings[scheme][view]
        if scheme == 'episode':
            return float(spec['threshold'])
        mu, g = template_values(spec['template'], source_time_s-prepared_origin_s)
        return float(mu + float(spec['h'])*g)
    return lookup


def replay_paired(payload, adapter, bindings, views, method, horizon=0, history_len=1, **timing):
    """One actual scored stream, independent first alarms for frozen protocols.

    Continue until every protocol/view alarms or the source is exhausted. Later
    work cannot change an earlier first alarm. No per-episode outcome/onset enters
    scheduling, and different windows are never batched together.
    """
    class SharedAdapter:
        def reset_episode(self):
            adapter.reset_episode()
        @property
        def last_stage_times(self):
            return getattr(adapter, 'last_stage_times', {})
        def score(self, payload, window):
            scores = adapter.score(payload, window)
            return {scheme+'/'+view: scores[view] for scheme in bindings for view in views}
    detectors = [scheme+'/'+view for scheme in bindings for view in views]
    return replay_episode(payload, SharedAdapter(), paired_thresholds(bindings, float(payload['elapsed'][0])),
                          detectors, method, horizon, history_len, **timing)


def full_streaming_plan(config_path, chosen_variants=None):
    """Plan every test episode from verified labels, never from detector alarms."""
    import subprocess
    import unified_benchmark as offline
    from functional_conformal import load_template_manifest
    config = yaml.safe_load(config_path.read_text())
    schemes = config.get('calibrations', [])
    if (config.get('schema_version') != 2 or config.get('selection') != 'all-test'
            or schemes != ['episode', 'functional_band']
            or config.get('window_batch_size') != 1 or config.get('replay_speed') != 1):
        raise ValueError('Schema 2 requires all-test, paired episode/functional bands, window batch 1 and replay speed 1')
    root = repo_path(config['offline_comparison'])
    manifest = json.loads((root/'artifact_manifest.json').read_text())
    if json.loads((root/'state.json').read_text())['status'] != 'complete':
        raise ValueError('Complete the offline comparison before all-test streaming')
    hashes = {}
    def digest(path):
        path = Path(path).resolve()
        if str(path) not in hashes:
            hashes[str(path)] = offline.digest(path)
        return hashes[str(path)]
    def verified(path):
        if digest(path) != manifest['files'].get(str(path.relative_to(root))):
            raise ValueError(f'Offline artifact changed: {path}')
        return path
    targets = config['targets']
    names = [t['id'] for t in targets]
    if (len(names) != len(set(names)) or any(not n.replace('_', '').isalnum() for n in names)
            or chosen_variants and not set(chosen_variants) <= set(names)):
        raise ValueError('Invalid/unknown streaming variants')
    plan = {'schema_version': 2, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'selection': 'all-test', 'calibrations': schemes, 'trials': [], 'summary': [],
            'bindings': {}, 'labels': {}, 'sources': [], 'config': config,
            'experiment_config_sha256': digest(config_path), 'script_sha256': digest(Path(__file__)),
            'shared_stream': True, 'window_batch_size': 1, 'replay_speed': 1,
            'deadline_seconds': config.get('deadline_seconds', [1, 2, 5, 10]),
            'clock': 'recorded wall time, one latest eligible window, raw-image inference',
            'stop': 'each protocol/view at its first alarm; replay until all alarm or source end'}
    seen = set()
    for target in targets:
        variant = target['id']
        if chosen_variants and variant not in chosen_variants:
            continue
        directory = root/variant
        reports = {}
        for scheme in schemes:
            folder = directory/scheme
            for filename in ['report.json', 'labels_snapshot.yaml', 'calibration.csv', 'episode_results.csv']:
                verified(folder/filename)
            reports[scheme] = json.loads((folder/'report.json').read_text())
        if reports['episode']['labels_sha256'] != reports['functional_band']['labels_sha256']:
            raise ValueError('Paired protocols have different annotation snapshots')
        labels = yaml.safe_load((directory/'episode/labels_snapshot.yaml').read_text())
        plan['labels'][variant] = labels
        templates = load_template_manifest(verified(directory/'templates_manifest.json'))
        for template in templates.values():
            if template['time_origin'] != 'first_prepared_observation_wall_timestamp':
                raise ValueError('Unknown functional template time origin')
        tasks = target.get('tasks', config['tasks'])
        if not set(tasks) <= set(reports['episode']['tasks']) or not set(target['methods']) <= set(reports['episode']['methods']):
            raise ValueError('Requested task or method is absent from the offline comparison')
        # Establish all expected IDs from annotations; reject missing cached score
        # rows instead of silently changing denominators to available detections.
        expected = {task: {ep: group for ep, group in offline.episode_labels(labels['tasks'][task]).items()
                           if group['split'] == 'test'} for task in tasks}
        for scheme in schemes:
            rows = csv_rows(directory/scheme/'episode_results.csv')
            grouped = defaultdict(list)
            for row in rows:
                if row['split'] == 'test':
                    grouped[row['task'], row['method'], row['view']].append(int(row['episode_index']))
            for inventory in reports[scheme]['score_caches']:
                task, method = inventory['task'], inventory['method']
                if task not in tasks or method not in target['methods']:
                    continue
                for view in inventory['views']:
                    ids = grouped[task, method, view]
                    if len(ids) != len(set(ids)) or set(ids) != set(expected[task]):
                        raise ValueError(f'Incomplete test coverage: {variant}/{scheme}/{task}/{method}/{view}')
        for inventory in reports['episode']['score_caches']:
            task, method, views = inventory['task'], inventory['method'], inventory['views']
            if task not in tasks or method not in target['methods']:
                continue
            settings = inventory['scoring_settings']
            cache_path = Path(inventory['path'])
            if digest(cache_path) != inventory['sha256']:
                raise ValueError(f'PRL score cache changed: {cache_path}')
            other = [c for c in reports['functional_band']['score_caches']
                     if (c['task'], c['method'], c['views']) == (task, method, views)]
            if len(other) != 1 or other[0]['sha256'] != inventory['sha256']:
                raise ValueError('Paired protocols must share exactly the same raw scorer')
            checkpoint = resolve_model_path(settings['checkpoint'])
            unique = (task, method, tuple(views), str(checkpoint.resolve()))
            if unique in seen:
                raise ValueError('Shared checkpoint repeated: include the K4/L4 reference once')
            seen.add(unique)
            if target.get('shape') and any(settings[k] != v for k, v in target['shape'].items()):
                raise ValueError('K/L settings differ from experiment config')
            with np.load(cache_path, allow_pickle=False) as saved:
                identity = json.loads(str(saved['provenance_json'].item()))['identity']
            frozen = None
            for filename, expected_hash in identity['sha256'].items():
                path = resolve_model_path(filename)
                if path == REPO/'lerobot/utils/unified_benchmark.py' and digest(path) != expected_hash:
                    source = subprocess.check_output(['git', 'show', config['offline_scorer_revision']+':lerobot/utils/unified_benchmark.py'], cwd=REPO)
                    if hashlib.sha256(source).hexdigest() != expected_hash:
                        raise ValueError('Recorded offline scorer revision does not match PRL cache')
                    snapshot = repo_path(config['source_snapshot_dir'])/('offline_'+expected_hash+'.py')
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    if snapshot.exists() and snapshot.read_bytes() != source:
                        raise ValueError('Frozen scorer snapshot changed')
                    if not snapshot.exists():
                        snapshot.write_bytes(source)
                    frozen = str(snapshot)
                    digest(snapshot)
                elif digest(path) != expected_hash:
                    raise ValueError(f'Frozen scoring input changed: {path}')
            binding_key = '/'.join([variant, task, method, ','.join(views)])
            bindings = {s: {} for s in schemes}
            for scheme in schemes:
                for view in views:
                    records = [r for r in reports[scheme]['calibration']
                               if (r['task'], r['method'], r['view'], r['protocol']) == (task, method, view, 'latency')]
                    if len(records) != 1:
                        raise ValueError('Need one causal calibrated threshold per detector/protocol')
                    record = records[0]
                    if scheme == 'episode':
                        bindings[scheme][view] = {'threshold': record['threshold'], 'record': record}
                    else:
                        template = templates[task, method, view, 'latency']
                        if template['source']['checkpoint_sha256'] != digest(checkpoint):
                            raise ValueError('Functional template belongs to a different checkpoint')
                        if record['calibration_episode_ids'] != list(range(19)) or record['n_cal'] != 19:
                            raise ValueError('Keep all 19 original successful calibration episodes')
                        bindings[scheme][view] = {'h': record['h'], 'template': template, 'record': record}
            plan['bindings'][binding_key] = bindings
            dataset = Path(settings['dataset'])
            preparation = json.loads((dataset/'meta/preparation.json').read_text())
            mapping = {r['dataset_episode_index']: r for r in preparation['episode_mapping']}
            timeline = load_timeline(dataset)
            duration = 0.
            for ep, label in sorted(expected[task].items()):
                if ep not in timeline or ep not in mapping:
                    raise ValueError('Test episode missing from dataset')
                first = settings['K']+settings['L']-1 if method.startswith('wm_') else 0
                if len(timeline[ep]['frame']) <= first:
                    raise ValueError('Test episode has no eligible window; report it explicitly before running')
                item = mapping[ep]
                seconds = float(item['wall_duration_seconds'])
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError('Invalid wall duration')
                duration += seconds
                input_views = ['fix', 'wrist'] if views == ['both'] else views
                plan['trials'].append({'trial_id': f'{variant}_{task}_{method}_{"_".join(input_views)}_{ep:06d}',
                    'variant': variant, 'task': task, 'method': method, 'episode_index': ep,
                    **{k: label[k] for k in ['split', 'outcome', 'failure_mode', 'failure_origin']},
                    'input_views': input_views, 'requested_detector_views': views,
                    'binding_key': binding_key, 'dataset': str(dataset), 'checkpoint': str(checkpoint),
                    'K': settings['K'], 'L': settings['L'], 'raw_file': item['raw_file'],
                    'recorded_duration_s': seconds, 'first_eligible_endpoint': first,
                    'total_prepared_frames': len(timeline[ep]['frame']), 'offline_ODE_settings': settings,
                    'offline_score_cache': str(cache_path), 'frozen_offline_scorer': frozen})
            plan['summary'].append({'variant': variant, 'task': task, 'method': method, 'views': views,
                                    'trials': len(expected[task]), 'full_replay_seconds': duration})
        plan['sources'].append({'variant': variant, 'offline_comparison': str(directory)})
    plan['full_replay_seconds'] = sum(r['full_replay_seconds'] for r in plan['summary'])
    digest(REPO/'lerobot/utils/functional_conformal.py')
    plan['source_hashes'] = hashes
    if len({t['trial_id'] for t in plan['trials']}) != len(plan['trials']):
        raise ValueError('Duplicate streaming trials')
    return plan, config


def full_trial_records(trial, result, labels, raw_wall, schemes):
    """Outcome counts are independent of annotation count/availability."""
    outcomes, latencies = [], []
    fields = {k: trial[k] for k in ['variant', 'task', 'method', 'episode_index', 'outcome', 'failure_mode', 'failure_origin']}
    for scheme in schemes:
        for view in trial['requested_detector_views']:
            alarm = result['alarms'].get(scheme+'/'+view)
            common = {**fields, 'calibration': scheme, 'view': view, 'alarm': alarm is not None}
            outcomes.append({**common, 'first_alarm': alarm})
            if trial['outcome'] == 'success':
                continue
            annotations = labels['tasks'][trial['task']].get('failure_onsets', {}).get(trial['episode_index'], {})
            for annotator_id, annotator in labels.get('annotators', {}).items():
                for onset_view in (['fix', 'wrist'] if view == 'both' else [view]):
                    onset = annotations.get(annotator_id, {}).get('views', {}).get(onset_view, {})
                    record = latency_record(alarm, onset.get('frame_index'), raw_wall)
                    latencies.append({**common, **record, 'annotator_id': annotator_id, 'annotator': annotator['name'],
                                      'onset_view': onset_view, 'onset_raw_frame': onset.get('frame_index'),
                                      'copied_from_annotator': onset.get('copied_from_annotator'),
                                      'alarm_before_onset': record.get('signed_alarm_delay_s', 0) < 0})
    return outcomes, latencies


def full_streaming_metrics(plan, outcomes, latencies):
    """Explicit complete/planned and annotated denominators; never pool annotators."""
    keys = ['variant', 'task', 'method', 'view', 'calibration']
    expected, observed, annotated = defaultdict(list), defaultdict(list), defaultdict(list)
    def add(mapping, fields, row):
        mapping[tuple(row[k] for k in fields)].append(row)
        mapping[tuple('Overall' if k == 'task' else row[k] for k in fields)].append(row)
    for trial in plan['trials']:
        for scheme in plan['calibrations']:
            for view in trial['requested_detector_views']:
                add(expected, keys, {**trial, 'view': view, 'calibration': scheme})
    identities = set()
    for row in outcomes:
        identity = tuple(row[k] for k in keys)+ (row['episode_index'],)
        if identity in identities:
            raise ValueError('Duplicate measured detector/episode outcome')
        identities.add(identity)
        add(observed, keys, row)
    latency_keys = keys+['annotator_id', 'annotator', 'onset_view']
    for row in latencies:
        add(annotated, latency_keys, row)
    metrics, onset_metrics, deadlines = [], [], []
    for key, planned in sorted(expected.items()):
        rows = observed[key]
        successes = [r for r in rows if r['outcome'] == 'success']
        failures = [r for r in rows if r['outcome'] == 'failure']
        fa, tp = sum(r['alarm'] for r in successes), sum(r['alarm'] for r in failures)
        metrics.append(dict(zip(keys, key)) | {'successes_planned': sum(r['outcome']=='success' for r in planned),
            'failures_planned': sum(r['outcome']=='failure' for r in planned),
            'successes_tested': len(successes), 'failures_tested': len(failures),
            'false_alarms': fa, 'alarmed_failures': tp, 'missed_failures': len(failures)-tp,
            'FAR': fa/len(successes) if successes else None, 'FDR_any_alarm': tp/len(failures) if failures else None,
            'status': 'complete' if len(rows)==len(planned) else 'partial'})
    for key, rows in sorted(annotated.items()):
        valid = [r for r in rows if r['status'] != 'unannotated']
        delays = [r['delay_s'] for r in valid if r['status'] == 'detected']
        common = dict(zip(latency_keys, key)) | {'failures_tested': len(rows), 'annotated_failures': len(valid),
            'missing_onsets': len(rows)-len(valid), 'post_onset_detections': len(delays),
            'post_onset_detection_rate': len(delays)/len(valid) if valid else None,
            'no_post_onset_detection': len(valid)-len(delays),
            'pre_onset_alarms': sum(r['alarm_before_onset'] for r in valid),
            'pre_onset_evidence': sum(r['status']=='pre_onset_evidence' for r in valid),
            'no_alarm_failures': sum(not r['alarm'] for r in valid),
            'delay_n': len(delays), 'delay_mean_s': float(np.mean(delays)) if delays else None,
            'delay_sd_s': float(np.std(delays, ddof=1)) if len(delays)>1 else None,
            'delay_median_s': float(np.median(delays)) if delays else None,
            'delay_p95_s': float(np.quantile(delays, .95)) if delays else None}
        onset_metrics.append(common)
        for deadline in plan['deadline_seconds']:
            n = sum(d <= deadline for d in delays)
            deadlines.append(dict(zip(latency_keys, key)) | {'deadline_s': deadline, 'detections_by_deadline': n,
                             'annotated_failures': len(valid), 'missing_onsets': len(rows)-len(valid),
                             'rate': n/len(valid) if valid else None})
    return {'metrics': metrics, 'onset_metrics': onset_metrics, 'deadline_rates': deadlines}


def streaming_dashboard(output, plan, state, outcomes, latencies):
    import unified_benchmark as offline
    summary = full_streaming_metrics(plan, outcomes, latencies)
    for name, rows in summary.items():
        offline.write_csv(output/(name+'.csv'), rows, ['status'])
    offline.write_csv(output/'episode_results.csv', outcomes, ['status'])
    offline.write_csv(output/'latency_episodes.csv', latencies, ['status'])
    atomic_json(output/'report.json', {'state': state, 'selection': 'all-test', 'calibrations': plan['calibrations'],
                                     'shared_stream': True, **summary})
    pct = lambda x: '—' if x is None else f'{100*x:.1f}%'
    fmt = lambda x: '—' if x is None else f'{x:.2f}'
    def table(headers, rows):
        return '\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in [headers, ['---']*len(headers), *rows])
    status = f"{state['status']}: {state['completed_trials']}/{state['total_trials']} trials. Current: {state.get('current', '—')}."
    notes = ('All verified test successes and failures, including offline misses; calibration episodes excluded. '
             'Constant and functional thresholds are frozen before replay and share each actually processed input. '
             'FDR_any_alarm counts any first alarm, including pre-onset alarms; FAR counts false alarms on successes. '
             'Post-onset and deadline rates use ALL annotated failures as denominator (early evidence and misses count as non-detections); '
             'missing onsets are reported separately. Pre-onset alarms use alarm completion time; pre-onset evidence uses source-frame time. '
             'Conditional delay statistics include only detections based on post-onset evidence. Annotators/camera onsets stay separate. '
             'Partial rates use completed episodes only and show planned denominators. No offline-detection filtering is applied.')
    headers = ['Variant','Task','Method','View','Calibration','Success tested/planned','Failure tested/planned','FA/n','FAR','Alarm/n','FDR any alarm','No-alarm failures']
    rows = [[*[r[k] for k in ['variant','task','method','view','calibration']],
             f"{r['successes_tested']}/{r['successes_planned']}", f"{r['failures_tested']}/{r['failures_planned']}",
             f"{r['false_alarms']}/{r['successes_tested']}", pct(r['FAR']), f"{r['alarmed_failures']}/{r['failures_tested']}",
             pct(r['FDR_any_alarm']), r['missed_failures']] for r in summary['metrics'] if r['task']=='Overall']
    oh = ['Variant','Method','View','Calibration','Annotator','Onset view','Annotated/tested failures','Post-onset/n','Early alarms','Early evidence','No post-onset detection','Delay mean±SD s (n)']
    onset = [[*[r[k] for k in ['variant','method','view','calibration','annotator','onset_view']],
              f"{r['annotated_failures']}/{r['failures_tested']}", f"{r['post_onset_detections']}/{r['annotated_failures']}",
              r['pre_onset_alarms'],r['pre_onset_evidence'],r['no_post_onset_detection'],
              f"{fmt(r['delay_mean_s'])}±{fmt(r['delay_sd_s'])} ({r['delay_n']})"]
             for r in summary['onset_metrics'] if r['task']=='Overall']
    dh = ['Variant','Method','View','Calibration','Annotator','Onset view','Deadline s','Detected/annotated','Rate']
    deadline = [[*[r[k] for k in ['variant','method','view','calibration','annotator','onset_view','deadline_s']],
                 f"{r['detections_by_deadline']}/{r['annotated_failures']}",pct(r['rate'])]
                for r in summary['deadline_rates'] if r['task']=='Overall']
    md = '# Full-test streaming benchmark\n\n'+status+'\n\n'+notes+'\n\n'+table(headers, rows)
    md += '\n\n## Onset and conditional delays\n\n'+table(oh,onset)+'\n\n## Detection deadlines\n\n'+table(dh,deadline)+'\n'
    (output/'report.md').write_text(md)
    def ht(headers, rows):
        return '<div style="overflow:auto"><table><tr>'+''.join('<th>'+html.escape(s)+'</th>' for s in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(s))+'</td>' for s in r)+'</tr>' for r in rows)+'</table></div>'
    page = '<!doctype html><meta charset="utf-8">'+('<meta http-equiv="refresh" content="60">' if state['status']=='running' else '')
    page += '<title>Full-test streaming benchmark</title><style>body{font:14px system-ui;margin:24px;color:#193142}table{border-collapse:collapse}th,td{padding:7px;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#edf3f7}</style>'
    page += '<h1>Full-test streaming benchmark</h1><p>'+html.escape(status)+'</p><p>'+html.escape(notes)+'</p>'
    page += '<p>'+ ' · '.join(f'<a href="{n}">{n}</a>' for n in ['report.md','metrics.csv','onset_metrics.csv','deadline_rates.csv','episode_results.csv','latency_episodes.csv'])+'</p>'
    page += ht(headers,rows)+'<h2>Onset and conditional delays</h2>'+ht(oh,onset)+'<h2>Detection deadlines</h2>'+ht(dh,deadline)
    temporary=output/'.report.html.tmp';temporary.write_text(page);temporary.replace(output/'report.html')


def run_full_streaming(plan, config, output, resume=False):
    """Detached/resumable full coverage; old conditional result archives stay intact."""
    import fcntl
    import gc
    assert_exclusive_gpu(config)
    import torch
    if output.exists() and not resume:
        raise FileExistsError('Use a new output, or --resume for unchanged completed trials')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if resume:
        previous=json.loads((output/'plan.json').read_text())
        for key in ['experiment_config_sha256','script_sha256','source_hashes','trials','bindings','calibrations']:
            if previous[key]!=plan[key]:raise ValueError('Streaming resume inputs changed: '+key)
    else:atomic_json(output/'plan.json',plan)
    (output/'trials').mkdir(exist_ok=True)
    expected={t['trial_id']:t for t in plan['trials']}
    done=set();outcomes=[];latencies=[]
    for path in sorted((output/'trials').glob('*.json')):
        r=json.loads(path.read_text());tid=r['trial_id']
        if tid not in expected or r['trial']!=expected[tid] or tid in done:
            raise ValueError('Unexpected or changed completed trial: '+tid)
        done.add(tid);outcomes.extend(r['outcomes']);latencies.extend(r['latencies'])
    state={'status':'running','pid':os.getpid(),'started':datetime.now(timezone.utc).isoformat(),
           'completed_trials':len(done),'total_trials':len(plan['trials'])}
    environment={'python':sys.version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
                 'gpu':torch.cuda.get_device_name(torch.device(config['device'])),
                 'window_batch_size':1,'replay_speed':1,'selection':'all-test','shared_stream':True,
                 'calibrations':plan['calibrations']}
    atomic_json(output/'environment.json',environment)
    groups=defaultdict(list)
    for t in plan['trials']:groups[t['binding_key']].append(t)
    last_report=0.
    try:
        for binding_key,trials in groups.items():
            remaining=[t for t in trials if t['trial_id'] not in done]
            if not remaining:continue
            first=remaining[0];method=first['method'];variant=first['variant'];task=first['task']
            print(f'Loading {binding_key}: {len(remaining)} test episodes',flush=True)
            state['current']='loading '+binding_key;atomic_json(output/'state.json',state)
            loading=time.perf_counter();adapter=EndToEndAdapter(first,config['device']);load_s=time.perf_counter()-loading
            payload=preload_episode(Path(first['dataset']),first['episode_index'],first['input_views'],method=method)
            payload['episode_index']=first['episode_index']
            parity=validate_adapter(adapter,payload,first,config['validation'])
            window=input_window(first['first_eligible_endpoint'],payload['raw'],payload['elapsed'],method,first['K'] or 0,first['L'] or 1)
            started=time.perf_counter()
            for _ in range(config['warmup_windows']):adapter.reset_episode();adapter.score(payload,window)
            atomic_json(output/('validation_'+binding_key.replace('/','_')+'.json'),{**parity,'model_load_s':load_s,'warmup_s':time.perf_counter()-started})
            for trial in remaining:
                state['current']=trial['trial_id'];atomic_json(output/'state.json',state)
                if time.perf_counter()-last_report>=config.get('report_interval_seconds',60):
                    streaming_dashboard(output,plan,state,outcomes,latencies);last_report=time.perf_counter()
                if trial is not first:
                    payload=preload_episode(Path(trial['dataset']),trial['episode_index'],trial['input_views'],method=method)
                    payload['episode_index']=trial['episode_index']
                result=replay_paired(payload,adapter,plan['bindings'][binding_key],trial['requested_detector_views'],
                                     method,trial['K'] or 0,trial['L'] or 1)
                eps,lats=full_trial_records(trial,result,plan['labels'][variant],payload['raw_wall'],plan['calibrations'])
                result.update(trial_id=trial['trial_id'],trial=trial,outcomes=eps,latencies=lats,
                              preload_s=payload['preload_s'],hardware=environment,offline_ODE_settings=trial['offline_ODE_settings'])
                atomic_json(output/'trials'/(trial['trial_id']+'.json'),result)
                done.add(trial['trial_id']);outcomes.extend(eps);latencies.extend(lats)
                state['completed_trials']=len(done);atomic_json(output/'state.json',state)
                print(f'{len(done)}/{len(plan["trials"])} {trial["trial_id"]} ({trial["outcome"]}): '
                      f'{result["elapsed_s"]:.2f}s; {len(result["alarms"])} protocol/view alarms; '
                      f'{result["processed_endpoints"]} processed, {result["skipped_endpoints"]} skipped',flush=True)
            del adapter,payload;gc.collect();torch.cuda.empty_cache()
        state.update(status='complete',finished=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(status='failed',error=str(exc),finished=datetime.now(timezone.utc).isoformat());raise
    finally:
        atomic_json(output/'state.json',state)
        streaming_dashboard(output,plan,state,outcomes,latencies)
        print(f"Full-test streaming {state['status']}: {len(done)}/{len(plan['trials'])}. Report: {output/'report.html'}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--offline-report', type=Path)
    parser.add_argument('--config', type=Path, help='Experiment YAML with main methods and checkpoint variants')
    parser.add_argument('--variants', nargs='+', help='Choose experiment variant IDs; default all')
    parser.add_argument('--run', action='store_true', help='Explicitly run end-to-end inference; default is CPU planning only')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--methods', nargs='+', choices=list(METHODS) + [
        'wm_nc_and', 'wm_nc_or', 'wm_nc_logpzo_native_and', 'wm_nc_logpzo_native_or'])
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--calibration', choices=['episode', 'time_varying'], default='episode')
    parser.add_argument('--selection', choices=['offline-detected', 'failures', 'all-test'], default='offline-detected')
    parser.add_argument('--output', type=Path, required=True, help='New draft plan directory; existing paths rejected')
    args = parser.parse_args()
    if args.config:
        if args.offline_report or args.methods or args.tasks:
            parser.error('With --config use --variants to restrict the experiment')
        specification = yaml.safe_load(args.config.read_text())
        if specification.get('schema_version') == 2:
            plan, config = full_streaming_plan(args.config.resolve(), args.variants)
            if args.run:
                run_full_streaming(plan, config, args.output.resolve(), args.resume)
            else:
                args.output.mkdir(parents=True, exist_ok=False)
                atomic_json(args.output/'plan.json',plan)
                print(f'Full-test plan: {len(plan["trials"])} trials; {plan["full_replay_seconds"]/3600:.2f} hours '
                      f'full-duration replay, shared by both thresholds. No GPU inference performed. Plan: {args.output}')
            return
        plan, config = experiment_plan(args.config.resolve(), args.variants)
        if args.run:
            run_experiment(plan, config, args.output.resolve(), args.resume)
        else:
            write_plan(plan, args.output.resolve())
        return
    if args.run or args.resume or args.variants or not args.offline_report:
        parser.error('--run/--resume/--variants require --config; otherwise supply --offline-report')
    report_dir = args.offline_report.resolve()
    report = json.loads((report_dir / 'report.json').read_text())
    methods, tasks = args.methods or report['methods'], args.tasks or report['tasks']
    if len(set(methods)) != len(methods) or len(set(tasks)) != len(tasks):
        parser.error('Choose unique tasks and methods')
    if args.output.exists():
        parser.error('Output already exists; choose a new plan directory')
    plan = build_plan(report_dir, methods, tasks, args.calibration, args.selection)
    write_plan(plan, args.output.resolve())


if __name__ == '__main__':
    main()
