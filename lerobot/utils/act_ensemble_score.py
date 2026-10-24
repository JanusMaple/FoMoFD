"""ACT adaptation of Thompson et al. (arXiv:2501.10561), not a reproduction.

Three independently initialized policies, deterministic z=0, first chunk action.
Position variance and mean SO(3) geodesic dispersion are differenced at adjacent
prepared observations. Higher signed differences are anomalous. No rectification,
smoothing, division by dt, gripper contribution, or failure-label tuning.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import numpy as np

METHODS = ('act_ensemble_pos', 'act_ensemble_rot')
BOUNDARY = ('ACT ensemble adaptation of Thompson et al., arXiv:2501.10561: three RGB/state '
            'policies instead of five goal-pointcloud DeformerNets; rolling adjacent-observation '
            'disagreement differences instead of initial slope; success-only conformal calibration. '
            'Recorded single-policy rollouts, no ensemble control or intervention.')


def rotations_from_6d(values):
    """Repository convention: concatenate the first TWO COLUMNS, Gram-Schmidt."""
    x = np.asarray(values, dtype=np.float64)
    a, b = x[..., :3], x[..., 3:6]
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    if (norm < 1e-10).any():
        raise ValueError('Degenerate first rotation column')
    a = a / norm
    b = b - np.sum(a*b, axis=-1, keepdims=True)*a
    norm = np.linalg.norm(b, axis=-1, keepdims=True)
    if (norm < 1e-10).any():
        raise ValueError('Degenerate second rotation column')
    b = b / norm
    return np.stack((a, b, np.cross(a, b)), axis=-1)


def pose_dispersion(actions):
    """[observations,members,10] physical actions -> [observations,2], m^2/rad.

    Paper's rotational 'variance' is mean geodesic angle, NOT squared angle.
    Project the mean rotation to SO(3), including determinant correction.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 3 or actions.shape[1] < 2 or actions.shape[2] != 10 or not np.isfinite(actions).all():
        raise ValueError('Expected finite [observations,>=2 members,10] physical actions')
    pos = actions[..., :3]
    variance = np.mean(np.sum((pos-pos.mean(axis=1, keepdims=True))**2, axis=-1), axis=1)
    rotations = rotations_from_6d(actions[..., 3:9])
    u, _, vh = np.linalg.svd(rotations.mean(axis=1))
    signs = np.ones((len(actions), 3))
    signs[:, -1] = np.linalg.det(u @ vh)
    mean = (u * signs[:, None, :]) @ vh
    relative = rotations @ mean[:, None].swapaxes(-1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)
    rotation = np.arccos(cosine).mean(axis=1)
    return np.stack((variance, rotation), axis=1)


def difference_scores(episodes, frames, dispersion):
    """Never difference across episode boundaries; frame zero has no score."""
    episodes, frames, dispersion = np.asarray(episodes), np.asarray(frames), np.asarray(dispersion)
    if len(episodes) != len(frames) or dispersion.shape != (len(frames), 2):
        raise ValueError('Invalid ensemble trajectory shape')
    keep = np.zeros(len(frames), dtype=bool)
    if len(frames) > 1:
        keep[1:] = episodes[1:] == episodes[:-1]
        if ((frames[1:]-frames[:-1])[keep[1:]] != 1).any():
            raise ValueError('Ensemble slope requires consecutive prepared observations')
    return episodes[keep], frames[keep], np.diff(dispersion, axis=0)[keep[1:]]


class Ensemble:
    def __init__(self, checkpoints, device):
        import torch
        from unified_benchmark import load_act_decoder, processor_stats
        from npz_to_le_v21 import _load_frozen_visual_tokenizer
        self.torch, self.device, self.members = torch, device, []
        torch.set_float32_matmul_precision('highest')
        for checkpoint in map(Path, checkpoints):
            tokenizer = _load_frozen_visual_tokenizer(str(checkpoint.parent), str(device))
            if self.members:
                first = self.members[0][0].dinov2
                other = tokenizer.dinov2.state_dict()
                if any(not torch.equal(v, other[k]) for k, v in first.state_dict().items()):
                    raise ValueError('Policy DINO weights differ; cannot share visual backbone')
                tokenizer.dinov2 = first
            policy = load_act_decoder(checkpoint, device)
            if policy.config.chunk_size != 4 or not policy.config.use_vae:
                raise ValueError('Expected the frozen chunk-4 ACT policies')
            prep, eps = processor_stats(checkpoint, 'policy_preprocessor.json', 'normalizer_processor')
            post, _ = processor_stats(checkpoint, 'policy_postprocessor.json', 'unnormalizer_processor')
            self.members.append((tokenizer, policy, prep, eps, post))
        if len(self.members) != 3:
            raise ValueError('This frozen adaptation requires exactly three policies')

    def predict(self, payload, indices):
        """Raw images -> shared frozen DINO -> each member's OWN compressor/ACT."""
        torch, device = self.torch, self.device
        start = time.perf_counter()
        raw = payload['raw'][indices]
        with torch.inference_mode():
            patches = {}
            for view in ('fix', 'wrist'):
                rgb = payload['images'][view][raw]
                if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                    raise ValueError('Expected raw uint8 RGB observations')
                tensor = torch.as_tensor(np.ascontiguousarray(rgb), device=device).float().permute(0, 3, 1, 2)/255
                patch, cls = self.members[0][0]._extract_dino(tensor)
                patches[f'observation.images.{view}_camera'] = {'x_norm_patchtokens': patch, 'x_norm_clstoken': cls}
            torch.cuda.synchronize(device)
            encoded = time.perf_counter()
            state = np.asarray(payload['prepared']['observation.state'][indices], dtype=np.float32)
            actions = []
            for tokenizer, policy, prep, eps, post in self.members:
                tokens = tokenizer.compress_from_patch_features(patches).compressed_tokens
                normalized = (state-prep['observation.state.mean'])/(prep['observation.state.std']+eps)
                batch = {'observation.state': torch.as_tensor(normalized, device=device),
                         'compressed_visual_tokens': tokens, 'observation.images': list(tokens.values())}
                latent = torch.zeros((len(indices), policy.config.latent_dim), device=device)
                action = policy.model.decode_with_given_latent(batch, latent)[:, 0].float().cpu().numpy()
                actions.append(action*post['action.std']+post['action.mean'])
            torch.cuda.synchronize(device)
        finished = time.perf_counter()
        self.last_stage_times = {'encoder_s': encoded-start, 'input_preparation_s': 0.,
                                 'detector_s': finished-encoded, 'ode_calls': []}
        return np.stack(actions, axis=1)


def run_worker(spec, device):
    from online_benchmark import preload_episode
    from unified_benchmark import write_json
    dataset = Path(spec['dataset'])
    # Both scalar detectors share this physical-action cache. Its identity covers
    # every member, processor, raw recording, scalar input, code and batch setting.
    key = spec.get('ensemble_cache_key')
    root = dataset.parent / 'score_result/act_ensemble'
    target = root / f'{key}.npz' if key else Path(spec['output'])/'development_predictions.npz'
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            episodes, frames, dispersion = (saved[k] for k in ('episode_index', 'frame_index', 'dispersion'))
        print('Reusing verified-identity ensemble predictions:', target, flush=True)
    else:
        ensemble = Ensemble(spec['merged']['ensemble']['checkpoints'], device)
        prep = json.loads((dataset/'meta/preparation.json').read_text())
        episodes, frames, actions = [], [], []
        for entry in prep['episode_mapping']:
            ep = entry['dataset_episode_index']
            if spec.get('episode_ids') is not None and ep not in spec['episode_ids']:
                continue
            payload = preload_episode(dataset, ep, ['fix', 'wrist'], method='act_ensemble_pos')
            count = min(len(payload['raw']), spec.get('max_frames') or len(payload['raw']))
            for start in range(0, count, spec['batch_size']):
                ids = np.arange(start, min(count, start+spec['batch_size']))
                actions.extend(ensemble.predict(payload, ids))
                episodes.extend([ep]*len(ids)); frames.extend(ids.tolist())
            print(f'ACT ensemble episode {ep}: {count} observations', flush=True)
        actions = np.asarray(actions)
        dispersion = pose_dispersion(actions)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f'.{os.getpid()}.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, episode_index=episodes, frame_index=frames,
                                dispersion=dispersion, actions=actions)
        os.replace(temporary, target)
        write_json(target.with_suffix('.json'), {'spec': spec, 'reproduction_boundary': BOUNDARY,
            'prediction_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'dispersion_min': dispersion.min(axis=0).tolist(), 'dispersion_max': dispersion.max(axis=0).tolist()})
    metadata = json.loads(target.with_suffix('.json').read_text())
    if hashlib.sha256(target.read_bytes()).hexdigest() != metadata['prediction_sha256']:
        raise ValueError('Ensemble prediction cache checksum mismatch')
    ep, frame, slopes = difference_scores(episodes, frames, dispersion)
    return ep, frame, slopes[:, [METHODS.index(spec['method'])]]


class EnsembleAdapter:
    def __init__(self, trial, device):
        import torch
        from unified_benchmark import digest
        with np.load(trial['offline_score_cache'], allow_pickle=False) as saved:
            identity = json.loads(str(saved['provenance_json'].item()))['identity']
        if identity['task'] != trial['task'] or identity['method'] != trial['method']:
            raise ValueError('Ensemble cache identity mismatch')
        for path, expected in identity['sha256'].items():
            if digest(Path(path)) != expected:
                raise ValueError(f'Ensemble input changed: {path}')
        self.engine = Ensemble(identity['merged_config']['ensemble']['checkpoints'], torch.device(device))
        self.method, self.views = trial['method'], ('both',)
        self.reset_episode()

    def reset_episode(self):
        self.cache = {}
        self.last_stage_times = {}

    def score(self, payload, window):
        # Always the adjacent RECORDED observation, never the last processed one.
        # This is one two-observation eligible window; not multi-window replay.
        start = time.perf_counter()
        ids = [window.endpoint-1, window.endpoint]
        if ids[0] < 0:
            raise ValueError('Ensemble slope unavailable at the first observation')
        stages = {'encoder_s': 0., 'input_preparation_s': 0., 'detector_s': 0., 'ode_calls': []}
        for index in ids:
            if index not in self.cache:
                prediction = self.engine.predict(payload, np.asarray([index]))
                self.cache[index] = pose_dispersion(prediction)[0]
                for name in ('encoder_s', 'input_preparation_s', 'detector_s'):
                    stages[name] += self.engine.last_stage_times[name]
        self.last_pair = np.stack([self.cache[i] for i in ids])
        value = np.diff(self.last_pair, axis=0)[0, METHODS.index(self.method)]
        self.cache = {window.endpoint: self.cache[window.endpoint]}
        stages['total_adapter_s'] = time.perf_counter()-start
        self.last_stage_times = stages
        return {'both': float(value)}


def validate_adapter(adapter, payload, trial, options):
    from online_benchmark import input_window
    first = trial['first_eligible_endpoint']
    adapter.reset_episode()
    measured = adapter.score(payload, input_window(first, payload['raw'], payload['elapsed'], trial['method']))
    with np.load(trial['offline_score_cache'], allow_pickle=False) as cache:
        keep = (cache['episode_index'] == trial['episode_index']) & (cache['frame_index'] == first)
        expected = cache['scores'][keep]
    if expected.shape != (1, 1):
        raise ValueError('Missing ensemble parity input')
    component = METHODS.index(trial['method'])
    # Differences can nearly cancel. Use a relative bound on the underlying
    # physical dispersions, with explicit per-unit absolute floors.
    tolerance = .02*float(np.max(np.abs(adapter.last_pair[:, component]))) + (1e-10 if component == 0 else 1e-6)
    if abs(measured['both']-expected[0, 0]) > tolerance:
        raise ValueError(f'Ensemble raw/offline parity failed: {measured} vs {expected.tolist()}, tolerance={tolerance}')
    adapter.reset_episode()
    return {'status': 'sampled_parity_passed', 'online_scores': measured,
            'offline_scores': expected[0].tolist(), 'score_absolute_bound': tolerance,
            'note': 'Disagreement-scale numerical bound; no threshold or orientation adjustment'}
