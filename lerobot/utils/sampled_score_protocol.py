"""Normal-data validation for sampled failure scores.

ACE is an ACT-based adaptation of FIPER's action branch. WM-PU is a
Ward-inspired score on FoMo-FD's WM, not a reproduction of Ward's model.
Settings are chosen using expert data, before policy calibration/test scoring.
No failure labels, performance metrics or sign selection enter these helpers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BOUNDARIES = {
    'ace': 'ACT-based adaptation of FIPER action branch; prior-sampled Cartesian histogram entropy with frozen normal-spread bin widths; not full FIPER.',
    'wm_pu': 'Ward-inspired mean latent coordinate SD on FoMo-FD FM-WM; not a reproduction of the Ward world model.',
}


def normal_partition(episode_ids, seed, fit_fraction=.6):
    ids = sorted(episode_ids)
    if len(ids) < 4 or len(set(ids)) != len(ids) or not 0 < fit_fraction < 1:
        raise ValueError('Need unique normal episode IDs and nonempty independent partitions')
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids).tolist()
    n = max(2, min(len(ids)-2, int(len(ids)*fit_fraction)))
    return sorted(ids[:n]), sorted(ids[n:])


def histogram_health(positions, widths):
    """[frames,samples,horizon,XYZ]; equal weight per selected frame."""
    from unified_benchmark import ace_histogram_entropy
    p = np.asarray(positions, dtype=float)
    if p.ndim != 4 or p.shape[1] < 2 or p.shape[-1] != 3 or not np.isfinite(p).all():
        raise ValueError('Malformed position samples')
    scores = np.array([ace_histogram_entropy(row, widths) for row in p])
    per_step = scores/p.shape[2]
    return {'frames': len(p), 'zero_fraction': float(np.mean(per_step < .05)),
            'saturated_fraction': float(np.mean(per_step >= np.log2(p.shape[1])-.05)),
            'median_bits_per_step': float(np.median(per_step)),
            'unique_scores': int(len(np.unique(np.round(scores, 10)))),
            'min': float(scores.min()), 'max': float(scores.max()),
            'std': float(scores.std()), 'mean': float(scores.mean())}


def health_passes(health, criteria):
    return (health['zero_fraction'] <= criteria['max_zero_fraction']
            and health['saturated_fraction'] <= criteria['max_saturated_fraction']
            and criteria['median_entropy_min'] <= health['median_bits_per_step'] <= criteria['median_entropy_max']
            and health['unique_scores'] >= min(criteria['min_unique_scores'], health['frames']))


def fit_ace_resolution(fit_positions, validation_positions, settings):
    """Choose on fit episodes, then independently gate on validation episodes.

    Median conditional IQR keeps fixed bin widths in physical units while
    resolving this policy's very small prior-induced variation. Widths never
    depend on the query observation at evaluation time. All candidates and
    acceptance criteria must be fixed before calling this function.
    """
    fit = np.asarray(fit_positions, dtype=float)
    reference = np.median(np.quantile(fit, .75, axis=1)-np.quantile(fit, .25, axis=1), axis=(0, 1))
    if reference.shape != (3,) or not np.isfinite(reference).all() or (reference < 0).any():
        raise ValueError('Invalid normal conditional IQR')
    floor = float(settings['min_width_m'])
    factors = list(settings['width_multipliers'])
    if floor <= 0 or not np.isfinite(floor) or any(x <= 0 or not np.isfinite(x) for x in factors):
        raise ValueError('Positive finite bin widths required')
    candidates = []
    for factor in sorted(set(factors), reverse=True):
        widths = np.maximum(reference*factor, floor)
        health = histogram_health(fit, widths)
        candidates.append({'multiplier': factor, 'widths_m': widths.tolist(), 'fit': health,
                           'fit_passed': health_passes(health, settings['criteria'])})
    selected = next((c for c in candidates if c['fit_passed']), None)
    result = {'reference_conditional_iqr_m': reference.tolist(), 'candidates': candidates,
              'selection': 'coarsest fit-passing fixed widths; independent episode validation gate',
              'passed': False, 'adaptation': BOUNDARIES['ace']}
    if selected is None:
        return result | {'reason': 'No candidate resolves normal samples without degeneracy or saturation'}
    validation = histogram_health(validation_positions, selected['widths_m'])
    passed = health_passes(validation, settings['criteria'])
    return result | {'widths_m': selected['widths_m'], 'multiplier': selected['multiplier'],
                     'fit': selected['fit'], 'validation': validation, 'passed': passed,
                     'reason': 'passed' if passed else 'Independent normal validation failed; do not evaluate failures'}


def validate_ode_steps(steps):
    if type(steps) is not int or not 1 <= steps <= 1024:
        raise ValueError('ode_steps must be an integer in [1,1024]')
    return steps


def ace_cell_widths(settings, repo):
    if 'cell_widths_m' in settings:
        widths = np.asarray(settings['cell_widths_m'], dtype=float)
    else:
        path = Path(settings['expert_stats'])
        if not path.is_absolute(): path = Path(repo)/path
        stats = json.loads(path.read_text())['observation.state']
        widths = np.maximum((np.asarray(stats['max']).reshape(-1)[:3]-
                             np.asarray(stats['min']).reshape(-1)[:3])*settings['cell_size_factor'],
                            settings['min_cell_width'])
    if widths.shape != (3,) or not np.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError('ACE requires three positive finite frozen cell widths')
    return widths


def sample_wm_endpoints(model, inputs, noise, sample_batch_size, precision, ode_steps=1):
    """Yield [batch,sample,view,H,W,C] endpoint groups with fixed random inputs."""
    import torch
    from torchdiffeq import odeint
    from unified_benchmark import repeat_batch
    validate_ode_steps(ode_steps)
    if model.config.fm_head.ode_solver != 'rk4':
        raise ValueError('Explicit step-count protocol currently supports RK4 only')
    if type(sample_batch_size) is not int or sample_batch_size < 1:
        raise ValueError('sample_batch_size must be positive')
    b, samples = noise.shape[:2]
    device = inputs['action_chunk'].device
    context = {k: inputs[k] for k in ('fix_hist', 'wrist_hist', 'state_hist', 'action_chunk')}
    with torch.inference_mode(), torch.autocast(device_type=device.type,
            dtype=torch.bfloat16, enabled=precision == 'bfloat16' and device.type == 'cuda'):
        for start in range(0, samples, sample_batch_size):
            n = min(sample_batch_size, samples-start)
            repeated = repeat_batch(context, n)
            x0 = torch.as_tensor(noise[:, start:start+n].copy(), device=device)
            x0 = x0.reshape(b*n, *noise.shape[2:])*model.config.fm_head.base_std
            if model._num_views() == 1:
                x0 = x0[:, 0]
            model.odefunc.set_context(repeated['fix_hist'], repeated['wrist_hist'],
                                     repeated['state_hist'], repeated['action_chunk'])
            head = model.config.fm_head
            # Preserve legacy one-step path exactly. rtol/atol do not refine RK4.
            options = {} if ode_steps == 1 else {'options': {'step_size': 1/ode_steps}}
            drawn = odeint(model.odefunc, x0, torch.tensor([0., 1.], device=device),
                           rtol=head.ode_rtol, atol=head.ode_atol, method='rk4', **options)[-1]
            yield drawn.float().reshape(b, n, *noise.shape[2:])


def score_wm_pu_batch(model, inputs, noise, sample_batch_size, precision, ode_steps=1):
    import torch
    mean = m2 = None
    count = 0
    for group in sample_wm_endpoints(model, inputs, noise, sample_batch_size, precision, ode_steps):
        for offset in range(group.shape[1]):
            sample = group[:, offset]
            count += 1
            if mean is None:
                mean, m2 = sample.clone(), torch.zeros_like(sample)
            else:
                delta = sample-mean
                mean += delta/count
                m2 += delta*(sample-mean)
    if count < 2:
        raise ValueError('At least two endpoint samples are required')
    return (m2/count).clamp_min(0).sqrt().flatten(2).mean(-1)


def refinement_error(coarse, fine):
    """Same windows AND samples; errors returned separately for each camera."""
    a, b = np.asarray(coarse, dtype=np.float64), np.asarray(fine, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 6 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite paired [window,sample,view,H,W,C] endpoints')
    sd_a = a.std(1).reshape(a.shape[0], a.shape[2], -1).mean(-1)
    sd_b = b.std(1).reshape(b.shape[0], b.shape[2], -1).mean(-1)
    rel = np.abs(sd_a-sd_b)/np.maximum(sd_b, 1e-8)
    axes = (1, 3, 4, 5)
    nrmse = np.sqrt(np.mean((a-b)**2, axis=axes))/np.maximum(np.sqrt(np.mean(b**2, axis=axes)), 1e-8)
    return {'score_relative_median': np.median(rel, axis=0).tolist(),
            'score_relative_p95': np.quantile(rel, .95, axis=0).tolist(),
            'endpoint_relative_p95': np.quantile(nrmse, .95, axis=0).tolist()}


def refinement_passes(error, criteria):
    return all(np.all(np.asarray(error[key]) <= limit) for key, limit in criteria.items())


def select_converged_steps(records, criteria):
    """Require two successive refinement agreements; never fall back to the cap."""
    for i in range(len(records)-1):
        a, b = records[i:i+2]
        if a['fine_steps'] != b['coarse_steps']:
            raise ValueError('Refinement records must form a contiguous sequence')
        if refinement_passes(a['error'], criteria) and refinement_passes(b['error'], criteria):
            return a['coarse_steps']
    return None


def load_frozen_sampling(path, repo, task, method, checkpoint_sha256):
    path = Path(path)
    if not path.is_absolute():
        path = Path(repo)/path
    manifest = json.loads(path.read_text())
    if manifest.get('status') != 'frozen_before_policy_calibration_and_test':
        raise ValueError('Sampling settings must be frozen before PRL scoring')
    entry = manifest['tasks'][task][method]
    if not entry['passed'] or entry['checkpoint_sha256'] != checkpoint_sha256:
        raise ValueError('Frozen sampling does not match validated checkpoint')
    if entry['source_role'] != 'expert_validation':
        raise ValueError('Only expert-normal data may choose revised settings')
    return entry


def phase_length_summary(curves):
    """Normal-only descriptive analysis; relative phase is offline hindsight.

    Prefix maxima quantify opportunity effects without changing the deployed
    score. Phase never becomes an online feature or a retimed CP threshold.
    """
    result = {'episodes': [], 'phase': [], 'prefixes': [],
              'phase_note': 'Relative episode phase uses final duration; diagnostic only, never online calibration.'}
    for episode, curve in curves.items():
        scores = np.asarray(curve['score'], dtype=float)
        t = np.asarray(curve['wall'], dtype=float)
        if not len(scores) or not np.isfinite(scores).all() or np.any(np.diff(t) < 0):
            raise ValueError('Invalid normal trajectory')
        span = max(float(t[-1]-t[0]), 1e-12)
        phase = (t-t[0])/span
        result['episodes'].append({'episode': episode, 'windows': len(scores),
            'duration_s': float(t[-1]-t[0]), 'maximum': float(scores.max()), 'mean': float(scores.mean())})
        for fraction in (.25, .5, .75, 1.):
            values = scores[phase <= fraction]
            result['prefixes'].append({'episode': episode, 'fraction': fraction,
                'windows': len(values), 'maximum': float(values.max())})
        for lo in (0., .25, .5, .75):
            keep = (phase >= lo) & ((phase < lo+.25) if lo < .75 else (phase <= 1))
            if keep.any():
                result['phase'].append({'episode': episode, 'phase_start': lo,
                    'windows': int(keep.sum()), 'mean': float(scores[keep].mean()), 'maximum': float(scores[keep].max())})
    return result
