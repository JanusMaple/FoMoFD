"""Upper-sided, trajectory-wide split conformal bands (NumPy only).

The expert template is frozen before the 19 successful policy calibration
rollouts are read. Variable lengths use elapsed recorded wall seconds, including
simulation; no normalization by a trajectory's eventual length is performed.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np

REFERENCE = 'https://arxiv.org/html/2503.08558v3#A2'
REFERENCE_CODE = 'https://github.com/CXU-TRI/FAIL-Detect/blob/main/UQ_test/timeseries_cp/methods/functional_predictor.py'
ADAPTATION = (
    'FAIL-Detect-style adaptation, not an exact reproduction: expert-validation '
    'demonstrations fit the template; all 19 original successful policy rollouts '
    'calibrate its trajectory-wide multiplier. Elapsed recorded wall time, '
    'variable-length available-support interpolation, and boundary-hold '
    'extrapolation replace equal-length control-step arrays. Default g(t)=1. '
    'The released one-sided FunctionalPredictor uses np.quantile(R, 1-alpha) '
    '(default linear interpolation); we use rank ceil((n+1)*(1-alpha)), with '
    '+infinity if rank>n. We use signed (score-mu)/g and strict >; the released '
    'plotting code uses >= for upper-band crossing. Nominal marginal '
    'trajectory-level coverage assumes exchangeable successful calibration/test '
    'trajectories conditional on the frozen model/template and identical scoring '
    'and observation rules; it does not bound FAR in every finite test batch.'
)


def canonical_scheme(scheme):
    return 'wall_time_bin' if scheme == 'time_varying' else scheme


def load_template_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    result = {}
    for entry in manifest['templates']:
        key = tuple(entry[k] for k in ['task', 'method', 'view', 'protocol'])
        source = path.parent/entry['path']
        if key in result or hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Duplicate or changed frozen template: {source}')
        template = json.loads(source.read_text())
        if template['source']['role'] != 'expert_validation':
            raise ValueError('Template source must be expert validation')
        template_values(template, [0.])
        result[key] = template
    return result


def order_statistic(values, alpha):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 < alpha < 1:
        raise ValueError('Calibration residuals must be a finite vector; alpha must lie in (0,1)')
    rank = math.ceil((len(values)+1)*(1-alpha))
    if rank > len(values):
        warnings.warn(f'Conformal rank {rank} exceeds {len(values)} calibration trajectories; h=+infinity', RuntimeWarning)
        return math.inf, rank
    return float(np.sort(values)[rank-1]), rank


def valid_trace(curve, clock, *, episode):
    score = np.asarray(curve['score'], dtype=float)
    wall = np.asarray(curve['wall'], dtype=float)
    if score.ndim != 1 or not len(score) or wall.shape != score.shape:
        raise ValueError(f'Episode {episode}: no valid scores or malformed trace')
    if not np.isfinite(score).all() or not np.isfinite(wall).all():
        raise ValueError(f'Episode {episode}: nonfinite score/time; refusing to silently drop inputs')
    origin = float(clock['wall'][0])
    elapsed = wall-origin
    if not np.isfinite(origin) or (elapsed < 0).any() or (np.diff(elapsed) < 0).any():
        raise ValueError(f'Episode {episode}: invalid elapsed wall times')
    return elapsed, score


def fit_template(source, timeline, episode_ids, provenance, options=None):
    """Fit ONLY the explicit saved expert-validation IDs, never policy rollouts.

    Grid = regular elapsed-second knots plus each expert's first/last scoring
    time. Interpolate each expert only inside its valid support; average the
    contributors equally at each knot. Unsupported grid knots are filled by
    linear interpolation of the fitted means; outer knots hold boundary means.
    This does not pad individual expert scores or disable late monitoring.
    """
    options = options or {}
    ids = list(episode_ids)
    if (provenance.get('role') != 'expert_validation' or not ids or len(ids) != len(set(ids))
            or set(ids) != set(source) or set(ids)-set(timeline)):
        raise ValueError('Template requires exactly the saved expert-validation IDs and explicit dataset role')
    step = float(options.get('grid_seconds', 1.0))
    if not math.isfinite(step) or step <= 0:
        raise ValueError('Template grid_seconds must be finite and positive')
    modulation = options.get('modulation', 'constant')
    if modulation not in {'constant', 'robust_mad'}:
        raise ValueError('modulation must be constant or robust_mad')
    floor = float(options.get('scale_floor', 1e-6))
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError('Positive scale_floor must be fixed before calibration/test')
    traces = [valid_trace(source[e], timeline[e], episode=e) for e in ids]
    end = max(t[-1] for t, _ in traces)
    grid = np.unique(np.r_[np.arange(0, end, step), [v for t, _ in traces for v in (t[0], t[-1])]])
    matrix = np.full((len(traces), len(grid)), np.nan)
    for i, (times, values) in enumerate(traces):
        unique, inverse = np.unique(times, return_inverse=True)
        collapsed = np.full(len(unique), -np.inf)
        np.maximum.at(collapsed, inverse, values)
        valid = (grid >= unique[0]) & (grid <= unique[-1])
        matrix[i, valid] = np.interp(grid[valid], unique, collapsed)
    counts = np.isfinite(matrix).sum(axis=0)
    supported = counts > 0
    mu = np.nansum(matrix[:, supported], axis=0)/counts[supported]
    mean = np.interp(grid, grid[supported], mu)
    g = np.ones(len(grid))
    if modulation == 'robust_mad':
        values = matrix[:, supported]
        med = np.nanmedian(values, axis=0)
        scale = np.maximum(floor, 1.4826*np.nanmedian(np.abs(values-med), axis=0))
        g = np.interp(grid, grid[supported], scale)
    return {'schema_version': 1, 'grid_s': grid.tolist(), 'mu': mean.tolist(), 'g': g.tolist(),
            'contributors': counts.tolist(), 'expert_validation_episode_ids': ids,
            'source': provenance, 'grid_seconds': step, 'modulation': modulation,
            'scale_floor': floor if modulation != 'constant' else None,
            'time_coordinate': 'elapsed_recorded_wall_seconds', 'score_orientation': 'higher_is_more_abnormal',
            'time_origin': 'first_prepared_observation_wall_timestamp',
            'grid_rule': 'regular knots plus expert score-support endpoints; available-support mean',
            'interpolation': 'linear', 'extrapolation': 'hold_boundary_values',
            'duplicate_timestamp_rule': 'maximum_score', 'nonfinite_policy': 'error_with_episode_id',
            'reference': REFERENCE, 'reference_implementation': REFERENCE_CODE, 'adaptation': ADAPTATION}


def template_values(template, elapsed):
    grid, mu, scale = (np.asarray(template[k], dtype=float) for k in ['grid_s', 'mu', 'g'])
    if (grid.ndim != 1 or not len(grid) or mu.shape != grid.shape or scale.shape != grid.shape
            or not all(np.isfinite(a).all() for a in [grid, mu, scale])
            or (np.diff(grid) <= 0).any() or (scale <= 0).any()
            or template['extrapolation'] != 'hold_boundary_values'):
        raise ValueError('Invalid frozen functional template')
    return np.interp(elapsed, grid, mu), np.interp(elapsed, grid, scale)


def calibrate_band(source, timeline, calibration_ids, template, alpha):
    ids = list(calibration_ids)
    if not ids or len(ids) != len(set(ids)) or set(ids)-set(source):
        raise ValueError('Missing or duplicated policy calibration episodes')
    residuals = []
    for e in ids:
        elapsed, scores = valid_trace(source[e], timeline[e], episode=e)
        mu, g = template_values(template, elapsed)
        residuals.append({'episode_index': e, 'R': float(np.max((scores-mu)/g)), 'n_valid_scores': len(scores)})
    h, rank = order_statistic([r['R'] for r in residuals], alpha)
    return {'calibration_episode_ids': ids, 'residuals': residuals, 'n_cal': len(ids),
            'alpha': alpha, 'rank': rank, 'h': h if math.isfinite(h) else '+inf',
            'quantile_rule': 'ceil((n_cal+1)*(1-alpha))-th smallest; +inf if rank>n_cal',
            'residual': 'max_t((raw_score(t)-mu(t))/g(t))', 'alarm_rule': 'strict >',
            'status': 'finite' if math.isfinite(h) else 'insufficient_calibration_trajectories'}


def apply_band(source, timeline, template, band):
    transformed = {}
    h = float(band['h'])
    for e, curve in source.items():
        elapsed, raw = valid_trace(curve, timeline[e], episode=e)
        mu, g = template_values(template, elapsed)
        residual = (raw-mu)/g
        transformed[e] = {**curve, 'raw_score': raw, 'score': residual-h,
                          'roc_score': residual, 'template_mu': mu, 'template_g': g,
                          'threshold': mu+h*g, 'support': np.full(len(raw), band['n_cal'])}
    return transformed
