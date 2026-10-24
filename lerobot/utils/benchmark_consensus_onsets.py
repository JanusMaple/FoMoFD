"""CPU-only, derived mean onsets. Never modify individual annotations.

Frame indices select evidence; exact mean wall-clock times measure latency.
The first frame at/after the mean is NOT the timestamp used for latency.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np


REFERENCE = 'consensus_mean'


def mean_annotation(annotations, wall):
    wall = np.asarray(wall, dtype=np.float64)
    if not len(wall) or not np.isfinite(wall).all() or (np.diff(wall) < 0).any():
        raise ValueError('Invalid raw wall-clock timestamps')
    frames = [a.get('frame_index') for a in annotations.values()]
    if not frames or any(type(f) is not int or not 0 <= f < len(wall) for f in frames):
        raise ValueError('All specified annotators must supply a valid onset')
    elapsed = float(np.mean(wall[frames] - wall[0]))
    timestamp = float(wall[0] + elapsed)
    # Use the same absolute timestamp representation as score caches. No
    # rounding of mean frame numbers or interpolation of detector decisions.
    frame = int(np.searchsorted(wall, timestamp, side='left'))
    return dict(frame_index=frame, wall_timestamp=timestamp, elapsed_seconds=elapsed,
                reference='mean_annotated_wall_time',
                contributing_frames={a: x['frame_index'] for a, x in annotations.items()},
                copied_contributors=[a for a, x in annotations.items() if x.get('copied_from_annotator')])


def derive_labels(labels, spec, resolve=None):
    rule = spec.get('detection', {}).get('onset_reference', {}) if spec else {}
    if rule.get('kind') != 'mean_wall_time':
        return labels
    if REFERENCE in labels['annotators']:
        return labels
    from benchmark_onset_results import RawClocks
    from unified_benchmark import annotation_views, episode_labels
    repo = Path(__file__).resolve().parents[2]
    resolve = resolve or (lambda p: repo / p)
    clocks = RawClocks(resolve)
    required = rule['annotator_ids']
    if len(required) != len(set(required)) or set(required) - set(labels['annotators']):
        raise ValueError('Consensus annotator roster does not match labels')
    result = deepcopy(labels)
    result['annotators'] = {REFERENCE: {'name': 'Mean onset (3 annotators)', 'derived': True}, **result['annotators']}
    for task, node in result['tasks'].items():
        dataset = f'fomofd/tasks/{task}/prl_data/processed_le_v21_vfmf_vae_fix_wrist'
        for ep, group in episode_labels(node).items():
            entries = node.get('failure_onsets', {}).get(ep, {})
            if group['outcome'] != 'failure':
                if entries:
                    raise ValueError(f'Success episode has onset annotations: {task}/{ep}')
                continue
            wall = clocks(dataset, ep)
            views = {}
            for view in ('fix', 'wrist'):
                annotations = {aid: annotation_views(entries.get(aid)).get(view, {}) for aid in required}
                views[view] = mean_annotation(annotations, wall)
            entries[REFERENCE] = {'mode': 'separate', 'views': views, 'derived': True}
    return result


def select_annotation(entry, view):
    from unified_benchmark import annotation_views
    views = annotation_views(entry)
    required = ('fix', 'wrist') if view == 'both' else (view,)
    values = [views.get(v) for v in required]
    if any(v is None or v.get('frame_index') is None for v in values):
        return None
    # Wall time resolves means falling within the same frame interval.
    return min(values, key=lambda a: a.get('wall_timestamp', a['frame_index']))


def annotation_timestamp(annotation, fallback):
    return annotation['wall_timestamp'] if annotation and 'wall_timestamp' in annotation else fallback()


def elapsed_onset(annotation, wall):
    if not annotation:
        return None
    return annotation.get('elapsed_seconds', float(wall[annotation['frame_index']] - wall[0]))


def missing_mean_replays(data):
    """Expose newly eligible failures; never turn offline scores into replay."""
    from unified_benchmark import METHODS
    fields = ('variant', 'task', 'method', 'view', 'calibration', 'episode_index')
    primary = data['evaluation_protocol']['detection']['primary_annotator']
    observed = {tuple({**t, **r}[k] for k in fields) for t in data['online']
                for r in t['latencies'] if r['annotator_id'] == primary}
    rows = []
    for values in data['episodes']['rows']:
        r = dict(zip(data['episodes']['columns'], values))
        if (r['method'] not in METHODS or r['method'] in data.get('supplementary_methods', [])
                or r['variant'] != data.get('summary_sources', {}).get(r['method'], 'main')
                or r['calibration'] != 'episode' or r['outcome'] != 'failure' or not r['alarm']):
            continue
        if tuple(r[k] for k in fields) not in observed:
            rows.append({k: r[k] for k in fields})
    return rows
