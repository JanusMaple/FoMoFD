"""Conditional online latency on shared detections; cached records only.

Groups are selected by intersection size, never by latency. All qualifying
episodes are retained. Thresholds, scores, annotations and primary reports are
inputs, not fitted or changed here.
"""
from collections import Counter
import csv
import itertools
import json
import math
from pathlib import Path
import statistics

METHODS = {'wm_nc': 'WM-NC', 'wm_pe': 'WM-PE', 'wm_pu': 'WM-PU',
           'logpzo': 'logpZO VAE', 'rnd': 'RND VAE', 'logpzo_native': 'logpZO policy',
           'rnd_native': 'RND policy', 'ace': 'ACE',
           'act_ensemble_pos': 'AEU', 'act_ensemble_rot': 'AEU-Rot'}
VIEWS = {'fix': 'fixed', 'wrist': 'wrist', 'both': 'joint'}


def detector_id(row):
    return '|'.join(row[k] for k in ('variant', 'method', 'view'))


def default_detectors(data):
    """Current main revisions plus joint models; exclude ablations and alarm fusions."""
    runs = {r['id']: r for r in data['runs']}
    result = {}
    for values in data['episodes']['rows']:
        row = dict(zip(data['episodes']['columns'], values))
        method, variant = row['method'], row['variant']
        if method not in METHODS or method in data.get('supplementary_methods', []) or not (variant == data.get('summary_sources', {}).get(method, 'main')
                or runs[variant]['family'] == 'view_fusion'):
            continue
        label = METHODS[method] + ('' if method in ('ace','act_ensemble_pos','act_ensemble_rot') and row['view']=='both' else ' ' + VIEWS[row['view']])
        if runs[variant]['family'] == 'view_fusion':
            label = 'Joint-view ' + METHODS[method]
        result[detector_id(row)] = {'variant': variant, 'method': method, 'view': row['view'], 'label': label}
    return dict(sorted(result.items()))


def online_index(data, annotator_id, combined_onset='earlier_view', task=None, origin=None, failure=None,
                 include_vlm=False, on_time_only=False):
    """Use an identical (task, episode, onset time) as the pairing unit.

    Fixed/wrist output keeps its own annotation; a joint output uses the earlier
    camera onset by default. Different onset timestamps cannot be silently paired.
    """
    from benchmark_onset_results import automatic_latencies
    index = {}
    trials = list(data['online'])
    if include_vlm:
        trials += [{**row, 'path': row['trace_path']} for run in data.get('vlm_runs', [])
                   for row in run['episodes'] if row['outcome'] == 'failure']
    for trial in trials:
        if trial['calibration'] not in ('episode', 'binary_anytime') or (task and trial['task'] != task):
            continue
        if (origin and trial['failure_origin'] != origin) or (failure and trial['failure_mode'] != failure):
            continue
        for row in (automatic_latencies(trial['latencies']) if combined_onset == 'earlier_view' else trial['latencies']):
            if row['annotator_id'] != annotator_id or (combined_onset != 'earlier_view' and row['view'] == 'both' and row['onset_view'] != combined_onset):
                continue
            key = detector_id(row)
            bucket = index.setdefault(key, {'records': {}, 'detected': {}})
            episode = (trial['task'], trial['episode_index'])
            if episode in bucket['records']:
                raise ValueError(f'Duplicate detector/episode latency: {key}/{episode}')
            record = {**row, 'task': trial['task'], 'episode_index': trial['episode_index'],
                      'failure_origin': trial['failure_origin'], 'failure_mode': trial['failure_mode'],
                      'trace': trial['path']}
            bucket['records'][episode] = record
            delay, onset = row.get('delay_s'), row.get('onset_time_s')
            if (row['status'] == 'detected' and delay is not None and onset is not None
                    and math.isfinite(delay) and math.isfinite(onset) and (delay >= 0 or row.get('detected_before_human_label'))
                    and not (on_time_only and row.get('late_alarm', False))):
                bucket['detected'][(*episode, onset)] = record
    return index


def pairwise_detectors(data):
    """All requested comparisons, including unmeasured action variants."""
    runs = {r['id']: r for r in data['runs']}
    result = {}
    for values in data['episodes']['rows']:
        row = dict(zip(data['episodes']['columns'], values))
        method, variant = row['method'], row['variant']
        family = runs[variant]['family']
        if family in ('K', 'L', 'action') and method == 'wm_nc':
            label = runs[variant]['title'] + ' · ' + VIEWS[row['view']]
        elif method in METHODS and variant == data.get('summary_sources', {}).get(method, 'main'):
            label = METHODS[method] + ('' if method in ('ace','act_ensemble_pos','act_ensemble_rot') and row['view']=='both' else ' · ' + VIEWS[row['view']])
            family = 'supplementary' if method in data.get('supplementary_methods', []) else 'baseline'
        else:
            continue
        result[detector_id(row)] = {k: row[k] for k in ('variant', 'method', 'view')}
        result[detector_id(row)].update(label=label, family=family)
    for run in data.get('vlm_runs', []):
        for row in run['episodes']:
            result[detector_id(row)] = {k: row[k] for k in ('variant', 'method', 'view')}
            result[detector_id(row)].update(label='Qwen v9 · '+VIEWS[row['view']], family='baseline')
    order = {method: i for i, method in enumerate([*METHODS, 'qwen_v9'])}
    return dict(sorted(result.items(), key=lambda p: (['baseline','K','L','action','supplementary'].index(p[1]['family']),
                                                       order[p[1]['method']], p[1]['label'])))


def pairwise_comparisons(index, detectors, reference, minimum=1, same_view=False):
    if reference not in detectors:
        raise ValueError('Unknown pairwise reference')
    rows = []
    for comparator, meta in detectors.items():
        if comparator == reference or (same_view and meta['view'] not in (detectors[reference]['view'], 'both')):
            continue
        result = cohort(index, [reference, comparator], minimum, match_onset=False)
        result.update(reference=reference, comparator=comparator, **meta)
        missing = not result['summaries'][1]['measured_n']
        result['status'] = ('not_measured_online' if missing else 'no_shared_detections' if not result['shared_n']
                            else 'below_minimum' if not result['eligible'] else 'available')
        rows.append(result)
    return rows


def write_pairwise_report(data, root):
    """Pairwise comparison archive; the main-model mean is recomputed per pair."""
    root = Path(root)/'pairwise_latency'
    root.mkdir(parents=True, exist_ok=True)
    annotator = next(iter(data['annotators']))
    detectors = pairwise_detectors(data)
    report = {'created_utc': data['created'], 'annotator_id': annotator, 'annotator': data['annotators'][annotator],
              'onset_labels_sha256': data['onset_analysis']['labels_sha256'], 'detectors': detectors,
              'reference': 'main|wm_nc|wrist',
              'definition': 'Main wrist WM-NC is the sole reference. Pair the same task and episode detected by both methods for the same annotator. Each delay uses that detector’s own view-specific onset; onset frames/timestamps need not match. Joint outputs use the earlier camera onset. Each pair retains its full intersection. Cross-view delay differences can reflect different onset times as well as alarm times. Original thresholds; not matched FAR. Historical ablation replay is excluded; unavailable comparisons remain explicit.',
              'comparisons': []}
    episode_rows, summary_rows = [], []
    for on_time in (True, False):
        index = online_index(data, annotator, include_vlm=True, on_time_only=on_time)
        for view in ('wrist',):
            reference = f'main|wm_nc|{view}'
            for result in pairwise_comparisons(index, detectors, reference):
                result['on_time_only'] = on_time
                report['comparisons'].append(result)
                left, right = result['summaries']
                difference = result['paired_differences'][0]
                summary_rows.append({'reference': reference, 'comparator': result['comparator'], 'label': result['label'],
                    'family': result['family'], 'on_time_only': on_time, 'status': result['status'],
                    'shared_n': result['shared_n'], 'reference_own_n': left['own_post_onset_n'],
                    'comparator_own_n': right['own_post_onset_n'], 'comparator_measured_n': right['measured_n'],
                    'different_onset_excluded_n': result['different_onset_excluded_n'],
                    'different_onset_included_n': result['different_onset_included_n'],
                    'reference_mean_s': left['mean_s'], 'reference_sd_s': left['sd_s'],
                    'comparator_mean_s': right['mean_s'], 'comparator_sd_s': right['sd_s'],
                    'mean_difference_s': difference['mean_s'], 'sd_difference_s': difference['sd_s'],
                    'reference_faster_n': difference['a_faster_n']})
                for ep in result['episodes']:
                    key = ep['task'], ep['episode_index']
                    a, b = index[reference]['records'][key], index[result['comparator']]['records'][key]
                    episode_rows.append({**ep, 'reference': reference, 'comparator': result['comparator'],
                        'on_time_only': on_time, 'annotator_id': annotator,
                        'reference_onset_time_s': a['onset_time_s'], 'comparator_onset_time_s': b['onset_time_s'],
                        'reference_onset_raw_frame': a.get('onset_raw_frame'), 'comparator_onset_raw_frame': b.get('onset_raw_frame'),
                        'reference_delay_s': a['delay_s'], 'comparator_delay_s': b['delay_s'],
                        'difference_s': a['delay_s']-b['delay_s'], 'failure_origin': a['failure_origin'],
                        'failure_mode': a['failure_mode'], 'reference_trace': a['trace'], 'comparator_trace': b['trace']})
    (root/'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for name, rows in [('summary', summary_rows), ('episodes', episode_rows)]:
        with (root/(name+'.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines = ['# Main wrist WM-NC versus each comparator', '', report['definition'], '',
             'Primary plots use on-time alarms for BOTH detectors. An optional view includes late alarms for BOTH. '
             'Pairwise means use different episode sets; compare the two methods within each row, not comparator means across rows. '
             'SD describes episode variability, not a confidence interval.', '',
             '[Interactive comparison](../index.html#pairwise) · [Statistics](summary.csv) · [Episode IDs and delays](episodes.csv)', '']
    for view in ('wrist',):
        lines += [f'## Main WM-NC {view} · on-time alarms', '',
                  '| Comparator | Shared n | Different onsets included | Wrist mean ± SD (s) | Comparator mean ± SD (s) | Status |',
                  '|---|---:|---:|---:|---:|---|']
        def value(mean, sd):
            return '—' if mean is None else f'{mean:.3f} ± ' + (f'{sd:.3f}' if sd is not None else '—')
        for row in summary_rows:
            if row['on_time_only'] and row['reference'] == f'main|wm_nc|{view}':
                lines.append(f"| {row['label']} | {row['shared_n']} | {row['different_onset_included_n']} | {value(row['reference_mean_s'], row['reference_sd_s'])} | {value(row['comparator_mean_s'], row['comparator_sd_s'])} | {row['status']} |")
        lines.append('')
    (root/'report.md').write_text('\n'.join(lines))
    return report


def stats(values):
    return {'n': len(values), 'mean_s': statistics.mean(values) if values else None,
            'sd_s': statistics.stdev(values) if len(values) > 1 else None,
            'median_s': statistics.median(values) if values else None}


def cohort(index, detectors, minimum=1, *, match_onset=True):
    if minimum < 1 or len(set(detectors)) != len(detectors):
        raise ValueError('Positive minimum and unique detectors required')
    detected = {d: {(key if match_onset else key[:2]): row
                    for key, row in index.get(d, {}).get('detected', {}).items()} for d in detectors}
    sets = [set(detected[d]) for d in detectors]
    common = sorted(set.intersection(*sets)) if sets else []
    same_episode = [set(k[:2] for k in s) for s in sets]
    overlap = set.intersection(*same_episode) if same_episode else set()
    summaries, pairs = [], []
    for d in detectors:
        bucket = index.get(d, {'records': {}, 'detected': {}})
        summaries.append({'detector': d, **stats([detected[d][k]['delay_s'] for k in common]),
                          'own_post_onset_n': len(bucket['detected']),
                          'measured_n': len(bucket['records']),
                          'excluded_from_own_detections': len(bucket['detected']) - len(common)})
    for a, b in itertools.combinations(detectors, 2):
        differences = [detected[a][k]['delay_s'] - detected[b][k]['delay_s'] for k in common]
        pairs.append({'detector_a': a, 'detector_b': b, **stats(differences),
                      'a_faster_n': sum(v < 0 for v in differences), 'tie_n': sum(v == 0 for v in differences)})
    records = [detected[detectors[0]][k] for k in common] if common else []
    different_onsets = sum(len({(detected[d][k].get('onset_time_s'), detected[d][k].get('onset_raw_frame'))
                               for d in detectors}) > 1 for k in common)
    return {'detectors': list(detectors), 'minimum': minimum, 'shared_n': len(common),
            'eligible': len(detectors) >= 2 and len(common) >= minimum,
            'different_onset_excluded_n': len(overlap) - len(common),
            'different_onset_included_n': different_onsets,
            'episodes': [{'task': k[0], 'episode_index': k[1], **({'onset_time_s': k[2]} if match_onset else {})} for k in common],
            'by_task': dict(Counter(k[0] for k in common)),
            'by_origin': dict(Counter(r['failure_origin'] for r in records)),
            'by_failure': dict(Counter(r['task'] + ': ' + str(r['failure_mode']) for r in records)),
            'summaries': summaries, 'paired_differences': pairs}


def largest_groups(index, detectors, minimum, max_candidates=20):
    """Exact maximum-cardinality search, deterministic coverage-based ordering.

    Every detector must share the SAME intersection. Pairwise overlap alone is
    insufficient. The bounded candidate count prevents an unbounded UI search.
    """
    if minimum < 1:
        raise ValueError('minimum must be positive')
    candidates = sorted(set(d for d in detectors if len(index.get(d, {}).get('detected', {})) >= minimum))
    if len(candidates) > max_candidates:
        return {'complete': False, 'reason': f'Select at most {max_candidates} eligible detectors for group search',
                'groups': [], 'candidate_count': len(candidates)}
    universe = {k: i for i, k in enumerate(sorted({k for d in candidates for k in index[d]['detected']}))}
    masks = [sum(1 << universe[k] for k in index[d]['detected']) for d in candidates]
    best_size, best = 1, []

    def visit(chosen, mask, start):
        nonlocal best_size, best
        if len(chosen) >= 2:
            if len(chosen) > best_size:
                best_size, best = len(chosen), []
            if len(chosen) == best_size:
                best.append({'detectors': list(chosen), 'shared_n': mask.bit_count()})
        if len(chosen) + len(candidates) - start < best_size:
            return
        for i in range(start, len(candidates)):
            intersection = masks[i] if mask is None else mask & masks[i]
            if intersection.bit_count() >= minimum:
                visit(chosen + [candidates[i]], intersection, i + 1)

    visit([], None, 0)
    best.sort(key=lambda g: (-g['shared_n'], g['detectors']))
    return {'complete': True, 'candidate_count': len(candidates), 'maximum_detectors': best_size if best else 0,
            'groups': best}


def write_report(data, root):
    """Publish all largest tied groups for the fixed, declared default pool."""
    root = Path(root) / 'shared_latency'
    root.mkdir(parents=True, exist_ok=True)
    annotator = next(iter(data['annotators']), None)
    index = online_index(data, annotator, on_time_only=True)
    detectors = default_detectors(data)
    searches, groups, latency_rows, episode_rows, paired_rows = [], [], [], [], []
    for minimum in (150, 100, 50):
        search = largest_groups(index, detectors, minimum)
        searches.append({'minimum': minimum, **search})
        for number, found in enumerate(search['groups'], 1):
            result = cohort(index, found['detectors'], minimum)
            result['id'] = f'min{minimum}_group{number}'
            groups.append(result)
            base = {'group': result['id'], 'minimum': minimum}
            latency_rows += [{**base, **r, 'label': detectors[r['detector']]['label']} for r in result['summaries']]
            paired_rows += [{**base, **r} for r in result['paired_differences']]
            for e in result['episodes']:
                k = e['task'], e['episode_index'], e['onset_time_s']
                for detector in result['detectors']:
                    record = index[detector]['detected'][k]
                    episode_rows.append({**base, **e, 'detector': detector, 'label': detectors[detector]['label'],
                                         'delay_s': record['delay_s'], 'failure_mode': record['failure_mode'],
                                         'failure_origin': record['failure_origin'], 'trace': record['trace']})
    report = {'schema_version': 1, 'created_utc': data['created'], 'annotator_id': annotator,
              'annotator': data['annotators'].get(annotator), 'calibration': 'episode', 'combined_onset': 'earlier_view',
              'selection': 'All tasks and failures; current main detectors plus joint models. Maximum detector count, then largest shared intersection, then detector ID. Never rank by latency.',
              'interpretation': 'Conditional common-detection latency; no change to thresholds and no matched-FAR latency claim. Minima are floors, not subsampling targets. All shared episodes retained. Cross-group means use different populations.',
              'onset_rule': 'Same task, episode, annotator and exact onset timestamp. Early alarms, misses, missing labels/measurements and unequal camera onsets are excluded.',
              'detectors': detectors, 'searches': searches, 'groups': groups,
              'source_archives': [{'path': r['path'], 'sha256': r['archive_sha256']} for r in data['online_runs']]}
    (root / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, rows in [('latency', latency_rows), ('episodes', episode_rows), ('paired_differences', paired_rows)]:
        with (root / (name + '.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['group'])
            writer.writeheader(); writer.writerows(rows)
    lines = ['# Shared-episode online latency', '', report['selection'], '', report['interpretation'], '',
             report['onset_rule'], '', f"Annotator: {report['annotator']}. Episode calibration; joint outputs use the earlier fixed/wrist onset.", '']
    for result in groups:
        lines += [f"## {result['id']}: {len(result['detectors'])} detectors, {result['shared_n']} shared episodes", '',
                  'Task counts: ' + ', '.join(f'{k}={v}' for k, v in result['by_task'].items()) + '.',
                  'Failure origins: ' + ', '.join(f'{k}={v}' for k, v in result['by_origin'].items()) + '.', '',
                  '| Detector | Mean ± sample SD (s) | Median (s) | Shared / own post-onset detections |',
                  '|---|---:|---:|---:|']
        for r in result['summaries']:
            lines.append(f"| {detectors[r['detector']]['label']} | {r['mean_s']:.3f} ± {r['sd_s']:.3f} | {r['median_s']:.3f} | {r['n']}/{r['own_post_onset_n']} |")
        lines.append('')
    lines += ['[All episode IDs and delays](episodes.csv) · [Paired differences: A − B](paired_differences.csv) · [Machine-readable results](analysis.json)', '']
    (root / 'report.md').write_text('\n'.join(lines))
    return report
