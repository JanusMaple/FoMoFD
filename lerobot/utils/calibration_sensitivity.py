#!/usr/bin/env python3
"""CPU-only 19/20 calibration-split sensitivity on frozen, cached PRL scores.

No inference, training, template fitting, or writes to primary results. The
original split is a separate reference from the requested random replicates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import yaml

import functional_conformal as functional
import unified_benchmark as benchmark
from results_inspector import resolve_repo_path

REPO = benchmark.REPO
TASKS = ('needle_pickup', 'ring_over_post', 'tissue_retraction', 'shunt_insertion')
SCHEMES = ('episode', 'functional_band')
METHODS = ('wm_nc', 'wm_pe', 'wm_pu', 'logpzo', 'rnd', 'logpzo_native', 'rnd_native', 'ace')
NOTE = (
    'These are overlapping resampling sensitivity statistics conditional on the '
    'same 39 successes, frozen detectors/templates and fixed failure set; they '
    'are not independent experiments, confidence intervals, or seed robustness. '
    'All splits are retained. Higher scores are more abnormal; alarms use strict >. '
    'With distinct success scores, the expected FAR over uniform 19/20 splits is '
    '1/20 = 5% by rank symmetry (ties can lower it); a mean near 5% does not validate '
    'calibration or exchangeability. Original FAR/FDR are reproduced from the '
    'saved paper frame-selection protocol, with no smoothing or template refitting. '
    'FDR counts any episode alarm, including pre-onset alarms. No online latency '
    'is evaluated. Paired FDR advantages must be read alongside paired FAR; these '
    'are not matched-FAR comparisons. Percentile ranges describe the retained '
    'random splits, not population uncertainty. Threshold units differ across '
    'detectors and protocols; functional thresholds are multipliers h.'
)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return benchmark.digest(Path(path))


def stats(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Sensitivity summaries require at least two finite observations')
    return {'mean': float(a.mean()), 'sd': float(a.std(ddof=1)),
            'p05': float(np.quantile(a, .05)), 'p95': float(np.quantile(a, .95)),
            'min': float(a.min()), 'max': float(a.max())}


def original_position(values, original):
    a = np.asarray(values)
    return {'fraction_below_original': float(np.mean(a < original)),
            'fraction_equal_original': float(np.mean(a == original)),
            'fraction_at_or_above_original': float(np.mean(a >= original)),
            'original_above_resampling_p95': bool(original > np.quantile(a, .95))}


def make_splits(success_ids, *, task, seed, n_splits=200, unique=False):
    """Task-specific PCG64 stream, invariant to task/method iteration order.

    The historical default retains repeated draws. unique=True rejects only
    duplicate subsets, without inspecting outcomes. Within each draw sampling
    is without replacement. Save IDs to reproduce the actual experiment.
    """
    ids = sorted(success_ids)
    if len(ids) != 39 or len(set(ids)) != 39 or any(type(e) is not int for e in ids):
        raise ValueError('Expected 39 distinct integer success episode IDs')
    if type(seed) is not int or seed < 0 or type(n_splits) is not int or n_splits < 2:
        raise ValueError('Require nonnegative integer seed and at least two splits')
    task_code = int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], 'little')
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, task_code])))
    if unique and n_splits > math.comb(39, 19):
        raise ValueError('More unique splits requested than available combinations')
    result, seen = [], set()
    while len(result) < n_splits:
        shuffled = rng.permutation(ids).tolist()
        key = tuple(sorted(shuffled[:19]))
        if unique and key in seen:
            continue  # Outcome-blind duplicate rejection only, never performance selection.
        seen.add(key)
        result.append({'split_id': len(result), 'calibration_ids': list(key),
                       'test_success_ids': sorted(shuffled[19:])})
    return result


def validate_partition(split, success_ids):
    cal, test = split['calibration_ids'], split['test_success_ids']
    if (len(cal) != 19 or len(test) != 20 or len(set(cal)) != 19 or len(set(test)) != 20
            or set(cal) & set(test) or set(cal) | set(test) != set(success_ids)):
        raise ValueError('Invalid 19/20 partition: overlap, missing IDs, or duplicates')


def episode_q(source, timeline, template=None, *, orientation='higher_is_more_abnormal'):
    if orientation != 'higher_is_more_abnormal':
        raise ValueError('Expected cached nonconformity scores, not log likelihood or reversed scores')
    if template is not None and template.get('score_orientation') != orientation:
        raise ValueError('Frozen template score orientation mismatch')
    result = {}
    for episode, curve in source.items():
        elapsed, score = functional.valid_trace(curve, timeline[episode], episode=episode)
        if template is not None:
            mu, g = functional.template_values(template, elapsed)
            score = (score - mu) / g  # signed upper residual, never absolute value
        result[episode] = float(np.max(score))
    return result


def evaluate_splits(q, success_ids, failure_ids, splits, *, failure_detection_scores=None):
    """Vectorized strict comparisons; calibration sees only its 19 successes."""
    if set(success_ids) & set(failure_ids) or len(failure_ids) != len(set(failure_ids)) or not failure_ids:
        raise ValueError('Failure evaluation IDs must be distinct and disjoint from successes')
    if set(q) != set(success_ids) | set(failure_ids) or not np.isfinite(list(q.values())).all():
        raise ValueError('Missing/extra/nonfinite episode maxima')
    for split in splits:
        validate_partition(split, success_ids)
    if len({s['split_id'] for s in splits}) != len(splits):
        raise ValueError('Duplicate split IDs')
    thresholds = np.array([max(q[e] for e in s['calibration_ids']) for s in splits])
    test_q = np.array([[q[e] for e in s['test_success_ids']] for s in splits])
    failure_q = np.array([q[e] for e in failure_ids])
    fa = (test_q > thresholds[:, None]).sum(axis=1)
    detection_q = failure_q
    if failure_detection_scores is not None:
        if set(failure_detection_scores) != set(failure_ids):
            raise ValueError('Post-onset maxima must cover every failure')
        detection_q = np.asarray([failure_detection_scores[e] for e in failure_ids], dtype=float)
        if np.isnan(detection_q).any() or np.isposinf(detection_q).any():
            raise ValueError('Invalid post-onset maxima')
    detected = (detection_q[None, :] > thresholds[:, None]).sum(axis=1)
    # Pairwise rank probabilities: subtraction of a common calibrated threshold
    # cannot change AUROC; changing the held-out success IDs can.
    auc = ((failure_q[None, :, None] > test_q[:, None, :]) +
           .5 * (failure_q[None, :, None] == test_q[:, None, :])).mean(axis=(1, 2))
    return [{'split_id': s['split_id'], 'threshold': float(h),
             'false_alarms': int(a), 'test_successes': 20, 'FAR': float(a / 20),
             'detected_failures': int(d), 'test_failures': len(failure_ids),
             'FDR': float(d / len(failure_ids)), 'episode_AUROC': float(u),
             'true_negatives': int(20-a), 'missed_failures': int(len(failure_ids)-d)}
            for s, h, a, d, u in zip(splits, thresholds, fa, detected, auc)]


def summarize(rows, original):
    result = {'n_splits': len(rows)}
    for metric in ('FAR', 'FDR', 'threshold', 'episode_AUROC'):
        if metric not in original:
            continue
        values = [r[metric] for r in rows]
        result.update({metric + '_' + k: v for k, v in stats(values).items()})
        result['original_' + metric] = original[metric]
        result.update({metric + '_' + k: v for k, v in original_position(values, original[metric]).items()})
    return result


def paired_rows(left, right):
    a, b = ({r['split_id']: r for r in rows} for rows in (left, right))
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError('Paired comparisons require identical unique split IDs')
    return [{'split_id': i, 'delta_FAR': a[i]['FAR'] - b[i]['FAR'],
             'delta_FDR': a[i]['FDR'] - b[i]['FDR']} for i in sorted(a)]


def paired_summary(rows):
    result = {'n_splits': len(rows)}
    for metric in ('delta_FAR', 'delta_FDR'):
        result.update({metric + '_' + k: v for k, v in stats([r[metric] for r in rows]).items()})
    result.update(fraction_FDR_win=sum(r['delta_FDR'] > 0 for r in rows) / len(rows),
                  fraction_FDR_tie=sum(r['delta_FDR'] == 0 for r in rows) / len(rows),
                  fraction_FDR_loss=sum(r['delta_FDR'] < 0 for r in rows) / len(rows),
                  fraction_FAR_lower=sum(r['delta_FAR'] < 0 for r in rows) / len(rows),
                  fraction_FDR_win_without_higher_FAR=sum(
                      r['delta_FDR'] > 0 and r['delta_FAR'] <= 0 for r in rows) / len(rows))
    return result


class Inputs:
    """Hash-check the frozen reports/templates and each original score cache."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.used = {}
        if json.loads((self.root / 'state.json').read_text())['status'] != 'complete':
            raise ValueError('Require a completed functional comparison')
        manifest_path = self.root / 'artifact_manifest.json'
        self.manifest = json.loads(manifest_path.read_text())['files']
        self.used[str(manifest_path)] = sha(manifest_path)

    def verified(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Frozen source path escapes its report directory')
        expected = self.manifest[str(path.relative_to(self.root))]
        self.external(path, expected)
        return path

    def external(self, path, expected):
        path = Path(path)
        if sha(path) != expected:
            raise ValueError(f'Frozen input hash mismatch: {path}')
        self.used[str(path)] = expected

    def read(self, relative):
        return json.loads(self.verified(relative).read_text())

    def assert_unchanged(self):
        for path, expected in self.used.items():
            if sha(path) != expected:
                raise ValueError(f'Input changed during sensitivity analysis: {path}')


def cache_key(identity):
    ckpt = identity['checkpoint']
    return (identity['task'], identity['method'], tuple(identity['views']),
            identity['sha256'][ckpt], identity['seed'])


def cache_coverage(selected, *, scoped=False):
    """Every available checkpoint/view/score-seed is represented, once.

    Historical duplicates with other batching/code versions do not masquerade
    as extra trained checkpoints. Prefer the exact primary-report caches.
    """
    keys = {cache_key(i) for i in selected.values()}
    rows = []
    paths = (map(Path, selected) if scoped else
             (REPO / 'fomofd/tasks').glob('*/prl_data/score_result/unified_benchmark/*/*/*.npz'))
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as data:
            identity = json.loads(str(data['provenance_json'].item()))['identity']
        if identity['method'] not in METHODS or identity['task'] not in TASKS:
            continue
        status = 'selected' if str(path.resolve()) in selected else 'alternate_cache_same_checkpoint_view_seed'
        if cache_key(identity) not in keys:
            raise ValueError(f'Unrepresented checkpoint/view/seed cache: {path}. Add a frozen report/template; no inference will be run.')
        rows.append({'path': str(path.relative_to(REPO)), 'task': identity['task'],
                     'method': identity['method'], 'views': ','.join(identity['views']),
                     'checkpoint': identity['checkpoint'], 'score_seed': identity['seed'], 'status': status})
    return rows


def load_sources(inputs, selection=None, *, include_offline_metrics=False):
    selection = selection or {}
    records = []
    for variant in sorted({Path(k).parts[0] for k in inputs.manifest if k.endswith('/episode/report.json')}):
        if selection.get('variants') and variant not in selection['variants']:
            continue
        reports = {scheme: inputs.read(f'{variant}/{scheme}/report.json') for scheme in SCHEMES}
        configs = {scheme: inputs.read(f'{variant}/{scheme}/benchmark_config.json') for scheme in SCHEMES}
        if reports['episode']['score_caches'] != reports['functional_band']['score_caches']:
            raise ValueError('Threshold protocols must reuse identical score caches')
        if configs['episode']['paper_protocol'] != configs['functional_band']['paper_protocol']:
            raise ValueError('Threshold protocols use different frame-selection conventions')
        protocol = configs['episode']['paper_protocol']
        benchmark.validate_protocol(protocol)
        if protocol['window'] != 1 or any(reports[s]['alpha'] != .05 for s in SCHEMES):
            raise ValueError('This analysis expects alpha=.05 and existing unsmoothed scores')
        label_path = inputs.verified(f'{variant}/episode/labels_snapshot.yaml')
        other_labels = inputs.verified(f'{variant}/functional_band/labels_snapshot.yaml')
        if sha(label_path) != sha(other_labels):
            raise ValueError('Threshold protocols have different label snapshots')
        labels, _ = benchmark.load_labels(label_path, list(TASKS))
        original = {}
        latency_original = {}
        for scheme in SCHEMES:
            with inputs.verified(f'{variant}/{scheme}/episode_results.csv').open() as stream:
                original[scheme] = list(csv.DictReader(stream))
            if include_offline_metrics:
                with inputs.verified(f'{variant}/{scheme}/latency_episodes.csv').open() as stream:
                    latency_original[scheme] = list(csv.DictReader(stream))
        manifest = inputs.read(f'{variant}/templates_manifest.json')
        templates, latency_templates = {}, {}
        for entry in manifest['templates']:
            if entry['protocol'] != 'paper' and not (include_offline_metrics and entry['protocol'] == 'latency'):
                continue
            if selection.get('methods') and entry['method'] not in selection['methods']:
                continue
            template_path = inputs.verified(f'{variant}/{entry["path"]}')
            if sha(template_path) != entry['sha256']:
                raise ValueError('Template manifest digest mismatch')
            key = tuple(entry[k] for k in ('task', 'method', 'view'))
            target = templates if entry['protocol'] == 'paper' else latency_templates
            if key in target:
                raise ValueError('Duplicate frozen template')
            template = json.loads(template_path.read_text())
            if template['source']['role'] != 'expert_validation':
                raise ValueError('Template must be frozen from expert validation')
            target[key] = (template, str(template_path), entry['sha256'])
        for cache in reports['episode']['score_caches']:
            if selection.get('methods') and cache['method'] not in selection['methods']:
                continue
            if cache['method'] not in METHODS:
                raise ValueError('No fixed continuous-score definition for this detector')
            path = resolve_repo_path(cache['path'])
            inputs.external(path, cache['sha256'])
            with np.load(path, allow_pickle=False) as data:
                identity = json.loads(str(data['provenance_json'].item()))['identity']
            if (identity['task'], identity['method'], list(identity['views'])) != (cache['task'], cache['method'], cache['views']):
                raise ValueError('Cache identity differs from report')
            records.append({'variant': variant, 'cache': cache, 'path': path, 'identity': identity,
                            'protocol': protocol, 'labels': labels, 'templates': templates,
                            'original': original, 'reports': reports,
                            'latency_templates': latency_templates, 'latency_original': latency_original,
                            'latency_protocol': configs['episode']['latency_protocol']})
    if not records:
        raise ValueError('No frozen scores match the requested selection')
    for key, field in [('variants', 'variant'), ('methods', None)]:
        actual = {r[field] if field else r['cache']['method'] for r in records}
        if selection.get(key) and set(selection[key]) != actual:
            raise ValueError(f'Missing requested {key}: {set(selection[key]) - actual}')
    return records


def split_manifest(records, seed, n_splits, *, unique=False):
    tasks = {}
    for record in records:
        task = record['cache']['task']
        groups = benchmark.episode_labels(record['labels']['tasks'][task])
        successes = sorted(e for e, g in groups.items() if g['outcome'] == 'success')
        failures = sorted(e for e, g in groups.items() if g['outcome'] == 'failure' and g['split'] == 'test')
        original = {'split_id': 'original',
                    'calibration_ids': sorted(e for e, g in groups.items() if g['split'] == 'calibration'),
                    'test_success_ids': sorted(e for e, g in groups.items() if g['split'] == 'test' and g['outcome'] == 'success')}
        validate_partition(original, successes)
        definition = {'success_ids': successes, 'failure_ids': failures, 'original': original}
        if task in tasks:
            if any(tasks[task][k] != definition[k] for k in definition):
                raise ValueError('Different checkpoint reports use different episode groups')
        else:
            tasks[task] = definition | {'random_splits': make_splits(successes, task=task, seed=seed, n_splits=n_splits, unique=unique)}
    if set(tasks) != set(TASKS):
        raise ValueError('Expected complete cached coverage of all four tasks')
    return {'schema_version': 1, 'seed': seed, 'n_splits': n_splits, 'n_calibration': 19,
            'n_test_successes': 20, 'alpha': .05, 'rank': 19,
            'rng': 'NumPy PCG64; SeedSequence([seed, uint32_le(SHA256(task)[:4])])',
            'draw_rule': ('uniform unique calibration subsets; only duplicate subsets rejected without inspecting scores'
                          if unique else 'uniform without replacement within split; retain every draw including repeats'),
            'unique_splits': unique,
            'tasks': tasks}


def verify_original(record, view, scheme, q, definition, result):
    selected = [r for r in record['original'][scheme] if
                (r['task'], r['method'], r['view']) == (record['cache']['task'], record['cache']['method'], view)]
    lookup = {int(r['episode_index']): r for r in selected}
    if len(lookup) != len(selected) or set(lookup) != set(q):
        raise ValueError('Original report does not contain exactly the scored episodes')
    cal = [r for r in record['reports'][scheme]['calibration'] if
           (r['task'], r['method'], r['view'], r['protocol']) ==
           (record['cache']['task'], record['cache']['method'], view, 'paper')]
    if len(cal) != 1:
        raise ValueError('Ambiguous original calibration row')
    saved_h = float(cal[0]['h' if scheme == 'functional_band' else 'threshold'])
    if not np.isclose(saved_h, result['threshold'], rtol=1e-12, atol=1e-12):
        raise ValueError('Original threshold does not reproduce')
    for episode, score in q.items():
        expected_alarm = str(lookup[episode]['alarm']).lower() == 'true'
        if bool(score > result['threshold']) != expected_alarm:
            raise ValueError(f'Original strict alarm mismatch: {record["variant"]}/{view}/{scheme}/{episode}')
        if scheme == 'functional_band' and not np.isclose(float(lookup[episode]['score_max']), score, rtol=1e-12, atol=1e-12):
            raise ValueError('Frozen functional maxima do not reproduce')
    fa = sum(str(lookup[e]['alarm']).lower() == 'true' for e in definition['original']['test_success_ids'])
    tp = sum(str(lookup[e]['alarm']).lower() == 'true' for e in definition['failure_ids'])
    if (fa, tp) != (result['false_alarms'], result['detected_failures']):
        raise ValueError('Original FAR/FDR do not reproduce')
    metric = next(r for r in record['reports'][scheme]['summary_metrics'] if
                  (r['task'], r['method'], r['view']) == (record['cache']['task'], record['cache']['method'], view))
    if not np.isclose(metric['episode_AUROC'], result['episode_AUROC'], rtol=1e-12, atol=1e-12):
        raise ValueError('Original episode AUROC does not reproduce')
    return {'passed': True, 'episodes_checked': len(q), 'original_threshold': saved_h,
            'false_alarms': fa, 'detected_failures': tp}


def analyze_records(records, splits, output, *, include_offline_metrics=False):
    extras = {'subgroup_results': [], 'offline_latency_split_results': [], 'offline_latency_original': [],
              'offline_latency_verification': []}
    timelines, cases, all_summary, all_results, original_rows, q_rows, validations = {}, [], [], [], [], [], []
    for index, record in enumerate(records):
        identity, cache = record['identity'], record['cache']
        task, method = cache['task'], cache['method']
        dataset = resolve_repo_path(identity['dataset'])
        if dataset not in timelines:
            timelines[dataset] = benchmark.load_timeline(dataset)  # timestamp columns only
        timeline = timelines[dataset]
        job = SimpleNamespace(task=task, method=method, views=tuple(cache['views']),
                              mode=benchmark.METHODS[method][3], merged=identity['merged_config'])
        with np.load(record['path'], allow_pickle=False) as data:
            arrays = {k: data[k] for k in ('episode_index', 'frame_index', 'raw_frame_index', 'wall_timestamp', 'scores')}
        benchmark.validate_scores(arrays, job, timeline)
        definition = splits['tasks'][task]
        for view_index, view in enumerate(cache['views']):
            template, template_path, template_sha = record['templates'][(task, method, view)]
            provenance = template['source']
            checkpoint_sha = identity['sha256'][identity['checkpoint']]
            if (provenance['prl_score_cache_sha256'] != cache['sha256']
                    or provenance['checkpoint_sha256'] != checkpoint_sha
                    or provenance['score_seed'] != identity['seed']):
                raise ValueError('Frozen template belongs to a different checkpoint/cache/seed')
            meta = {'variant': record['variant'], 'task': task, 'method': method, 'view': view,
                    'checkpoint': identity['checkpoint'], 'checkpoint_sha256': checkpoint_sha,
                    'training_seed': provenance['training_seed'], 'score_seed': identity['seed'],
                    'cache_path': str(record['path']), 'cache_sha256': cache['sha256'],
                    'template_path': template_path, 'template_sha256': template_sha,
                    'paper_protocol': record['protocol'], 'scoring_settings': cache['scoring_settings']}
            case_id = hashlib.sha256(benchmark.canonical(meta).encode()).hexdigest()[:16]
            common = {k: meta[k] for k in ('variant', 'task', 'method', 'view', 'training_seed', 'score_seed')}
            common['case_id'] = case_id
            source = benchmark.curves(arrays, job, view_index, record['protocol'])
            case = {'metadata': meta | {'case_id': case_id}, 'summaries': [], 'original': [], 'results': [], 'validation': []}
            for scheme in SCHEMES:
                q = episode_q(source, timeline, template if scheme == 'functional_band' else None)
                random_rows = evaluate_splits(q, definition['success_ids'], definition['failure_ids'], definition['random_splits'])
                original = evaluate_splits(q, definition['success_ids'], definition['failure_ids'], [definition['original']])[0]
                validation = common | {'calibration': scheme} | verify_original(record, view, scheme, q, definition, original)
                validation.update(original_false_alarms=original['false_alarms'], original_detected_failures=original['detected_failures'])
                validations.append(validation); case['validation'].append(validation)
                context = common | {'calibration': scheme}
                summaries = context | summarize(random_rows, original)
                originals = context | original
                results = [context | r for r in random_rows]
                case['summaries'].append(summaries); case['original'].append(originals); case['results'].extend(results)
                all_summary.append(summaries); original_rows.append(originals); all_results.extend(results)
                if include_offline_metrics:
                    from calibration_sensitivity_metrics import subgroup_results, latency_results
                    extras['subgroup_results'].extend(subgroup_results(
                        q, definition, record['labels']['tasks'][task], context, evaluate_splits))
                    causal = benchmark.curves(arrays, job, view_index, record['latency_protocol'])
                    frozen = record['latency_templates'][(task, method, view)][0] if scheme == 'functional_band' else None
                    rows, original_latency, verification = latency_results(
                        causal, timeline, frozen, definition, record, view, scheme, context)
                    extras['offline_latency_split_results'].extend(rows)
                    extras['offline_latency_original'].extend(original_latency)
                    extras['offline_latency_verification'].append(verification)
                for episode, maximum in q.items():
                    q_rows.append(context | {'episode_index': episode, 'Q': maximum,
                                           'outcome': 'success' if episode in definition['success_ids'] else 'failure'})
            folder = output / 'cases' / task / view / method / (record['variant'] + '__' + case_id)
            save(folder / 'results.json', case)
            benchmark.write_csv(folder / 'results.csv', case['results'], [])
            benchmark.write_csv(folder / 'summary.csv', case['summaries'], [])
            cases.append(case)
        print(f'{index+1}/{len(records)} {record["variant"]}/{task}/{method}/{",".join(cache["views"])}: original verified, all splits retained', flush=True)
    if include_offline_metrics:
        from calibration_sensitivity_metrics import write_extras
        write_extras(output, extras, all_results, original_rows)
    return cases, all_summary, all_results, original_rows, q_rows, validations


def comparisons(cases):
    """Within-variant method pairs, plus main WM-NC versus its K/L variants.

    ACE has joint input; cross-view comparisons are explicit, never relabeled
    as single-camera ACE. Positive deltas always mean left minus right.
    """
    pairs = []
    for a, b in combinations(cases, 2):
        x, y = a['metadata'], b['metadata']
        if x['task'] != y['task']:
            continue
        same_view = x['view'] == y['view']
        ace_pair = x['method'] == 'ace' or y['method'] == 'ace'
        method_pair = x['variant'] == y['variant'] and x['method'] != y['method'] and (same_view or ace_pair)
        ablation_pair = (x['method'] == y['method'] == 'wm_nc' and same_view
                         and (x['variant'] == 'main') != (y['variant'] == 'main'))
        if not (method_pair or ablation_pair):
            continue
        # WM-NC is always the left reference; main precedes its ablations.
        if (y['method'] == 'wm_nc' and x['method'] != 'wm_nc') or (ablation_pair and y['variant'] == 'main'):
            a, b, x, y = b, a, y, x
        for scheme in SCHEMES:
            left = [r for r in a['results'] if r['calibration'] == scheme]
            right = [r for r in b['results'] if r['calibration'] == scheme]
            rows = paired_rows(left, right)
            left_original = next(r for r in a['original'] if r['calibration'] == scheme)
            right_original = next(r for r in b['original'] if r['calibration'] == scheme)
            pair_id = f'{x["case_id"]}__{y["case_id"]}__{scheme}'
            meta = {'pair_id': pair_id, 'task': x['task'], 'calibration': scheme,
                    'left_case_id': x['case_id'], 'right_case_id': y['case_id'],
                    'left_variant': x['variant'], 'right_variant': y['variant'],
                    'left_method': x['method'], 'right_method': y['method'],
                    'left_view': x['view'], 'right_view': y['view'],
                    'comparison': 'main_vs_ablation' if ablation_pair else 'between_methods',
                    'same_input_views': same_view,
                    'original_delta_FAR': left_original['FAR'] - right_original['FAR'],
                    'original_delta_FDR': left_original['FDR'] - right_original['FDR']}
            pairs.append({'metadata': meta, 'summary': meta | paired_summary(rows), 'results': rows})
    return pairs


def shunt_analysis(summaries, results, historical_far):
    rows = []
    for s in summaries:
        if (s['task'], s['view']) != ('shunt_insertion', 'fix'):
            continue
        values = [r['FAR'] for r in results if (r['case_id'], r['calibration']) == (s['case_id'], s['calibration'])]
        row = {k: s[k] for k in ('case_id', 'variant', 'method', 'calibration', 'original_FAR', 'FAR_mean', 'FAR_sd', 'FAR_p05', 'FAR_p95')}
        row.update(original_position(values, s['original_FAR']))
        if s['variant'] == 'main' and s['method'] == 'wm_nc':
            row.update(historical_paper_FAR=historical_far,
                       historical_reference_provenance='User-supplied paper table: 6/20; not reproduced by current cached original split',
                       historical_vs_current_protocol='historical reference only; functional band was not the paper protocol',
                       fraction_at_or_above_historical=float(np.mean(np.asarray(values) >= historical_far)),
                       historical_above_current_resampling_p95=bool(historical_far > s['FAR_p95']))
        rows.append(row)
    return rows


def label(meta):
    name = benchmark.METHODS[meta['method']][0]
    if meta['variant'] != 'main':
        name += ' ' + meta['variant'].removeprefix('wm_nc_').upper().replace('_', ' ')
    if meta['view'] == 'both':
        name += ' (joint cameras)'
    return name


def plots(output, cases, pairs):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    colors = {'episode': '#2477a8', 'functional_band': '#d57826'}
    figures = []
    folder = output / 'figures'; folder.mkdir(exist_ok=True)
    def export(fig, stem, task, view, kind):
        fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=9)
        fig.tight_layout(rect=(0, .065, 1, 1))
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(folder / (stem + '.' + extension), dpi=140, bbox_inches='tight')
        plt.close(fig)
        figures.append({'stem': 'figures/' + stem, 'task': task, 'view': view, 'kind': kind})
    def boxes(ax, values, positions, color, original):
        bp = ax.boxplot(values, positions=positions, vert=False, widths=.28, whis=(5, 95),
                        showfliers=False, patch_artist=True, manage_ticks=False)
        for patch in bp['boxes']:
            patch.set(facecolor=color, alpha=.35, edgecolor=color)
        for line in bp['medians']:
            line.set_color(color)
        ax.scatter([np.mean(v) for v in values], positions, color=color, s=16, zorder=4)
        ax.scatter(original, positions, marker='D', facecolors='none', edgecolors='black', s=26, zorder=5)
    legend = [Patch(facecolor=c, alpha=.5, label=s.replace('_', ' ')) for s, c in colors.items()]
    legend += [Line2D([], [], marker='D', color='black', markerfacecolor='none', linestyle='None', label='original split'),
               Line2D([], [], marker='o', color='gray', linestyle='None', label='resampling mean')]
    for task in TASKS:
        for view in ('fix', 'wrist'):
            selected = [c for c in cases if c['metadata']['task'] == task and c['metadata']['view'] in (view, 'both')]
            selected.sort(key=lambda c: (c['metadata']['variant'] != 'main', c['metadata']['variant'], METHODS.index(c['metadata']['method'])))
            y = np.arange(len(selected))
            if not selected:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(15, max(3.5, len(selected) * .49)), sharey=True)
            for offset, scheme in zip((-.18, .18), SCHEMES):
                for ax, metric in zip(axes, ('FAR', 'FDR', 'episode_AUROC')):
                    values = [[100*r[metric] for r in c['results'] if r['calibration'] == scheme] for c in selected]
                    originals = [100*next(r[metric] for r in c['original'] if r['calibration'] == scheme) for c in selected]
                    boxes(ax, values, y + offset, colors[scheme], originals)
                    ax.set_xlabel(metric + ' (%)'); ax.grid(axis='x', alpha=.2)
            axes[0].set_yticks(y, [label(c['metadata']) for c in selected]); axes[0].invert_yaxis()
            axes[0].set_xlim(-2, max(40, axes[0].get_xlim()[1])); axes[1].set_xlim(-2, 102)
            count = len(selected[0]['results']) // len(SCHEMES)
            fig.suptitle(f'{task} / {view}: {count} paired 19/20 splits\nBoxes: 25–75%; whiskers: 5–95% resampling percentiles (not confidence intervals)', fontsize=11)
            export(fig, task + '_' + view + '_sensitivity', task, view, 'sensitivity')
            selected_pairs = [p for p in pairs if p['metadata']['task'] == task and p['metadata']['left_view'] == view
                              and p['metadata']['left_variant'] == p['metadata']['right_variant'] == 'main'
                              and p['metadata']['left_method'] == 'wm_nc']
            names = sorted({(p['metadata']['right_method'], p['metadata']['right_view']) for p in selected_pairs}, key=lambda k: METHODS.index(k[0]))
            if not names:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
            y = np.arange(len(names))
            for offset, scheme in zip((-.18, .18), SCHEMES):
                lookup = {(p['metadata']['right_method'], p['metadata']['right_view']): p for p in selected_pairs if p['metadata']['calibration'] == scheme}
                for ax, metric in zip(axes, ('delta_FDR', 'delta_FAR')):
                    values = [[100*r[metric] for r in lookup[k]['results']] for k in names]
                    originals = [100*lookup[k]['metadata']['original_' + metric] for k in names]
                    boxes(ax, values, y + offset, colors[scheme], originals)
                    ax.axvline(0, color='gray', linewidth=1); ax.grid(axis='x', alpha=.2)
                    ax.set_xlabel(('FDR gain (higher better)' if metric == 'delta_FDR' else 'FAR change (lower better)') + ', percentage points')
            axes[0].set_yticks(y, [benchmark.METHODS[m][0] + (' (joint)' if v == 'both' else '') for m, v in names]); axes[0].invert_yaxis()
            fig.suptitle(f'{task} / {view}: WM-NC minus baseline on identical splits\nWhiskers: 5–95%; comparisons use calibrated thresholds, not matched FAR', fontsize=11)
            export(fig, task + '_' + view + '_paired', task, view, 'paired')
    return figures


def reports(output, summaries, pairs, shunt, figures, metadata):
    def percent(value): return f'{100*value:.1f}'
    def interval(row, metric):
        return f'{percent(row[metric+"_mean"])} ± {percent(row[metric+"_sd"])} [{percent(row[metric+"_p05"])}, {percent(row[metric+"_p95"])}]'
    headers = ['Task', 'View', 'Protocol', 'Original FAR/FDR %', 'FAR mean ± SD [5–95%] %', 'FDR mean ± SD [5–95%] %', 'Episode AUROC mean ± SD [5–95%] %']
    cells = [[r['task'], r['view'], r['calibration'], percent(r['original_FAR'])+'/'+percent(r['original_FDR']),
              interval(r, 'FAR'), interval(r, 'FDR'), interval(r, 'episode_AUROC')] for r in summaries if r['variant'] == 'main' and r['method'] == 'wm_nc']
    note = metadata.get('note', NOTE)
    md = '# Calibration-split sensitivity\n\n' + note + '\n\n'
    md += f'{metadata["case_count"]} checkpoint/view cases × two protocols × {metadata["n_splits"]} shared splits per task. No scoring or training.\n\n'
    md += '## Main WM-NC\n\n' + benchmark.markdown_table(headers, cells) + '\n\n'
    shunt_notes = []
    for row in shunt:
        if row['variant'] == 'main' and row['method'] == 'wm_nc':
            shunt_notes.append(f'{row["calibration"]}: saved original FAR {percent(row["original_FAR"])}%; '
                               f'resampling mean {percent(row["FAR_mean"])}%, 5–95% range [{percent(row["FAR_p05"])}, {percent(row["FAR_p95"])}]%; '
                               f'{percent(row["fraction_at_or_above_original"])}% of splits have FAR ≥ this original; '
                               f'{percent(row["fraction_at_or_above_historical"])}% reach the historical 30% reference.')
    md += '## Shunt fixed-view FAR\n\n' + '\n\n'.join(shunt_notes) + '\n\n'
    md += ('The user-supplied paper table reported 6/20 (30%). The current cached original split does not reproduce that value; '
           'its tail position here is a descriptive comparison to current scores, not an explanation or validation of the historical result.\n\n')
    main_pairs = [p['summary'] for p in pairs if p['metadata']['left_variant'] == p['metadata']['right_variant'] == 'main'
                  and p['metadata']['left_method'] == 'wm_nc']
    paired_headers = ['Protocol', 'Baseline', 'Mean paired ΔFDR pp (task/view range)', 'WM-NC FDR win fraction range', 'Win without higher FAR range']
    paired_cells = []
    for scheme in SCHEMES:
        for method in METHODS[1:]:
            rows = [r for r in main_pairs if r['calibration'] == scheme and r['right_method'] == method]
            if not rows: continue
            bounds = lambda key: f'{percent(min(r[key] for r in rows))}–{percent(max(r[key] for r in rows))}'
            paired_cells.append([scheme, benchmark.METHODS[method][0], bounds('delta_FDR_mean'),
                                 bounds('fraction_FDR_win')+'%', bounds('fraction_FDR_win_without_higher_FAR')+'%'])
    md += '## WM-NC paired comparisons\n\nRanges below span four tasks × two WM-NC views; inspect per-task CSV/plots before claiming consistent superiority. ACE uses joint cameras.\n\n'
    md += benchmark.markdown_table(paired_headers, paired_cells)
    md += f'\n\nAll {metadata["n_splits"]} split results, thresholds, originals, paired deltas and provenance are retained in CSV/JSON. No favorable split selection.\n'
    if metadata['config'].get('include_offline_metrics'):
        md += '\nEpisode AUROC varies with the held-out success IDs, not with the threshold itself. '
        md += 'Additional [offline metrics](offline_metrics.md) include failure-mode/origin breakdowns, pooled FAR/FDR and macro AUROC, and dense offline onset delays with explicit denominators. These delays exclude runtime and are not online latency.\n'
    (output / 'summary.md').write_text(md)
    def table(head, rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    page = '<!doctype html><html><meta charset="utf-8"><title>Calibration sensitivity</title><style>body{font:14px system-ui;color:#183344;margin:28px}p{max-width:1200px;line-height:1.5}.scroll{overflow:auto}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#edf3f7}img{max-width:1200px;width:100%}section{margin-top:30px}a{color:#176ca0}select{padding:8px;margin-right:12px}</style>'
    page += '<h1>Calibration-split sensitivity</h1><p>'+html.escape(note)+'</p><p>'
    page += ' · '.join(f'<a href="{name}">{name}</a>' for name in ('summary.md', 'summary.csv', 'original_results.csv', 'split_results.csv', 'splits.json', 'paired_summary.csv', 'shunt_fixed_analysis.json', 'manifest.json'))+'</p>'
    page += '<h2>Original split versus resampling: main WM-NC</h2>'+table(headers, cells)
    if metadata['config'].get('include_offline_metrics'):
        page += '<p><a href="offline_metrics.html">Offline delays, subgroup results, and overall metrics</a>. AUROC changes with held-out success IDs, not with threshold shifts.</p>'
    page += '<h2>Shunt fixed-view FAR</h2>'+''.join('<p>'+html.escape(s)+'</p>' for s in shunt_notes)
    page += '<p>The historical paper FAR of 30% is a separate reference; these cached original results do not reproduce it.</p>'
    page += '<h2>Paired WM-NC differences</h2>'+table(paired_headers, paired_cells)
    page += '<h2>Plots</h2><label>Task <select id="task"><option value="all">All tasks</option>'+''.join(f'<option>{t}</option>' for t in TASKS)+'</select></label><label>View <select id="view"><option value="all">Both views</option><option>fix</option><option>wrist</option></select></label>'
    for figure in figures:
        stem = figure['stem']
        page += f'<section data-task="{figure["task"]}" data-view="{figure["view"]}"><h3>{figure["task"]} / {figure["view"]} / {figure["kind"]}</h3><a href="{stem}.pdf">PDF</a> · <a href="{stem}.svg">SVG</a><br><img loading="lazy" src="{stem}.png" alt="{figure["kind"]}"></section>'
    page += '<script>function filter(){document.querySelectorAll("section[data-task]").forEach(s=>s.hidden=!( (task.value==="all"||s.dataset.task===task.value)&&(view.value==="all"||s.dataset.view===view.value)));}const task=document.getElementById("task"),view=document.getElementById("view");task.onchange=view.onchange=filter;</script></html>'
    (output / 'report.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'lerobot/configs/benchmark/calibration_sensitivity.yaml')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing outputs are never overwritten')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if (config['schema_version'] != 1 or config['alpha'] != .05
            or type(config['n_splits']) is not int or config['n_splits'] < 2):
        raise ValueError('Require at least two splits and alpha=.05')
    output = args.output.resolve()
    inputs = Inputs(resolve_repo_path(config['source']))
    if output.exists() or output.is_relative_to(inputs.root) or inputs.root.is_relative_to(output):
        raise ValueError('Use a new output directory separate from primary inputs')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    save(output / 'state.json', {'status': 'running', 'started_utc': benchmark.utc_now(), 'inference': False})
    try:
        records = load_sources(inputs, config.get('selection'), include_offline_metrics=config.get('include_offline_metrics', False))
        coverage = cache_coverage({str(r['path'].resolve()): r['identity'] for r in records}, scoped=bool(config.get('selection')))
        splits = split_manifest(records, config['seed'], config['n_splits'], unique=config.get('unique_splits', False))
        save(output / 'splits.json', splits)
        save(output / 'config.json', config)
        benchmark.write_csv(output / 'cache_coverage.csv', coverage, [])
        cases, summaries, results, originals, q_rows, validations = analyze_records(
            records, splits, output, include_offline_metrics=config.get('include_offline_metrics', False))
        print('All original thresholds and episode alarms reproduced. Computing paired differences and plots.', flush=True)
        pairs = comparisons(cases)
        shunt = shunt_analysis(summaries, results, config['historical_shunt_fixed_wm_nc_far'])
        for name, rows in [('summary', summaries), ('split_results', results), ('original_results', originals),
                           ('episode_maxima', q_rows), ('paired_summary', [p['summary'] for p in pairs]),
                           ('paired_split_results', [dict(pair_id=p['metadata']['pair_id'], **r) for p in pairs for r in p['results']]),
                           ('original_verification', validations), ('shunt_fixed_analysis', shunt)]:
            benchmark.write_csv(output / (name + '.csv'), rows, [])
        save(output / 'summary.json', summaries)
        save(output / 'paired_results.json', pairs)
        save(output / 'shunt_fixed_analysis.json', shunt)
        figures = plots(output, cases, pairs)
        metadata = {'created_utc': benchmark.utc_now(), 'note': NOTE, 'config': config,
                    'n_splits': config['n_splits'], 'case_count': len(cases), 'protocol_case_count': len(summaries),
                    'paired_comparisons': len(pairs), 'original_verification': 'all thresholds and per-episode alarms match',
                    'score_orientation': 'higher_is_more_abnormal; use cached scores, never negate logpZO scores',
                    'n_calibration': 19, 'n_test_successes': 20, 'split_manifest_sha256': sha(output / 'splits.json'),
                    'scoring': False, 'training': False, 'template_refitting': False, 'source_inputs': inputs.used,
                    'code_sha256': {str(p.relative_to(REPO)): sha(p) for p in (
                        Path(__file__), Path(benchmark.__file__), Path(functional.__file__),
                        Path(__file__).with_name('calibration_sensitivity_metrics.py'))},
                    'numpy': np.__version__, 'python': sys.version,
                    'excluded': ['calibration-dependent AND/OR fusions: no frozen pre-calibration continuous score/template',
                                 'alternate caches of the same checkpoint/view/seed (see cache_coverage.csv)',
                                 'models without cached scores; no new scoring permitted'],
                    'cases': [c['metadata'] for c in cases], 'figures': figures}
        snapshots = output / 'frozen_sources'
        for relative in metadata['code_sha256']:
            path = snapshots / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((REPO / relative).read_bytes())
        reports(output, summaries, pairs, shunt, figures, metadata)
        inputs.assert_unchanged()
        if 'torch' in sys.modules:
            raise RuntimeError('CPU-only analysis unexpectedly imported torch')
        metadata.update(elapsed_seconds=time.perf_counter() - started, torch_imported=False)
        save(output / 'manifest.json', metadata)
        save(output / 'state.json', {'status': 'complete', 'finished_utc': benchmark.utc_now(),
                                     'elapsed_seconds': metadata['elapsed_seconds'], 'cases': len(cases),
                                     'original_checks_passed': len(validations), 'models_loaded': 0})
        print(f'Complete in {metadata["elapsed_seconds"]:.1f}s: {output / "report.html"}', flush=True)
    except BaseException as error:
        save(output / 'state.json', {'status': 'failed', 'error': str(error), 'finished_utc': benchmark.utc_now()})
        raise


if __name__ == '__main__':
    main()
