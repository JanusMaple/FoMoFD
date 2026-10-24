"""Cached-score offline metrics for resampled calibration sets; NumPy only."""
from collections import defaultdict
import html
import json

import numpy as np

import functional_conformal as functional
import unified_benchmark as benchmark


def subgroup_results(q, definition, task_labels, context, evaluate):
    labels = benchmark.episode_labels(task_labels)
    splits = [definition['original'], *definition['random_splits']]
    rows = []
    for kind in ('failure_mode', 'failure_origin'):
        for value in sorted({labels[e][kind] for e in definition['failure_ids']}):
            ids = [e for e in definition['failure_ids'] if labels[e][kind] == value]
            selected = {e: q[e] for e in definition['success_ids'] + ids}
            rows.extend(context | {'subgroup_kind': kind, 'subgroup': value} | r
                        for r in evaluate(selected, definition['success_ids'], ids, splits))
    return rows


def first_alarms(source, threshold):
    result = {}
    for episode, curve in source.items():
        indices = np.flatnonzero(curve['score'] > threshold)
        i = int(indices[0]) if len(indices) else None
        result[episode] = None if i is None else {'raw': int(curve['raw'][i]), 'wall': float(curve['wall'][i])}
    return result


def latency_rows(source, annotations, threshold, *, frozen=False, first_post_onset=False):
    """Match unified_benchmark's first-alarm/onset rules, including early alarms."""
    alarms = first_alarms(source, threshold)
    rows = []
    for original in annotations:
        episode = int(original['episode_index'])
        onset = int(original['onset_raw_frame']) if original['onset_raw_frame'] != '' else None
        wall = float(original['onset_wall_timestamp']) if original['onset_wall_timestamp'] != '' else None
        alarm = alarms[episode]
        row = {k: original[k] for k in ['task', 'method', 'view', 'calibration', 'annotator_id',
                                       'annotator', 'onset_view', 'copied_from_annotator']}
        row.update(episode_index=episode, status='unannotated', detection_delay_s=None,
                   signed_first_alarm_delay_s=None,
                   first_alarm_raw_frame=alarm['raw'] if alarm else None,
                   first_alarm_wall_timestamp=alarm['wall'] if alarm else None)
        if onset is not None:
            if wall is None:
                row['status'] = 'timestamp_unavailable'
            elif alarm and alarm['raw'] < onset:
                row.update(status='pre_onset_alarm', signed_first_alarm_delay_s=alarm['wall']-wall)
            elif onset > int(source[episode]['raw'][-1]):
                row['status'] = 'onset_after_scored_end'
            elif alarm:
                row.update(status='detected', detection_delay_s=alarm['wall']-wall,
                           signed_first_alarm_delay_s=alarm['wall']-wall)
            else:
                row['status'] = 'missed'
        if frozen:
            from benchmark_frozen_protocol import alarm_run, allows_anticipation
            curve = source[episode]
            result = alarm_run(dict(curve, score=curve['score']-threshold), onset, wall,
                allow_anticipation=allows_anticipation(original['task'], original['failure_origin']),
                first_post_onset=first_post_onset)
            row.update(result)
        rows.append(row)
    return rows


def latency_results(source, timeline, template, definition, record, view, scheme, context):
    transformed = {}
    for episode, curve in source.items():
        elapsed, scores = functional.valid_trace(curve, timeline[episode], episode=episode)
        if template is not None:
            mu, g = functional.template_values(template, elapsed)
            scores = (scores-mu)/g
        transformed[episode] = dict(curve, score=scores)
    maxima = {e: float(c['score'].max()) for e, c in transformed.items()}
    annotations = [r for r in record['latency_original'][scheme] if
                   (r['task'], r['method'], r['view']) == (context['task'], context['method'], view)]
    if not annotations:
        raise ValueError('Missing frozen onset/latency records')
    key = lambda r: (int(r['episode_index']), r['annotator_id'], r['onset_view'])
    if len({key(r) for r in annotations}) != len(annotations):
        raise ValueError('Duplicate frozen latency record')
    if {int(r['episode_index']) for r in annotations} != set(definition['failure_ids']):
        raise ValueError('Frozen onset records do not cover the failure set')
    saved_cal = [r for r in record['reports'][scheme]['calibration'] if
                 (r['task'], r['method'], r['view'], r['protocol']) ==
                 (context['task'], context['method'], view, 'latency')]
    original_threshold = max(maxima[e] for e in definition['original']['calibration_ids'])
    if len(saved_cal) != 1 or not np.isclose(original_threshold,
            float(saved_cal[0]['h' if template is not None else 'threshold']), rtol=1e-12, atol=1e-12):
        raise ValueError('Original dense-latency threshold does not reproduce')
    original_rows = latency_rows(transformed, annotations, original_threshold)
    lookup = {key(r): r for r in annotations}
    for row in original_rows:
        saved = lookup[key(row)]
        if saved['status'] != row['status']:
            raise ValueError('Original offline latency status differs')
        for field in ['first_alarm_raw_frame', 'first_alarm_wall_timestamp',
                      'signed_first_alarm_delay_s', 'detection_delay_s']:
            expected = float(saved[field]) if saved[field] != '' else None
            if (expected is None) != (row[field] is None) or (
                    expected is not None and not np.isclose(expected, row[field], rtol=0, atol=1e-8)):
                raise ValueError(f'Original offline latency mismatch: {key(row)}/{field}')
    summaries, originals, memo = [], [], {}
    for split in [definition['original'], *definition['random_splits']]:
        threshold = max(maxima[e] for e in split['calibration_ids'])
        if threshold not in memo:
            memo[threshold] = benchmark.latency_summary(latency_rows(transformed, annotations, threshold))
        rows = [context | r | {'split_id': split['split_id'], 'latency_threshold': threshold,
                              'timing_protocol': 'dense_offline_all_endpoints_no_runtime'} for r in memo[threshold]]
        (originals if split['split_id'] == 'original' else summaries).extend(rows)
    return summaries, originals, context | {'passed': True, 'failure_annotation_records': len(annotations),
                                           'original_latency_threshold': original_threshold}


def distribution(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite offline metric')
    if not len(values):
        return {'defined_splits': 0, **dict.fromkeys(['mean', 'sd', 'p05', 'p95', 'min', 'max'])}
    return {'defined_splits': len(values), 'mean': float(values.mean()),
            'sd': float(values.std(ddof=1)) if len(values) > 1 else None,
            'p05': float(np.quantile(values, .05)), 'p95': float(np.quantile(values, .95)),
            'min': float(values.min()), 'max': float(values.max())}


def grouped_summary(rows, originals, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    reference = {tuple(row[k] for k in keys): row for row in originals}
    result = []
    for key, selected in groups.items():
        row = dict(zip(keys, key)) | {'n_splits': len(selected)}
        for metric in metrics:
            row.update({metric+'_'+k: v for k, v in distribution([r[metric] for r in selected]).items()})
            row['original_'+metric] = reference[key][metric]
        result.append(row)
    return result


def overall_results(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['variant'], row['method'], row['view'], row['calibration'], row['split_id'])].append(row)
    result = []
    for key, selected in groups.items():
        if len({r['task'] for r in selected}) != 4 or len(selected) != 4:
            raise ValueError('Overall metrics require exactly all four tasks per split')
        counts = {k: sum(r[k] for r in selected) for k in
                  ['false_alarms', 'test_successes', 'detected_failures', 'test_failures']}
        result.append(dict(zip(['variant', 'method', 'view', 'calibration', 'split_id'], key)) |
                      {'task': 'Overall', **counts, 'FAR': counts['false_alarms']/counts['test_successes'],
                       'FDR': counts['detected_failures']/counts['test_failures'],
                       'episode_AUROC': float(np.mean([r['episode_AUROC'] for r in selected]))})
    return result


def write_extras(output, extras, results, originals, protocol_note=None):
    common = ['variant', 'task', 'method', 'view', 'calibration']
    subgroups = extras['subgroup_results']
    extras['subgroup_summary'] = grouped_summary(
        [r for r in subgroups if r['split_id'] != 'original'],
        [r for r in subgroups if r['split_id'] == 'original'],
        common + ['subgroup_kind', 'subgroup'], ['FAR', 'FDR', 'episode_AUROC'])
    metrics = ['latency_threshold', 'n_detected', 'n_pre_onset_alarm', 'n_missed',
               'n_onset_after_scored_end', 'n_unannotated', 'n_timestamp_unavailable',
               'n_failure_episodes', 'n_annotated', 'delay_mean_s', 'delay_std_s', 'delay_median_s', 'delay_p95_s']
    extras['offline_latency_summary'] = grouped_summary(
        extras['offline_latency_split_results'], extras['offline_latency_original'],
        common + ['annotator_id', 'annotator', 'onset_view'], metrics)
    extras['overall_split_results'] = overall_results(results)
    extras['overall_original'] = overall_results(originals)
    extras['overall_summary'] = grouped_summary(extras['overall_split_results'], extras['overall_original'],
                                                common, ['FAR', 'FDR', 'episode_AUROC'])
    for name, rows in extras.items():
        benchmark.write_csv(output/(name+'.csv'), rows, [])
        (output/(name+'.json')).write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    def metric(row, name, scale=1):
        if row[name+'_mean'] is None:
            return 'undefined (no detected episodes)'
        def f(key):
            value = row[name+'_'+key]
            return f'{scale*value:.3f}' if value is not None else 'undefined'
        return f'{f("mean")} ± {f("sd")} [{f("p05")}, {f("p95")}]'
    sections = []
    sections.append(('Overall: pooled FAR/FDR; equal-task macro AUROC',
        ['View', 'Protocol', 'FAR %', 'FDR %', 'Episode AUROC %'],
        [[r['view'], r['calibration'], *[metric(r, m, 100) for m in ['FAR','FDR','episode_AUROC']]]
         for r in extras['overall_summary']]))
    sections.append(('Offline delay conditional on a first alarm at/after onset',
        ['Task','View','Protocol','Annotator','Delay (s)','Defined splits','Detected episodes','Pre-onset alarms'],
        [[r['task'],r['view'],r['calibration'],r['annotator'],metric(r,'delay_mean_s'),
          f'{r["delay_mean_s_defined_splits"]}/{r["n_splits"]}',metric(r,'n_detected'),metric(r,'n_pre_onset_alarm')]
         for r in extras['offline_latency_summary']]))
    sections.append(('Failure groups: unchanged failure IDs in every split',
        ['Task','View','Protocol','Group type','Group','FDR %','Episode AUROC %'],
        [[r['task'],r['view'],r['calibration'],r['subgroup_kind'],r['subgroup'],
          metric(r,'FDR',100),metric(r,'episode_AUROC',100)] for r in extras['subgroup_summary']]))
    note = ('All intervals are mean ± sample SD [5th, 95th resampling percentile], not confidence intervals. '
            'Paper FAR/FDR/AUROC preserve the existing paper frame selection; delay analysis separately recalibrates '
            'on every eligible endpoint, exactly as the existing offline latency protocol. Delay includes no inference/runtime cost. '
            'Pre-onset alarms are not counted as successful post-onset detections. Missing annotations and undefined delays '
            'remain explicit; defined_splits records the denominator for each metric. Annotation snapshots and templates are frozen. '
            'Overall row i combines the independent task-specific split i; it is not an additional experiment.')
    if protocol_note is not None:
        note = protocol_note
    md = '# Additional offline sensitivity metrics\n\n'+note+'\n\n'
    page = '<!doctype html><meta charset="utf-8"><title>Offline calibration sensitivity</title><style>body{font:14px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}section{overflow:auto}</style>'
    page += '<h1>Additional offline sensitivity metrics</h1><p>'+html.escape(note)+'</p><a href="report.html">Main report</a> · <a href="offline_metrics.md">Markdown</a>'
    for title, headers, cells in sections:
        md += '## '+title+'\n\n'+benchmark.markdown_table(headers,cells)+'\n\n'
        page += '<section><h2>'+html.escape(title)+'</h2><table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'
        page += ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in cells)+'</table></section>'
    (output/'offline_metrics.md').write_text(md)
    (output/'offline_metrics.html').write_text(page)
