#!/usr/bin/env python3
"""Publish CPU-only, Git-friendly tables from the current verified HTML snapshot.

No images, checkpoints, score arrays or model inference are loaded. Existing
episode decisions are aggregated; ROC uses the saved episode maxima. Run from
the repo root with the lerobot environment:
    python lerobot/utils/export_benchmark_tables.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics

import yaml

from benchmark_roc import curve, operating_point
from benchmark_review_results import METHODS
from results_inspector import read_viewer_snapshot

REPO = Path(__file__).resolve().parents[2]
KEYS = ('variant', 'task', 'method', 'view', 'calibration')


def unpack(table):
    return [dict(zip(table['columns'], row)) for row in table['rows']]


def key(row):
    return tuple(row[k] for k in KEYS)


def unique_index(rows):
    result = {}
    for row in rows:
        identity = (*key(row), row['episode_index'])
        if identity in result:
            raise ValueError(f'Duplicate episode: {identity}')
        result[identity] = row
    return result


def ratio(k, n):
    return k / n if n else None


def latency_summary(rows):
    delays = [r['detection_delay_s'] for r in rows
              if r.get('alarm') is True and not r.get('late_alarm', False)
              and r.get('detection_delay_s') is not None]
    if any(not math.isfinite(v) or v < 0 for v in delays):
        raise ValueError('Invalid first-post-onset delay')
    return {
        'latency_n': len(delays),
        'latency_mean_s': statistics.mean(delays) if delays else None,
        'latency_sd_sample_s': statistics.stdev(delays) if len(delays) > 1 else None,
        'latency_sd_population_s': statistics.pstdev(delays) if delays else None,
        'latency_median_s': statistics.median(delays) if delays else None,
        **{f'detected_within_{t}s_n': sum(v <= t for v in delays) for t in (1, 2, 5, 10)},
    }


def subsets(task_labels):
    success = set()
    failures = set()
    modes = []
    for group in task_labels['groups']:
        if group['split'] != 'test':
            continue
        ids = set(range(group['episode_range'][0], group['episode_range'][1] + 1))
        if group['outcome'] == 'success':
            success |= ids
        else:
            if failures & ids:
                raise ValueError('Overlapping failure groups')
            failures |= ids
            modes.append(('failure_mode', group['failure_mode'], group['failure_origin'], ids))
    if success & failures:
        raise ValueError('Overlapping success/failure groups')
    return [('success_test', 'No Failure', 'not_applicable', success),
            ('all_failures', 'All failures', 'all', failures), *modes]


def metadata(data, group):
    result = dict(zip(KEYS, group))
    runs = {r['id']: r for r in data['runs']}
    run = runs.get(result['variant'], {})
    method = result['method']
    is_vlm = method.startswith('qwen')
    result.update(
        method_name=METHODS.get(method, 'Qwen v9' if is_vlm else method),
        variant_title=run.get('title', result['variant']),
        family=run.get('family', 'vlm' if is_vlm else 'unknown'),
        main_comparison=(is_vlm or (method in METHODS and result['variant'] ==
                         data.get('summary_sources', {}).get(method, 'main')))
                        and method not in data.get('supplementary_methods', []),
        supplementary_method=method in data.get('supplementary_methods', []),
        annotator_id=data['evaluation_protocol']['detection']['primary_annotator'],
        evaluation_protocol=data['evaluation_protocol']['protocol_id'],
    )
    return result


def observed_counts(rows, expected):
    ids = {r['episode_index'] for r in rows}
    if not ids <= expected or len(ids) != len(rows):
        raise ValueError('Duplicate or mislabeled episodes in a group')
    if any(r['alarm'] is not None and type(r['alarm']) is not bool for r in rows):
        raise ValueError('Alarm must be true, false or unknown')
    detected = sum(r['alarm'] is True for r in rows)
    unknown = sum(r['alarm'] is None for r in rows)
    missing = len(expected) - len(rows)
    return dict(population_n=len(expected), recorded_n=len(rows), not_recorded_n=missing,
                alarm_n=detected, observed_negative_n=sum(r['alarm'] is False for r in rows),
                unknown_n=unknown, rate=ratio(detected, len(expected)) if not unknown + missing else None,
                rate_lower=ratio(detected, len(expected)),
                rate_upper=ratio(detected + unknown + missing, len(expected)),
                episode_ids=';'.join(map(str, sorted(ids))))


def offline_tables(data, labels):
    primary = data['evaluation_protocol']['detection']['primary_annotator']
    episodes = [r for r in unpack(data['episodes']) if r['split'] == 'test']
    unique_index(episodes)
    latencies = unique_index(r for r in unpack(data['latencies']) if r['annotator_id'] == primary)
    groups = defaultdict(list)
    for row in episodes:
        groups[key(row)].append(row)
    output = []
    for group, records in sorted(groups.items()):
        negative = [r for r in records if r['outcome'] == 'success']
        success_ids = subsets(labels['tasks'][group[1]])[0][3]
        success_counts = observed_counts(negative, success_ids)
        for kind, mode, origin, ids in subsets(labels['tasks'][group[1]]):
            rows = [r for r in records if r['episode_index'] in ids]
            counts = observed_counts(rows, ids)
            failure = kind != 'success_test'
            if any((r['outcome'] == 'failure') != failure or
                   (kind == 'failure_mode' and (r['failure_mode'], r['failure_origin']) != (mode, origin))
                   for r in rows):
                raise ValueError('Episode labels disagree with the frozen label file')
            timing = []
            for row in rows if failure else []:
                assessment = (row.get('onset_assessments') or {}).get(primary)
                if assessment is None or assessment['post_alarm'] != row['alarm']:
                    raise ValueError('Snapshot is not using the frozen primary onset assessment')
                latency = latencies[(*group, row['episode_index'])]
                if latency['post_alarm'] != row['alarm']:
                    raise ValueError('Offline latency and detection populations disagree')
                timing.append({**latency, 'alarm': row['alarm']})
            row = {**metadata(data, group), 'group_type': kind, 'failure_mode': mode,
                   'failure_origin': origin, 'measurement': 'dense_offline', **counts,
                   'FAR': success_counts['rate'], 'false_alarm_n': success_counts['alarm_n'],
                   'success_test_n': len(success_ids), 'FDR': counts['rate'] if failure else None,
                   'early_alarm_n': sum(r.get('early_alarm') is True for r in rows) if failure else None,
                   'any_pre_onset_alarm_n': sum(r.get('any_pre_onset_alarm') is True for r in rows) if failure else None,
                   'sustained_pre_onset_alarm_n': sum(r.get('sustained_pre_onset_alarm') is True for r in rows) if failure else None,
                   **latency_summary(timing)}
            # Mode-specific ROC keeps every success-TEST episode as a negative.
            if failure and not counts['unknown_n'] + counts['not_recorded_n'] + success_counts['not_recorded_n']:
                selected = negative + rows
                _, row['episode_AUROC'] = curve(selected)
                matched = operating_point(selected, .05)
                row.update(matched_FAR_target=.05, matched_FAR=matched['FAR'],
                           matched_FDR=matched['FDR'], matched_detected_n=matched['detected_failures'],
                           matched_false_alarm_n=matched['false_alarms'], matched_threshold=matched['threshold'])
            output.append(row)
    return output, groups


def online_records(data):
    primary = data['evaluation_protocol']['detection']['primary_annotator']
    records, planned = [], defaultdict(set)
    for plan in data['plans']:
        for view in plan['requested_detector_views']:
            planned[key({**plan, 'view': view})].add(plan['episode_index'])
    for trial in data['online']:
        for latency in trial['latencies']:
            if latency['annotator_id'] != primary or latency['view'] not in trial['requested_detector_views']:
                continue
            row = {**{k: v for k, v in trial.items() if k not in ('latencies', 'timings', 'hardware')}, **latency}
            row.update(alarm=latency['post_alarm'], replay_scope='offline_detected_failures_only',
                       trace_path=trial['path'])
            records.append(row)
    for run in data.get('vlm_runs', []):
        for episode in run['episodes']:
            latencies = [l for l in episode.get('latencies', []) if l['annotator_id'] == primary]
            if episode['outcome'] == 'failure' and len(latencies) != 1:
                raise ValueError('VLM failure lacks a unique primary onset assessment')
            row = {**episode, **(latencies[0] if latencies else {})}
            row.pop('latencies', None)
            row.pop('onset_assessments', None)
            row.update(replay_scope='all_test_episodes', annotator_id=primary,
                       evaluation_protocol=data['evaluation_protocol']['protocol_id'])
            records.append(row)
            planned[key(row)].add(row['episode_index'])
    unique_index(records)
    for row in records:
        if row['episode_index'] not in planned[key(row)]:
            raise ValueError('Recorded replay absent from its plan')
    return records, planned


def online_tables(data, labels, offline_groups):
    records, planned = online_records(data)
    groups = defaultdict(list)
    for row in records:
        groups[key(row)].append(row)
    output = []
    for group in sorted(set(offline_groups) | set(groups)):
        available = groups[group]
        full_test = bool(available) and available[0]['replay_scope'] == 'all_test_episodes'
        for kind, mode, origin, ids in subsets(labels['tasks'][group[1]]):
            rows = [r for r in available if r['episode_index'] in ids]
            selected = planned[group] & ids
            counts = observed_counts(rows, ids)
            failure = kind != 'success_test'
            scope = 'all_test_episodes' if full_test else 'offline_detected_failures_only' if available else 'not_replayed'
            status = ('complete' if rows and len(rows) == len(selected) else 'incomplete' if selected
                      else 'not_replayed')
            late = sum(r['alarm'] is True and r.get('late_alarm', False) for r in rows)
            on_time = counts['alarm_n'] - late
            row = {**metadata(data, group), 'group_type': kind, 'failure_mode': mode,
                   'failure_origin': origin, 'measurement': 'recorded_online', 'replay_scope': scope,
                   'coverage_status': status, **{k: v for k, v in counts.items() if not k.startswith('rate')},
                   'selected_n': len(selected), 'missing_selected_n': len(selected) - len(rows),
                   'selected_detection_rate': ratio(counts['alarm_n'], len(selected))
                       if len(rows) == len(selected) and not counts['unknown_n'] else None,
                   'on_time_alarm_n': on_time, 'late_alarm_n': late,
                   'FDR': counts['rate'] if full_test and failure else None,
                   'FAR': counts['rate'] if full_test and not failure else None,
                   'full_test_rate_lower': counts['rate_lower'] if full_test else None,
                   'full_test_rate_upper': counts['rate_upper'] if full_test else None,
                   'full_test_on_time_rate': ratio(on_time, len(ids))
                       if full_test and not counts['unknown_n'] + counts['not_recorded_n'] else None,
                   'early_alarm_n': sum(r.get('early_alarm') is True for r in rows) if failure and rows else None,
                   'any_pre_onset_alarm_n': sum(r.get('any_pre_onset_alarm') is True for r in rows) if failure and rows else None,
                   'sustained_pre_onset_alarm_n': sum(r.get('sustained_pre_onset_alarm') is True for r in rows) if failure and rows else None,
                   **latency_summary(rows if failure else [])}
            output.append(row)
    return output, records


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    def cell(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: cell(v) for k, v in row.items()} for row in rows)
    return {'rows': len(rows), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def relocate_markdown(text, source_dir, destination_dir, local_links=()):
    def link(match):
        target = match[1]
        if target in local_links or target.startswith(('#', '/', 'http:', 'https:', 'mailto:')):
            return match[0]
        path, separator, fragment = target.partition('#')
        return '](' + os.path.relpath(source_dir / path, destination_dir) + separator + fragment + ')'
    return re.sub(r'\]\(([^)]+)\)', link, text)


def verify_aggregates(data, offline, online):
    """Reconcile new mode tables with independently stored report aggregates."""
    counts = defaultdict(lambda: [0, 0, 0, 0])
    for row in offline + [r for r in online if r['replay_scope'] == 'all_test_episodes']:
        if row['group_type'] != 'failure_mode':
            continue
        bucket = counts[(*key(row), row['failure_category'])]
        for i, field in enumerate(('alarm_n', 'population_n', 'unknown_n', 'not_recorded_n')):
            bucket[i] += row[field]
    verified = 0
    for row in data['failure_categories']['records']:
        if row['task'] == 'Overall':
            continue
        if counts[(*key(row), row['category_title'])] != [row[k] for k in ('detected','total','unknown','missing')]:
            raise ValueError(f'Category reconciliation failed: {key(row)}, {row["category_title"]}')
        verified += 1
    vlm_verified = 0
    for run in data.get('vlm_runs', []):
        for expected in run['summary']:
            rows = [r for r in online if r['method'] == run['id']
                    and r['task'] == expected['task'] and r['view'] == expected['view']]
            failure = next(r for r in rows if r['group_type'] == 'all_failures')
            success = next(r for r in rows if r['group_type'] == 'success_test')
            for row, metric in ((failure, 'FDR'), (success, 'FAR')):
                actual = [row[k] for k in ('alarm_n','on_time_alarm_n','population_n')]
                if actual != [expected[metric + suffix] for suffix in ('_any_count','_on_time_count','_denominator')]:
                    raise ValueError('VLM rate reconciliation failed')
            if failure['latency_n'] != expected['conditional_delay_n']:
                raise ValueError('VLM latency population differs')
            for column, original in (('latency_mean_s', 'conditional_delay_mean_s'),
                                     ('latency_sd_population_s', 'conditional_delay_sd_s')):
                if failure['latency_n'] and not math.isclose(failure[column], expected[original], abs_tol=1e-12):
                    raise ValueError('VLM latency statistics differ')
            vlm_verified += 1
    return {'category_groups_exact': verified, 'vlm_task_view_groups_exact': vlm_verified,
            'duplicate_episode_checks': 'passed', 'offline_alarm_latency_agreement': 'passed',
            'full_test_rates_absent_for_conditional_replay': all(r['FDR'] is None and r['FAR'] is None
                      for r in online if r['replay_scope'] != 'all_test_episodes')}


def export(source, output):
    data = read_viewer_snapshot(source / 'index.html')
    spec = data['evaluation_protocol']
    summary = yaml.safe_load((source / 'summary.yaml').read_text())
    if summary['created_utc'] != data['created'] or summary['evaluation_protocol'] != spec:
        raise ValueError('Summary and viewer snapshot differ; refresh reports first')
    config = REPO / 'lerobot/configs/benchmark'
    catalog = yaml.safe_load((config / 'results_catalog.yaml').read_text())
    if yaml.safe_load((REPO / catalog['evaluation_protocol']).read_text()) != spec:
        raise ValueError('Snapshot and current protocol differ; refresh the viewer first')
    for item in spec['frozen_inputs'].values():
        if sha(REPO / item['path']) != item['sha256']:
            raise ValueError(f'Frozen input changed: {item["path"]}')
    labels = yaml.safe_load((REPO / spec['frozen_inputs']['onsets']['path']).read_text())
    offline, groups = offline_tables(data, labels)
    online, episodes = online_tables(data, labels, groups)
    details = output / 'details'
    details.mkdir(parents=True, exist_ok=True)
    files = {}
    tables = {'offline_by_failure.csv': offline, 'online_by_failure.csv': online,
              'online_episode_outcomes.csv': episodes, 'scoring_settings.csv': data['settings']}
    category_lookup = {(m['task'], m['failure_mode']): c['title']
                       for c in data['failure_categories']['categories'] for m in c['modes']}
    for rows in (offline, online):
        for row in rows:
            row['failure_category'] = category_lookup.get((row['task'], row['failure_mode']))
    validation = verify_aggregates(data, offline, online)
    for row in episodes:
        row.update(metadata(data, key(row)))
    for name, rows in tables.items():
        files['details/' + name] = write_csv(details / name, rows)
    for section in data.get('reviewer_evidence', {}).get('sections', []):
        if section['id'] == 'calibration_matched_far':
            name = 'calibration_matched_far_summary.csv'
            files['details/'+name] = write_csv(details/name, section['records'])
    copies = {
        'details/failure_categories.csv': source / 'failure_categories/metrics.csv',
        'details/pairwise_latency_summary.csv': source / 'pairwise_latency/summary.csv',
        'details/pairwise_latency_episodes.csv': source / 'pairwise_latency/episodes.csv',
        'evaluation_protocol.yaml': REPO / catalog['evaluation_protocol'],
    }
    if data.get('fusion_level_comparison'):
        fusion = source / data['fusion_level_comparison']['path']
        copies.update({'details/fusion_'+name+'.csv': fusion/(name+'.csv')
                       for name in ('metrics','episode_scores','component_calibration')})
    study = next((s for s in data.get('studies', []) if s['id'] == 'expert_policy_score_distributions'), None)
    if study and study.get('dataset_validity_applied'):
        folder = source / study['path']
        copies.update({'details/expert_policy_episode_maxima_summary.csv': folder/'episode_maximum_summary.csv',
                       'details/expert_policy_episode_maxima.csv': folder/'episodes.csv'})
    for name, path in copies.items():
        destination = output / name
        if path.suffix == '.csv':
            destination.write_text(path.read_text())  # Normalize CRLF for Git text diffs.
        else:
            shutil.copyfile(path, destination)
        files[name] = {'bytes': destination.stat().st_size, 'sha256': sha(destination),
                       'source_sha256': sha(path)}
    protocol = config / 'PROTOCOL.md'
    (output / 'frozen_protocol.md').write_text(relocate_markdown(protocol.read_text(), config, output))
    files['frozen_protocol.md'] = {'bytes': (output / 'frozen_protocol.md').stat().st_size,
                                 'sha256': sha(output / 'frozen_protocol.md')}
    if data.get('fusion_level_comparison'):
        target = output / 'fusion_protocol.md'
        target.write_text(relocate_markdown((config/'FUSION_PROTOCOL.md').read_text(),config,output))
        files['fusion_protocol.md'] = {'bytes':target.stat().st_size,'sha256':sha(target)}
    publication = {'detailed_exports': {
        'guide': 'README.md', 'manifest': 'details/manifest.json',
        'offline_by_failure': 'details/offline_by_failure.csv',
        'online_by_failure': 'details/online_by_failure.csv',
        'online_episodes': 'details/online_episode_outcomes.csv',
        'frozen_protocol_markdown': 'frozen_protocol.md', 'frozen_protocol_yaml': 'evaluation_protocol.yaml',
    }}
    if data.get('fusion_level_comparison'):
        publication['detailed_exports'].update(fusion_metrics='details/fusion_metrics.csv',
            fusion_scores='details/fusion_episode_scores.csv',fusion_protocol='fusion_protocol.md')
    if study and study.get('dataset_validity_applied'):
        publication['detailed_exports'].update(
            expert_policy_episode_maxima_summary='details/expert_policy_episode_maxima_summary.csv',
            expert_policy_episode_maxima='details/expert_policy_episode_maxima.csv')
    (output / 'summary.yaml').write_text((source / 'summary.yaml').read_text().rstrip() + '\n' +
                                       yaml.safe_dump(publication, sort_keys=False))
    (output / 'summary.md').write_text('[Detailed per-mode tables](README.md) · [Frozen protocol](frozen_protocol.md)\n\n' +
        relocate_markdown((source / 'summary.md').read_text(), source, output, local_links=('summary.yaml',)))
    for name in ('summary.md', 'summary.yaml'):
        files[name] = {'bytes': (output / name).stat().st_size, 'sha256': sha(output / name)}
    manifest = {
        'schema_version': 1, 'exported_utc': datetime.now(timezone.utc).isoformat(),
        'source_snapshot_utc': data['created'], 'presentation_refreshed_utc': data.get('presentation_refreshed_utc'),
        'source_viewer_sha256': sha(source / 'index.html'), 'protocol_id': spec['protocol_id'],
        'primary_annotator': spec['detection']['primary_annotator'], 'frozen_inputs': spec['frozen_inputs'],
        'source_protocol_markdown_sha256': sha(protocol), 'exporter_sha256': sha(Path(__file__)),
        'source_summary_yaml_sha256': sha(source / 'summary.yaml'),
        'source_variants': data['runs'], 'summary_sources': data['summary_sources'],
        'online_coverage': data['online_coverage'], 'source_online_runs': data['online_runs'],
        'vlm_runs': [{k: r[k] for k in ('id', 'path', 'sha256', 'protocol_id')} for r in data['vlm_runs']],
        'offline_detection_groups': len(groups), 'online_trial_views': len(episodes), 'files': files,
        'validation': validation,
        'fusion_level_comparison': {k:v for k,v in data.get('fusion_level_comparison',{}).items()
                                   if k in ('analysis_id','evaluation_protocol','note','source_hashes','verification','labels_sha256','script_sha256')},
        'rules': {
            'rates': 'Fractions in [0,1], not percentages. Blank = unavailable, never imputed zero.',
            'offline': 'Post-onset FDR; whole-episode AUROC. Per-mode ROC retains all task success-test negatives.',
            'online': 'Continuous-score replay measures selected failures only: FDR/FAR blank. Selected detection rate is conditional, not full-test FDR.',
            'latency': 'First completed post-onset-source alarm. Online excludes late completions; offline excludes inference time.',
            'scope': 'All current curated variants, including archived seed 1 and superseded ACE scores, are explicitly identified. Main-comparison flag chooses the current baseline revisions.',
            'unknown': 'Unknown and unrecorded episodes remain explicit. No interpolation or reconstruction from aggregate totals.',
        },
    }
    (details / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'offline_rows': len(offline), 'online_rows': len(online),
                      'online_episode_rows': len(episodes), 'total_bytes': sum(f['bytes'] for f in files.values()),
                      'output': str(output)}, indent=2))


def export_study_refresh(source, output):
    """Publish auxiliary statistics without reinterpreting primary/onset results."""
    data = read_viewer_snapshot(source / 'index.html')
    current = yaml.safe_load((source / 'summary.yaml').read_text())
    previous = yaml.safe_load((output / 'summary.yaml').read_text())
    study_ids = {s['id'] for s in data['studies'] if s['kind'] == 'score_distributions'}
    if current['created_utc'] != previous['created_utc'] or current['evaluation_protocol'] != previous['evaluation_protocol']:
        raise ValueError('Study-only export requires the same primary snapshot and protocol')
    for name, table in previous['tables'].items():
        if name not in study_ids and current['tables'].get(name) != table:
            raise ValueError(f'Primary table changed during study-only export: {name}')
    study = next(s for s in data['studies'] if s['id'] == 'expert_policy_score_distributions')
    if study['status'] != 'complete' or not study.get('dataset_validity_applied') or study.get('episode_quantile_method') != 'linear':
        raise ValueError('Expected a completed corrected episode median/IQR study')
    folder = source / study['path']
    details = output / 'details'
    manifest_path = details / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    publication = {'detailed_exports': dict(previous['detailed_exports'])}
    for key, origin, filename in (
        ('expert_policy_episode_maxima_summary', 'episode_maximum_summary.csv', 'expert_policy_episode_maxima_summary.csv'),
        ('expert_policy_episode_maxima', 'episodes.csv', 'expert_policy_episode_maxima.csv'),
    ):
        target = details / filename
        target.write_text((folder / origin).read_text())
        publication['detailed_exports'][key] = f'details/{filename}'
        manifest['files'][f'details/{filename}'] = {'bytes': target.stat().st_size, 'sha256': sha(target),
                                                 'source_sha256': sha(folder / origin)}
    (output / 'summary.yaml').write_text((source / 'summary.yaml').read_text().rstrip() + '\n' +
                                        yaml.safe_dump(publication, sort_keys=False))
    (output / 'summary.md').write_text('[Detailed per-mode tables](README.md) · [Frozen protocol](frozen_protocol.md)\n\n' +
        relocate_markdown((source / 'summary.md').read_text(), source, output, local_links=('summary.yaml',)))
    for name in ('summary.md', 'summary.yaml'):
        target = output / name
        manifest['files'][name] = {'bytes': target.stat().st_size, 'sha256': sha(target)}
    manifest.update(source_viewer_sha256=sha(source/'index.html'), source_summary_yaml_sha256=sha(source/'summary.yaml'),
                    exporter_sha256=sha(Path(__file__)))
    manifest['auxiliary_study_refresh'] = {
        'refreshed_utc': datetime.now(timezone.utc).isoformat(), 'study': study['id'], 'path': study['path'],
        'source_sha256': study['source_sha256'], 'quantile_method': 'linear', 'iqr': 'Q3-Q1',
        'primary_tables_preserved': True, 'primary_frozen_inputs_revalidated': False,
        'inference_performed': False, 'gpu_used': False,
        'note': 'Study-only refresh. Primary results retain their frozen snapshot; current onset edits were not applied.'}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print('Published episode-max median/IQR summaries and per-episode values; primary tables preserved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO / 'fomofd/tasks/benchmark_results')
    parser.add_argument('--output', type=Path, default=REPO / 'logs/benchmarks/result_tables')
    parser.add_argument('--studies-only', action='store_true', help='Publish corrected auxiliary study summaries while preserving primary tables')
    args = parser.parse_args()
    if args.studies_only:
        export_study_refresh(args.source.resolve(), args.output.resolve())
    else:
        export(args.source.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
