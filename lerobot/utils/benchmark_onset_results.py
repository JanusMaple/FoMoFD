"""Read-only onset-aware report derivatives; no model loading or recalibration.

Original episode AUROC and calibration remain unchanged. Failure FDR can use
any crossing at/after the selected annotator's raw onset frame, even after an
early alarm. Combined detectors use the earlier of their two view onsets.
"""
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


def onset_frame(entry, view):
    from benchmark_consensus_onsets import select_annotation
    annotation = select_annotation(entry, view)
    return annotation.get('frame_index') if annotation else None


def assess(curve, onset):
    if onset is None:
        return {'post_alarm': None, 'early_alarm': None, 'post_score_max': None, 'early_score_max': None,
                'status': 'missing_onset', 'onset_frame': None}
    raw, scores = np.asarray(curve['raw']), np.asarray(curve['score'])
    post = raw >= onset
    early_max = float(np.max(np.asarray(curve.get('roc_score', scores))[~post])) if (~post).any() else -np.inf
    maximum = float(np.max(np.asarray(curve.get('roc_score', scores))[post])) if post.any() else -np.inf
    return {'post_alarm': bool(np.any(scores[post] > 0)),
            'early_alarm': bool(np.any(scores[~post] > 0)),
            'early_score_max': early_max if np.isfinite(early_max) else ('-inf' if early_max < 0 else '+inf'),
            'post_score_max': maximum if np.isfinite(maximum) else ('-inf' if maximum < 0 else '+inf'),
            'status': 'observed' if post.any() else 'onset_after_scored_end', 'onset_frame': onset}


def assess_stream(result, view, onset):
    events = result.get('replay', {}).get('events', [])
    raw = [e['raw_frame'] for e in events]
    margins = [e['scores'][view] - e['thresholds'][view] for e in events]
    row = assess({'raw': raw, 'score': margins}, onset)
    if row['post_alarm'] is False:
        replay = result.get('replay', {})
        if result.get('status') not in ('complete', 'completed') or replay.get('stop_reason') != 'source_exhausted':
            row.update(post_alarm=None, status='unobserved_after_stop')
    return row


def automatic_latencies(rows):
    """One combined-view row per episode/annotator, retaining its actual view."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in ('variant', 'task', 'method', 'view', 'calibration', 'episode_index', 'annotator_id'))
        groups[key].append(row)
    output = []
    for group in groups.values():
        if group[0]['view'] != 'both':
            output.extend(group)
            continue
        field = ('onset_wall_timestamp' if group[0].get('onset_wall_timestamp') is not None else
                 'onset_time_s' if group[0].get('onset_time_s') is not None else 'onset_raw_frame')
        valid = [r for r in group if r.get(field) is not None]
        if len(valid) == 2:
            output.append({**min(valid, key=lambda r: (r[field], r['onset_view'])), 'onset_rule': 'earlier_view'})
        elif len(group) == 1 and group[0].get('onset_rule'):
            output.extend(group)
        else:
            output.append({**group[0], field: None, 'status': 'unannotated',
                           'detection_delay_s': None, 'delay_s': None, 'onset_rule': 'missing_view_onset'})
    return output


def frozen_curves(source, records, timeline, template=None):
    from functional_conformal import apply_band
    if template is not None:
        return apply_band(source, timeline, template, records[0])
    fitted = {r['bin_index']: (float(r['threshold']), r['scale']) for r in records}
    width = next((r['bin_end_s'] - r['bin_start_s'] for r in records if r['bin_index'] is not None), None)
    output = {}
    for episode, curve in source.items():
        bins = [None] * len(curve['score']) if width is None else np.floor((curve['wall'] - timeline[episode]['wall'][0]) / width).astype(int)
        params = [fitted.get(b, (np.inf, 1.0)) for b in bins]
        threshold, scale = np.asarray(params).T
        output[episode] = {**curve, 'score': (curve['score'] - threshold) / scale}
    return output


class RawClocks:
    """Read only timestamp members, preserving each prepared-to-raw mapping."""

    def __init__(self, resolve):
        self.resolve, self.mappings, self.clocks = resolve, {}, {}

    def __call__(self, dataset, episode):
        dataset = self.resolve(str(dataset))
        if dataset not in self.mappings:
            preparation = json.loads((dataset/'meta/preparation.json').read_text())
            self.mappings[dataset] = {r['dataset_episode_index']: r for r in preparation['episode_mapping']}
        entry = self.mappings[dataset][episode]
        filename = entry['raw_file']
        if Path(filename).name != filename:
            raise ValueError('Invalid raw recording filename')
        path = dataset.parent/'raw'/filename
        if path not in self.clocks:
            with np.load(path, allow_pickle=False) as source:
                wall = np.asarray(source['timestamps'], dtype=np.float64).reshape(-1)
            if len(wall) != entry['raw_frames'] or not np.isfinite(wall).all() or (np.diff(wall) < 0).any():
                raise ValueError(f'Invalid raw clock: {path}')
            self.clocks[path] = wall
        return self.clocks[path]


def refresh_offline_latency(row, annotation, wall):
    """Relabel the saved first alarm; never replace it with a later crossing."""
    row = {**row, 'onset_raw_frame': annotation.get('frame_index') if annotation else None,
           'copied_from_annotator': annotation.get('copied_from_annotator') if annotation else None,
           'onset_wall_timestamp': None, 'signed_first_alarm_delay_s': None,
           'detection_delay_s': None, 'status': 'unannotated'}
    if not annotation:
        return row
    onset = annotation['frame_index']
    if type(onset) is not int or not 0 <= onset < len(wall):
        raise ValueError('Onset outside raw recording')
    timestamp = annotation.get('wall_timestamp', float(wall[onset]))
    row['onset_wall_timestamp'] = timestamp
    alarm_frame = row['first_alarm_raw_frame']
    if alarm_frame is not None and alarm_frame < onset:
        row['status'] = 'pre_onset_alarm'
        row['signed_first_alarm_delay_s'] = row['first_alarm_wall_timestamp'] - timestamp
    elif timestamp > row['last_scored_wall_timestamp']:
        row['status'] = 'onset_after_scored_end'
    elif alarm_frame is not None:
        row['status'] = 'detected'
        row['detection_delay_s'] = row['signed_first_alarm_delay_s'] = row['first_alarm_wall_timestamp'] - timestamp
    else:
        row['status'] = 'missed'
    return row


def refresh_online_latencies(trial, result, labels, wall):
    from benchmark_consensus_onsets import elapsed_onset
    from online_benchmark import latency_record
    from unified_benchmark import annotation_views
    rows = []
    annotations = labels['tasks'][trial['task']].get('failure_onsets', {}).get(trial['episode_index'], {})
    for view in trial['requested_detector_views']:
        for aid, annotator in labels['annotators'].items():
            for onset_view in (('fix', 'wrist') if view == 'both' else (view,)):
                annotation = annotation_views(annotations.get(aid)).get(onset_view, {})
                alarm = result['alarms'].get(view)
                record = latency_record(alarm, annotation.get('frame_index'), wall)
                record['onset_time_s'] = elapsed_onset(annotation, wall)
                rows.append({**{k: trial[k] for k in ('variant', 'task', 'method', 'episode_index', 'calibration')},
                             **record, 'view': view, 'onset_view': onset_view,
                             'onset_raw_frame': annotation.get('frame_index'),
                             'annotator_id': aid, 'annotator': annotator['name'],
                             'copied_from_annotator': annotation.get('copied_from_annotator'),
                             'late_alarm': bool(alarm and alarm['alarm_time_s'] > float(wall[-1] - wall[0])),
                             'alarm': alarm is not None})
    return automatic_latencies(rows)


def augment(data, root, resolve, pack, spec=None):
    from unified_benchmark import curves, load_timeline, combine_curves, aggregate_wm_scores, detector_definitions, annotation_views
    from functional_conformal import load_template_manifest, canonical_scheme
    root = Path(root)
    rows = [dict(zip(data['episodes']['columns'], r)) for r in data['episodes']['rows']]
    by_source = defaultdict(list)
    for row in rows:
        by_source[(row['variant'], canonical_scheme(row['calibration']))].append(row)
    labels_path = Path(__file__).resolve().parents[1] / 'configs/benchmark/prl_episode_labels.yaml'
    label_text = labels_path.read_text()
    labels = yaml.safe_load(label_text)
    from benchmark_consensus_onsets import derive_labels
    labels = derive_labels(labels, spec, resolve)
    data['annotators'] = {k: v['name'] for k, v in labels['annotators'].items()}
    sources = [(run['id'], root/run['path']) for run in data['runs']]
    sources += [(v['variant'], root/v['path']) for run in data['functional_runs'] for v in run['variants']]
    timelines, datasets, verified, exports = {}, {}, 0, []
    for variant, folder in sources:
        report = json.loads((folder/'report.json').read_text())
        config_path = folder/'benchmark_config.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        functional = 'templates_manifest' in report
        template_path = folder/report['templates_manifest'] if functional else None
        if not template_path and any(r['calibration'] == 'functional_band' for r in report['calibration']):
            run = next(r for r in data['runs'] if r['id'] == variant)
            template_path = resolve(run['source']).parent/'templates_manifest.json'
        templates = load_template_manifest(template_path) if template_path else {}
        schemes = {canonical_scheme(r['calibration']) for r in report['calibration']}
        streams = defaultdict(dict)
        for cache in report['score_caches']:
            task, method = cache['task'], cache['method']
            settings = cache['scoring_settings']
            with np.load(resolve(cache['path']), allow_pickle=False) as loaded:
                arrays = {k: loaded[k] for k in ['episode_index', 'frame_index', 'raw_frame_index', 'wall_timestamp', 'scores']}
            views = cache['views']
            if config.get('wm_score_aggregation') == 'joint' and method.startswith('wm_'):
                arrays['scores'] = aggregate_wm_scores(method, arrays['scores'], 'joint')
                views = ['both']
            job = SimpleNamespace(task=task, mode=method.startswith('wm_'), merged={'data': {'history_len': settings['L'], 'pred_horizon': settings['K']}})
            dataset = resolve(settings['dataset'])
            datasets[variant, task, method] = dataset
            if str(dataset) not in timelines:
                timelines[str(dataset)] = load_timeline(dataset)
            timeline = timelines[str(dataset)]
            for i, view in enumerate(views):
                source = curves(arrays, job, i, report['protocols']['paper_protocol'])
                for scheme in schemes:
                    records = [r for r in report['calibration'] if r['task'] == task and r['method'] == method and r['view'] == view and r['protocol'] == 'paper' and canonical_scheme(r['calibration']) == scheme]
                    if records:
                        streams[(task, scheme)][(method, view)] = frozen_curves(source, records, timeline, templates.get((task, method, view, 'paper')) if scheme == 'functional_band' else None)
        definitions = detector_definitions(report['methods'], config) if not functional else []
        for (task, scheme), detectors in streams.items():
            for method, view, fusion in definitions:
                if scheme != 'functional_band' and fusion and fusion[0] in detectors and fusion[1] in detectors:
                    detectors[(method, view)] = combine_curves(detectors[fusion[0]], detectors[fusion[1]], fusion[2])
            for row in by_source[(variant, scheme)]:
                if row['task'] != task:
                    continue
                group = next(g for g in labels['tasks'][task]['groups']
                             if g['episode_range'][0] <= row['episode_index'] <= g['episode_range'][1])
                if any((row[k] or None) != (group[k] or None) for k in ('split', 'outcome', 'failure_mode', 'failure_origin')):
                    raise ValueError('Current evaluation groups differ from frozen benchmark groups')
                curve = detectors[(row['method'], row['view'])][row['episode_index']]
                maximum = float(np.max(curve.get('roc_score', curve['score'])))
                any_time = row['any_time_alarm'] if row.get('any_time_alarm') is not None else row['alarm']
                if bool(np.any(curve['score'] > 0)) != any_time or not np.isclose(maximum, float(row['score_max']), rtol=1e-7, atol=1e-9):
                    raise ValueError(f'Onset report reconstruction differs from original: {variant}/{task}/{row["method"]}/{row["view"]}/{scheme}/{row["episode_index"]}')
                verified += 1
                if row['outcome'] != 'failure':
                    continue
                annotations = labels['tasks'][task].get('failure_onsets', {}).get(row['episode_index'], {})
                row['onset_assessments'] = {aid: assess(curve, onset_frame(annotations.get(aid), row['view'])) for aid in data['annotators']}
                for aid, assessment in row['onset_assessments'].items():
                    exports.append({**{k: row[k] for k in ('variant','task','method','view','calibration','episode_index')}, 'annotator_id': aid, **assessment})
        print(f'Onset curves verified: {variant} ({"functional" if functional else "episode/bin"})', flush=True)
    if verified != len(rows):
        raise ValueError(f'Onset coverage {verified} != {len(rows)}')
    data['episodes'] = pack(rows)
    clocks = RawClocks(resolve)
    revised_latencies = []
    for values in data['latencies']['rows']:
        row = dict(zip(data['latencies']['columns'], values))
        method = 'wm_nc' if row['method'].startswith('wm_nc_') else row['method']
        wall = clocks(datasets[row['variant'], row['task'], method], row['episode_index'])
        entry = labels['tasks'][row['task']].get('failure_onsets', {}).get(row['episode_index'], {}).get(row['annotator_id'])
        annotation = annotation_views(entry).get(row['onset_view'])
        revised_latencies.append(refresh_offline_latency(row, annotation, wall))
    revised_latencies = automatic_latencies(revised_latencies)
    data['latencies'] = pack(revised_latencies)
    for trial in data['online']:
        result = json.loads((root/trial['path']).read_text())
        wall = clocks(result['trial']['dataset'], trial['episode_index'])
        trial['latencies'] = refresh_online_latencies(trial, result, labels, wall)
    for run in data.get('vlm_runs', []):
        if spec:
            run['primary_annotator_id'] = spec['detection']['primary_annotator']
        revised_vlm = []
        for row in run['episodes']:
            result = json.loads((root/row['trace_path']).read_text())
            annotations = labels['tasks'][row['task']].get('failure_onsets', {}).get(row['episode_index'], {})
            if row['outcome'] == 'failure':
                row['onset_assessments'] = {aid: assess_stream(result, row['view'], onset_frame(annotations.get(aid), row['view'])) for aid in data['annotators']}
            from benchmark_vlm_results import refresh_trial_onset, latency_rows
            wall = clocks(result['dataset'], row['episode_index'])
            revised_vlm.append(refresh_trial_onset(result, labels, run['primary_annotator_id'], wall))
            row['latencies'] = latency_rows(result, labels, wall, row['variant']) if row['outcome'] == 'failure' else []
            row['pre_onset_alarm'] = revised_vlm[-1]['pre_onset_alarm']
        from benchmark_vlm_results import refresh_onset_summary
        refresh_onset_summary(run, revised_vlm)
    import hashlib
    if labels_path.read_text() != label_text:
        raise ValueError('Onset labels changed during report generation; rerun for a consistent snapshot')
    metadata = {'labels_sha256': hashlib.sha256(label_text.encode()).hexdigest(), 'verified_original_episode_rows': verified,
                'created_utc': data['created'], 'labels_snapshot': 'labels_snapshot.yaml',
                'offline_latency_rows': len(revised_latencies), 'online_trials_reannotated': len(data['online']),
                'vlm_trials_reannotated': sum(len(run['episodes']) for run in data.get('vlm_runs', [])),
                'latency_rule': 'First recorded alarm, with updated onset; original alarm times and traces unchanged',
                'FDR_rule': 'Any strict crossing at raw frame >= onset; earlier alarms reported separately',
                'combined_onset': 'Earlier fixed/wrist onset; both annotations required',
                'AUROC': 'Original whole-episode AUROC, unchanged', 'streaming': 'Early-stopped trials are unknown, not misses',
                'source_archives': [{'variant': r['id'], 'path': r['path'], 'sha256': r['archive_sha256']} for r in data['runs']],
                'score_caches': data['caches'],
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = root/'analyses/onset_detection'
    output.mkdir(parents=True, exist_ok=True)
    (output/'labels_snapshot.yaml').write_text(label_text)
    if spec and spec['detection'].get('onset_reference'):
        (output/'derived_onsets.yaml').write_text(yaml.safe_dump(labels, sort_keys=False))
        metadata['onset_reference'] = spec['detection']['onset_reference']
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    import csv
    with (output/'episode_assessments.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(exports[0]))
        writer.writeheader(); writer.writerows(exports)
    for name, records in [('offline_latencies', revised_latencies),
                          ('online_latencies', [r for trial in data['online'] for r in trial['latencies']])]:
        with (output/(name+'.csv')).open('w') as f:
            fields = list(dict.fromkeys(k for row in records for k in row))
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(records)
    data['onset_analysis'] = metadata
