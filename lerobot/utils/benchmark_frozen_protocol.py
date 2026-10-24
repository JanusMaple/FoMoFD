"""Frozen revision protocol; pure CPU interpretation of observed alarm traces.

No smoothing, interpolation, score inversion or imputation of unobserved alarms.
Historical results are inputs and are never rewritten by this module.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / 'lerobot/configs/benchmark/evaluation_protocol.yaml'


def load_protocol(path=PROTOCOL):
    path = Path(path)
    if not path.is_absolute():
        path = REPO / path
    spec = yaml.safe_load(path.read_text())
    for entry in spec['frozen_inputs'].values():
        actual = hashlib.sha256((REPO / entry['path']).read_bytes()).hexdigest()
        if actual != entry['sha256']:
            raise ValueError(f'Frozen protocol input changed: {entry["path"]}. Create a new protocol version explicitly.')
    return spec


def allows_anticipation(task, origin):
    return not (task in ('needle_pickup', 'ring_over_post') and origin == 'induced')


def resampling_post_max(curve, onset):
    """Keep the same score units as the recalibrated Q maxima, not alarm margins."""
    values = np.asarray(curve.get('roc_score', curve.get('raw_score', curve['score'])))
    post = np.asarray(curve['raw']) >= onset
    return float(values[post].max()) if post.any() else -np.inf


def validate_config(config):
    spec = load_protocol(config['evaluation_protocol'])
    corrections = yaml.safe_load((REPO / spec['score_support']['dataset_validity']).read_text())['corrections']
    expected = {(r['task'], r['episode_index'], r['raw_frame_stop']) for r in corrections}
    for key in ('paper_protocol', 'latency_protocol'):
        p = config[key]
        actual = {(r['task'], r['episode_index'], r['raw_frame_stop']) for r in p.get('frame_limits', [])}
        if p['window'] != 1 or p['wm_frame_policy'] != 'all_endpoints' or actual != expected:
            raise ValueError(f'{key} differs from frozen endpoint/correction/smoothing protocol')
    if config['alpha'] != spec['calibration']['alpha']:
        raise ValueError('Alpha differs from frozen protocol')
    return spec


def uses_post_onset(spec):
    return spec.get('latency', {}).get('detection_start') == 'first_positive_decision_with_source_endpoint_at_or_after_onset'


def alarm_run(curve, onset, onset_time, *, allow_anticipation, complete=True, first_post_onset=False):
    """A run spans onset only if adjacent observed decisions on both sides alarm.

    ``raw`` identifies the source evidence; ``wall`` is the score availability
    time (offline endpoint wall clock, or online decision completion clock).
    Negative delay requires an alarm returned before onset AND observed positive
    decisions continuing through the first source endpoint at/after onset.
    v2 always times the first post-onset-source decision. Sustained pre-onset
    runs are classified separately; v1 remains available for historical runs.
    """
    raw = np.asarray(curve['raw'], dtype=int)
    score = np.asarray(curve['score'], dtype=float)
    wall = np.asarray(curve['wall'], dtype=float)
    if not (len(raw) == len(score) == len(wall)) or (np.diff(raw) <= 0).any() or (np.diff(wall) < 0).any():
        raise ValueError('Alarm trace must contain ordered, distinct source endpoints and ordered completion times')
    # A -inf margin from a +inf conformal threshold is a known non-alarm.
    # NaN represents an actually unobserved/indeterminate decision.
    valid = ~np.isnan(score)
    positive = valid & (score > 0)
    out = dict(post_alarm=None, early_alarm=None, persistent_early=False,
               any_pre_onset_alarm=None, sustained_pre_onset_alarm=False,
               early_completion=None,
               detected_before_human_label=False, detection_delay_s=None,
               signed_first_alarm_delay_s=None, detection_raw_frame=None,
               detection_time=None, status='unannotated', onset_raw_frame=onset)
    if onset is None or onset_time is None:
        return out
    post = raw >= onset
    early = ~post
    out['early_alarm'] = bool(np.any(positive & early))
    out['any_pre_onset_alarm'] = out['early_alarm']
    out['early_completion'] = bool(np.any(positive & (wall < onset_time)))
    hits = np.flatnonzero(positive)
    if len(hits):
        out['signed_first_alarm_delay_s'] = float(wall[hits[0]] - onset_time)
    post_hits = np.flatnonzero(positive & post)
    if not len(post_hits):
        unknown = not complete or bool(np.any(post & ~valid))
        out.update(post_alarm=None if unknown else False,
                   status='unobserved_after_stop' if unknown else
                   'onset_after_scored_end' if not post.any() else
                   'pre_onset_alarm' if out['early_alarm'] else 'missed')
        return out
    index = int(post_hits[0])
    # A negative/indeterminate decision breaks persistence. Later alarms cannot
    # retroactively validate an isolated earlier crossing.
    first_post = int(np.flatnonzero(post)[0])
    start = index
    if index == first_post:
        while start > 0 and positive[start - 1]:
            start -= 1
    spans = start < first_post and raw[start] < onset
    out['persistent_early'] = bool(spans)
    sustained = bool(spans and allow_anticipation and wall[start] < onset_time)
    out['sustained_pre_onset_alarm'] = sustained
    if first_post_onset and sustained:
        # Exempt only this run; earlier disconnected alarms remain early.
        other_early = positive & early
        other_early[start:first_post] = False
        out['early_alarm'] = bool(other_early.any())
    if spans and allow_anticipation and not first_post_onset:
        index = start
    delay = float(wall[index] - onset_time)
    if first_post_onset and delay < 0:
        raise ValueError('Post-onset source decision completed before onset; check clocks')
    out.update(post_alarm=True, status='detected', detection_delay_s=delay,
               detection_raw_frame=int(raw[index]), detection_time=float(wall[index]),
               detected_before_human_label=sustained)
    return out


def apply_offline(evaluated, job, view, paper, causal, timeline, labels, spec):
    from benchmark_onset_results import assess, onset_frame
    from unified_benchmark import onset_time
    from benchmark_consensus_onsets import select_annotation, annotation_timestamp
    primary = spec['detection']['primary_annotator']
    for row in evaluated['episodes']:
        row['any_time_alarm'] = row['alarm']
        if row['outcome'] != 'failure':
            continue
        entries = labels['tasks'][job.task].get('failure_onsets', {}).get(row['episode_index'], {})
        row['onset_assessments'] = {aid: assess(paper[row['episode_index']], onset_frame(entries.get(aid), view))
                                    for aid in labels['annotators']}
        if uses_post_onset(spec):
            for aid, assessment in row['onset_assessments'].items():
                onset = assessment['onset_frame']
                annotation = select_annotation(entries.get(aid), view)
                t = annotation_timestamp(annotation, lambda: onset_time(job.dataset, row['episode_index'], onset, timeline)) if onset is not None else None
                result = alarm_run(paper[row['episode_index']], onset, t,
                    allow_anticipation=allows_anticipation(job.task, row['failure_origin']), first_post_onset=True)
                assessment.update({k: result[k] for k in ('early_alarm','any_pre_onset_alarm','sustained_pre_onset_alarm')})
        selected = row['onset_assessments'][primary]
        if selected['post_alarm'] is None:
            raise ValueError(f'Missing frozen primary onset: {job.task}/{row["episode_index"]}/{view}')
        row.update(alarm=selected['post_alarm'], early_alarm=selected['early_alarm'],
                   post_score_max=selected['post_score_max'], early_score_max=selected['early_score_max'],
                   detection_definition='at_or_after_onset', primary_annotator=primary)
        if uses_post_onset(spec):
            row.update({k:selected[k] for k in ('any_pre_onset_alarm','sustained_pre_onset_alarm')})
    for row in evaluated['latencies']:
        if row['onset_raw_frame'] is None:
            continue
        t = row.get('onset_wall_timestamp')
        if t is None:
            t = onset_time(job.dataset, row['episode_index'], row['onset_raw_frame'], timeline)
        result = alarm_run(causal[row['episode_index']], row['onset_raw_frame'], t,
                           allow_anticipation=allows_anticipation(job.task, row['failure_origin']),
                           first_post_onset=uses_post_onset(spec))
        row.update(result)
        row['evaluation_protocol'] = spec['protocol_id']


def update_recorded_online(data, root, resolve, spec):
    """Reinterpret available traces, without claiming new calibrated replay.

    Historical early-stopped traces remain censored. Offline persistence never
    supplies missing online decisions. Late decisions remain separate.
    """
    from benchmark_onset_results import RawClocks
    clocks = RawClocks(resolve)
    trials = list(data['online'])
    trials += [row for run in data.get('vlm_runs', [])
               for row in run['episodes'] if row['outcome'] == 'failure']
    for trial in trials:
        source = json.loads((Path(root) / (trial.get('path') or trial['trace_path'])).read_text())
        vlm = 'replay' in source
        replay = source['replay'] if vlm else source
        wall = clocks(source['dataset'] if vlm else source['trial']['dataset'], trial['episode_index'])
        for row in trial['latencies']:
            events = replay['events']
            view = row['view']
            onset = row.get('onset_raw_frame')
            t = row.get('onset_time_s')
            if t is None and onset is not None:
                t = float(wall[onset] - wall[0])
            curve = recorded_curve(events, view, trial['calibration'])
            result = alarm_run(curve, onset, t, allow_anticipation=allows_anticipation(trial['task'], trial['failure_origin']),
                               complete=replay['stop_reason'] == 'source_exhausted' and (not vlm or source['status'] == 'complete'),
                               first_post_onset=uses_post_onset(spec))
            row['historical_status'] = row['status']
            row['historical_late_alarm'] = row.get('late_alarm', False)
            late = result['detection_time'] is not None and result['detection_time'] > wall[-1] - wall[0]
            row.update(result, delay_s=result['detection_delay_s'], evaluation_protocol=spec['protocol_id'],
                       late_alarm=bool(late if uses_post_onset(spec) else row['historical_late_alarm'] or late))
            if uses_post_onset(spec):
                row.update(any_time_alarm=bool(replay['alarms'].get(view)), alarm=result['post_alarm'], onset_time_s=t)
                if vlm:
                    assessment = trial['onset_assessments'][row['annotator_id']]
                    assessment.update({k:result[k] for k in ('post_alarm','early_alarm','any_pre_onset_alarm','sustained_pre_onset_alarm')})
                    if row['annotator_id'] == spec['detection']['primary_annotator']:
                        trial.update(alarm=result['post_alarm'], late_alarm=bool(late))
            row['threshold_protocol'] = ('uncalibrated_binary_vlm' if vlm else
                source['trial'].get('evaluation_protocol', 'historical_uncorrected_replay'))
    if uses_post_onset(spec):
        from benchmark_vlm_results import summarize_post_onset
        for run in data.get('vlm_runs', []):
            summarize_post_onset(run, spec)


def recorded_curve(events, view, calibration):
    # Historical replay stops thresholding a view after its first alarm, but may
    # still log its actual score while processing the other view. A saved,
    # verified constant threshold can classify those observed scores. Never
    # forward-fill a time-varying threshold or invent a missing model score.
    key = view if any(view in e['thresholds'] for e in events) else calibration+'/'+view
    known = {e['thresholds'][key] for e in events if key in e['thresholds']}
    fallback = next(iter(known)) if len(known)==1 and calibration in ('episode','binary_anytime') else np.nan
    return {'raw': [e['raw_frame'] for e in events],
            'score': [e['scores'].get(view, np.nan)-e['thresholds'].get(key, fallback) for e in events],
            'wall': [e['alarm_ready_s'] for e in events]}


def export_interpreted(data, root):
    """Publish CPU-derived records separately from immutable replay artifacts."""
    import csv
    folder = Path(root)/data['evaluation_protocol'].get('report_output', 'analyses/2026-10-16_post_onset_v2')
    folder.mkdir(parents=True, exist_ok=True)
    offline = [dict(zip(data['latencies']['columns'], r)) for r in data['latencies']['rows']]
    online = [dict(r, trace_path=t['path'], failure_origin=t['failure_origin'], failure_mode=t['failure_mode'])
              for t in data['online'] for r in t['latencies']]
    vlm = [dict(r, trace_path=e['trace_path']) for run in data.get('vlm_runs', [])
           for e in run['episodes'] for r in e['latencies']]
    for name, rows in [('offline_latencies',offline),('online_latencies',online),('vlm_latencies',vlm),
                       ('vlm_summary',[r for run in data.get('vlm_runs',[]) for r in run['summary']+run['overall']])]:
        if any(r.get('detection_delay_s') is not None and r['detection_delay_s'] < 0 for r in rows):
            raise ValueError('v2 export contains negative latency')
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (folder/(name+'.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    metadata=dict(created_utc=data['created'],evaluation_protocol=data['evaluation_protocol']['protocol_id'],
                  inference=False,original_traces_preserved=True,online_trials=len(data['online']),
                  vlm_trials=sum(len(r['episodes']) for r in data.get('vlm_runs',[])),
                  source_online_runs=data['online_runs'],labels_sha256=data['onset_analysis']['labels_sha256'],
                  definition='First completed positive decision with source endpoint at/after onset. Other early alarms exclude only qualifying sustained pre-onset runs. No clipping, interpolation or imputation.',
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.csv')})
    if data['evaluation_protocol'].get('detection', {}).get('onset_reference'):
        from benchmark_consensus_onsets import missing_mean_replays
        metadata['onset_reference'] = data['evaluation_protocol']['detection']['onset_reference']
        metadata['offline_detected_without_replay'] = missing_mean_replays(data)
        metadata['consensus_code_sha256'] = hashlib.sha256(Path(__file__).with_name('benchmark_consensus_onsets.py').read_bytes()).hexdigest()
        (folder/'missing_replays.json').write_text(json.dumps(metadata['offline_detected_without_replay'], indent=2)+'\n')
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (folder/'report.md').write_text('# First-post-onset report revision\n\n'+metadata['definition']+'\n\n'
        'All metrics derive from saved scores/traces; no inference was rerun. Continuous-score replay is conditional on offline post-onset detection. '
        'Qwen evaluates all test episodes. Late alarms are separate from on-time latency.\n\n'
        '[Offline delays](offline_latencies.csv) · [Online delays](online_latencies.csv) · '
        '[Qwen delays](vlm_latencies.csv) · [Qwen summary](vlm_summary.csv) · [Provenance](metadata.json)\n')
    # Replace only the mutable current onset derivatives, not source archives.
    # The previous main report and these CSVs were backed up before migration.
    onset_folder = Path(root)/'analyses/onset_detection'
    onset_folder.mkdir(parents=True, exist_ok=True)
    for name in ('offline_latencies.csv','online_latencies.csv','vlm_latencies.csv'):
        (onset_folder/name).write_bytes((folder/name).read_bytes())
    (onset_folder/'metadata.json').write_text(json.dumps(dict(data['onset_analysis'],
        evaluation_protocol=metadata['evaluation_protocol'],derived_report=str(folder.relative_to(root)),
        latency_rule=metadata['definition']),indent=2)+'\n')
    episodes = [dict(zip(data['episodes']['columns'], r)) for r in data['episodes']['rows']]
    episodes += [r for run in data.get('vlm_runs',[]) for r in run['episodes']]
    assessments = [dict({k:e[k] for k in ('variant','task','method','view','calibration','episode_index')},
                       annotator_id=aid,**a) for e in episodes for aid,a in (e.get('onset_assessments') or {}).items()]
    with (onset_folder/'episode_assessments.csv').open('w') as stream:
        fields=list(dict.fromkeys(k for row in assessments for k in row))
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(assessments)
    return dict(path=str(folder.relative_to(root)),**metadata)
