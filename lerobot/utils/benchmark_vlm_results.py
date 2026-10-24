"""CPU-only ingestion of frozen VLM summaries for the shared results explorer."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from os.path import relpath

REPO = Path(__file__).resolve().parents[2]


NOTE = (
    'Qwen v9 is a Sentinel-inspired adaptation, not a full Sentinel reproduction. '
    'Each camera runs separately, using full observed history at approximately 1 FPS, '
    'greedy decoding, BF16/SDPA and a 256-token cap. All test successes and failures '
    'are replayed. Main-method FAR/FDR use any-time alarms, including decisions '
    'returned after episode end. On-time rates are retained separately. '
    'FDR includes pre-onset alarms; delay summaries '
    'require post-onset source evidence and an on-time decision. Binary decisions '
    'provide neither continuous-score AUROC nor a matched-5%-FAR sweep. '
    'These full-test streaming rates are separate from dense offline metrics and '
    'the other monitors’ conditional online replay. Prompt development inspected '
    'some episodes from this evaluation pool; this is not an untouched held-out VLM test.'
)


def episode_row(trial, result, group, variant, method='qwen_v9'):
    """Use the returned label, including late alarms; never invent a score."""
    if result['status'] != 'complete':
        raise ValueError('Main VLM comparison requires complete trials; retain partial runs separately')
    if group['split'] != 'test' or result['outcome'] != group['outcome']:
        raise ValueError('VLM result does not match frozen test labels')
    return {'variant':variant, 'method':method, 'task':trial['task'], 'view':trial['view'],
            'episode_index':trial['episode'], 'split':'test', 'outcome':group['outcome'],
            'failure_mode':group['failure_mode'], 'failure_origin':group['failure_origin'],
            'calibration':'binary_anytime', 'score_kind':'binary_decision',
            'alarm':bool(result.get('alarm')), 'late_alarm':result['late_alarm'],
            'pre_onset_alarm':result['pre_onset_alarm'], 'score_max':None,
            'n_supported_scores':None, 'trial_status':result['status']}


def load_episode_rows(plan, directory, variant, root=None, method='qwen_v9'):
    rows = []
    for trial in plan['trials']:
        groups = plan['labels']['tasks'][trial['task']]['groups']
        group = next(g for g in groups if g['episode_range'][0] <= trial['episode'] <= g['episode_range'][1])
        prefix = plan['config']['outputs']['per_task_root'].format(
            task=trial['task'], protocol_id=plan['config']['protocol_id'], run_id=directory.name)
        path = (REPO/prefix/(trial['id']+'.json')).resolve()
        if not path.is_relative_to(REPO/'fomofd/tasks'):
            raise ValueError('VLM trial evidence must remain under tasks')
        raw = path.read_bytes()
        result = json.loads(raw)
        if result.get('fingerprint') != plan['fingerprint'] or any(result[k] != trial[k] for k in ['id','model','task','view','episode']):
            raise ValueError(f'VLM trial identity mismatch: {path}')
        row = episode_row(trial, result, group, variant, method=method)
        row.update(any_time_alarm=bool(result.get('any_time_alarm', result.get('alarm'))),
                   trace_path=relpath(path, root if root is not None else directory.parent.parent),
                   trace_sha256=hashlib.sha256(raw).hexdigest())
        rows.append(row)
    return rows


def aggregate(rows):
    """Pool counts, retaining errors and population-SD delay conventions."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['model'], row['view']].append(row)
    result = []
    for (model, view), items in groups.items():
        total = {'model': model, 'task': 'Overall', 'view': view}
        for key in items[0]:
            if (key in {'planned', 'complete', 'indeterminate', 'unattempted', 'late_alarms',
                        'late_decisions', 'pre_onset_alarms', 'pre_onset_evidence',
                        'valid_misses', 'on_time_misses', 'annotated_failures',
                        'annotated_unknown', 'conditional_delay_n'}
                    or key.endswith(('_count', '_denominator', '_unknown', '_requests'))):
                total[key] = sum(r[key] for r in items)
        for metric in ['FAR', 'FDR']:
            n, unknown = total[metric+'_denominator'], total[metric+'_unknown']
            for timing in ['any', 'on_time']:
                total[f'{metric}_{timing}'] = total[f'{metric}_{timing}_count']/n if n and not unknown else None
            total[metric+'_any_lower_bound'] = total[metric+'_any_count']/n if n else None
            total[metric+'_any_upper_bound'] = (total[metric+'_any_count']+unknown)/n if n else None
        for key in items[0]:
            if key.startswith('detected_by_') and key.endswith('_rate'):
                stem = key[:-5]
                n = total[stem+'_denominator']
                total[key] = total[stem+'_count']/n if n and not total['annotated_unknown'] else None
        n = total['conditional_delay_n']
        measured = [r for r in items if r['conditional_delay_n']]
        mean = sum(r['conditional_delay_n']*r['conditional_delay_mean_s'] for r in measured)/n if n else None
        variance = sum(r['conditional_delay_n']*(r['conditional_delay_sd_s']**2
                       +(r['conditional_delay_mean_s']-mean)**2) for r in measured)/n if n else None
        total['conditional_delay_mean_s'] = mean
        total['conditional_delay_sd_s'] = math.sqrt(variance) if n else None
        result.append(total)
    return result


def refresh_trial_onset(result, labels, annotator, wall):
    """Re-annotate a saved decision without changing its label or completion time."""
    from unified_benchmark import annotation_views
    onset = None
    if result['outcome'] == 'failure':
        entry = labels['tasks'][result['task']].get('failure_onsets', {}).get(result['episode'], {}).get(annotator)
        annotation = annotation_views(entry).get(result['view'])
        if annotation:
            frame = annotation['frame_index']
            if type(frame) is not int or not 0 <= frame < len(wall):
                raise ValueError('VLM onset outside raw recording')
            from benchmark_consensus_onsets import elapsed_onset
            onset = elapsed_onset(annotation, wall)
    alarm = result.get('alarm')
    delay = alarm['alarm_time_s'] - onset if alarm and onset is not None else None
    return {**result, 'onset_s': onset,
            'pre_onset_alarm': bool(delay is not None and delay < 0),
            'pre_onset_evidence': bool(alarm and onset is not None and alarm['source_time_s'] < onset),
            'post_onset_evidence_delay_s': delay if delay is not None and delay >= 0 and alarm['source_time_s'] >= onset else None}


def refresh_onset_summary(run, results):
    """Keep frozen anytime/on-time counts; update only onset-dependent metrics."""
    import numpy as np
    for row in run['summary']:
        failures = [r for r in results if r['outcome'] == 'failure'
                    and (r['model'], r['task'], r['view']) == (row['model'], row['task'], row['view'])]
        annotated = [r for r in failures if r['onset_s'] is not None]
        delays = [r['post_onset_evidence_delay_s'] for r in annotated
                  if r['post_onset_evidence_delay_s'] is not None and not r['late_alarm']]
        row.update(pre_onset_alarms=sum(r['pre_onset_alarm'] for r in failures),
                   pre_onset_evidence=sum(r['pre_onset_evidence'] for r in failures),
                   annotated_failures=len(annotated),
                   annotated_unknown=sum(r['status'] != 'complete' for r in annotated),
                   conditional_delay_n=len(delays),
                   conditional_delay_mean_s=float(np.mean(delays)) if delays else None,
                   conditional_delay_sd_s=float(np.std(delays)) if delays else None)
        for key in list(row):
            if key.startswith('detected_by_') and key.endswith('s_count'):
                deadline = float(key[len('detected_by_'):-len('s_count')])
                stem = key[:-len('_count')]
                count = sum(d <= deadline for d in delays)
                row.update({key: count, stem+'_denominator': len(annotated),
                            stem+'_rate': count/len(annotated) if annotated and not row['annotated_unknown'] else None})
    run['overall'] = aggregate(run['summary'])
    run['onset_annotation_source'] = 'current prl_episode_labels.yaml; snapshot in analyses/onset_detection'
    run['note'] += ' Onset-dependent summaries use the current annotation snapshot; original decisions and source reports remain preserved.'


def latency_rows(result, labels, wall, variant):
    """Saved first decision, per annotator; lateness is an explicit separate flag."""
    from online_benchmark import latency_record
    from unified_benchmark import annotation_views
    entries = labels['tasks'][result['task']].get('failure_onsets', {}).get(result['episode'], {})
    rows = []
    for aid, annotator in labels['annotators'].items():
        annotation = annotation_views(entries.get(aid)).get(result['view'], {})
        record = latency_record(result.get('alarm'), annotation.get('frame_index'), wall)
        from benchmark_consensus_onsets import elapsed_onset
        record['onset_time_s'] = elapsed_onset(annotation, wall)
        rows.append({**record, 'variant': variant, 'method': 'qwen_v9', 'task': result['task'],
                     'view': result['view'], 'onset_view': result['view'], 'episode_index': result['episode'],
                     'annotator_id': aid, 'annotator': annotator['name'], 'calibration': 'binary_anytime',
                     'onset_raw_frame': annotation.get('frame_index'),
                     'late_alarm': bool(result['late_alarm']), 'alarm': bool(result.get('alarm')),
                     'copied_from_annotator': annotation.get('copied_from_annotator')})
    return rows


def load_runs(entries, root):
    runs = []
    root = root.resolve()
    for entry in entries:
        directory = (root/entry['path']).resolve()
        if not directory.is_relative_to(root):
            raise ValueError('VLM evidence must be under benchmark_results')
        for name, expected in entry['sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'VLM report changed: {directory/name}')
        report = json.loads((directory/'report.json').read_text())
        plan = json.loads((directory/'plan.json').read_text())
        if report['fingerprint'] != plan['fingerprint'] or report['protocol_id'] != plan['config']['protocol_id']:
            raise ValueError('VLM plan/report identity mismatch')
        rows = report['summary']
        planned = defaultdict(int)
        ids = set()
        for trial in plan['trials']:
            if trial['id'] in ids:
                raise ValueError('Duplicate VLM trial')
            ids.add(trial['id'])
            planned[trial['model'], trial['task'], trial['view']] += 1
        reported = set()
        for row in rows:
            key = row['model'], row['task'], row['view']
            if key in reported or row['planned'] != planned[key]:
                raise ValueError('VLM planned denominator mismatch')
            reported.add(key)
            if row['complete']+row['indeterminate']+row['unattempted'] != row['planned']:
                raise ValueError('VLM errors omitted from denominator')
            if row['FAR_denominator']+row['FDR_denominator'] != row['planned']:
                raise ValueError('VLM outcome denominator mismatch')
        if reported != set(planned):
            raise ValueError('Missing VLM summary group')
        episodes = load_episode_rows(plan, directory, 'vlm_'+entry['id'], root, method=entry['id'])
        for summary in rows:
            selected = [r for r in episodes if (r['task'],r['view']) == (summary['task'],summary['view'])]
            for outcome, metric in [('success','FAR'), ('failure','FDR')]:
                subset = [r for r in selected if r['outcome']==outcome]
                if (len(subset) != summary[metric+'_denominator']
                        or sum(r['alarm'] for r in subset) != summary[metric+'_any_count']
                        or sum(r['alarm'] and not r['late_alarm'] for r in subset) != summary[metric+'_on_time_count']):
                    raise ValueError('VLM trial alarms disagree with frozen summary')
        runs.append({**entry, 'protocol_id': report['protocol_id'], 'fingerprint':report['fingerprint'],
                     'source_evaluation_protocol': report.get('evaluation_protocol'),
                     'source_note': report.get('note'),
                     'primary_annotator_id': plan['config']['evaluation']['primary_annotator_id'],
                     'episodes':episodes,
                     'summary':rows, 'overall':aggregate(rows), 'note':NOTE,
                     'planned':len(ids), 'complete':sum(r['complete'] for r in rows),
                     'indeterminate':sum(r['indeterminate'] for r in rows),
                     'unattempted':sum(r['unattempted'] for r in rows)})
    return runs


def summarize_post_onset(run, spec):
    """Derive current onset metrics from observed traces; preserve source summaries."""
    from copy import deepcopy
    import numpy as np
    run.setdefault('source_summary', deepcopy(run['summary']))
    primary = spec['detection']['primary_annotator']
    for row in run['summary']:
        episodes = [r for r in run['episodes'] if (r['task'],r['view']) == (row['task'],row['view'])]
        failures = [r for r in episodes if r['outcome']=='failure']
        records = [next(r for r in ep['latencies'] if r['annotator_id']==primary) for ep in failures]
        delays = [r['delay_s'] for r in records if r['post_alarm'] and not r['late_alarm']]
        detected = sum(r['post_alarm'] is True for r in records)
        ontime = len(delays)
        unknown = sum(r['post_alarm'] is None for r in records)
        n = len(failures)
        row.update(FDR_any_count=detected, FDR_on_time_count=ontime, FDR_unknown=unknown,
            FDR_any=detected/n if n and not unknown else None,
            FDR_on_time=ontime/n if n and not unknown else None,
            FDR_any_lower_bound=detected/n if n else None,
            FDR_any_upper_bound=(detected+unknown)/n if n else None,
            FDR_anytime_raw_count=sum(bool(r['any_time_alarm']) for r in failures),
            valid_misses=sum(r['post_alarm'] is False for r in records), on_time_misses=n-unknown-ontime,
            pre_onset_alarms=sum(bool(r['early_alarm']) for r in records),
            pre_onset_evidence=sum(bool(r['any_pre_onset_alarm']) for r in records),
            sustained_pre_onset_count=sum(r['sustained_pre_onset_alarm'] for r in records),
            late_alarms=sum(bool(r['late_alarm']) for r in episodes),
            annotated_failures=sum(r['onset_raw_frame'] is not None for r in records),
            annotated_unknown=unknown, conditional_delay_n=len(delays),
            conditional_delay_mean_s=float(np.mean(delays)) if delays else None,
            conditional_delay_sd_s=float(np.std(delays)) if delays else None)
        for key in list(row):
            if key.startswith('detected_by_') and key.endswith('s_count'):
                deadline=float(key[len('detected_by_'):-len('s_count')]);stem=key[:-len('_count')]
                count=sum(d<=deadline for d in delays)
                row.update({key:count,stem+'_denominator':n,stem+'_rate':count/n if n and not unknown else None})
    run['overall']=aggregate(run['summary'])
    run['evaluation_protocol']=spec['protocol_id']
    run['note']=('Qwen v9, Sentinel-inspired adaptation: separate cameras, full observed history at 1 FPS, '
        'greedy BF16/SDPA, 256-token cap. All 800 test episode/view trials replayed through source end. '
        'FDR requires post-onset source evidence; incl-late and on-time rates are separate. FAR uses any alarm on successes. '
        'Delays use first completed post-onset decisions; sustained pre-onset and other early alarms are separate. '
        'No continuous AUROC or matched-FAR sweep. Prompt development inspected some test episodes. '
        f'Source summaries and raw outputs are preserved; saved traces are reinterpreted under {spec["protocol_id"]}.')
