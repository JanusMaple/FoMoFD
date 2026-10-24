"""Compare temporal aggregation orders using immutable cached component margins.

Both levels use common raw endpoints and frozen component thresholds/scales.
Endpoint Q=max_t op(a_t,b_t); episode Q=op(max_t a_t,max_t b_t).
Failure decisions restrict t to the frozen output-view post-onset interval.
No new calibration, inference, online replay or latency is performed.
"""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json
import math
import statistics

import numpy as np
import yaml

from benchmark_onset_results import onset_frame
from benchmark_roc import curve, operating_point, safe

ANALYSIS_ID = 'fusion_aggregation_levels_v1'
RELATIVE_PATH = 'analyses/fusion_aggregation_levels_v1'
DEFINITIONS = [
    ('wm_nc_or', 'both', ('wm_nc', 'fix'), ('wm_nc', 'wrist'), 'or', 'Fixed OR wrist'),
    ('wm_nc_and', 'both', ('wm_nc', 'fix'), ('wm_nc', 'wrist'), 'and', 'Fixed AND wrist'),
    *[(f'wm_nc_logpzo_native_{op}', view, ('wm_nc', view), ('logpzo_native', view), op,
       'WM-NC '+op.upper()+' native logpZO') for view in ('fix','wrist') for op in ('or','and')],
]
NOTE = ('Both levels use the same common eligible raw endpoints and frozen component thresholds/scales. '
        'Endpoint OR/AND takes the maximum/minimum margin at each endpoint, then the episode maximum. '
        'Episode OR/AND combines each component’s episode maximum; AND does not require simultaneous detections. '
        'FDR restricts both components to endpoints at/after the output-view onset (earlier camera onset for view fusion). '
        'Whole-episode maxima determine FAR and AUROC. OR is identical between levels on this shared support; '
        'AND can differ. Native-only endpoints outside WM support remain excluded. '
        'No joint recalibration or unchanged-alpha OR guarantee is claimed. These are offline comparisons; '
        'episode aggregation is not a newly replayed online detector and has no measured latency.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combine_peak(left, right, operation, level):
    if operation not in ('or','and') or level not in ('endpoint','episode'):
        raise ValueError('Unknown fusion rule')
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Fusion needs finite, aligned component margins')
    if not len(a):
        return -math.inf
    op = np.maximum if operation == 'or' else np.minimum
    return float(np.max(op(a, b)) if level == 'endpoint' else op(a.max(), b.max()))


def aligned(left, right):
    for c in (left, right):
        if len(np.unique(c['raw'])) != len(c['raw']):
            raise ValueError('Duplicate raw endpoints')
    raw, a, b = np.intersect1d(left['raw'], right['raw'], return_indices=True)
    if not len(raw) or not np.allclose(left['wall'][a], right['wall'][b], atol=1e-6, rtol=0):
        raise ValueError('Common raw endpoints / wall timestamps disagree')
    return raw, left['score'][a], right['score'][b]


def summarize(rows, primary):
    groups = defaultdict(list)
    for r in rows:
        if r['split'] == 'test':
            assessment = r['onset_assessments'].get(primary, {})
            groups[r['method'],r['view'],r['level'],r['task']].append({**r,
                'alarm': assessment.get('post_alarm') if r['outcome']=='failure' else r['any_time_alarm'],
                'post_score_max': assessment.get('post_score_max')})
    result = []
    for (method,view,level,task), items in sorted(groups.items()):
        neg = [r for r in items if r['outcome']=='success']
        pos = [r for r in items if r['outcome']=='failure']
        subsets = [('all','All failures',pos)] + [(o,o.title(),[r for r in pos if r['failure_origin']==o])
                  for o in ('natural','induced')]
        subsets += [('failure_mode',mode,[r for r in pos if r['failure_mode']==mode]) for mode in sorted({r['failure_mode'] for r in pos})]
        for kind, name, failures in subsets:
            if not failures:
                continue
            selected = neg + failures
            _, auc = curve(selected)
            unknown = sum(r['alarm'] is None for r in failures)
            matched = operating_point(selected,.05) if not unknown else {}
            fa, tp = sum(r['alarm'] is True for r in neg), sum(r['alarm'] is True for r in failures)
            result.append(dict(method=method, view=view, level=level, task=task, label=items[0]['label'],
                subset=kind, failure_mode=name, annotator_id=primary, success_n=len(neg), false_alarm_n=fa,
                failure_n=len(failures), detected_n=tp, unknown_n=unknown,
                FAR=fa/len(neg) if neg else None, FDR=tp/len(failures) if not unknown else None,
                episode_AUROC=auc, matched_FAR=matched.get('FAR'), matched_FDR=matched.get('FDR'),
                matched_threshold=matched.get('threshold'), matched_false_alarm_n=matched.get('false_alarms'),
                matched_detected_n=matched.get('detected_failures')))
    totals = defaultdict(list)
    for row in result:
        if row['subset'] != 'failure_mode':
            totals[row['method'],row['view'],row['level'],row['subset']].append(row)
    for items in totals.values():
        total = {**items[0], 'task':'Overall', 'matched_threshold':None}
        for name in ('success_n','false_alarm_n','failure_n','detected_n','unknown_n'):
            total[name] = sum(r[name] for r in items)
        for name in ('matched_false_alarm_n','matched_detected_n'):
            total[name] = sum(r[name] for r in items) if all(r[name] is not None for r in items) else None
        total.update(FAR=total['false_alarm_n']/total['success_n'],
            FDR=total['detected_n']/total['failure_n'] if not total['unknown_n'] else None,
            episode_AUROC=statistics.mean(r['episode_AUROC'] for r in items if r['episode_AUROC'] is not None),
            matched_FAR=total['matched_false_alarm_n']/total['success_n'] if total['matched_false_alarm_n'] is not None else None,
            matched_FDR=total['matched_detected_n']/total['failure_n'] if total['matched_detected_n'] is not None else None)
        result.append(total)
    return result


def build(data, root, resolve):
    from unified_benchmark import curves, episode_labels
    root = Path(root)
    run = next(r for r in data['runs'] if r['id']=='main')
    folder = root/run['path']
    report_path = folder/'report.json'
    report = json.loads(report_path.read_text())
    if digest(report_path) != run['archive_sha256']:
        raise ValueError('Current main report hash changed')
    spec = data['evaluation_protocol']
    label_path = resolve(spec['frozen_inputs']['onsets']['path'])
    if digest(label_path) != spec['frozen_inputs']['onsets']['sha256']:
        raise ValueError('Frozen onsets changed')
    labels = yaml.safe_load(label_path.read_text())
    from benchmark_consensus_onsets import derive_labels
    labels = derive_labels(labels, spec, resolve)
    streams, sources, calibration = {}, {str(report_path):digest(report_path)}, []
    for cache in report['score_caches']:
        if cache['method'] not in ('wm_nc','logpzo_native'):
            continue
        path = resolve(cache['path'])
        if digest(path) != cache['sha256']:
            raise ValueError(f'Changed component score cache: {path}')
        sources[str(path)] = cache['sha256']
        with np.load(path, allow_pickle=False) as z:
            arrays = {k:z[k] for k in ('episode_index','frame_index','raw_frame_index','wall_timestamp','scores')}
        job = SimpleNamespace(task=cache['task'],mode=cache['method']=='wm_nc')
        for i, view in enumerate(cache['views']):
            records = [r for r in report['calibration'] if (r['task'],r['method'],r['view'],r['calibration'],r['protocol']) ==
                       (cache['task'],cache['method'],view,'episode','paper')]
            if len(records)!=1 or not math.isfinite(float(records[0]['threshold'])) or records[0]['scale']<=0:
                raise ValueError('Expected one finite, frozen component calibration')
            fit = records[0]; calibration.append(fit)
            source = curves(arrays,job,i,report['protocols']['paper_protocol'])
            streams[cache['task'],cache['method'],view] = {e:{**c,'score':(c['score']-fit['threshold'])/fit['scale']} for e,c in source.items()}
    original = {tuple(r[k] for k in ('task','method','view','episode_index')):r
                for r in (dict(zip(data['episodes']['columns'],v)) for v in data['episodes']['rows'])
                if r['variant']=='main' and r['calibration']=='episode'}
    primary = spec['detection']['primary_annotator']
    rows, verified = [], 0
    for task, task_labels in labels['tasks'].items():
        lookup = episode_labels(task_labels)
        for method, view, left, right, operation, label in DEFINITIONS:
            lhs, rhs = streams[(task,*left)], streams[(task,*right)]
            if set(lhs)!=set(rhs) or set(lhs)!=set(lookup):
                raise ValueError('Component episode populations differ')
            for ep in sorted(lhs):
                raw,a,b = aligned(lhs[ep],rhs[ep])
                annotation = task_labels.get('failure_onsets',{}).get(ep,{})
                for level in ('endpoint','episode'):
                    maximum = combine_peak(a,b,operation,level)
                    assessments = {}
                    for aid in labels['annotators']:
                        onset = onset_frame(annotation.get(aid),view) if lookup[ep]['outcome']=='failure' else None
                        post = None if onset is None else combine_peak(a[raw>=onset],b[raw>=onset],operation,level)
                        assessments[aid] = dict(onset_frame=onset,post_score_max=post,
                            post_alarm=None if post is None else post>0)
                    row = dict(analysis_id=ANALYSIS_ID,variant='main',task=task,method=method,view=view,
                        level=level,operation=operation,label=label,calibration='episode',episode_index=ep,**lookup[ep],
                        score_max=maximum,any_time_alarm=maximum>0,n_supported_scores=len(raw),
                        left_max=float(a.max()),right_max=float(b.max()),onset_assessments=assessments,
                        detection_definition='at_or_after_onset',left_component='/'.join(left),right_component='/'.join(right))
                    if level=='endpoint':
                        old = original[task,method,view,ep]
                        old_alarm = old['any_time_alarm'] if old.get('any_time_alarm') is not None else old['alarm']
                        if (bool(maximum>0)!=old_alarm or not math.isclose(maximum,float(old['score_max']),rel_tol=1e-7,abs_tol=1e-9)
                                or (lookup[ep]['outcome']=='failure' and assessments[primary]['post_alarm']!=old['alarm'])):
                            raise ValueError(f'Endpoint reconstruction changed: {task}/{method}/{view}/{ep}')
                        if lookup[ep]['outcome']=='failure':
                            expected = old['onset_assessments'][primary]['post_score_max']
                            actual = assessments[primary]['post_score_max']
                            if (expected is None) != (actual is None) or (actual is not None and
                                    not math.isclose(actual,float(expected),rel_tol=1e-7,abs_tol=1e-9)):
                                raise ValueError('Post-onset endpoint score reconstruction differs')
                        verified += 1
                    rows.append(row)
                endpoint, episode = rows[-2:]
                if operation=='or' and endpoint['score_max']!=episode['score_max']:
                    raise ValueError('OR equivalence failed')
                if endpoint['score_max']>episode['score_max']:
                    raise ValueError('Episode AND dominance failed')
                for aid in labels['annotators']:
                    a_post,b_post = [r['onset_assessments'][aid]['post_score_max'] for r in (endpoint,episode)]
                    if a_post is not None and (a_post>b_post or (operation=='or' and a_post!=b_post)):
                        raise ValueError('Post-onset fusion order invariant failed')
    metrics = summarize(rows,primary)
    result = dict(analysis_id=ANALYSIS_ID,evaluation_protocol=spec['protocol_id'],note=NOTE,
        primary_annotator=primary,source_snapshot_utc=data['created'],source_hashes=sources,
        source_protocol=report['protocols']['paper_protocol'],component_calibration=calibration,
        labels_sha256=digest(label_path),script_sha256=digest(__file__),
        verification={'endpoint_episode_rows_matched':verified,'OR_equivalence':'passed','AND_dominance':'passed'},
        rows=safe(rows),metrics=safe(metrics),path=RELATIVE_PATH,online_latency='not_measured')
    output = root/RELATIVE_PATH; output.mkdir(parents=True,exist_ok=True)
    for name, table in [('metrics',metrics),('component_calibration',calibration),('episode_scores',[
            {**{k:v for k,v in r.items() if k!='onset_assessments'},'annotator_id':primary,
             **r['onset_assessments'][primary]} for r in rows])]:
        with (output/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(safe(table))
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# Endpoint versus episode alarm fusion','',NOTE,'',
           'Normalized margins are (score − frozen threshold) / frozen positive scale; alarm threshold is zero. '
           'Episode-level AND is agreement within the eligible episode interval, not simultaneous confirmation. '
           'This is a separately identified supplemental analysis; frozen primary results remain unchanged.','',
           '| Monitor | View | Level | False alarms / successes | Detections / failures | AUROC |',
           '|---|---|---|---|---|---|']
    for r in metrics:
        if r['task']=='Overall' and r['subset']=='all':
            lines.append(f"| {r['label']} | {r['view']} | {r['level']} | {r['false_alarm_n']}/{r['success_n']} | {r['detected_n']}/{r['failure_n']} | {r['episode_AUROC']:.3f} |")
    lines+=['','[Per-task/per-mode metrics](metrics.csv) · [Both levels’ episode scores](episode_scores.csv) · [Frozen component calibration](component_calibration.csv)','']
    (output/'report.md').write_text('\n'.join(lines))
    return result
