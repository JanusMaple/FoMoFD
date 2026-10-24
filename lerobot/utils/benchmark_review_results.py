"""CPU-only reviewer evidence derived from verified reports and recorded clocks.

This module does not score models, change thresholds, or alter archived results.
It supplements the concise report with explicit denominators and coverage gaps.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import html
import json
import math

import numpy as np
import yaml

TASKS = {'needle_pickup': 'Needle', 'ring_over_post': 'Ring',
         'tissue_retraction': 'Tissue', 'shunt_insertion': 'Shunt'}
METHODS = {'wm_nc': 'WM-NC', 'wm_pe': 'WM-PE', 'wm_pu': 'WM-PU',
           'logpzo': 'logpZO VAE', 'rnd': 'RND VAE', 'logpzo_native': 'logpZO policy',
           'rnd_native': 'RND policy', 'ace': 'ACE',
           'act_ensemble_pos': 'AEU', 'act_ensemble_rot': 'AEU-Rot'}


def unpack(value):
    return [dict(zip(value['columns'], row)) for row in value['rows']]


def wilson(k, n):
    """Two-sided 95% Wilson interval, conditional on fixed models/thresholds."""
    if not n:
        return None
    if not 0 <= k <= n:
        raise ValueError('Invalid binomial counts')
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., center-half), min(1., center+half)]


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'n': 0, 'mean': None, 'sd': None, 'median': None, 'p05': None,
                'p95': None, 'min': None, 'max': None}
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite measurement')
    return {'n': len(a), 'mean': float(a.mean()), 'sd': float(a.std(ddof=1)) if len(a)>1 else None,
            'median': float(np.median(a)), 'p05': float(np.quantile(a,.05)),
            'p95': float(np.quantile(a,.95)), 'min': float(a.min()), 'max': float(a.max())}


def latency_counts(records):
    """Early evidence, alarm completion and lateness are separate conditions."""
    ontime = [r for r in records if r['status']=='detected' and not r.get('late_alarm', False)]
    delays = [r['delay_s'] for r in ontime]
    if any(r['delay_s'] is None or (r['delay_s'] < 0 and not r.get('detected_before_human_label')) for r in ontime):
        raise ValueError('Detected record lacks a valid onset delay or verified spanning early run')
    return {'selected': len(records), 'alarms': sum(bool(r.get('alarm')) for r in records),
            'early_evidence': sum(bool(r.get('early_alarm',r['status']=='pre_onset_evidence')) for r in records),
            'early_completion': sum(r.get('signed_alarm_delay_s') is not None and r['signed_alarm_delay_s']<0 for r in records),
            'no_alarm': sum(not r.get('alarm') for r in records),
            'sustained_before_human': sum(bool(r.get('detected_before_human_label')) and not r.get('late_alarm',False) for r in records),
            'censored': sum(r['status']=='unobserved_after_stop' for r in records),
            'missing_onset': sum(r.get('onset_raw_frame') is None for r in records),
            'late_alarms': sum(bool(r.get('late_alarm')) and bool(r.get('alarm')) for r in records),
            'delay_s': describe(delays),
            'within_deadline': {str(s): sum(v<=s for v in delays) for s in (1,2,5,10)}}


def display(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def mean_sd(stat, scale=1):
    return '—' if not stat['n'] else f"{display(stat['mean']*scale)} ± {display(None if stat['sd'] is None else stat['sd']*scale)}"


def verify_population_counts(audit, episodes, online):
    """A saved population explanation must match the currently displayed groups."""
    for row in audit['rows']:
        task, view = row['task'], row['view']
        paper = {r['episode_index'] for r in episodes
                 if (r['variant'], r['method'], r['calibration'], r['task'], r['view'], r['outcome'])
                 == ('main', 'wm_nc', 'episode', task, view, 'failure') and r['alarm']}
        replay = {r['episode_index'] for r in online
                  if (r['variant'], r['method'], r['calibration'], r['task'])
                  == ('main', 'wm_nc', 'episode', task) and view in r['requested_detector_views']}
        if (len(paper) != row['paper_detected'] or len(replay) != row['completed_replay']
                or paper-replay != set(row['paper_only_ids'])
                or replay-paper != set(row['replay_only_ids'])
                or paper & replay != set(row['intersection_ids'])):
            raise ValueError('Population reconciliation is stale; re-audit the exact episode sets')


def rate(k, n, interval=False):
    if not n:
        return '—'
    s = f'{k}/{n} ({100*k/n:.1f}%)'
    if interval:
        lo, hi = wilson(k,n)
        s += f' [{100*lo:.1f}, {100*hi:.1f}]'
    return s


def sections_for(data, root, config, resolve):
    import benchmark_roc as roc
    from benchmark_onset_results import RawClocks
    root = Path(root)
    sections = []
    def add(id, title, note, columns, rows, records=None, figures=()):
        sections.append({'id': id, 'title': title, 'note': note, 'columns': columns,
                         'rows': rows, 'records': records if records is not None else rows,
                         'figures': list(figures)})
    episodes = unpack(data['episodes'])
    test = [r for r in episodes if r['split']=='test']
    sources = {m:data.get('summary_sources',{}).get(m,'main') for m in METHODS}
    main = [r for r in test if r['method'] in METHODS and r['variant']==sources[r['method']]
            and r['method'] not in data.get('supplementary_methods', [])
            and r['calibration'] in ('episode','functional_band')]
    aid = next(iter(data['annotators']))
    titles = {r['id']:r['title'] for r in data['runs']}
    reference = config.get('population_reconciliation')
    if data.get('offline_protocol'):
        selected = [r for r in test if r['variant']=='main' and r['method']=='wm_nc' and r['calibration']=='episode' and r['outcome']=='failure']
        rows=[]
        for task in TASKS:
            for view in ('fix','wrist'):
                current={r['episode_index'] for r in selected if r['task']==task and r['view']==view and r['alarm']}
                replay={r['episode_index'] for r in data['online'] if r['variant']=='main' and r['method']=='wm_nc' and r['calibration']=='episode' and r['task']==task and view in r['requested_detector_views']}
                rows.append([TASKS[task],view,len(current),len(replay),len(current-replay),len(replay-current)])
        add('populations','Current offline versus recorded replay populations',
            data['offline_protocol']['note']+' '+data['offline_protocol']['online_note']+
            ' Offline-only episodes have no measured replay latency. Current main-method replay uses corrected thresholds; historical ablation timings are excluded. The frozen crop is a data-quality decision and was inspected after prior results.',
            ['Task','View','Offline detected','Recorded replay population','Offline-only','Replay-only'],rows)
    elif reference:
        path = root/reference['path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference['sha256']:
            raise ValueError('Population audit changed')
        audit = json.loads(raw)
        if hashlib.sha256((root/audit['main_report']/'report.json').read_bytes()).hexdigest() != audit['main_report_sha256']:
            raise ValueError('Population audit refers to a different offline report')
        verify_population_counts(audit, test, data['online'])
        add('populations', 'Offline versus replay population reconciliation',
            'These are different endpoint/calibration protocols, not missing runs or different checkpoints. '
            'Paper scores omit the last K−1 endpoints (three at K=4); replay uses all endpoints and their separately calibrated thresholds. '
            'Needle-wrist calibration episode 5 raises the threshold from 923.4774 (frame 125) to 1059.8646 (frame 127), '
            'excluding 28 paper-detected failures. Fixed thresholds are unchanged; the additional tail scores add needle 136, '
            'ring 134/141/143 and shunt 62. All 424 selected detector-view replays are complete. '
            'Wrist: 302 paper detections versus 274 replays = 265 post-onset + 9 early-evidence alarms. '
            'Fixed: 145 paper detections versus 150 replays = 138 post-onset + 12 early-evidence alarms. '
            'No late alarms or online misses occurred in these selected WM-NC trials. '
            'The 1.0479 s wrist mean does not measure latency at the paper table’s 94.375% FDR operating point. '
            'Matching all-endpoint offline FDR is 85.625% wrist / 46.875% fixed, with FAR 2.5% / 0%; '
            'these remain dense offline rates, not full-test online measurements. '
            'Exact IDs, source hashes and threshold drivers: analyses/2026-10-15/wm_nc_population_reconciliation/report.md and audit.json.',
            ['Task', 'View', 'Paper detected / failures', 'All-endpoint detected / failures',
             'Completed / selected replay', 'Paper threshold', 'Replay threshold'],
            [[TASKS[r['task']], r['view'], f"{r['paper_detected']}/{r['failure_total']}",
              f"{r['all_endpoint_detected']}/{r['failure_total']}",
              f"{r['completed_replay']}/{r['selected_replay']}",
              display(r['thresholds']['paper'],4),display(r['thresholds']['latency'],4)] for r in audit['rows']], audit)
    def detector(m,v):
        return METHODS.get(m,m)+('' if v=='both' and m in ('ace','act_ensemble_pos','act_ensemble_rot') else ' '+v)
    if data.get('failure_categories'):
        from benchmark_failure_categories import format_rate
        category_data = data['failure_categories']
        records = [r for r in category_data['main_records'] if r['calibration'] != 'wall_time_bin']
        add('failure_categories', 'Failure-category FDR', category_data['definition'],
            ['Category', 'Detector', 'Calibration', 'Measurement', 'Detected / total (FDR)', 'Late detections'],
            [[r['category_title'],detector(r['method'],r['view']),r['calibration'],r['measurement'],
              format_rate(r),r['late_detections']] for r in records], records)
    metrics = roc.analyze(main, targets=(.05,), subsets=('all',))
    mkey = lambda r: tuple(r[k] for k in ('variant','task','method','view','calibration'))
    matched = {mkey(r):r for r in metrics['matched_far']}
    groups = defaultdict(list)
    for r in main:
        groups[mkey(r)].append(r)
        groups[(r['variant'],'Overall',r['method'],r['view'],r['calibration'])].append(r)
    records=[]
    for key, rows in sorted(groups.items()):
        neg=[r for r in rows if r['outcome']=='success']; pos=[r for r in rows if r['outcome']=='failure']
        fp=sum(r['alarm'] for r in neg); tp=sum(r['alarm'] for r in pos); point=matched[key]
        records.append(dict(zip(('variant','task','method','view','calibration'),key)) |
                       {'false_alarms':fp,'successes':len(neg),'detections':tp,'failures':len(pos),
                        'FAR':fp/len(neg),'FDR':tp/len(pos),'FAR_CI95':wilson(fp,len(neg)),
                        'FDR_CI95':wilson(tp,len(pos)), 'AUROC':point['episode_AUROC'],
                        'matched_FAR':point['FAR'],'matched_FDR':point['FDR']})
    add('uncertainty','Main performance with uncertainty and matched FAR',
        'Post-onset episode detection. Brackets are descriptive 95% Wilson intervals for fixed thresholds and curated episodes; pooled intervals do not account for task heterogeneity or calibration/training variation. AUROC is task-macro for Overall. Matched FAR uses retrospective per-task success-test thresholds, not deployment calibration. No confidence interval for AUROC is inferred.',
        ['Task','Detector','Protocol','FAR [95% CI]','FDR [95% CI]','AUROC','Matched FAR% / FDR%'],
        [[TASKS.get(r['task'],r['task']),detector(r['method'],r['view']),r['calibration'],
          rate(r['false_alarms'],r['successes'],True),rate(r['detections'],r['failures'],True),display(r['AUROC'],3),
          f"{display(100*r['matched_FAR'],1)} / {display(100*r['matched_FDR'],1)}"] for r in records],records)

    # All measured baseline/ablation latencies, explicit selection and censoring.
    families={r['id']:r['family'] for r in data['runs']}
    lat=defaultdict(list)
    for trial in data['online']:
        if trial['calibration']!='episode':continue
        for row in trial['latencies']:
            if row['annotator_id']!=aid:continue
            v,m=trial['variant'],trial['method']
            if not (v==sources.get(m) or families.get(v) in ('K','L','action','view_fusion') or v=='main'):continue
            lat[(v,m,row['view'],'Overall','All failures')].append(row)
            if v=='main' and m=='wm_nc':
                lat[(v,m,row['view'],trial['task'],'All failures')].append(row)
                lat[(v,m,row['view'],trial['task'],trial['failure_mode'])].append(row)
    all_failures={}
    for r in test:
        if r['variant']=='main' and r['method']=='wm_nc' and r['calibration']=='episode' and r['view']=='wrist' and r['outcome']=='failure':
            for key in [('Overall','All failures'),(r['task'],'All failures'),(r['task'],r['failure_mode'])]:
                all_failures.setdefault(key,set()).add((r['task'],r['episode_index']))
    latency_records=[]
    for (v,m,view,task,mode), rows in sorted(lat.items()):
        s=latency_counts(rows)
        latency_records.append({'variant':v,'method':m,'view':view,'task':task,'failure_mode':mode,
                                'total_failure_pool':len(all_failures[task,mode]),**s})
    add('latency','Onset latency, deadlines and missing detections',
        f"Annotator: {data['annotators'][aid]}. Wall-clock onset to completed alarm, including visual preprocessing and monitor computation. Delay statistics require a verified detection and completion by episode end; latency uses the first completed post-onset-source alarm and is nonnegative. Sustained pre-onset runs are reported separately from other early alarms. Other summary tables may include late alarms; this table explicitly separates them. Only offline-detected failures were replayed for continuous-score detectors. Deadline counts are observed counts / selected trials, NOT full-test online FDR. Early-stopped trials do not establish whether a later post-onset detection would occur. Early evidence and early completion are different; late and early-evidence flags can overlap. No irreversible-state annotation is available.",
        ['Model / score / view','Task / failure','Selected / full failures','Alarms / selected','On-time detected n','Sustained before human label n','Censored n','Other early alarms / selected','Any pre-onset completion / selected','Late','No alarm','Missing onset','Delay mean ± SD s','Median / p95 s','≤1 / ≤2 / ≤5 / ≤10 s (counts / selected)'],
        [[titles.get(r['variant'],r['variant'])+' / '+detector(r['method'],r['view']),TASKS.get(r['task'],r['task'])+' / '+r['failure_mode'],
          f"{r['selected']}/{r['total_failure_pool']}",f"{r['alarms']}/{r['selected']}",r['delay_s']['n'],r['sustained_before_human'],r['censored'],rate(r['early_evidence'],r['selected']),rate(r['early_completion'],r['selected']),r['late_alarms'],r['no_alarm'],r['missing_onset'],
          mean_sd(r['delay_s']),f"{display(r['delay_s']['median'])} / {display(r['delay_s']['p95'])}",
          ' / '.join(str(r['within_deadline'][str(t)]) for t in (1,2,5,10))+f" (of {r['selected']})"] for r in latency_records],latency_records)
    paired=[]
    for row in data.get('pairwise_latency',{}).get('comparisons',[]):
        if not row['on_time_only']:continue
        paired.append({k:v for k,v in row.items() if k not in ('episodes',)})
    add('paired','Paired onset latency: main wrist WM-NC versus each comparator',
        'Wrist WM-NC is the sole reference; fixed WM-NC is a comparator. Both methods must detect the same episode for the same annotator, with on-time alarms established at/after their respective view-specific onsets (joint outputs use the earlier camera onset). Onsets need not match. Every pair retains its full intersection and recomputes the wrist WM-NC mean. Δ = wrist WM-NC − comparator; negative is shorter onset-relative delay, which can reflect different onset times as well as alarm times. Compare within a row, not between rows with different cohorts. Original operating points, not matched FAR. Both onset labels and paired delays: pairwise_latency/analysis.json and episodes.csv.',
        ['WM-NC reference','Comparator','Shared n','Different onsets included','WM-NC mean ± SD s','Comparator mean ± SD s','Paired Δ mean ± SD s','WM-NC shorter delay n','Status'],
        [[r['reference'].split('|')[-1],r['label'],r['shared_n'],r.get('different_onset_included_n',0),
          *[f"{display(s['mean_s'])} ± {display(s['sd_s'])}" for s in r.get('summaries',[])],
          *([f"{display(r['paired_differences'][0]['mean_s'])} ± {display(r['paired_differences'][0]['sd_s'])}",r['paired_differences'][0]['a_faster_n']] if r.get('paired_differences') else ['—','—']),r['status']]
         if r.get('summaries') else [r['reference'].split('|')[-1],r['label'],0,0,'—','—','—','—',r['status']] for r in paired],paired,
        figures=[p['paths']['svg'] for p in data.get('figure_exports',[]) if 'pairwise' in p['paths']['svg']])

    # Dense post-onset and any-time action results use the same cached episode identities.
    action_variants=['wm_k4_l4_action_none','wm_k4_l4_action_adaln','wm_k4_l4_action_cross_attention','main']
    action=defaultdict(dict); origin=defaultdict(list)
    for r in test:
        if r['calibration']!='episode' or r['outcome']!='failure':continue
        if r['method']=='wm_nc' and r['variant'] in action_variants:
            action[(r['task'],r['failure_mode'],r['failure_origin'],r['view'])].setdefault(r['variant'],[]).append(r)
        if r['variant']==sources.get(r['method']):origin[(r['method'],r['view'],r['failure_origin'])].append(r)
    def detection_cell(rows):
        if not rows:return '—'
        a=[r.get('onset_assessments',{}).get(aid,{}) for r in rows]
        post=sum(r.get('post_alarm') is True for r in a); unknown=sum(r.get('post_alarm') is None for r in a)
        return f"{sum(r.get('any_time_alarm', r['alarm']) for r in rows)}/{len(rows)}; {post}/{len(rows)}"+(f' +{unknown} unknown' if unknown else '')
    add('action_failure','Action conditioning by failure mode',
        'Each cell = any-time detected/total; any crossing at/after onset detected/total. K=L=4, seed 0. NoAct, AdaLN and cross-attention share the main evaluation set; early and post-onset crossings can overlap. Ring was retrained from K=3; other action checkpoints were reused. See analyses/2026-10-14/action_protocol_audit/report.md for the historical-result audit.',
        ['Task','Failure','Origin','View','NoAct any; post','AdaLN any; post','Cross-attention any; post','Both any; post'],
        [[TASKS[t],mode,o,v,*[detection_cell(groups.get(a,[])) for a in action_variants]] for (t,mode,o,v),groups in sorted(action.items())])
    add('origins','Natural versus induced failure detection',
        'Origin labels are the manually specified natural/induced groups. This does not equate natural with manipulation failure, or induced with environmental failure. Per-failure results preserve the distinctions; an agreed mapping to the paper’s four failure categories is still needed.',
        ['Detector','Origin','Any-time; post-onset detected/total'],
        [[detector(m,v),o,detection_cell(rows)] for (m,v,o),rows in sorted(origin.items())])

    # Durations use real collection time, never simulated clock or assumed FPS.
    clocks=RawClocks(resolve)
    datasets={r['task']:r['dataset'] for r in data['settings'] if r['variant']=='main' and r['method']=='wm_nc'}
    unique={(r['task'],r['episode_index']):r for r in episodes if r['variant']=='main' and r['method']=='wm_nc' and r['view']=='wrist' and r['calibration']=='episode'}
    durations=defaultdict(list); duration_records=[]
    for (task,ep), r in sorted(unique.items()):
        wall=clocks(datasets[task],ep); seconds=float(wall[-1]-wall[0])
        group=('Success calibration' if r['split']=='calibration' else 'Success test') if r['outcome']=='success' else r['failure_mode']
        durations[task,group,r['failure_origin']].append(seconds)
        duration_records.append({'task':task,'episode_index':ep,'group':group,'origin':r['failure_origin'],'frames':len(wall),'wall_duration_s':seconds})
    add('durations','Recorded episode duration by failure class',
        'Last raw wall timestamp minus first, for both simulation and real tasks. Recording length includes any initial/terminal dwell and is not semantic task-completion time. Calibration and test successes are separate. These descriptive durations do not demonstrate that episode maxima are free of length/phase effects.',
        ['Task','Class','Origin','Episodes','Mean ± SD s','Median s','Min–max s'],
        [[TASKS[t],g,o,len(vals),mean_sd(s),display(s['median']),f"{display(s['min'])}–{display(s['max'])}"]
         for (t,g,o),vals in sorted(durations.items()) for s in [describe(vals)]],duration_records)

    # Between-seed variation must not be confused with a confidence interval.
    seed_ids={'main'}|{r['id'] for r in data['runs'] if r['family']=='seed'}
    seed_rows=[r for r in test if r['variant'] in seed_ids and r['method']=='wm_nc' and r['calibration']=='episode']
    seed_roc=roc.analyze(seed_rows, targets=(.05,),subsets=('all',))
    sg=defaultdict(list)
    for r in seed_roc['matched_far']:
        rows=[x for x in seed_rows if x['variant']==r['variant'] and x['view']==r['view'] and (r['task']=='Overall' or x['task']==r['task'])]
        neg=[x for x in rows if x['outcome']=='success']; pos=[x for x in rows if x['outcome']=='failure']
        sg[r['task'],r['view']].append({'variant':r['variant'],'FAR':sum(x['alarm'] for x in neg)/len(neg),
            'FDR':sum(x['alarm'] for x in pos)/len(pos),'AUROC':r['episode_AUROC'],'matched_FDR':r['FDR']})
    add('seeds','WM-NC variation across active training seeds',
        'Main seed 0 plus seeds 1–7, one realization per seed. Mean ± sample SD across trained models on the same test episodes, not eight independent datasets or confidence intervals. Active seed 1 is the post-evaluation-selected RTX 5090 rerun; archived PACE seed 1 remains in the main report. K/L/action/data-size sweeps use one training seed each; seed 5–7 have offline results only.',
        ['Task','View','Seeds n','FAR% mean ± SD','FDR% mean ± SD','AUROC mean ± SD','Matched-5%-FAR FDR% mean ± SD'],
        [[TASKS.get(t,t),v,len(rows),*[mean_sd(describe([r[k] for r in rows]),100 if k!='AUROC' else 1) for k in ('FAR','FDR','AUROC','matched_FDR')]] for (t,v),rows in sorted(sg.items())],
        [{'task':t,'view':v,'seeds':rows} for (t,v),rows in sorted(sg.items())])

    repeats = {r['id'] for r in data['runs'] if r['family']=='seed_repeat'}
    if repeats:
        comparison_ids = repeats | {'main','wm_k4_l4_seed2','wm_k4_l4_seed3'}
        selected = [r for r in test if r['variant'] in comparison_ids
                    and r['method']=='wm_nc' and r['calibration']=='episode']
        comparison = []
        for metric in roc.analyze(selected, targets=(.05,), subsets=('all',))['matched_far']:
            rows = [r for r in selected if r['variant']==metric['variant'] and r['view']==metric['view']
                    and (metric['task']=='Overall' or r['task']==metric['task'])]
            neg = [r for r in rows if r['outcome']=='success']
            pos = [r for r in rows if r['outcome']=='failure']
            comparison.append(dict(variant=metric['variant'], task=metric['task'], view=metric['view'],
                false_alarms=sum(r['alarm'] for r in neg), success_episodes=len(neg),
                detected_failures=sum(r['alarm'] for r in pos), failure_episodes=len(pos),
                episode_AUROC=metric['episode_AUROC'], matched_FAR=metric['FAR'], matched_FDR=metric['FDR']))
        add('seed_repeats','Local seed-2/3 repeats: WM-NC offline comparison',
            'Additional realizations of seeds 2 and 3; existing primary models and eight-seed mean/SD remain unchanged. '
            'All use the frozen all-endpoint, visually corrected, unsmoothed, mean-onset protocol. '
            'FDR requires a post-onset crossing. AUROC uses whole-episode maxima and averages tasks equally in Overall; '
            'FAR/FDR pool episode counts. Matched FAR is a retrospective success-test diagnostic. '
            'WM-NC only was scored for the repeats; no WM-PE or new online latency measurements.',
            ['Model','Task','View','False alarms / successes','Detected / failures','Episode AUROC','Matched FAR% / FDR%'],
            [[titles[r['variant']],TASKS.get(r['task'],r['task']),r['view'],
              rate(r['false_alarms'],r['success_episodes']),rate(r['detected_failures'],r['failure_episodes']),
              display(r['episode_AUROC']),f"{100*r['matched_FAR']:.1f} / {100*r['matched_FDR']:.1f}"]
             for r in comparison],comparison)
    for entry in config.get('completed_training', []):
        path = root/entry['path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('Training inventory changed: '+str(path))
        rows = json.loads(raw)
        add(entry['id'],entry['title'],
            'All eight runs passed checkpoint/configuration verification and reached their configured caps. '
            'Offline scoring uses best validation-flow checkpoints, not necessarily the final step. '
            'Validation loss is a training diagnostic, not failure-detection performance. '
            'The inventory retains checkpoint SHA-256 and original completion-check paths.',
            ['Task','Seed','Best step','Final step','Best validation flow'],
            [[TASKS[r['task']],r['seed'],r['best_step'],r['final_step'],f"{r['best_val_flow']:.6f}"] for r in rows],rows)

    # Include detailed sensitivity findings, especially the historical shunt discrepancy.
    study=next((s for s in data.get('studies',[]) if s['id']=='main_calibration_sensitivity_500'),None)
    if study:
        def spread(r,m,scale=100):return f"{scale*r[m+'_mean']:.1f} ± {scale*r[m+'_sd']:.1f} [{scale*r[m+'_p05']:.1f}, {scale*r[m+'_p95']:.1f}]"
        add('calibration','500-split main-model calibration sensitivity by task',
            'All 500 reproducible 19/20 success splits retained; original failures fixed. With n=19 and alpha=0.05, ceil((n+1)*(1-alpha))=19: the threshold is the maximum of 19 calibration episode maxima. Functional calibration instead takes the maximum trajectory residual (s(t)-mu(t))/g(t); the expert-validation template is frozen, default g(t)=1, with wall-time interpolation and boundary hold. This is an expert-template adaptation of FAIL-Detect. Nominal marginal coverage assumes exchangeable successful trajectories conditional on the frozen model/template; it does not guarantee FAR ≤5% in each finite test batch. Mean ± SD and 5th–95th percentiles are resampling sensitivity, not confidence intervals or independent experiments. Mean FAR near 5% follows rank symmetry, not validation of exchangeability. Current shunt-fixed episode FAR is 0/20, not the historical paper’s 6/20; these current scores cannot explain that historical value. See the study report for its historical-reference tail calculation.',
            ['Task','View','Protocol','Original FAR% / FDR%','FAR% mean ± SD [5–95]','FDR% mean ± SD [5–95]','Threshold mean ± SD [5–95]'],
            [[TASKS[r['task']],r['view'],r['calibration'],f"{100*r['original_FAR']:.1f} / {100*r['original_FDR']:.1f}",spread(r,'FAR'),spread(r,'FDR'),spread(r,'threshold',1)] for r in study['summary']],study['summary'])

    if config.get('matched_far_sensitivity'):
        entry = config['matched_far_sensitivity']
        path = root/entry['path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Matched-FAR sensitivity summary changed: '+str(path))
        matched = json.loads(raw)
        if matched['protocol_id'] != data['evaluation_protocol']['protocol_id']:
            raise ValueError('Matched-FAR sensitivity uses a different frozen protocol')
        def interval(r, metric):
            return (f"{100*r[metric+'_mean']:.2f} ± {100*r[metric+'_sd']:.2f} "
                    f"[{100*r[metric+'_p05']:.2f}, {100*r[metric+'_p95']:.2f}]")
        add('calibration_matched_far','500-split WM-NC sensitivity at matched 5% FAR',
            matched['note'],
            ['Task','View','Protocol','Original FDR%','FDR% mean ± SD [5–95]','Achieved FAR% range'],
            [[TASKS.get(r['task'],r['task']),r['view'],r['calibration'],f"{100*r['original_FDR']:.2f}",
              interval(r,'FDR'),f"{100*r['FAR_min']:.1f}–{100*r['FAR_max']:.1f}"] for r in matched['summary']],
            matched['summary'],figures=[str(path.parent.relative_to(root)/'sensitivity.png')])

    # Source metadata for main model and numerical scorer configuration.
    training=[]
    for s in data['settings']:
        if s['variant']!='main' or s['method']!='wm_nc':continue
        checkpoint=resolve(s['checkpoint']); folder=checkpoint.parent.parent
        conf=json.loads((folder/'config.json').read_text()); conf=conf.get('fm_wm',conf)
        splits=json.loads((folder/'splits.json').read_text())
        # Dataset root is task/prl_data/representation; use the recorded expert path.
        expert=resolve(conf['data']['dataset_root'])
        if not expert.exists(): expert=resolve(s['dataset']).parents[1]/'data'/Path(conf['data']['dataset_root']).name
        info=json.loads((expert/'meta/info.json').read_text())
        def ids(name):
            value=splits.get(name, splits.get(name+'_episode_ids',splits.get(name+'_episodes')))
            if value is None:raise ValueError(f'Unknown split schema: {folder}')
            return value
        train_ids,val_ids=ids('train'),ids('val')
        if set(train_ids)&set(val_ids):raise ValueError('WM split overlaps')
        training.append({'task':s['task'],'expert_total':info['total_episodes'],'wm_train':len(train_ids),'wm_val':len(val_ids),
                         'seed':conf['seed'],'split_seed':conf['data']['split_seed'],
                         'base_std':conf['fm_head']['base_std'],'checkpoint':str(checkpoint),
                         'config':str(folder/'config.json'),'splits':str(folder/'splits.json'),
                         'operator_count':None,'operator_status':'Not recorded in the audited episode metadata'})
    add('training','Training-data and implementation evidence',
        'WM split counts come from each saved splits.json (episode disjointness checked); expert totals come from dataset metadata. Policy/VAE per-stage training counts and operator attribution still need explicit confirmation before manuscript claims. WM-PE integrates from x0=0 to t=1; it does not use a random sample as its deterministic prediction. Base standard deviation is read from the trained model config.',
        ['Task','Expert episodes','WM train / val','Training / split seed','Base σ','Operators'],
        [[TASKS[r['task']],r['expert_total'],f"{r['wm_train']} / {r['wm_val']}",f"{r['seed']} / {r['split_seed']}",r['base_std'],'Unrecorded'] for r in training],training)
    settings=[r for r in data['settings'] if r['variant']==sources.get(r['method']) and r['method'].startswith('wm_')]
    add('numerics','ODE and sampling settings',
        'Scoring configuration, not model-training integration. WM-PU is Ward-inspired endpoint uncertainty on our WM, with the existing one-step RK4 result; convergence is not established. ACE is an ACT adaptation of FIPER’s action branch. Neither is a full reproduction. Native observation baselines use ACT’s 16 × 768 compressed visual tokens; logpZO additionally uses robot state.',
        ['Task','Method','K / L','Solver','Steps / solve','Solves / score','Field evaluations / score','Samples'],
        [[TASKS[r['task']],METHODS[r['method']],f"{r['K']} / {r['L']}",r['ode_solver'],r['ode_steps_per_solve'],r['ode_solves_per_score'],r['vector_field_evals_per_score'],r['samples_per_score']] for r in settings],settings)

    # Prompt variants are supplementary evidence, never silently replacing v9.
    from benchmark_vlm_results import load_runs
    vlm=list(data.get('vlm_runs',[]))+load_runs(config.get('supplementary_vlm',[]),root)
    vr=[]
    for run in vlm:
        for r in run['summary']+run['overall']:
            vr.append({'protocol':run['id'],'task':r['task'],'view':r['view'],
                       **{k:v for k,v in r.items() if k.startswith(('FAR','FDR'))},
                       'complete':r['complete'],'planned':r['planned'],'indeterminate':r['indeterminate'],
                       'unattempted':r['unattempted'],'source':run['path']+'/report.json',
                       'evaluation_protocol':run.get('evaluation_protocol') or run.get('source_evaluation_protocol'),
                       'FDR_definition':'post-onset; full replay' if run.get('evaluation_protocol') or run.get('source_evaluation_protocol') else 'any-time; historical replay'})
    add('vlm_protocols','Qwen prompt experiments',
        'Separate fixed/wrist runs, full observed history near 1 FPS, BF16/SDPA, greedy decoding, 256-token cap. '
        'V9 remains the primary baseline; V12 and V13 are preserved separately. V13 adds full failure descriptions. '
        'Current V9 and V13 use mean-onset post-onset FDR and complete replay; historical V12 uses any-time FDR and early stopping, '
        'so V12 is not a controlled prompt-only comparison. Incl-late includes late returned alarms; on-time excludes them. '
        'Binary output has no continuous AUROC or matched-FAR sweep. Prompt development reused evaluation cases; '
        'this is not an untouched held-out prompt comparison. No V12 onset-sensitive statistic is recomputed here.',
        ['Protocol','Task','View','FDR definition','Any-time FAR','FDR incl. late','On-time FAR','On-time FDR','Complete / planned','Indeterminate / unattempted'],
        [[r['protocol'],TASKS.get(r['task'],r['task']),r['view'],r['FDR_definition'],*[rate(r[f'{m}_{t}_count'],r[m+'_denominator']) for t in ('any','on_time') for m in ('FAR','FDR')],f"{r['complete']}/{r['planned']}",f"{r['indeterminate']} / {r['unattempted']}"] for r in vr],vr)
    return sections


def markdown_sections(sections, heading=2):
    lines=[]
    for s in sections:
        lines += ['#'*heading+' '+s['title'],'',s['note'],'']
        escape=lambda v:str(v).replace('|','\\|').replace('\n',' ')
        lines += ['| '+' | '.join(map(escape,row))+' |' for row in [s['columns'],['---']*len(s['columns']),*s['rows']]]+['']
    return lines


def write_review_evidence(data, root, config_path, resolve, suffix=''):
    root=Path(root); config=yaml.safe_load(Path(config_path).read_text())
    sections=sections_for(data,root,config,resolve)
    coverage={'id':'coverage','title':'Reviewer-comment evidence map','note':'Available evidence is not a claim that every requested experiment has been completed. The status and limits below distinguish measured results from manuscript discussion or missing data.',
              'columns':['Comments','Question','Evidence','Remaining limitation / action'],
              'rows':[[r['comments'],r['question'],r['evidence'],r['gap']] for r in config['coverage']],
              'records':config['coverage'],'figures':[]}
    sections.insert(0,coverage)
    payload={'schema_version':1,'created_utc':data['created'],'annotator':next(iter(data['annotators'])),
             'labels_sha256':data.get('onset_analysis',{}).get('labels_sha256'),
             'configuration_sha256':hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
             'sections':sections}
    stem='reviewer_evidence'+suffix
    note=data.get('offline_protocol',{}).get('note','Original model scores and thresholds are unchanged.')
    (root/(stem+'.yaml')).write_text(yaml.safe_dump(payload,sort_keys=False,allow_unicode=True,width=140))
    (root/(stem+'.md')).write_text('\n'.join(['# Reviewer evidence supplement','',
        f'Companion to [summary](summary{suffix}.md) and the [interactive results](index.html). '+note,'',*markdown_sections(sections)]))
    output=['<!doctype html><html lang="en"><meta charset="utf-8"><title>FoMo-FD reviewer evidence</title>',
            '<style>body{font:16px system-ui;color:#163345;max-width:1450px;margin:30px auto;padding:0 24px}a{color:#146e84}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #d4e0e5;text-align:left}th{background:#eaf1f4}section{margin:32px 0}details{overflow:auto}summary{cursor:pointer;padding:12px;background:#edf3f5}img{width:100%;max-width:1100px}nav{display:flex;gap:12px;flex-wrap:wrap}p{line-height:1.5}</style>',
            '<h1>Reviewer evidence</h1><p><a href="index.html">Interactive charts</a> · <a href="summary.md">Complete summary</a> · <a href="reviewer_evidence.md">Evidence Markdown</a> · <a href="reviewer_evidence.yaml">Evidence YAML</a></p>',
            '<p>Snapshot: '+html.escape(data['created'])+'. '+html.escape(note)+'</p>',
            '<nav>'+''.join('<a href="#'+s['id']+'">'+html.escape(s['title'])+'</a>' for s in sections)+'</nav>']
    for s in sections:
        output += ['<section id="'+s['id']+'"><h2>'+html.escape(s['title'])+'</h2><p>'+html.escape(s['note'])+'</p>']
        for figure in s['figures']:
            output.append('<img src="'+html.escape(figure,quote=True)+'" alt="'+html.escape(s['title'],quote=True)+'">')
        output += ['<details'+(' open' if s['id']=='coverage' else '')+'><summary>Exact results ('+str(len(s['rows']))+' rows)</summary><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in s['columns'])+'</tr></thead><tbody>']
        output += ['<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in s['rows']]
        output += ['</tbody></table></details></section>']
    document='\n'.join(output+['</html>'])
    if suffix:
        for name in ('summary.md','reviewer_evidence.md','reviewer_evidence.yaml'):
            document=document.replace('href="'+name+'"','href="'+name.rsplit('.',1)[0]+suffix+'.'+name.rsplit('.',1)[1]+'"')
    (root/(stem+'.html')).write_text(document)
    return payload
