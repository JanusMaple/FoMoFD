"""Cache-only offline protocol revisions and paired report options.

Raw recordings, score caches, legacy archives and online replay are immutable.
The optional visual correction is an exclusive raw-frame cutoff, applied before
calibration and to the same episode if it moves into a resampled test split.
"""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil

import numpy as np
import yaml

import unified_benchmark as benchmark
from functional_conformal import canonical_scheme
from benchmark_onset_results import assess, onset_frame, automatic_latencies

OPTIONS = ('uncorrected', 'visual_corrected')
CORRECTION = {'task': 'needle_pickup', 'episode_index': 5, 'raw_frame_stop': 122,
              'reason': 'User-identified blue background entering the wrist view; truncate the synchronized episode at raw frame 122 inclusive.'}
NOTE = ('Every valid cached endpoint is retained; no WM tail trimming or smoothing. '
        'Calibration uses the original 19 successes and strict threshold crossing. '
        'The dataset visual correction retains raw frames 0–121 of needle PRL episode 5 in both views. '
        'FDR requires an alarm at/after onset; early alarms are reported separately. Whole-episode AUROC is retained. '
        'Latency uses the first completed post-onset-source alarm; sustained pre-onset runs are separate from other early alarms. '
        'No raw data, model or cached score was modified. Replay retains its recorded selection; missing newly eligible episodes remain unmeasured. '
        'The frozen specification identifies the onset reference; v3 uses the mean annotated wall time per view.')


def protocol(corrected=False):
    result = {'smoothing': 'trailing', 'window': 1, 'wm_frame_policy': 'all_endpoints'}
    if corrected:
        result['frame_limits'] = [dict(CORRECTION)]
    benchmark.validate_protocol(result)
    return result


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_templates(data, root, resolve):
    """Select frozen all-endpoint expert templates by exact PRL cache digest."""
    result, files = {}, {}
    for manifest in sorted(root.rglob('templates_manifest.json')):
        if 'smoke' in str(manifest) or 'offline_protocols' in manifest.parts:
            continue
        files[str(manifest)] = benchmark.digest(manifest)
        for entry in json.loads(manifest.read_text())['templates']:
            if entry['protocol'] != 'latency':
                continue
            path = manifest.parent/entry['path']
            if benchmark.digest(path) != entry['sha256']:
                raise ValueError(f'Frozen template changed: {path}')
            files[str(path)] = entry['sha256']
            template = json.loads(path.read_text())
            cache_sha = template['source'].get('prl_score_cache_sha256')
            if cache_sha is None:
                if manifest.parent.name != 'ace_histogram_revision':
                    raise ValueError('Unmapped functional template')
                run = next(r for r in data['runs'] if r['id']=='ace_normal_bins')
                folder = root/run['path']
                config = json.loads((folder/'benchmark_config.json').read_text())
                sampling = resolve(config['ace']['sampling_manifest'])
                if benchmark.digest(sampling) != template['source']['sampling_manifest_sha256']:
                    raise ValueError('ACE frozen sampling mismatch')
                files[str(sampling)] = benchmark.digest(sampling)
                report = json.loads((folder/'report.json').read_text())
                cache_sha = next(c['sha256'] for c in report['score_caches'] if c['task']==entry['task'] and c['method']=='ace')
            key = (cache_sha, entry['view'])
            if key in result and result[key] != template:
                raise ValueError('Ambiguous frozen template')
            result[key] = template
    return result, files


def evaluate_available_bands(scored, labels, config, annotators, templates, observer=None):
    """Report each completed detector without requiring unrelated templates."""
    ready = []
    for item in scored:
        job = item[0]
        views = ('both',) if config.get('wm_score_aggregation') == 'joint' and job.mode else job.views
        if all((job.task, job.method, view, name) in templates
               for view in views for name in ('paper', 'latency')):
            ready.append(item)
    reports = {key: [] for key in ('episodes', 'latencies', 'distributions', 'calibration')}
    band_config = {**config, 'fusion': {'wm_views': False, 'conditional_marginal': False}}
    for task in dict.fromkeys(job.task for job, _, _ in ready):
        jobs = [item for item in ready if item[0].task == task]
        methods = list(dict.fromkeys(job.method for job, _, _ in jobs))
        evaluated = benchmark.evaluate_suite(jobs, labels, band_config, annotators, methods,
                                             ['functional_band'], templates, observer)
        for key, rows in evaluated.items():
            reports[key].extend(rows)
    return reports, len(ready)


def export_run(folder, source, config, reports, labels_text, tasks, methods, schemes):
    """Standard report artifacts, with explicit provenance to the frozen source."""
    folder.mkdir(parents=True, exist_ok=True)
    metrics = benchmark.aggregate_metrics(reports['episodes'])
    tables = {}
    labels = yaml.safe_load(labels_text)
    for scheme in schemes:
        table_config = deepcopy(config)
        if scheme == 'functional_band':
            table_config['fusion'] = {'wm_views': False, 'conditional_marginal': False}
        selected = [r for r in reports['episodes'] if r['calibration'] == scheme]
        available_methods = [method for method in methods if any(r['method'] == method for r in selected)]
        available_tasks = [task for task in tasks if any(r['task'] == task for r in selected)]
        header, table = benchmark.paper_table(selected, table_config, labels, available_tasks, available_methods)
        tables[scheme] = (header, table)
        (folder/f'paper_table_{scheme}.md').write_text(benchmark.markdown_table(header, table))
    report = {**source, 'created_utc': benchmark.utc_now(), 'script_sha256': benchmark.digest(Path(__file__)),
              'labels_sha256': hashlib.sha256(labels_text.encode()).hexdigest(), 'calibration_schemes': schemes,
              'protocols': {k: config[k] for k in ('paper_protocol','latency_protocol')},
              'calibration': reports['calibration'], 'summary_metrics': metrics,
              'latency_summary': benchmark.latency_summary(reports['latencies']),
              'latency_note': 'Dense first-post-onset cached-score delay, excluding inference/runtime. Not a new online replay.',
              'offline_revision': config['offline_revision']}
    from benchmark_frozen_protocol import load_protocol, REPO
    report['evaluation_protocol'] = load_protocol(config['evaluation_protocol'])
    report['evaluation_protocol_sha256'] = benchmark.digest(REPO/config['evaluation_protocol'])
    write_json(folder/'report.json', report)
    write_json(folder/'benchmark_config.json', config)
    (folder/'labels_snapshot.yaml').write_text(labels_text)
    for name, rows in [('episode_results',reports['episodes']),('latency_episodes',reports['latencies']),
                       ('score_distributions',reports['distributions']),('calibration',reports['calibration']),
                       ('summary_metrics',metrics),('latency_summary',report['latency_summary']),
                       ('scoring_settings',source['scoring_settings'])]:
        benchmark.write_csv(folder/(name+'.csv'), rows, [])
    benchmark.write_inspection_reports(folder, report, tables, reports['distributions'])
    from benchmark_roc import analyze, write_reports
    write_reports(folder, analyze(reports['episodes']), {'created_utc':report['created_utc'],
        'protocols':report['protocols'],'score_source':'episode_results.csv',
        'score_source_sha256':benchmark.digest(folder/'episode_results.csv'),
        'offline_revision':config['offline_revision']}, report_stem='roc_report')
    return report


def recompute_offline(data, root, output, corrected, resolve, pack, templates):
    labels_path = Path(benchmark.__file__).parents[1]/'configs/benchmark/prl_episode_labels.yaml'
    labels_text = labels_path.read_text(); labels = yaml.safe_load(labels_text)
    episodes, latencies, distributions, runs, functional_runs, qrecords = [], [], [], [], [], {}
    from benchmark_frozen_protocol import REPO
    PROTOCOL = REPO/data['evaluation_protocol_path']
    from benchmark_consensus_onsets import derive_labels
    labels = derive_labels(labels, data['evaluation_protocol'], resolve)
    timelines, sources = {}, {str(labels_path): benchmark.digest(labels_path), str(PROTOCOL): benchmark.digest(PROTOCOL),
        str(PROTOCOL.parents[2]/'utils/benchmark_frozen_protocol.py'): benchmark.digest(PROTOCOL.parents[2]/'utils/benchmark_frozen_protocol.py')}
    selected_protocol = protocol(corrected)
    for i, run in enumerate(data['runs']):
        original_folder = root/run['path']
        source = json.loads((original_folder/'report.json').read_text())
        sources[str(original_folder/'report.json')] = benchmark.digest(original_folder/'report.json')
        config = json.loads((original_folder/'benchmark_config.json').read_text())
        sources[str(original_folder/'benchmark_config.json')] = benchmark.digest(original_folder/'benchmark_config.json')
        config.update(paper_protocol=deepcopy(selected_protocol), latency_protocol=deepcopy(selected_protocol))
        config['evaluation_protocol'] = data['evaluation_protocol_path']
        config['offline_revision'] = {'option': OPTIONS[int(corrected)], 'note':NOTE,
            'original_report':str(original_folder/'report.json'), 'original_report_sha256':sources[str(original_folder/'report.json')],
            'correction':CORRECTION if corrected else None, 'inference':False}
        scored, expert_templates, metadata = [], {}, {}
        for cache in source['score_caches']:
            path = resolve(cache['path'])
            if benchmark.digest(path) != cache['sha256']:
                raise ValueError(f'Cache changed: {path}')
            sources[str(path)] = cache['sha256']
            with np.load(path, allow_pickle=False) as z:
                identity = json.loads(str(z['provenance_json'].item()))['identity']
                arrays = {k:z[k] for k in ('episode_index','frame_index','raw_frame_index','wall_timestamp','scores')}
            dataset = resolve(identity['dataset'])
            if dataset not in timelines:
                timelines[dataset] = benchmark.load_timeline(dataset)
            job = SimpleNamespace(task=cache['task'],method=cache['method'],views=tuple(cache['views']),
                                  mode=benchmark.METHODS[cache['method']][3],merged=identity['merged_config'],dataset=dataset)
            benchmark.validate_scores(arrays, job, timelines[dataset])
            scored.append((job,arrays,timelines[dataset]))
            metadata[job.task,job.method] = (cache,identity)
            views = ['both'] if config.get('wm_score_aggregation')=='joint' and job.mode else job.views
            for view in views:
                template = templates.get((cache['sha256'],view))
                if template:
                    for name in ('paper','latency'):
                        expert_templates[job.task,job.method,view,name] = template
        schemes = list(dict.fromkeys(canonical_scheme(s) for s in source['calibration_schemes'] if s!='functional_band'))
        folder = output/'offline'/run['id']
        def observe(job,view,method,scheme,paper,causal,evaluated):
            for row in evaluated['episodes']:
                if row['outcome']=='failure':
                    annotations = labels['tasks'][job.task].get('failure_onsets',{}).get(row['episode_index'],{})
                    row.setdefault('onset_assessments', {aid:assess(paper[row['episode_index']],onset_frame(annotations.get(aid),view)) for aid in labels['annotators']})
            if method not in benchmark.METHODS or scheme not in ('episode','functional_band'):
                return
            cache,identity = metadata[job.task,method]
            key = (run['id'],job.task,method,view)
            record = qrecords.setdefault(key, {'metadata': {'variant':run['id'],'task':job.task,'method':method,'view':view,
                'score_seed':identity['seed'],'training_seed':None,'cache_path':cache['path'],
                'cache_sha256':cache['sha256'],'checkpoint':identity['checkpoint'],'protocol':selected_protocol}, 'q':{}, 'original':{}, 'folder':folder})
            record['q'][scheme] = {e:float(c.get('roc_score',c.get('raw_score',c['score'])).max()) for e,c in paper.items()}
            primary = data['evaluation_protocol']['detection']['primary_annotator']
            from benchmark_frozen_protocol import resampling_post_max
            record.setdefault('post_q', {})[scheme] = {r['episode_index']:resampling_post_max(paper[r['episode_index']],r['onset_assessments'][primary]['onset_frame'])
                for r in evaluated['episodes'] if r['outcome']=='failure'}
            record['original'][scheme] = evaluated['episodes']
            template = expert_templates.get((job.task,method,view,'paper'))
            if template:
                record['metadata']['training_seed'] = template['source'].get('training_seed')
            if run['id']=='main' and method=='wm_nc':
                # Small saved curve set also updates dense offline latency in the 500-split study.
                record.setdefault('curves',{})[scheme] = {e:dict(c,score=c.get('roc_score',c['raw_score'])) for e,c in paper.items()}
        reports = benchmark.evaluate_suite(scored,labels,config,list(labels['annotators']),source['methods'],schemes,expert_templates,observe)
        bands, ready_count = evaluate_available_bands(scored, labels, config, list(labels['annotators']),
                                                     expert_templates, observe)
        if 'functional_band' in source['calibration_schemes'] and ready_count != len(scored):
            raise ValueError(f'Missing existing functional templates: {run["id"]}')
        if ready_count:
            schemes.append('functional_band')
            for key, rows in bands.items():
                reports[key].extend(rows)
        config['calibrations'] = schemes
        new_report = export_run(folder,source,config,reports,labels_text,source['tasks'],source['methods'],schemes)
        episodes.extend(dict(r,variant=run['id']) for r in reports['episodes'])
        latencies.extend(dict(r,variant=run['id']) for r in reports['latencies'])
        distributions.extend(dict(r,variant=run['id']) for r in reports['distributions'])
        runs.append({**run,'path':str(folder.relative_to(root)),'created':new_report['created_utc'],
                     'protocols':new_report['protocols'],'labels_sha256':new_report['labels_sha256'],
                     'original_archive_path':run['path'],'original_archive_sha256':run['archive_sha256'],
                     'archive_sha256':benchmark.digest(folder/'report.json')})
        if ready_count:
            functional_runs.append({'variant':run['id'],'path':str(folder.relative_to(root)),
                'labels_sha256':new_report['labels_sha256'],'calibration':[r for r in reports['calibration'] if r['calibration']=='functional_band']})
        print(f'Offline {OPTIONS[int(corrected)]} {i+1}/{len(data["runs"])}: {run["id"]}',flush=True)
    if labels_path.read_text()!=labels_text:
        raise ValueError('Onsets changed during offline recomputation')
    patch = {'episodes':pack(episodes),'latencies':pack(automatic_latencies(latencies)),
             'distributions':pack(distributions),'runs':runs,
             'functional_runs':[{'path':str(output.relative_to(root)),'state':{'status':'complete'},'variants':functional_runs}],
             'onset_analysis':dict(data['onset_analysis'],AUROC='Recomputed on every valid endpoint',
                 offline_protocol=selected_protocol,verified_original_episode_rows=None,derived_episode_rows=len(episodes)),
             'offline_protocol':{'id':data['evaluation_protocol']['protocol_id'],'note':NOTE,'correction':CORRECTION if corrected else None,
                 'wm_frame_policy':'all_endpoints','path':str(output.relative_to(root)),
                 'online_note':'Main-baseline and K/L WM-NC replay use the corrected all-endpoint episode thresholds and offline post-onset selection. Conditional latency/retention only; Qwen separately replays all test episodes. Historical ablation replay is excluded.'},
             'created':benchmark.utc_now()}
    write_json(output/'input_provenance.json',{'inputs':sources,'script_sha256':benchmark.digest(Path(__file__)),
              'protocol':selected_protocol,'no_inference':True,'legacy_archives_preserved':True})
    return patch,qrecords,labels


def sensitivity(qrecords, labels, study, root, output, option):
    """Reuse exact historical split IDs; recompute both schemes and all replicates."""
    import calibration_sensitivity as cs
    import calibration_sensitivity_metrics as cm
    original_dir = root/study['path']
    original_meta = json.loads((original_dir/'manifest.json').read_text())
    split_bytes = (original_dir/'splits.json').read_bytes()
    splits = json.loads(split_bytes)
    output.mkdir(parents=True,exist_ok=True)
    (output/'splits.json').write_bytes(split_bytes)
    cases, summaries, results, originals, maxima, checks = [], [], [], [], [], []
    extras = {'subgroup_results':[],'offline_latency_split_results':[],'offline_latency_original':[],'offline_latency_verification':[]}
    include_extras = splits['n_splits']==500
    for source in original_meta['cases']:
        key = tuple(source[k] for k in ('variant','task','method','view'))
        # Use the corrected ACE histogram in the updated multi-method analysis.
        source_key = ('ace_normal_bins',*key[1:]) if key[0]=='main' and key[2]=='ace' else key
        record = qrecords[source_key]
        if set(record['q']) != {'episode','functional_band'}:
            raise ValueError(f'Both frozen calibration protocols required: {source_key}')
        meta = record['metadata'] | {'variant':key[0], 'source_variant':source_key[0]}
        meta['case_id'] = hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()[:16]
        common = {k:meta[k] for k in ('variant','task','method','view','training_seed','score_seed','case_id')}
        definition = splits['tasks'][meta['task']]
        case={'metadata':meta,'summaries':[],'original':[],'results':[]}
        for scheme in cs.SCHEMES:
            context = common | {'calibration':scheme}
            q = record['q'][scheme]
            def evaluate(q, success_ids, failure_ids, splits):
                return cs.evaluate_splits(q, success_ids, failure_ids, splits,
                    failure_detection_scores={e:record['post_q'][scheme][e] for e in failure_ids})
            random = [context|r for r in evaluate(q,definition['success_ids'],definition['failure_ids'],definition['random_splits'])]
            original = context|evaluate(q,definition['success_ids'],definition['failure_ids'],[definition['original']])[0]
            ref = record['original'][scheme]
            for outcome,field in [('success','false_alarms'),('failure','detected_failures')]:
                expected = sum(r['alarm'] for r in ref if r['split']=='test' and r['outcome']==outcome)
                if original[field] != expected:
                    raise ValueError(f'Resampling original split mismatch: {context}/{field}: {original[field]} vs {expected}')
            summary = context|cs.summarize(random,original)
            case['summaries'].append(summary);case['original'].append(original);case['results'].extend(random)
            summaries.append(summary);originals.append(original);results.extend(random)
            checks.append(context|{'passed':True,'reference':'new all-endpoint report for the same correction option'})
            maxima.extend(context|{'episode_index':e,'Q':v} for e,v in q.items())
            if include_extras:
                extras['subgroup_results'].extend(cm.subgroup_results(q,definition,labels['tasks'][meta['task']],context,evaluate))
                # Curves already contain raw scores or frozen functional residuals.
                curves = record['curves'][scheme]
                annotations = [r for r in benchmark_csv(record['folder']/'latency_episodes.csv') if
                    (r['task'],r['method'],r['view'],r['calibration'])==(meta['task'],meta['method'],meta['view'],scheme)]
                memo={}
                for split in [definition['original'],*definition['random_splits']]:
                    threshold=max(q[e] for e in split['calibration_ids'])
                    if threshold not in memo:
                        memo[threshold]=benchmark.latency_summary(cm.latency_rows(curves,annotations,threshold,frozen=True,first_post_onset=True))
                    values=[context|r|{'split_id':split['split_id'],'latency_threshold':threshold,
                            'timing_protocol':'dense_offline_all_endpoints_no_runtime'} for r in memo[threshold]]
                    extras['offline_latency_original' if split['split_id']=='original' else 'offline_latency_split_results'].extend(values)
                extras['offline_latency_verification'].append(context|{'passed':True,'same_support_and_threshold_as_offline_metrics':True})
        cases.append(case)
    pairs=cs.comparisons(cases);shunt=cs.shunt_analysis(summaries,results,.30)
    for name,rows in [('summary',summaries),('split_results',results),('original_results',originals),('episode_maxima',maxima),
                      ('paired_summary',[p['summary'] for p in pairs]),('paired_split_results',[dict(pair_id=p['metadata']['pair_id'],**r) for p in pairs for r in p['results']]),
                      ('original_verification',checks),('shunt_fixed_analysis',shunt)]:
        benchmark.write_csv(output/(name+'.csv'),rows,[])
    write_json(output/'summary.json',summaries);write_json(output/'paired_results.json',pairs)
    write_json(output/'shunt_fixed_analysis.json',shunt)
    if include_extras:
        cm.write_extras(output,extras,results,originals,protocol_note=NOTE+' These are dense offline delays, excluding runtime; all intervals describe overlapping resamples, not confidence intervals.')
    else:
        overall, orig = cm.overall_results(results),cm.overall_results(originals)
        overall_summary = cm.grouped_summary(overall,orig,['variant','task','method','view','calibration'],['FAR','FDR','episode_AUROC'])
        for name,value in [('overall_summary',overall_summary),('overall_split_results',overall),('overall_original',orig)]:
            write_json(output/(name+'.json'),value)
    figures=cs.plots(output,cases,pairs)
    note = (NOTE+' Exact previously saved split IDs are reused. The visual cutoff applies to episode 5 even when it belongs to the held-out success set. '
            'These overlapping resamples are not independent experiments or confidence intervals; mean FAR near 5% follows rank symmetry, not calibration validation. '
            'Frozen all-endpoint expert templates are reused without refitting. Corrected ACE replaces the obsolete histogram in the updated 200-split study.')
    metadata={'created_utc':benchmark.utc_now(),'note':note,'config':{'include_offline_metrics':include_extras},
        'n_splits':splits['n_splits'],'case_count':len(cases),'protocol_case_count':len(summaries),
        'cases':[c['metadata'] for c in cases],'figures':figures,'option':option,'training':False,'scoring':False,
        'source_study':str(original_dir),'source_manifest_sha256':benchmark.digest(original_dir/'manifest.json'),
        'split_manifest_sha256':benchmark.digest(output/'splits.json'),'template_refitting':False}
    cs.reports(output,summaries,pairs,shunt,figures,metadata)
    write_json(output/'manifest.json',metadata)
    write_json(output/'state.json',{'status':'complete','original_checks_passed':len(checks),'models_loaded':0})
    assert (output/'splits.json').read_bytes()==split_bytes
    print(f'Sensitivity {option}: {len(cases)} cases × {splits["n_splits"]} exact saved splits',flush=True)
    return {**study,'path':str(output.relative_to(root)),'date':study.get('date'),
        'context':'Updated all-endpoint offline protocol; exact original split IDs retained. '+('Visual correction enabled.' if option=='visual_corrected' else 'No visual correction.')}


def benchmark_csv(path):
    import csv
    with Path(path).open() as f:return list(csv.DictReader(f))


def build_options(data, root, catalog, resolve, pack):
    root=Path(root)
    catalog_data=yaml.safe_load(Path(catalog).read_text())
    base=root/catalog_data['offline_output']
    base.mkdir(parents=True,exist_ok=True)
    legacy=base/'legacy_report';legacy.mkdir(exist_ok=True)
    for name in ['index.html','summary.md','summary.yaml','reviewer_evidence.md','reviewer_evidence.yaml','reviewer_evidence.html',
                 'summary_visual_corrected.md','summary_visual_corrected.yaml',
                 'reviewer_evidence_visual_corrected.md','reviewer_evidence_visual_corrected.yaml','reviewer_evidence_visual_corrected.html']:
        path=root/name
        if path.exists() and not (legacy/name).exists():shutil.copy2(path,legacy/name)
    templates,template_inputs=read_templates(data,root,resolve)
    catalog_data=yaml.safe_load(Path(catalog).read_text())
    patches={}
    for option in ('visual_corrected',):
        output=base/option
        write_json(output/'state.json',{'status':'running','started_utc':benchmark.utc_now(),'inference':False})
        patch,qrecords,labels=recompute_offline(data,root,output,option=='visual_corrected',resolve,pack,templates)
        entries=[]
        for study in catalog_data.get('studies',[]):
            if study['kind']=='calibration_sensitivity':
                entries.append(sensitivity(qrecords,labels,study,root,output/'studies'/study['id'],option))
            else:
                entries.append(study)
        from benchmark_studies import load_studies
        patch['studies']=load_studies(catalog,root,entries=entries)
        # The unchanged historical distribution study must not be mistaken for a corrected one.
        if option=='visual_corrected':
            for study in patch['studies']:
                if study['kind']=='score_distributions' and not study.get('dataset_validity_applied'):
                    study['note']='Historical uncorrected score-distribution study; preserved separately from the frozen corrected protocol. '+study['note']
                    if not any('Historical distribution study' in h for h in study['highlights']):
                        study['highlights'].insert(0,'Historical distribution study — uncorrected episode 5; preserved separately.')
        write_json(output/'template_provenance.json',template_inputs)
        write_json(output/'state.json',{'status':'complete','finished_utc':benchmark.utc_now(),'models_loaded':0,'option':option})
        patch['offline_protocol']['legacy_report']=str(legacy.relative_to(root))+'/summary.md'
        write_option_index(patch, root)
        patches[option]=patch
    return patches


def write_option_index(patch, root):
    import html
    output = Path(root)/patch['offline_protocol']['path']
    links = ['<li><a href="offline/'+html.escape(r['id'])+'/report.html">'+html.escape(r['title'])+'</a></li>' for r in patch['runs']]
    page = '<!doctype html><meta charset="utf-8"><title>All-endpoint offline reports</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;line-height:1.6}</style>'
    page += '<h1>'+html.escape(patch['offline_protocol']['id'])+'</h1><p>'+html.escape(NOTE)+'</p><p><a href="../../../index.html">Frozen protocol interactive report</a></p><ul>'+''.join(links)+'</ul>'
    (output/'report.html').write_text(page)
