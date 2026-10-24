#!/usr/bin/env python3
"""Compare constant, diagnostic wall-bin, and functional trajectory thresholds.

Uses curated, hash-verified PRL score caches; never re-scores PRL data. Missing
EXPERT VALIDATION scores require --allow-inference and use the saved checkpoint,
score configuration and episode IDs. No training is performed. Each protocol
has a separate report directory; original reports and annotations are read-only.

python lerobot/utils/functional_benchmark.py --output <new-directory> --plan-only
python lerobot/utils/functional_benchmark.py --output <new-directory> --allow-inference
Add --tasks needle_pickup --methods rnd --variants main for an end-to-end smoke.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import yaml

import unified_benchmark as b
import functional_conformal as f
import benchmark_roc as roc
from results_inspector import resolve_repo_path

ROOT = b.REPO/'fomofd/tasks/benchmark_results'
ARRAY_KEYS = ['episode_index', 'frame_index', 'raw_frame_index', 'wall_timestamp', 'scores']


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(roc.safe(data), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sources(config, variants, tasks, methods):
    catalog = yaml.safe_load((b.REPO/config['catalog']).read_text())
    manifests = [json.loads(p.read_text()) | {'directory': p.parent} for p in ROOT.glob('offline/*/archive_manifest.json')]
    records = []
    for entry in catalog['offline']:
        if variants and entry['id'] not in variants:
            continue
        matches = [m for m in manifests if m['source']==entry['source'] and (not entry.get('sha256') or entry['sha256']==m['sha256'])]
        if len(matches)!=1:
            raise ValueError(f'Expected one immutable source archive: {entry}')
        m=matches[0]
        for name in ['report.json','benchmark_config.json','labels_snapshot.yaml']:
            if b.digest(m['directory']/name)!=m['files'][name]:
                raise ValueError(f'Changed archived input: {m["directory"]/name}')
        report=json.loads((m['directory']/'report.json').read_text())
        oldconfig=json.loads((m['directory']/'benchmark_config.json').read_text())
        for cache in report['score_caches']:
            if (tasks and cache['task'] not in tasks) or (methods and cache['method'] not in methods):
                continue
            if cache['method'] not in b.METHODS:
                records.append({'variant':entry['id'],'task':cache['task'],'method':cache['method'],'status':'not_applicable','reason':'No compatible continuous scorer'})
                continue
            path=resolve_repo_path(cache['path'])
            if b.digest(path)!=cache['sha256']:
                raise ValueError(f'Changed saved scores: {path}')
            with np.load(path,allow_pickle=False) as data:
                provenance=json.loads(str(data['provenance_json'].item()))
            records.append({'variant':entry['id'],'task':cache['task'],'method':cache['method'],
                            'views':list(cache['views']),'prl_cache':str(path),'prl_cache_sha256':cache['sha256'],
                            'identity':provenance['identity'],'settings':cache['scoring_settings'],
                            'archive':str(m['directory']),'archive_sha256':m['sha256'],
                            'benchmark_config':oldconfig,'status':'pending'})
    requested=set(variants or [])
    if requested-{r['variant'] for r in records}:
        raise ValueError('Requested variants have no matching score caches')
    return records


def original_job(record):
    i=record['identity'];method=i['method'];ckpt=resolve_repo_path(i['checkpoint'])
    family,config_key,mode=b.METHODS[method][1:]
    if method=='ace' or method.startswith('act_ensemble_'):
        train=ckpt.parent/'config.json';score=train
    else:
        names=['config.local.json','config.json','flow_matching_wm.yaml'] if mode else ['config.local.json',family+'.yaml','config.json']
        train=next((ckpt.parent.parent/n for n in names if (ckpt.parent.parent/n).is_file()),None)
        if train is None: raise FileNotFoundError(f'Training config: {ckpt}')
        candidates=[resolve_repo_path(p) for p in i['sha256'] if Path(p).suffix in {'.yaml','.yml'} and 'score' in Path(p).stem]
        if len(candidates)!=1: raise ValueError(f'Ambiguous original score config: {candidates}')
        score=candidates[0]
    return b.Job(i['task'],method,tuple(i['views']),ckpt,train,score,{},deepcopy(i['merged_config']),
                 resolve_repo_path(i['dataset']),family,config_key,mode,i['batch_size'],i['seed'],i['device'])


def expert_job(record, original):
    # Read exact saved IDs, with dataset identity checked BEFORE selecting them.
    policy_method = original.method == 'ace' or original.method.startswith('act_ensemble_')
    if policy_method:
        expert=b.REPO/'fomofd/tasks'/original.task/'data'/original.dataset.name
        split_path=expert/'meta/preparation_splits.json'
        training_seed=None
    else:
        trained=b.load_node(original.train_config,original.config_key)
        expert=b.rooted(trained['data']['dataset_root'],b.REPO/'lerobot')
        split_path=original.checkpoint.parent.parent/'splits.json'
        training_seed=trained.get('runtime',{}).get('seed',trained.get('seed'))
    if not expert.is_relative_to(b.REPO/'fomofd/tasks'/original.task/'data'):
        raise ValueError('No expert-validation dataset for this checkpoint; policy rollouts cannot substitute')
    split=json.loads(split_path.read_text());ids=split['val_episode_ids'];train=split['train_episode_ids']
    if not ids or any(type(e)!=int for e in ids+train) or len(ids)!=len(set(ids)) or set(ids)&set(train):
        raise ValueError('Invalid saved expert episode split')
    preparation=json.loads((expert/'meta/preparation.json').read_text())
    all_ids=sorted(e['dataset_episode_index'] for e in preparation['episode_mapping'])
    if not set(ids)<=set(all_ids):raise ValueError('Saved expert IDs missing from prepared dataset')
    merged=deepcopy(original.merged)
    merged['data']['dataset_root']=str(expert)
    if not policy_method:
        # Existing scorers reconstruct val IDs. Prove this matches the manifest
        # before running, and validate every returned ID/frame afterwards.
        shuffled=all_ids[:];random.Random(merged['data']['split_seed']).shuffle(shuffled)
        n=max(1,min(len(shuffled)-1,round(len(shuffled)*merged['data']['val_ratio'])))
        if sorted(shuffled[:n])!=sorted(ids):
            raise ValueError('Scorer val split differs from saved manifest; explicit-ID adapter required')
    job=replace(original,dataset=expert,merged=merged,override=deepcopy(merged),split='val',episode_ids=tuple(sorted(ids)))
    provenance={'role':'expert_validation','dataset':str(expert),'split_manifest':str(split_path),
                'split_sha256':b.digest(split_path),'episode_ids':sorted(ids),
                'selection':'saved validation IDs; reconstructed scorer selection verified before/after inference',
                'checkpoint':str(original.checkpoint),'training_seed':training_seed,
                'score_seed':original.seed,'prl_score_cache_sha256':record['prl_cache_sha256']}
    if policy_method:
        provenance['selection']='saved preparation validation manifest (ACT has no detector-training split)'
    return job,provenance


def validate_frozen_model(record, job):
    identity=record['identity']
    # Changes to calibration/report code do not invalidate saved PRL scores.
    # Model/scorer code, checkpoint, normalizers and feature exports must agree.
    required=[job.checkpoint,job.train_config]
    if job.method.startswith('act_ensemble_'):
        members = [resolve_repo_path(p) for p in job.merged['ensemble']['checkpoints']]
        required += members + [p.parent/'config.json' for p in members]
        required += [b.REPO/'lerobot/utils/act_ensemble_score.py']
    elif job.method!='ace':
        required += [b.REPO/'lerobot/models'/job.family/n for n in ['model.py','score.py','dataset.py','config.py']]
    for k in ['vfmf_vae_latent_stats_path','state_stats_path','action_stats_path']:
        if job.merged['data'].get(k): required.append(b.rooted(job.merged['data'][k],b.REPO/'lerobot'))
    recorded={resolve_repo_path(p):sha for p,sha in identity['sha256'].items()}
    for path in required:
        if path not in recorded or b.digest(path)!=recorded[path]:
            raise ValueError(f'Frozen scoring input no longer matches PRL cache: {path}')
    b.validate_native_provenance(job)


def wilson(k,n):
    if not n:return None,None
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-r),min(1,c+r)


def enrich_metrics(episodes):
    rows=b.aggregate_metrics(episodes)
    for r in rows:
        for metric,k,n in [('FAR','false_alarms','success_episodes'),('FDR','detected_failures','failure_episodes')]:
            r[metric+'_ci95_low'],r[metric+'_ci95_high']=wilson(r[k],r[n])
        r['ci_method']='Wilson 95% (descriptive episode proportions)'
    return rows


def deadlines(latencies,seconds):
    grouped=defaultdict(list)
    for r in latencies:
        grouped[tuple(r[k] for k in ['task','method','view','calibration','annotator_id','onset_view'])].append(r)
    output=[]
    for key,items in grouped.items():
        valid=[r for r in items if r['status'] not in {'unannotated','timestamp_unavailable'}]
        for deadline in seconds:
            hit=sum(r['status']=='detected' and r['detection_delay_s']<=deadline for r in valid)
            lo,hi=wilson(hit,len(valid))
            output.append(dict(zip(['task','method','view','calibration','annotator_id','onset_view'],key)) | {
                'deadline_s':deadline,'detected_by_deadline':hit,'annotated_failures':len(valid),
                'detection_rate':hit/len(valid) if valid else None,'ci95_low':lo,'ci95_high':hi,
                'pre_onset_alarms':sum(r['status']=='pre_onset_alarm' for r in valid),
                'missed':sum(r['status']=='missed' for r in valid),
                'onset_after_scored_end':sum(r['status']=='onset_after_scored_end' for r in valid),
                'excluded_missing_onsets':len(items)-len(valid)})
    return output


def comparison_figures(root, metrics):
    """Static, exportable FDR/FAR comparisons at deployed calibrated thresholds."""
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    images=[]
    for variant in sorted({r['variant'] for r in metrics}):
        rows=[r for r in metrics if r['variant']==variant and r['task']=='Overall']
        keys=sorted({(r['method'],r['view']) for r in rows})
        if not keys:continue
        fig,axes=plt.subplots(2,1,figsize=(max(8,len(keys)*.85),7),sharex=True)
        x=np.arange(len(keys));width=.25
        for index,scheme in enumerate(['episode','wall_time_bin','functional_band']):
            lookup={(r['method'],r['view']):r for r in rows if r['calibration']==scheme}
            for ax,metric in zip(axes,['FDR','FAR']):
                values=np.array([lookup[k][metric] if k in lookup else np.nan for k in keys])
                low=np.array([lookup[k][metric+'_ci95_low'] if k in lookup else np.nan for k in keys])
                high=np.array([lookup[k][metric+'_ci95_high'] if k in lookup else np.nan for k in keys])
                ax.bar(x+(index-1)*width,values,width,label=scheme,
                       yerr=np.maximum(0,np.array([values-low,high-values])),capsize=2)
        for ax,metric in zip(axes,['FDR','FAR']):
            ax.set_ylabel(metric);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        axes[0].set_ylim(0,1.05);axes[0].legend(ncol=3,fontsize=9)
        axes[1].set_xticks(x,[m+' / '+v for m,v in keys],rotation=50,ha='right')
        fig.suptitle(variant+' · calibrated thresholds · 95% Wilson intervals')
        fig.tight_layout();name=variant+'_threshold_comparison'
        for ext in ['png','svg','pdf']:fig.savefig(root/(name+'.'+ext),dpi=150,bbox_inches='tight')
        plt.close(fig);images.append(name)
    return images


def write_comparison(root, availability, metrics, latencies, episode_rows, metadata):
    b.write_csv(root/'availability.csv',availability,['status'])
    b.write_csv(root/'comparison_metrics.csv',metrics,['status'])
    b.write_csv(root/'comparison_latency.csv',latencies,['status'])
    save(root/'comparison.json',metadata | {'availability':availability,'metrics':metrics,'latency':latencies})
    notes=[f.ADAPTATION,
           'The existing time_varying alias is wall_time_bin: diagnostic per-bin calibration, without simultaneous coverage.',
           'All protocols use the same verified evaluation episodes. Paper endpoint selection and all-endpoint causal delay retain their historical conventions; templates and h are fitted separately for these two protocols.',
           '95% intervals are Wilson intervals (there were no confidence intervals in the original unified reports). Deadline rates retain pre-onset alarms and misses in the annotated-failure denominator; missing onsets are explicit.',
           'Component-calibrated AND/OR fusions are not assigned a new functional band: their calibration-dependent score definition lacks an independent expert-only template. No joint coverage claim is made.',
           'Only frozen-model expert-validation scores may be newly generated. PRL scores, annotations, checkpoints, state inclusion, normalization and score orientation are unchanged.']
    header=['Variant','Task','Method','View','Protocol','FAR','FDR','AUROC','FA/n','Detected/n']
    display=lambda x:'N/A' if x is None else f'{100*x:.2f}%'
    cells=[[r['variant'],r['task'],r['method'],r['view'],r['calibration'],display(r['FAR']),display(r['FDR']),f"{r['episode_AUROC']:.4f}" if r['episode_AUROC'] is not None else 'N/A',f"{r['false_alarms']}/{r['success_episodes']}",f"{r['detected_failures']}/{r['failure_episodes']}"] for r in metrics if r['task']=='Overall']
    md='# Functional conformal comparison\n\n'+'\n\n'.join(notes)+'\n\n'+b.markdown_table(header,cells)
    failures=[r for r in availability if r['status']!='complete']
    md+='\n## Missing or unavailable comparisons\n\n'+b.markdown_table(['Variant','Task','Method','Views','Status','Reason'],[[r.get(k,'') for k in ['variant','task','method','views','status','reason']] for r in failures])
    md+='\nDetailed metrics, 95% intervals, deadline rates, onset statuses, ODE settings, templates and calibration residuals are in the protocol subdirectories.\n'
    (root/'report.md').write_text(md)
    tables='<tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in header)+'</tr>'
    tables+=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in cells)
    links=[]
    for variant in sorted({r['variant'] for r in metrics}):
        for scheme in ['episode','wall_time_bin','functional_band']:
            path=root/variant/scheme/'report.html'
            if path.is_file():links.append(f'<a href="{variant}/{scheme}/report.html">{variant} / {scheme}</a>')
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Functional conformal comparison</title><style>body{font:14px system-ui;color:#193142;margin:28px}p{max-width:1100px}td,th{padding:7px;border-bottom:1px solid #ddd}th{position:sticky;top:0;background:#eef4f7}table{border-collapse:collapse}</style><h1>Functional conformal comparison</h1>'+''.join('<p>'+html.escape(n)+'</p>' for n in notes)+'<p><a href="report.md">Markdown</a> · <a href="comparison_metrics.csv">Metrics / confidence intervals</a> · <a href="availability.csv">Availability</a> · <a href="roc/roc_report.html">ROC / matched FAR</a></p><p>'+' · '.join(links)+'</p><table>'+tables+'</table><h2>Availability</h2><pre>'+html.escape(json.dumps(failures,indent=2))+'</pre>')
    if episode_rows:
        roc_results=roc.analyze(episode_rows)
        roc.write_reports(root/'roc',roc_results,metadata,report_stem='roc_report')
    images=comparison_figures(root,metrics)
    with (root/'report.md').open('a') as stream:
        for name in images:stream.write(f'\n![{name}]({name}.png)\n\n[PDF]({name}.pdf) · [SVG]({name}.svg)\n')
    with (root/'report.html').open('a') as stream:
        for name in images:stream.write(f'<h2>{html.escape(name)}</h2><p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p><img style="width:100%;max-width:1400px" src="{name}.png" alt="Threshold comparison">')


def write_variant(root,variant,scored,templates,template_entries,records,labels,labels_text,config):
    methods=list(dict.fromkeys(j.method for j,_,_ in scored));tasks=list(dict.fromkeys(j.task for j,_,_ in scored))
    # Only complete task/method camera sets reach evaluation.
    oldconfig=deepcopy(records[0]['benchmark_config'])
    oldconfig['fusion']={'wm_views':False,'conditional_marginal':False}
    oldconfig['alpha']=config['alpha']
    oldconfig['wall_time_bin']=config['wall_time_bin']
    if config.get('evaluation_protocol'):
        oldconfig['evaluation_protocol'] = config['evaluation_protocol']
    all_metrics,all_latencies,all_episodes=[],[],[]
    for scheme in ['episode','wall_time_bin','functional_band']:
        folder=root/variant/scheme;folder.mkdir(parents=True,exist_ok=False)
        reports={'episodes':[],'latencies':[],'distributions':[],'calibration':[]}
        for task in tasks:
            jobs=[s for s in scored if s[0].task==task]
            task_methods=list(dict.fromkeys(j.method for j,_,_ in jobs))
            out=b.evaluate_suite(jobs,labels,oldconfig,list(labels['annotators']),task_methods,[scheme],templates)
            for row in out['episodes']+out['latencies']:
                source=templates[(task,row['method'],row['view'],'paper')]['source']
                row.update(checkpoint=source['checkpoint'],training_seed=source['training_seed'],score_seed=source['score_seed'])
            for key,value in out.items():reports[key].extend(value)
        metrics=enrich_metrics(reports['episodes']);latency=b.latency_summary(reports['latencies'])
        deadline=deadlines(reports['latencies'],config['deadline_seconds'])
        settings=[r['settings'] for r in records]
        report={'created_utc':b.utc_now(),'script_sha256':b.digest(Path(__file__)),'alpha':config['alpha'],
                'tasks':tasks,'methods':methods,'labels_sha256':hashlib.sha256(labels_text.encode()).hexdigest(),
                'protocols':{k:oldconfig[k] for k in ['paper_protocol','latency_protocol']},'scoring_settings':settings,
                'calibration':reports['calibration'],'summary_metrics':metrics,'latency_summary':latency,
                'calibration_schemes':[scheme],'latency_note':'Dense causal onset-to-first-alarm delay in recorded wall seconds; not online compute latency.',
                'time_varying_note':f.ADAPTATION,'fusion_note':'Only base continuous scorers; component-calibrated fusions not applicable to this band comparison.',
                'templates_manifest':'../templates_manifest.json','score_caches':[{'task':r['task'],'method':r['method'],'views':r['views'],'path':r['prl_cache'],'sha256':r['prl_cache_sha256'],'scoring_settings':r['settings']} for r in records]}
        tables={}
        # All requested methods have the same completed task coverage in a full
        # run. Partial runs use summary metrics, never manufacture table cells.
        if all(set(j.method for j,_,_ in scored if j.task==task)==set(methods) for task in tasks):
            tables[scheme]=b.paper_table(reports['episodes'],oldconfig,labels,tasks,methods)
            header,table=tables[scheme]
            import csv
            with (folder/'paper_table.csv').open('w') as stream:
                writer=csv.writer(stream);writer.writerow(header);writer.writerows(table)
            table_stem='paper_table' if scheme=='episode' else 'paper_table_'+scheme
            if table_stem!='paper_table':
                (folder/(table_stem+'.csv')).write_bytes((folder/'paper_table.csv').read_bytes())
            (folder/(table_stem+'.md')).write_text(b.markdown_table(header,table))
        for name,rows in [('episode_results',reports['episodes']),('latency_episodes',reports['latencies']),('latency_summary',latency),('summary_metrics',metrics),('calibration',reports['calibration']),('score_distributions',reports['distributions']),('deadline_rates',deadline),('scoring_settings',settings)]:
            b.write_csv(folder/(name+'.csv'),rows,['status'])
        save(folder/'report.json',report);save(folder/'benchmark_config.json',oldconfig)
        (folder/'labels_snapshot.yaml').write_text(labels_text)
        b.write_inspection_reports(folder,report,tables,reports['distributions'])
        all_metrics.extend({'variant':variant,**r} for r in metrics)
        all_latencies.extend({'variant':variant,**r} for r in latency)
        all_episodes.extend({'variant':variant,**r} for r in reports['episodes'])
    save(root/variant/'templates_manifest.json',{'templates':template_entries})
    return all_metrics,all_latencies,all_episodes


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',type=Path,default=b.REPO/'lerobot/configs/benchmark/functional_band.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variants',nargs='+');parser.add_argument('--tasks',nargs='+');parser.add_argument('--methods',nargs='+',choices=list(b.METHODS))
    parser.add_argument('--plan-only',action='store_true');parser.add_argument('--allow-inference',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Resume score acquisition; completed protocol reports remain immutable')
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text())
    if config['schema_version']!=1 or not 0<config['alpha']<1:raise ValueError('Invalid functional config')
    root=args.output.resolve()
    selection={'variants':args.variants,'tasks':args.tasks,'methods':args.methods}
    if root.exists():
        if not args.resume:raise FileExistsError(root)
        previous=json.loads((root/'run_config.json').read_text())
        if previous!={'config':config,'selection':selection}:raise ValueError('Resume configuration differs')
        if (root/'comparison.json').is_file():raise ValueError('Completed reports exist: use a fresh output directory')
    root.mkdir(parents=True,exist_ok=True);save(root/'run_config.json',{'config':config,'selection':selection})
    (root/'functional_band.yaml').write_text(args.config.read_text())
    start=time.perf_counter();records=sources(config,args.variants,args.tasks,args.methods)
    labels,labels_text=b.load_labels(b.REPO/config['labels'],list(dict.fromkeys(r['task'] for r in records)))
    if config.get('evaluation_protocol'):
        from benchmark_frozen_protocol import load_protocol
        from benchmark_consensus_onsets import derive_labels
        from benchmark_endpoint_protocol import protocol
        spec = load_protocol(config['evaluation_protocol'])
        labels = derive_labels(labels, spec, resolve_repo_path)
        for record in records:
            record['benchmark_config'].update(paper_protocol=protocol(True), latency_protocol=protocol(True),
                                               evaluation_protocol=config['evaluation_protocol'])
    for task in {r['task'] for r in records}:
        cal=[e for e,g in b.episode_labels(labels['tasks'][task]).items() if g['split']=='calibration' and g['outcome']=='success']
        if sorted(cal)!=list(range(19)):raise ValueError('Keep all original 19 calibration successes (0–18)')
    availability=[];ready=defaultdict(list);templates=defaultdict(dict);template_entries=defaultdict(list);used_records=defaultdict(list)
    timelines={}
    def clock(path):
        if path not in timelines:timelines[path]=b.load_timeline(path)
        return timelines[path]
    # Record every requested comparison before inference; state is updated after
    # each job so interruptions expose exact missing caches rather than zeros.
    save(root/'state.json',{'status':'planning','jobs':len(records),'created_utc':b.utc_now()})
    for index,record in enumerate(records):
        item={k:record[k] for k in ['variant','task','method','views'] if k in record}
        item.update(status='pending',reason='')
        try:
            if record['status']=='not_applicable':raise ValueError(record['reason'])
            original=original_job(record);expert,provenance=expert_job(record,original)
            item.update(expert_dataset=str(expert.dataset),expert_validation_ids=list(expert.episode_ids),split_manifest=provenance['split_manifest'],checkpoint=str(original.checkpoint),training_seed=provenance['training_seed'],score_seed=original.seed)
            if args.plan_only:
                item.update(status='needs_expert_scores',reason='Exact saved validation IDs verified; no inference performed')
                continue
            validate_frozen_model(record,expert)
            identity=b.fingerprint(expert,b.REPO)
            target=expert.cache_dir/(hashlib.sha256(b.canonical(identity).encode()).hexdigest()+'.npz')
            item['expert_cache']=str(target)
            if not target.is_file() and not args.allow_inference:
                item.update(status='missing_expert_scores',reason='Use --allow-inference to generate only the saved expert-validation scores')
                continue
            fullclock=clock(expert.dataset);expert_clock={e:fullclock[e] for e in expert.episode_ids}
            options=SimpleNamespace(cache_only=not args.allow_inference,mode='normal',allow_inference=args.allow_inference)
            scores=b.acquire_scores(expert,b.REPO,expert_clock,identity,target,options)
            provenance.update(score_cache=str(target),score_cache_sha256=b.digest(target),checkpoint_sha256=b.digest(original.checkpoint))
            with np.load(record['prl_cache'],allow_pickle=False) as data:prl={k:data[k] for k in ARRAY_KEYS}
            prl_clock=clock(original.dataset);b.validate_scores(prl,original,prl_clock)
            if set(prl_clock)!=set(b.episode_labels(labels['tasks'][original.task])):raise ValueError('PRL IDs differ from verified labels')
            variant=record['variant']
            for i,view in enumerate(original.views):
                for name,protocol in [('paper',record['benchmark_config']['paper_protocol']),('latency',record['benchmark_config']['latency_protocol'])]:
                    trace=b.curves(scores,expert,i,protocol)
                    template=f.fit_template(trace,expert_clock,list(expert.episode_ids),provenance,config['template'])
                    key=(original.task,original.method,view,name);templates[variant][key]=template
                    path=root/variant/'templates'/('_'.join(key)+'.json');save(path,template)
                    template_entries[variant].append(dict(zip(['task','method','view','protocol'],key)) | {'path':str(path.relative_to(root/variant)),'sha256':b.digest(path)})
            ready[variant].append((original,prl,prl_clock));used_records[variant].append(record)
            item.update(status='complete',reason='Expert template frozen; original PRL cache reused')
        except (OSError,ValueError,KeyError,RuntimeError) as error:
            item.update(status='unavailable',reason=str(error))
        finally:
            availability.append(item)
            print(f"{index+1}/{len(records)} {item['variant']}/{item['task']}/{item['method']}/{item.get('views')}: {item['status']} {item['reason']}",flush=True)
            save(root/'availability.json',availability)
            save(root/'state.json',{'status':'acquiring_expert_scores','completed_jobs':index+1,'total_jobs':len(records),'updated_utc':b.utc_now()})
    if args.plan_only:
        save(root/'state.json',{'status':'planned','jobs':len(records),'models_loaded':0})
        print('Plan saved:',root);return
    all_metrics=[];all_latencies=[];all_episodes=[]
    for variant,scored in ready.items():
        complete={(j.task,j.method) for j,_,_ in scored if set(v for other,_,_ in scored if (other.task,other.method)==(j.task,j.method) for v in other.views)==({'both'} if j.method=='ace' or j.method.startswith('act_ensemble_') else set(b.CAMERAS))}
        scored=[s for s in scored if (s[0].task,s[0].method) in complete]
        if not scored:continue
        used=[r for r in used_records[variant] if (r['task'],r['method']) in complete]
        metrics,latencies,episodes=write_variant(root,variant,scored,templates[variant],template_entries[variant],used,labels,labels_text,config)
        all_metrics+=metrics;all_latencies+=latencies;all_episodes+=episodes
    metadata={'created_utc':b.utc_now(),'script_sha256':b.digest(Path(__file__)),'core_sha256':b.digest(Path(f.__file__)),
              'config':config,'selection':selection,'elapsed_seconds':time.perf_counter()-start,
              'adaptation':f.ADAPTATION,'prl_rescoring':False,'model_training':False,
              'sources':[{'variant':r['variant'],'cache':r.get('prl_cache'),'sha256':r.get('prl_cache_sha256')} for r in records]}
    write_comparison(root,availability,all_metrics,all_latencies,all_episodes,metadata)
    # Immutable protocol artifacts can be consumed by the existing results
    # inspector without changing the old archived reports/catalog comparisons.
    artifacts={str(p.relative_to(root)):b.digest(p) for p in sorted(root.rglob('*'))
               if p.is_file() and '.matplotlib_cache' not in p.parts and p.name not in {'state.json','artifact_manifest.json'}}
    save(root/'artifact_manifest.json',{'files':artifacts,'created_utc':b.utc_now()})
    status='complete' if all(r['status']=='complete' for r in availability) else 'partial'
    save(root/'state.json',{'status':status,'completed_scores':sum(r['status']=='complete' for r in availability),'total_scores':len(records),'finished_utc':b.utc_now(),'elapsed_seconds':time.perf_counter()-start})
    print(f'{status}: {root}/report.html',flush=True)


if __name__=='__main__':main()
