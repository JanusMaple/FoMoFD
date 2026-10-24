#!/usr/bin/env python3
"""Describe sampled-score duration/phase effects from immutable caches (CPU only).

Default: original calibration successes only. --include-failures requires a
frozen revised sampling manifest and matching revised score provenance. These
are descriptive checks, not a way to choose orientation, thresholds or bins.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

import sampled_score_protocol as p
import unified_benchmark as b


def diagnose(report_path, labels_path, output, include_failures=False, frozen=None):
    from scipy.stats import spearmanr
    report = json.loads(report_path.read_text())
    labels = yaml.safe_load(labels_path.read_text())
    frozen_sha = None
    if include_failures:
        manifest = json.loads(frozen.read_text()) if frozen else {}
        if manifest.get('status') != 'frozen_before_policy_calibration_and_test':
            raise ValueError('Failure diagnostics require frozen validated settings')
        frozen_sha = b.digest(frozen)
    output.mkdir(parents=True, exist_ok=False)
    all_rows = {name: [] for name in ('episodes', 'phase', 'prefixes')}
    summaries = []
    for cache in report['score_caches']:
        if cache['method'] not in {'ace', 'wm_pu'}:
            continue
        path = b.rooted(cache['path'], b.REPO)
        if b.digest(path) != cache['sha256']:
            raise ValueError('Changed source score cache: '+str(path))
        with np.load(path, allow_pickle=False) as data:
            identity = json.loads(str(data['provenance_json'].item()))['identity']
            if include_failures:
                source = identity['merged_config']['sampling'].get('sampling_manifest')
                if not source or b.digest(b.rooted(source, b.REPO)) != frozen_sha:
                    raise ValueError('Failure diagnostics require the revised frozen scores')
            groups = b.episode_labels(labels['tasks'][cache['task']])
            allowed = {e for e, g in groups.items() if
                       (g['split'] == 'calibration' and g['outcome'] == 'success') or
                       (include_failures and g['split'] == 'test')}
            keep = np.isin(data['episode_index'], sorted(allowed))
            arrays = {k: data[k][keep] for k in
                      ('episode_index', 'frame_index', 'raw_frame_index', 'wall_timestamp', 'scores')}
        job = SimpleNamespace(task=cache['task'], mode='reactive' if cache['method']=='wm_pu' else None,
                              merged=identity['merged_config'])
        for i, view in enumerate(cache['views']):
            curves = b.curves(arrays, job, i, report['protocols']['paper_protocol'])
            categories = ['calibration_success'] + (['test_success', 'failure'] if include_failures else [])
            for category in categories:
                chosen = {e: c for e, c in curves.items() if
                          ('failure' if groups[e]['outcome']=='failure' else
                           'calibration_success' if groups[e]['split']=='calibration' else 'test_success') == category}
                if not chosen:
                    continue
                stats = p.phase_length_summary(chosen)
                tags = dict(task=cache['task'], method=cache['method'], view=view, group=category)
                for name in all_rows:
                    all_rows[name] += [tags | r for r in stats[name]]
                lengths = np.array([r['windows'] for r in stats['episodes']])
                maxima = np.array([r['maximum'] for r in stats['episodes']])
                rho = float(spearmanr(lengths, maxima).statistic) if len(set(lengths))>1 and len(set(maxima))>1 else None
                summaries.append(tags | {'episodes': len(chosen), 'median_windows': float(np.median(lengths)),
                    'median_scored_duration_s': float(np.median([r['duration_s'] for r in stats['episodes']])),
                    'window_count_max_score_spearman': rho,
                    'median_maximum': float(np.median(maxima)), 'cache_sha256': cache['sha256']})
    for name, rows in all_rows.items():
        b.write_csv(output/(name+'.csv'), rows, ['task'])
    b.write_csv(output/'summary.csv', summaries, ['task'])
    notes = [
        'Descriptive duration/phase analysis; no score inversion or setting selection.',
        'Scored duration spans the eligible score endpoints, not the complete recording.',
        'Relative phase uses final episode duration and is an offline diagnostic only.',
        'Prefix maxima measure the opportunity effect: a longer prefix cannot reduce its maximum.',
        'Correlations and group differences do not establish why a detector succeeds or fails.',
        *p.BOUNDARIES.values()]
    payload = {'created_utc': b.utc_now(), 'report': str(report_path), 'report_sha256': b.digest(report_path),
               'frozen_sampling_sha256': frozen_sha, 'notes': notes, 'summary': summaries, **all_rows}
    (output/'diagnostics.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', str(output/'.matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = []
    for task in sorted({r['task'] for r in summaries}):
        keys = sorted({(r['method'],r['view']) for r in summaries if r['task']==task})
        fig, axes = plt.subplots(len(keys), 3, figsize=(12, 3.2*len(keys)), squeeze=False)
        for row, (method, view) in enumerate(keys):
            selected = lambda rows, group: [r for r in rows if
                (r['task'],r['method'],r['view'],r['group'])==(task,method,view,group)]
            for group in categories:
                episodes = selected(all_rows['episodes'],group)
                if not episodes: continue
                axes[row,0].scatter([r['windows'] for r in episodes], [r['maximum'] for r in episodes], s=15,label=group,alpha=.65)
                for column, name, key, value in [(1,'phase','phase_start','mean'),(2,'prefixes','fraction','maximum')]:
                    data = selected(all_rows[name],group)
                    xs=sorted({r[key] for r in data})
                    ys=[np.array([r[value] for r in data if r[key]==x]) for x in xs]
                    axes[row,column].plot(xs,[np.median(y) for y in ys],marker='.',label=group)
                    axes[row,column].fill_between(xs,[np.quantile(y,.25) for y in ys],[np.quantile(y,.75) for y in ys],alpha=.15)
            axes[row,0].set_ylabel(method+' / '+view)
            for ax,title in zip(axes[row],['Window count → episode max','Relative phase → mean score','Prefix fraction → max score']):
                ax.set_title(title); ax.grid(alpha=.2)
            axes[row,0].legend(fontsize=7)
        fig.suptitle(task+' · medians and interquartile bands (descriptive)')
        fig.tight_layout(); name=task+'_length_phase.png';fig.savefig(output/name,dpi=140);plt.close(fig);plots.append(name)
    header=['Task','Method/view','Group','n','Median windows','Count/max Spearman']
    rows=[[r['task'],r['method']+'/'+r['view'],r['group'],r['episodes'],r['median_windows'],
           'undefined' if r['window_count_max_score_spearman'] is None else f"{r['window_count_max_score_spearman']:.3f}"] for r in summaries]
    text='\n\n'.join(notes)+'\n\n'+b.markdown_table(header,rows)+'\n'+ '\n'.join(f'![{x}]({x})' for x in plots)
    (output/'report.md').write_text(text)
    import html
    (output/'report.html').write_text('<meta charset="utf-8"><title>Sampled-score diagnostics</title><style>body{max-width:1200px;margin:32px auto;font:16px sans-serif}img{max-width:100%}pre{white-space:pre-wrap}</style><pre>'+html.escape('\n\n'.join(notes))+'</pre>'+''.join('<img src="'+x+'">' for x in plots)+'<p><a href="report.md">Text report</a> · <a href="summary.csv">Summary CSV</a></p>')
    print('Saved CPU-only length/phase diagnostics:',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-failures',action='store_true')
    parser.add_argument('--frozen',type=Path)
    args=parser.parse_args()
    diagnose(args.report.resolve(),args.labels.resolve(),args.output.resolve(),args.include_failures,args.frozen)
