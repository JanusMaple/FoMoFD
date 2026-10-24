#!/usr/bin/env python3
"""Small CPU-only matched-FAR supplement using saved splits and episode scores.

python lerobot/utils/calibration_matched_far.py
Never loads a model, generates scores, refits templates, or edits primary results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import shutil

import numpy as np
import yaml

from benchmark_roc import curve, operating_point, safe

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO/'lerobot/configs/benchmark/calibration_matched_far_main_500.yaml'
KEYS = ('task', 'view', 'calibration')
NOTE = (
    'Main K4/L4 WM-NC only. All 500 saved 19/20 success splits are reused, with identical '
    'splits across views and protocols and unchanged failure episodes. For each split, '
    'select a strict score > threshold operating point using ONLY its 20 test successes '
    'at FAR <= 5% (second-highest success maximum; ties may lower achieved FAR). '
    'The other 19 successes do not determine this retrospective matched-FAR threshold. '
    'This measures sensitivity to the test-success subset, not a deployment calibration '
    'guarantee. Episode scores and frozen functional-band residuals are reused. '
    'FDR uses post-onset maxima; early-only crossings are misses. AUROC uses whole-episode '
    'maxima. Overall FAR/FDR pool counts across tasks; AUROC averages tasks equally. '
    'Mean, sample SD and linear 5th–95th percentiles describe overlapping resamples, '
    'not independent experiments or confidence intervals. No favorable splits are selected. '
    'The primary conformal results and original split are unchanged. No inference or latency '
    'recomputation. Thresholds use saved report score coordinates, not necessarily raw WM units.'
)


def validate_splits(manifest, n_splits):
    if manifest['n_splits'] != n_splits or len(manifest['tasks']) != 4:
        raise ValueError('Expected four tasks and the requested number of saved splits')
    for task in manifest['tasks'].values():
        success, failure = set(task['success_ids']), set(task['failure_ids'])
        if len(success) != 39 or success & failure:
            raise ValueError('Invalid success/failure population')
        draws = task['random_splits']
        if len(draws) != n_splits or len({s['split_id'] for s in draws}) != n_splits:
            raise ValueError('Missing or duplicate split IDs')
        if len({tuple(sorted(s['calibration_ids'])) for s in draws}) != n_splits:
            raise ValueError('Expected unique saved partitions')
        for split in [task['original'], *draws]:
            cal, test = split['calibration_ids'], split['test_success_ids']
            if (len(cal) != 19 or len(set(cal)) != 19 or len(test) != 20 or
                    len(set(test)) != 20 or set(cal) & set(test) or set(cal) | set(test) != success):
                raise ValueError('Split must partition all 39 successes into disjoint 19/20 sets')


def evaluate_split(by_id, task, split, target):
    rows = [by_id[i] for i in split['test_success_ids'] + task['failure_ids']]
    for r in rows:
        if r['outcome'] == 'failure':
            if r.get('detection_definition') != 'at_or_after_onset':
                raise ValueError('Failure FDR requires post-onset score maxima')
            value = float(r['post_score_max'])
            if math.isnan(value) or value == math.inf:
                raise ValueError('Invalid post-onset score')
    return dict(split_id=split['split_id'], **operating_point(rows, target),
                episode_AUROC=curve(rows)[1])


def summarize(rows, originals):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in KEYS)].append(row)
    references = {tuple(r[k] for k in KEYS): r for r in originals}
    result = []
    for key, items in sorted(groups.items()):
        row = dict(zip(KEYS, key), n_splits=len(items))
        for metric in ('FAR', 'FDR', 'episode_AUROC', 'threshold'):
            row['original_'+metric] = references[key][metric]
            if key[0] == 'Overall' and metric == 'threshold':
                continue
            values = np.array([r[metric] for r in items], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError('Unexpected nonfinite summary metric')
            row.update({metric+'_'+k: float(v) for k, v in dict(
                mean=values.mean(), sd=values.std(ddof=1),
                p05=np.quantile(values,.05), p95=np.quantile(values,.95),
                min=values.min(), max=values.max()).items()})
        result.append(row)
    return result


def pool_tasks(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['view'],r['calibration'],r['split_id']].append(r)
    result = []
    for (view, calibration, split), items in groups.items():
        if len(items) != 4 or len({r['task'] for r in items}) != 4:
            raise ValueError('Overall requires one result per task')
        row = dict(task='Overall', view=view, calibration=calibration, split_id=split,
                   threshold=None, comparison='per-task thresholds')
        for k in ('false_alarms','success_episodes','detected_failures','failure_episodes'):
            row[k] = sum(r[k] for r in items)
        row.update(FAR=row['false_alarms']/row['success_episodes'],
                   FDR=row['detected_failures']/row['failure_episodes'],
                   episode_AUROC=float(np.mean([r['episode_AUROC'] for r in items])))
        result.append(row)
    return result


def save_csv(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(safe(rows))


def run(config_path):
    config = yaml.safe_load(config_path.read_text())
    source = REPO/config['source_report']
    splits_path = REPO/config['split_manifest']
    splits = json.loads(splits_path.read_text())
    validate_splits(splits, config['n_splits'])
    report = json.loads((source/'report.json').read_text())
    protocol_path = REPO/config['evaluation_protocol']
    protocol = yaml.safe_load(protocol_path.read_text())
    if report['evaluation_protocol'] != protocol:
        raise ValueError('Source report does not match the frozen protocol')
    selected, groups = [], defaultdict(dict)
    for r in csv.DictReader((source/'episode_results.csv').open()):
        if (r['method'] != config['method'] or r['view'] not in config['views'] or
                r['calibration'] not in config['calibrations']):
            continue
        key, ep = tuple(r[k] for k in KEYS), int(r['episode_index'])
        if ep in groups[key]:
            raise ValueError('Duplicate episode evidence')
        groups[key][ep] = r
        selected.append({k:r[k] for k in (*KEYS, 'method','episode_index','outcome',
                         'score_max','post_score_max','detection_definition')})
    expected = {(t,v,c) for t in splits['tasks'] for v in config['views'] for c in config['calibrations']}
    if set(groups) != expected:
        raise ValueError('Incomplete detector/task/view/protocol coverage')
    results, originals = [], []
    for key, by_id in sorted(groups.items()):
        task = splits['tasks'][key[0]]
        if (set(by_id) != set(task['success_ids']+task['failure_ids']) or
            {i for i,r in by_id.items() if r['outcome']=='success'} != set(task['success_ids']) or
            {i for i,r in by_id.items() if r['outcome']=='failure'} != set(task['failure_ids'])):
            raise ValueError('Episode labels/populations differ from saved split manifest')
        base = dict(zip(KEYS, key))
        originals.append(dict(base, **evaluate_split(by_id,task,task['original'],config['target_far'])))
        results.extend(dict(base, **evaluate_split(by_id,task,s,config['target_far'])) for s in task['random_splits'])
    results += pool_tasks(results)
    originals += pool_tasks(originals)
    reference = {tuple(r[k] for k in KEYS):r for r in csv.DictReader((source/'matched_far.csv').open())
                 if r['variant']=='main' and r['method']==config['method'] and r['failure_subset']=='all'
                 and float(r['target_FAR'])==config['target_far']}
    for r in originals:
        old = reference[tuple(r[k] for k in KEYS)]
        for metric in ('false_alarms','success_episodes','detected_failures','failure_episodes',
                       'FAR','FDR','episode_AUROC') + (() if r['task']=='Overall' else ('threshold',)):
            if not math.isclose(r[metric],float(old[metric]),abs_tol=1e-12,rel_tol=1e-12):
                raise ValueError('Original matched-FAR reproduction failed: '+str((r,metric)))
    summary = summarize(results, originals)
    output = REPO/config['output']
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('split_results',results),('original_results',originals),('summary',summary),('source_scores',selected)]:
        save_csv(output/(name+'.csv'), rows)
    shutil.copyfile(splits_path, output/'splits.json')
    shutil.copyfile(protocol_path, output/'evaluation_protocol.yaml')
    shutil.copyfile(config_path, output/'config.yaml')
    inputs = [config_path,splits_path,protocol_path,source/'report.json',source/'episode_results.csv',source/'matched_far.csv',Path(__file__)]
    metadata = dict(experiment_id=config['experiment_id'],created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_id=protocol['protocol_id'], note=NOTE, target_far=config['target_far'],
        n_splits=config['n_splits'],original_checks=len(originals), inference=False,
        source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        summary=summary)
    (output/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    lines = ['# WM-NC matched-FAR split sensitivity', '', NOTE, '',
             '| Task | View | Protocol | Original FDR | FDR mean ± SD | FDR 5th–95th | Achieved FAR range |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['task']} | {r['view']} | {r['calibration']} | {100*r['original_FDR']:.2f}% | "
                     f"{100*r['FDR_mean']:.2f} ± {100*r['FDR_sd']:.2f}% | "
                     f"{100*r['FDR_p05']:.2f}–{100*r['FDR_p95']:.2f}% | {100*r['FAR_min']:.1f}–{100*r['FAR_max']:.1f}% |")
    lines += ['', 'All 20 original task/overall operating points reproduced exactly. '
              'Threshold variability and whole-episode AUROC are in summary.csv; all 10,000 split/task/overall rows are in split_results.csv.',
              '', '![Matched-FAR FDR sensitivity](sensitivity.png)', '']
    (output/'summary.md').write_text('\n'.join(lines))
    plot(summary, output)
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Matched-FAR split sensitivity</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;color:#222}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>WM-NC matched-FAR split sensitivity</h1><p>'+html.escape(NOTE)+'</p><img src="sensitivity.png" alt="FDR sensitivity">'
        '<p><a href="summary.md">Markdown</a> · <a href="summary.csv">Summary CSV</a> · '
        '<a href="split_results.csv">All splits CSV</a> · <a href="splits.json">Episode IDs</a></p>')
    (output/'state.json').write_text(json.dumps({'status':'complete','original_checks':len(originals)})+'\n')
    print(output)
    for r in summary:
        if r['task']=='Overall':
            print(f"{r['view']} {r['calibration']}: FDR {100*r['FDR_mean']:.2f} ± {100*r['FDR_sd']:.2f}%; "
                  f"5–95% {100*r['FDR_p05']:.2f}–{100*r['FDR_p95']:.2f}%; original {100*r['original_FDR']:.2f}%")


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks = ['needle_pickup','ring_over_post','tissue_retraction','shunt_insertion','Overall']
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax, view in zip(axes,('fix','wrist')):
        for offset, scheme, color in [(-.13,'episode','#bb8200'),(.13,'functional_band','#32797d')]:
            rows = [next(r for r in summary if (r['task'],r['view'],r['calibration'])==(t,view,scheme)) for t in tasks]
            y = np.arange(len(tasks))+offset
            ax.hlines(y,[100*r['FDR_p05'] for r in rows],[100*r['FDR_p95'] for r in rows],color=color,lw=3)
            ax.scatter([100*r['FDR_mean'] for r in rows],y,color=color,label=scheme.replace('_',' '),s=35)
            ax.scatter([100*r['original_FDR'] for r in rows],y,color=color,marker='x',s=45)
        ax.set(title='Fixed' if view=='fix' else 'Wrist',xlabel='Post-onset FDR at ≤5% FAR (%)',xlim=(0,102),
               yticks=range(len(tasks)),yticklabels=['Needle pickup','Ring over post','Tissue retraction','Shunt insertion','Overall'])
        ax.grid(axis='x',alpha=.2)
        ax.legend(fontsize=9,loc='lower left')
    axes[0].invert_yaxis()
    fig.suptitle('500 saved splits · dot: mean · line: 5th–95th percentile · ×: original split',fontsize=11)
    fig.tight_layout()
    fig.savefig(output/'sensitivity.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    run(parser.parse_args().config.resolve())
