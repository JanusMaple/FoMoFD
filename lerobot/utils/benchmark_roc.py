#!/usr/bin/env python3
"""CPU-only episode ROC and matched-FAR analysis of already saved scores.

python3 lerobot/utils/benchmark_roc.py --output fomofd/tasks/benchmark_results/analyses/2026-10-08/matched_far_all

No inference, checkpoints or images are loaded. Inputs are the curated archived
episode_results.csv files. Original reports, calibration and labels are read-only.
Thresholds are selected from success-TEST scores and are explicitly retrospective,
not deployable calibration estimates. Failures are positive; successful test
episodes are negative; calibration episodes never enter ROC estimation.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timezone
import csv
import html
import json
import math
from pathlib import Path
import statistics
import time

import yaml

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'fomofd/tasks/benchmark_results'
KEYS = ['variant', 'task', 'method', 'view', 'calibration']
NOTE = (
    'Retrospective episode-level ROC, not deployment calibration. Thresholds use '
    'successful TEST episodes; do not claim held-out FAR control. Deterministic '
    'FDR is measured at achieved FAR <= target; tied scores are never split. '
    'Interpolated FDR is a descriptive ROC value, possibly requiring randomized '
    'thresholds; it is not a measured deterministic operating point. AUROC is '
    'threshold-independent. Overall FAR/FDR pool task counts after selecting '
    'thresholds separately per task; overall AUROC is an equal-task macro mean. '
    'Calibration episodes are excluded. Failure filtering retains test successes. '
    'Wall-time-bin/fusion evidence uses the existing maximum standardized margin. '
    'Functional-band evidence is max_t((raw_score-mu)/g); ROC sweeps h while '
    'keeping the expert template frozen, separately from the deployed calibrated h. Unsupported episodes remain '
    'tied at -inf; their counts are reported. ROC includes the formal all-positive '
    'endpoint, which does not represent supported monitoring in missing-score bins. '
    'No online latency is recomputed at these thresholds.'
)


def score(row):
    value = float(row['score_max'])
    if math.isnan(value) or value == math.inf:
        raise ValueError('Episode score must be finite or -inf (unsupported)')
    return value


def curve(rows):
    """Exact tied-score ROC, with >= thresholds and a separate never-alarm origin."""
    positives = sum(r['outcome'] == 'failure' for r in rows)
    negatives = sum(r['outcome'] == 'success' for r in rows)
    if not positives or not negatives:
        return [], None
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        counts[score(r)][0 if r['outcome'] == 'success' else 1] += 1
    points = [{'threshold': None, 'comparison': 'never', 'FAR': 0., 'FDR': 0., 'fp': 0, 'tp': 0}]
    fp = tp = 0
    for value in sorted(counts, reverse=True):
        n, p = counts[value]
        fp += n
        tp += p
        points.append({'threshold': value, 'comparison': '>=', 'FAR': fp/negatives,
                       'FDR': tp/positives, 'fp': fp, 'tp': tp})
    auc = sum((b['FAR']-a['FAR'])*(a['FDR']+b['FDR'])/2 for a, b in zip(points, points[1:]))
    return points, auc


def roc_at(points, target, side='right'):
    if not points:
        return None
    xs = [p['FAR'] for p in points]
    lo = bisect_right(xs, target)-1
    if xs[lo] == target:
        return points[bisect_left(xs, target) if side == 'left' else lo]['FDR']
    hi = lo+1
    a, b = points[lo], points[hi]
    return a['FDR']+(b['FDR']-a['FDR'])*(target-a['FAR'])/(b['FAR']-a['FAR'])


def operating_point(rows, target):
    """Lowest strict-> threshold within the FAR cap, using negative scores only."""
    if not 0 <= target <= 1:
        raise ValueError('FAR targets must lie in [0,1]')
    negatives = sorted([score(r) for r in rows if r['outcome'] == 'success'], reverse=True)
    positives = [float(r['post_score_max']) if r.get('detection_definition') == 'at_or_after_onset' else score(r)
                 for r in rows if r['outcome'] == 'failure']
    if not negatives:
        return {'threshold': None, 'comparison': 'unavailable', 'false_alarms': 0,
                'success_episodes': 0, 'detected_failures': 0, 'failure_episodes': len(positives),
                'FAR': None, 'FDR': None}
    allowed = math.floor(target*len(negatives)+1e-10)
    threshold = negatives[allowed] if allowed < len(negatives) else -math.inf
    all_positive = allowed == len(negatives)
    alarm = lambda value: all_positive or value > threshold
    fp, tp = sum(map(alarm, negatives)), sum(map(alarm, positives))
    return {'threshold': threshold, 'comparison': 'always' if all_positive else '>',
            'false_alarms': fp, 'success_episodes': len(negatives),
            'detected_failures': tp, 'failure_episodes': len(positives),
            'FAR': fp/len(negatives), 'FDR': tp/len(positives) if positives else None}


def analyze(episodes, targets=(0., .05, .10, .20), subsets=('all', 'natural', 'induced', 'per_failure')):
    groups = defaultdict(list)
    seen = set()
    for original in episodes:
        if original['split'] != 'test':
            continue
        r = {'variant': 'main', **original}
        if r['outcome'] not in {'success', 'failure'}:
            raise ValueError('Unexpected test label')
        key = tuple(r[k] for k in KEYS)
        identity = (*key, r['episode_index'])
        if identity in seen:
            raise ValueError(f'Duplicate test episode: {identity}')
        seen.add(identity)
        score(r)
        groups[key].append(r)
    modes = sorted({r['failure_mode'] for rows in groups.values() for r in rows if r['outcome'] == 'failure'})
    subset_names = [s for s in subsets if s != 'per_failure']
    if 'per_failure' in subsets:
        subset_names += ['failure:'+m for m in modes]
    curves, summaries, matched = [], [], []
    for key, rows in sorted(groups.items()):
        base = dict(zip(KEYS, key))
        for subset in subset_names:
            selected = [r for r in rows if r['outcome'] == 'success' or subset == 'all' or
                        (subset in {'natural', 'induced'} and r['failure_origin'] == subset) or
                        (subset.startswith('failure:') and r['failure_mode'] == subset[8:])]
            common = {**base, 'failure_subset': subset}
            points, auc = curve(selected)
            neg = sum(r['outcome'] == 'success' for r in selected)
            pos = len(selected)-neg
            unsupported = sum(int(r['n_supported_scores']) == 0 for r in selected)
            summaries.append({**common, 'success_episodes': neg, 'failure_episodes': pos,
                              'episode_AUROC': auc, 'auc_tasks': int(auc is not None),
                              'unsupported_episodes': unsupported})
            curves.extend({**common, **p} for p in points)
            for target in targets:
                matched.append({**common, 'target_FAR': target, **operating_point(selected, target),
                                'interpolated_FDR': None if any(r.get('detection_definition') == 'at_or_after_onset' for r in selected) else roc_at(points, target), 'episode_AUROC': auc,
                                'auc_tasks': int(auc is not None), 'unsupported_episodes': unsupported})
    # Overall metrics are aggregated after task-specific thresholds, never by
    # pooling incompatible score scales. Empty positive classes do not enter AUC.
    def overall(rows, target=False):
        grouped = defaultdict(list)
        fields = ['variant', 'method', 'view', 'calibration', 'failure_subset']+(['target_FAR'] if target else [])
        for r in rows:
            grouped[tuple(r[k] for k in fields)].append(r)
        result = []
        for key, items in grouped.items():
            entry = {**dict(zip(fields, key)), 'task': 'Overall'}
            entry.update({k: sum(r[k] for r in items) for k in ['success_episodes', 'failure_episodes', 'unsupported_episodes']})
            aucs = [r['episode_AUROC'] for r in items if r['episode_AUROC'] is not None]
            entry.update(episode_AUROC=statistics.mean(aucs) if aucs else None, auc_tasks=len(aucs))
            if target:
                entry.update({k: sum(r[k] for r in items) for k in ['false_alarms', 'detected_failures']})
                entry.update(threshold=None, comparison='per-task thresholds',
                             FAR=entry['false_alarms']/entry['success_episodes'] if entry['success_episodes'] else None,
                             FDR=entry['detected_failures']/entry['failure_episodes'] if entry['failure_episodes'] else None,
                             interpolated_FDR=sum(r['interpolated_FDR']*r['failure_episodes'] for r in items if r['interpolated_FDR'] is not None)/entry['failure_episodes'] if entry['failure_episodes'] and any(r['interpolated_FDR'] is not None for r in items) else None)
            result.append(entry)
        return result
    summaries.extend(overall(summaries))
    matched.extend(overall(matched, True))
    return {'roc_points': curves, 'roc_summary': summaries, 'matched_far': matched}


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def write_reports(directory, results, metadata, report_stem='report'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in results.items():
        with (directory / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else KEYS)
            writer.writeheader()
            writer.writerows(safe(rows))
    report = {**metadata, 'methodology': NOTE,
              'table_rows': {k: len(v) for k, v in results.items()}}
    (directory / 'roc_report.json').write_text(json.dumps(safe(report), indent=2, allow_nan=False)+'\n')
    rows = [r for r in results['matched_far'] if r['task'] == 'Overall' and r['failure_subset'] == 'all' and r['target_FAR'] == .05]
    headers = ['Variant', 'Method', 'View', 'Score transform', 'Target FAR', 'Achieved FAR', 'FDR at FAR cap', 'Interpolated FDR', 'Episode AUROC']
    pct = lambda x: 'N/A' if x is None else f'{100*x:.1f}%'
    cells = [[r['variant'], r['method'], r['view'], r['calibration'], pct(r['target_FAR']), pct(r['FAR']), pct(r['FDR']), pct(r['interpolated_FDR']), 'N/A' if r['episode_AUROC'] is None else f"{r['episode_AUROC']:.4f}"] for r in rows]
    markdown = '# Empirical ROC and matched-FAR analysis\n\n'+NOTE+'\n\n'
    markdown += 'At 20 test successes/task, one false alarm is 5% FAR. AUROC is not tied to a chosen FAR.\n\n'
    markdown += '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    markdown += '\n'.join('| '+' | '.join(row)+' |' for row in cells)+'\n'
    figures = [f'roc_main_{scheme}' for scheme in ['episode', 'time_varying', 'wall_time_bin', 'functional_band']
               if (directory/f'roc_main_{scheme}.png').is_file()]
    for name in figures:
        markdown += f'\n![{name}]({name}.png)\n\n[PDF]({name}.pdf) · [SVG]({name}.svg)\n'
    settings = [(source['variant'], setting) for source in metadata.get('sources', [])
                for setting in source.get('scoring_settings', [])]
    settings += [('current run', s) for s in metadata.get('scoring_settings', [])]
    ode_headers = ['Variant', 'Task', 'Method', 'Views', 'K', 'L', 'ODE solver',
                   'Steps/solve', 'Solves/score', 'Steps/score', 'Field evaluations/score']
    ode_cells = [[variant]+[str(s.get(k)) if s.get(k) is not None else 'N/A' for k in
                 ['task', 'method', 'views', 'K', 'L', 'ode_solver', 'ode_steps_per_solve',
                  'ode_solves_per_score', 'ode_steps_per_score', 'vector_field_evals_per_score']]
                 for variant, s in settings if s.get('ode_solver') is not None]
    if ode_cells:
        markdown += '\n## Original scoring settings\n\nNo new ODE solves were run for this analysis.\n\n'
        markdown += '| '+' | '.join(ode_headers)+' |\n| '+' | '.join(['---']*len(ode_headers))+' |\n'
        markdown += '\n'.join('| '+' | '.join(row)+' |' for row in ode_cells)+'\n'
    (directory / f'{report_stem}.md').write_text(markdown)
    table = '<tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'
    table += ''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in cells)
    ode_table = '<tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in ode_headers)+'</tr>'
    ode_table += ''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in ode_cells)
    (directory / f'{report_stem}.html').write_text('<!doctype html><meta charset="utf-8"><title>Matched-FAR ROC</title>'
        '<style>body{font:14px system-ui;margin:28px;color:#193142}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}th{position:sticky;top:0;background:#eef4f7}p{max-width:1100px}</style>'
        '<h1>Empirical ROC and matched-FAR analysis</h1><p>'+html.escape(NOTE)+'</p>'
        '<p>Table: all failures, all tasks, 5% FAR cap. CSV files include all targets and failure subsets.</p>'
        '<p><a href="matched_far.csv">Matched FAR CSV</a> · <a href="roc_summary.csv">AUROC CSV</a> · <a href="roc_points.csv">Full ROC points</a></p>'+
        ''.join(f'<p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p><img style="max-width:1200px;width:100%" src="{name}.png" alt="{name}">' for name in figures)+
        '<table>'+table+'</table>'+
        ('<details><summary>Original scoring settings and ODE work</summary><p>No new ODE solves were run.</p><table>'+ode_table+'</table></details>' if ode_cells else ''))


def from_catalog(catalog, archive_root=ROOT):
    from results_inspector import csv_rows, sha256
    config = yaml.safe_load(Path(catalog).read_text())
    manifests = [json.loads(p.read_text()) | {'directory': p.parent} for p in archive_root.glob('offline/*/archive_manifest.json')]
    rows, sources = [], []
    for entry in config['offline']:
        matches = [m for m in manifests if m['source'] == entry['source'] and (not entry.get('sha256') or m['sha256'] == entry['sha256'])]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one archive: {entry['source']}")
        manifest = matches[0]
        for name in ['episode_results.csv', 'report.json']:
            if sha256(manifest['directory']/name) != manifest['files'][name]:
                raise ValueError(f'Changed archived input: {manifest["directory"] / name}')
        report = json.loads((manifest['directory']/'report.json').read_text())
        sources.append({'variant': entry['id'], 'archive': str(manifest['directory'].relative_to(REPO)),
                        'archive_sha256': manifest['sha256'], 'episode_csv_sha256': manifest['files']['episode_results.csv'],
                        'labels_sha256': report['labels_sha256'], 'protocols': report['protocols'],
                        'scoring_settings': report['scoring_settings']})
        rows.extend({'variant': entry['id'], **r} for r in csv_rows(manifest['directory']/'episode_results.csv'))
    for entry in config.get('functional_band_runs', []):
        directory=(archive_root/entry['path']).resolve()
        if not directory.is_relative_to(archive_root.resolve()):
            raise ValueError('Functional reports must be inside tasks/benchmark_results')
        state_path=directory/'state.json'
        state=json.loads(state_path.read_text()) if state_path.is_file() else {'status':'not_started'}
        if state.get('status') not in {'complete','partial'}:
            sources.append({'variant':'functional_band_pending','status':state,'scoring_settings':[]})
            continue
        manifest=json.loads((directory/'artifact_manifest.json').read_text())
        for variant in config['offline']:
            folder=directory/variant['id']/'functional_band'
            if not folder.is_dir():continue
            for name in ['report.json','episode_results.csv']:
                path=folder/name
                if sha256(path)!=manifest['files'][str(path.relative_to(directory))]:
                    raise ValueError(f'Changed functional report: {path}')
            report=json.loads((folder/'report.json').read_text())
            extra=csv_rows(folder/'episode_results.csv')
            if any(r['calibration']!='functional_band' for r in extra):
                raise ValueError('Functional source contains another calibration protocol')
            rows.extend({'variant':variant['id'],**r} for r in extra)
            sources.append({'variant':variant['id'],'protocol':'functional_band','archive':str(folder),
                            'episode_csv_sha256':sha256(folder/'episode_results.csv'),
                            'labels_sha256':report['labels_sha256'],'protocols':report['protocols'],
                            'scoring_settings':report['scoring_settings']})
    from functional_conformal import canonical_scheme
    for row in rows:row['calibration']=canonical_scheme(row['calibration'])
    return rows, sources


def export_plots(directory, results):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(directory/'.matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    methods = ['wm_nc', 'wm_pe', 'wm_pu', 'logpzo', 'rnd', 'logpzo_native', 'rnd_native', 'ace']
    points = results['roc_points']
    for scheme in sorted({r['calibration'] for r in points}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for method in methods:
            for view in ['fix', 'wrist', 'both']:
                selected = [r for r in points if r['variant'] == 'main' and r['calibration'] == scheme and r['failure_subset'] == 'all' and r['method'] == method and r['view'] == view]
                tasks = defaultdict(list)
                for r in selected:
                    tasks[r['task']].append(r)
                if not tasks:
                    continue
                grid = sorted({r['FAR'] for r in selected})
                xs, ys = [], []
                for x in grid:
                    for side in ['left', 'right']:
                        xs.append(x)
                        ys.append(statistics.mean(roc_at(p, x, side) for p in tasks.values()))
                for ax in axes:
                    ax.plot(xs, ys, label=f'{method} · {view}', color=plt.get_cmap('tab10')(methods.index(method)), linestyle='--' if view == 'wrist' else '-')
        for ax, xmax in zip(axes, [1, .2]):
            ax.plot([0, 1], [0, 1], ':', color='gray', linewidth=1)
            ax.set(xlim=(0, xmax), ylim=(0, 1.015), xlabel='Episode FAR', ylabel='Task-macro FDR')
            ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.2)
        axes[0].set_title('Full ROC'); axes[1].set_title('Low-FAR region')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8)
        fig.suptitle(f'Empirical episode ROC · {scheme} score transform · all tasks')
        fig.tight_layout(rect=(0, .14, 1, .95))
        for extension in ['pdf', 'svg', 'png']:
            fig.savefig(directory/f'roc_main_{scheme}.{extension}', dpi=160, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=REPO/'lerobot/configs/benchmark/matched_far.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from results_inspector import sha256
    config = yaml.safe_load(args.config.read_text())
    if config.get('schema_version') != 1 or set(config['failure_subsets'])-{'all', 'natural', 'induced', 'per_failure'}:
        raise ValueError('Unsupported analysis configuration')
    targets = config['far_targets']
    if not targets or len(targets) != len(set(targets)) or any(not 0 <= x <= 1 for x in targets):
        raise ValueError('Choose distinct FAR targets in [0,1]')
    if args.output.exists():
        raise FileExistsError(f'Analysis output already exists: {args.output}')
    started = time.perf_counter()
    episodes, sources = from_catalog(REPO/config['catalog'])
    from functional_conformal import canonical_scheme
    episodes = [r for r in episodes if r['calibration'] in {canonical_scheme(s) for s in config['calibrations']}]
    results = analyze(episodes, targets, config['failure_subsets'])
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
                'sources': sources, 'script_sha256': sha256(Path(__file__)),
                'input_episode_rows': len(episodes), 'analysis_seconds': time.perf_counter()-started,
                'models_loaded': 0, 'scores_recomputed': 0}
    args.output.mkdir(parents=True)
    (args.output/'analysis_config.yaml').write_text(args.config.read_text())
    export_plots(args.output, results)
    metadata['total_seconds'] = time.perf_counter()-started
    write_reports(args.output, results, metadata)
    print(f'Completed ROC/matched-FAR analysis in {time.perf_counter()-started:.2f}s. No inference.')
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
