"""Descriptive expert-validation versus successful-policy score distributions.

Invoked through analyze_experiments.py score-distributions. Cache-only; no model
loading or template fitting. Input integrity and saved episode partitions are
checked before comparing raw, unsmoothed scores at identical eligible endpoints.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import yaml

import analyze_experiments as analysis
import unified_benchmark as benchmark
from calibration_sensitivity import Inputs
from results_inspector import resolve_repo_path

REPO = analysis.REPO
COHORTS = ('expert_validation', 'policy_calibration', 'policy_test', 'policy_success')
COLORS = ('#2563a6', '#d99018', '#bf4781', '#15936d')
LEVELS = ('window_pooled', 'window_episode_balanced', 'episode_maximum')
PLOT_NOTE = (
    'Plots show probability-density histograms, not cumulative probabilities. '
    'Every cohort uses the same bin edges within a panel and has total area 1. '
    'Bin height is probability per score unit; probability in a bin is its height '
    'times its width. Filled blue/green histograms compare expert validation '
    'with all policy successes; dashed outlines show calibration and test separately. '
    'No KDE or score smoothing is applied. Episode-max histograms use fewer bins '
    'because only 10/25 expert and 39 policy episodes are available.'
)
NOTE = (
    'Descriptive comparison of main K4/L4 WM-NC raw scores on expert validation '
    'and successful policy rollouts. All eligible endpoint windows are retained '
    '(without smoothing or functional normalization). '
    'Pooled windows give longer episodes more weight; episode-balanced windows '
    'give each episode equal total weight. Episode maxima give each episode equal '
    'weight but depend on window count and execution phase. Overlapping windows '
    'are correlated, so no window-independent significance tests are reported. '
    'Expert validation selected the checkpoint during training and is not an '
    'independent held-out expert test set. The 39 policy successes pool the original '
    '19 calibration and 20 test episodes for description only; their split and '
    'primary benchmark results are unchanged. This comparison does not establish '
    'exchangeability, calibration validity, or failure detection performance.'
)


def distribution(curves, episode_ids, level):
    """Return values and normalized masses; never discard a nonfinite score."""
    if level not in LEVELS or not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError('Unknown distribution level or empty/duplicate episode selection')
    values, weights = [], []
    for episode in episode_ids:
        a = np.asarray(curves[episode]['score'], dtype=float)
        if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
            raise ValueError(f'Invalid score trace for episode {episode}')
        if level == 'episode_maximum':
            values.append(np.array([a.max()]))
            weights.append(np.ones(1))
        else:
            values.append(a)
            weights.append(np.full(len(a), 1 / len(a) if level == 'window_episode_balanced' else 1.))
    values, weights = np.concatenate(values), np.concatenate(weights)
    return values, weights / weights.sum()


def ecdf(values, weights):
    order = np.argsort(values, kind='stable')
    x = np.asarray(values)[order]
    unique, starts = np.unique(x, return_index=True)
    mass = np.add.reduceat(np.asarray(weights)[order], starts)
    cumulative = np.cumsum(mass / mass.sum())
    cumulative[-1] = 1.
    return unique, cumulative


def describe_distribution(values, weights, quantile_method='inverted_cdf'):
    x, cumulative = ecdf(values, weights)
    mean = float(np.dot(values, weights))
    result = {'mean': mean, 'sd_population': float(np.sqrt(np.dot((values - mean)**2, weights)))}
    if quantile_method not in ('inverted_cdf', 'linear'):
        raise ValueError('Unknown descriptive quantile method')
    if quantile_method == 'linear' and not np.allclose(weights, 1 / len(values), rtol=1e-12, atol=1e-15):
        raise ValueError('Linear quantiles require equal episode weights')
    for label, q in [('p05', .05), ('q25', .25), ('median', .5), ('q75', .75), ('p95', .95)]:
        result[label] = (float(np.quantile(values, q, method='linear')) if quantile_method == 'linear'
                         else float(x[np.searchsorted(cumulative, q, side='left')]))
    return result | {'min': float(x[0]), 'max': float(x[-1]), 'iqr': result['q75'] - result['q25']}


def policy_protocol(config, task, view, corrections):
    """Validity limits apply to PRL only, never same-numbered expert episodes."""
    protocol = deepcopy(config['window_protocol'])
    limits = [{'task': r['task'], 'episode_index': r['episode_index'], 'raw_frame_stop': r['raw_frame_stop']}
              for r in corrections if r['task'] == task and view in r['views']]
    if limits:
        protocol['frame_limits'] = limits
    return protocol


def compare_distributions(expert, policy):
    """Weighted effect sizes only; windows are not independent replicates."""
    x, fx = ecdf(*expert)
    y, fy = ecdf(*policy)
    grid = np.union1d(x, y)
    a = np.r_[0., fx][np.searchsorted(x, grid, side='right')]
    b = np.r_[0., fy][np.searchsorted(y, grid, side='right')]
    values, weights = policy
    below = np.r_[0., fx][np.searchsorted(x, values, side='left')]
    at_or_below = np.r_[0., fx][np.searchsorted(x, values, side='right')]
    return {'KS_distance_descriptive': float(np.max(np.abs(a-b))),
            'wasserstein_raw_units': float(np.sum(np.diff(grid) * np.abs(a[:-1]-b[:-1]))),
            'prob_policy_higher_half_ties': float(np.dot(weights, (below+at_or_below)/2))}


def validate_pair(expert, policy, saved_ids):
    for key in ('task', 'method', 'views', 'seed', 'batch_size', 'device'):
        if expert[key] != policy[key]:
            raise ValueError(f'Expert/policy scoring identity differs: {key}')
    if expert['sha256'][expert['checkpoint']] != policy['sha256'][policy['checkpoint']]:
        raise ValueError('Expert/policy checkpoint hashes differ')
    configs = [deepcopy(i['merged_config']) for i in (expert, policy)]
    for config in configs:
        config['data'].pop('dataset_root')
    if configs[0] != configs[1]:
        raise ValueError('Expert/policy scorer settings differ beyond dataset_root')
    # Compare every shared input: scorer/model source, normalizers, checkpoint,
    # and config. Dataset-specific files and the expert split are separate.
    shared = set(expert['sha256']) & set(policy['sha256'])
    if any(expert['sha256'][p] != policy['sha256'][p] for p in shared):
        raise ValueError('Expert/policy frozen shared scoring inputs differ')
    if expert.get('split') != 'val' or sorted(expert.get('episode_ids', [])) != sorted(saved_ids):
        raise ValueError('Expert cache is not exactly the saved validation selection')


def load_cache(inputs, path, expected):
    path = resolve_repo_path(path)
    inputs.external(path, expected)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in ('episode_index', 'frame_index', 'raw_frame_index', 'wall_timestamp', 'scores')}
        identity = json.loads(str(data['provenance_json'].item()))['identity']
        if list(data['views']) != identity['views']:
            raise ValueError('Cache view order differs from identity')
    return arrays, identity, path


def load_cases(config, inputs):
    variant, method = config['variant'], config['method']
    if method != 'wm_nc':
        raise ValueError('This experiment currently compares WM-NC only')
    corrections = []
    if config.get('dataset_validity'):
        path = REPO / config['dataset_validity']
        inputs.external(path, benchmark.digest(path))
        validity = yaml.safe_load(path.read_text())
        if validity['frame_id_space'] != 'zero_based_raw_recording':
            raise ValueError('Expected raw-frame dataset validity corrections')
        corrections = validity['corrections']
    report = inputs.read(f'{variant}/episode/report.json')
    templates = inputs.read(f'{variant}/templates_manifest.json')['templates']
    labels_path = inputs.verified(f'{variant}/episode/labels_snapshot.yaml')
    labels, _ = benchmark.load_labels(labels_path, config['tasks'])
    cases, selections, provenance = [], {}, []
    for task in config['tasks']:
        matches = [c for c in report['score_caches'] if c['task'] == task and c['method'] == method]
        if len(matches) != 1:
            raise ValueError(f'Expected one main WM-NC cache for {task}')
        cache = matches[0]
        policy_arrays, policy_identity, policy_path = load_cache(inputs, cache['path'], cache['sha256'])
        if (policy_identity['task'], policy_identity['method'], policy_identity['views']) != (task, method, config['views']):
            raise ValueError('Unexpected task/method/views in policy cache')
        sources = []
        for view in config['views']:
            entries = [e for e in templates if (e['task'], e['method'], e['view'], e['protocol']) == (task, method, view, 'paper')]
            if len(entries) != 1:
                raise ValueError(f'Missing/ambiguous expert source for {task}/{view}')
            entry = entries[0]
            path = inputs.verified(f'{variant}/{entry["path"]}')
            inputs.external(path, entry['sha256'])
            sources.append(json.loads(path.read_text())['source'])
        if any(source != sources[0] for source in sources[1:]):
            raise ValueError('Expert sources differ across views')
        source = sources[0]
        if source['role'] != 'expert_validation' or source['prl_score_cache_sha256'] != cache['sha256']:
            raise ValueError('Unpaired expert/policy caches')
        split_path = resolve_repo_path(source['split_manifest'])
        inputs.external(split_path, source['split_sha256'])
        split = json.loads(split_path.read_text())
        expert_ids = sorted(split['val_episode_ids'])
        if (len(set(expert_ids)) != len(expert_ids) or set(expert_ids) & set(split['train_episode_ids'])
                or expert_ids != source['episode_ids']
                or len(expert_ids) != config['expected_expert_validation_episodes'][task]):
            raise ValueError('Saved expert train/validation partition differs')
        expert_arrays, expert_identity, expert_path = load_cache(inputs, source['score_cache'], source['score_cache_sha256'])
        validate_pair(expert_identity, policy_identity, expert_ids)
        if source['checkpoint_sha256'] != policy_identity['sha256'][policy_identity['checkpoint']]:
            raise ValueError('Expert source checkpoint hash mismatch')
        groups = benchmark.episode_labels(labels['tasks'][task])
        selected = {'expert_validation': expert_ids}
        for role in ('calibration', 'test'):
            selected['policy_'+role] = sorted(e for e, g in groups.items() if g['outcome'] == 'success' and g['split'] == role)
            if len(selected['policy_'+role]) != config['expected_policy_success_episodes'][role]:
                raise ValueError('Unexpected policy success partition')
        if set(selected['policy_calibration']) & set(selected['policy_test']):
            raise ValueError('Policy calibration and test overlap')
        selected['policy_success'] = sorted(selected['policy_calibration'] + selected['policy_test'])
        selections[task] = selected
        job = SimpleNamespace(task=task, method=method, views=config['views'], mode='nc', merged=policy_identity['merged_config'])
        timelines = []
        for identity, arrays, role in [(expert_identity, expert_arrays, 'data'), (policy_identity, policy_arrays, 'prl_data')]:
            dataset = resolve_repo_path(identity['dataset'])
            if not dataset.is_relative_to(REPO/'fomofd/tasks'/task/role):
                raise ValueError('Dataset population mismatch')
            timeline = benchmark.load_timeline(dataset)
            if role == 'data':
                timeline = {ep: timeline[ep] for ep in expert_ids}
            elif set(timeline) != set(groups):
                raise ValueError('Policy dataset and label episode IDs differ')
            benchmark.validate_scores(arrays, job, timeline)
            timelines.append(timeline)
        for index, view in enumerate(config['views']):
            expert = benchmark.curves(expert_arrays, job, index, config['window_protocol'])
            policy = benchmark.curves(policy_arrays, job, index, policy_protocol(config, task, view, corrections))
            cases.append({'task': task, 'view': view, 'selected': selected,
                          'curves': {cohort: expert if cohort == 'expert_validation' else policy for cohort in COHORTS},
                          'timelines': {cohort: timelines[0] if cohort == 'expert_validation' else timelines[1] for cohort in COHORTS}})
        provenance.append({'task': task, 'checkpoint': str(resolve_repo_path(policy_identity['checkpoint'])),
                           'checkpoint_sha256': source['checkpoint_sha256'], 'expert_cache': str(expert_path),
                           'policy_cache': str(policy_path), 'expert_split_manifest': str(split_path),
                           'scoring_settings': cache['scoring_settings'], 'training_seed': source['training_seed'],
                           'score_seed': policy_identity['seed'], 'shared_score_identity_verified': True})
        print(f'{task}: verified {len(expert_ids)} expert validation / {len(selected["policy_success"])} policy successes', flush=True)
    return cases, selections, provenance, labels_path


def tables(cases, episode_quantile_method='inverted_cdf'):
    summaries, comparisons, episodes, windows = [], [], [], []
    for case in cases:
        key = {k: case[k] for k in ('task', 'view')}
        case['distributions'] = {}
        for cohort in COHORTS:
            curves, ids = case['curves'][cohort], case['selected'][cohort]
            case['distributions'][cohort] = {level: distribution(curves, ids, level) for level in LEVELS}
            for level, (values, weights) in case['distributions'][cohort].items():
                method = episode_quantile_method if level == 'episode_maximum' else 'inverted_cdf'
                summaries.append(key | {'cohort': cohort, 'level': level, 'n_episodes': len(ids),
                                        'n_windows': sum(len(curves[e]['score']) for e in ids), 'quantile_method': method} |
                                 describe_distribution(values, weights, method))
            # Export each policy episode once; policy_success is the union.
            if cohort == 'policy_success':
                continue
            for episode in ids:
                curve = curves[episode]
                origin = case['timelines'][cohort][episode]['wall'][0]
                clock = case['timelines'][cohort][episode]['wall']
                episodes.append(key | {'cohort': cohort, 'episode_index': episode,
                                       'n_windows': len(curve['score']), 'max_score': float(np.max(curve['score'])),
                                       'mean_score': float(np.mean(curve['score'])),
                                       'prepared_wall_duration_s': float(clock[-1]-clock[0]),
                                       'first_scored_elapsed_s': float(curve['wall'][0]-origin),
                                       'last_scored_elapsed_s': float(curve['wall'][-1]-origin)})
                windows.extend(key | {'cohort': cohort, 'episode_index': episode, 'frame_index': int(frame),
                                      'raw_frame_index': int(raw), 'elapsed_wall_s': float(wall-origin), 'score': float(score)}
                               for frame, raw, wall, score in zip(curve['frame'], curve['raw'], curve['wall'], curve['score']))
        for cohort in COHORTS[1:]:
            for level in LEVELS:
                expert = case['distributions']['expert_validation'][level]
                policy = case['distributions'][cohort][level]
                comparisons.append(key | {'policy_cohort': cohort, 'level': level} | compare_distributions(expert, policy))
    return summaries, comparisons, episodes, windows


def figures(root, cases, episodes):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fomofd-score-distribution-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names, histogram_rows = [], []
    for case in cases:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, level in zip(axes.flat, LEVELS):
            # Select common edges from expert + all policy successes once; the
            # calibration/test subsets must not be counted twice in bin choice.
            support = np.concatenate([case['distributions'][cohort][level][0]
                                      for cohort in ('expert_validation', 'policy_success')])
            if level == 'episode_maximum':
                bins = int(np.clip(np.ceil(np.sqrt(len(support))), 5, 12))
            else:
                bins = int(np.clip(len(np.histogram_bin_edges(support, bins='fd'))-1, 12, 40))
            edges = np.histogram_bin_edges(support, bins=bins)
            # Filled primary populations first, then the split-specific outlines.
            for cohort in ('expert_validation', 'policy_success', 'policy_calibration', 'policy_test'):
                color = COLORS[COHORTS.index(cohort)]
                values, weights = case['distributions'][cohort][level]
                density, _ = np.histogram(values, bins=edges, weights=weights, density=True)
                if not np.isclose(np.sum(density * np.diff(edges)), 1.):
                    raise ValueError('Histogram probability density must integrate to one')
                primary = cohort in ('expert_validation', 'policy_success')
                if primary:
                    ax.stairs(density, edges, fill=True, color=color, alpha=.16, linewidth=0)
                labels = {'expert_validation': 'Expert validation', 'policy_success': 'Policy successes',
                          'policy_calibration': 'Policy calibration', 'policy_test': 'Policy test'}
                ax.stairs(density, edges, color=color, linewidth=1.8 if primary else 1.2,
                          linestyle='-' if primary else '--',
                          label=f'{labels[cohort]} (n={len(case["selected"][cohort])})')
                histogram_rows.extend({'task': case['task'], 'view': case['view'], 'level': level,
                                       'cohort': cohort, 'bin_left': float(left), 'bin_right': float(right),
                                       'density': float(height), 'probability_mass': float(height*(right-left))}
                                      for left, right, height in zip(edges[:-1], edges[1:], density))
            ax.set(title=level.replace('_', ' '), xlabel='Raw WM-NC score', ylabel='Probability density', ylim=(0, None))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
            ax.grid(alpha=.2)
        ax = axes[1, 1]
        for cohort, color in zip(COHORTS[:3], COLORS[:3]):
            rows = [r for r in episodes if (r['task'], r['view'], r['cohort']) == (case['task'], case['view'], cohort)]
            ax.scatter([r['n_windows'] for r in rows], [r['max_score'] for r in rows], color=color, alpha=.7, s=22)
        ax.set(title='Length effect (descriptive)', xlabel='Scored windows per episode', ylabel='Episode maximum score')
        ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'{analysis.TASKS[case["task"]]} / {case["view"]} · main K4/L4 WM-NC')
        fig.tight_layout()
        name = f'{case["task"]}_{case["view"]}_density'
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(root/f'{name}.{extension}', dpi=150)
        plt.close(fig)
        names.append(name)
    analysis.write_csv(root/'density_histograms.csv', histogram_rows)
    return names


def report(root, summaries, comparisons, provenance, names, note=NOTE):
    lookup = {(r['task'], r['view'], r['cohort'], r['level']): r for r in summaries}
    rows = []
    for task, view in dict.fromkeys((r['task'], r['view']) for r in summaries):
        values = [lookup[task, view, cohort, 'episode_maximum'] for cohort in ('expert_validation', 'policy_success')]
        rows.append([task, view, f'{values[0]["n_episodes"]} / {values[1]["n_episodes"]}',
                     *[f'{r["median"]:.3f} ({r["q75"]-r["q25"]:.3f})' for r in values],
                     f'{values[1]["median"]-values[0]["median"]:+.3f}'])
    headers = ['Task', 'View', 'Expert / policy episodes', 'Expert episode-max median (IQR)',
               'Policy-success episode-max median (IQR)', 'Median shift']
    table = benchmark.markdown_table(headers, rows)
    episode_rows = [r for r in summaries if r['level'] == 'episode_maximum']
    cohort_headers = ['Task', 'View', 'Cohort', 'Episodes', 'Median', 'Q1', 'Q3', 'IQR']
    cohort_rows = [[r['task'], r['view'], r['cohort'], r['n_episodes'],
                    *[f'{r[k]:.3f}' for k in ('median', 'q25', 'q75')], f'{r["q75"]-r["q25"]:.3f}'] for r in episode_rows]
    methods = sorted({r.get('quantile_method', 'inverted_cdf') for r in episode_rows})
    quantile_note = ('Episode maxima give one observation per episode. IQR = Q3 − Q1 (75th minus 25th percentile); '
                     f'episode quantile method: {", ".join(methods)}. '
                     'The linear method interpolates at index (n−1)p, including the midpoint median for even n. '
                     'Window-level diagnostics retain inverse-empirical-CDF quantiles. These are descriptive spreads, not confidence intervals.')
    lines = ['# Expert validation versus policy success: episode-maximum WM-NC scores', '', note, '',
             '## Median and interquartile range', '', quantile_note, '', table, '',
             'Positive shifts mean larger policy-success scores, not detected failures. Scores are compared within each task/view only.', '',
             '## All cohorts', '', benchmark.markdown_table(cohort_headers, cohort_rows), '',
             'Policy successes combine the disjoint calibration (19) and test-success (20) cohorts; they are not a fourth independent sample.', '',
             '## Scoring and selection', '',
             'Existing expert-validation and policy score caches were SHA-256 verified and reused. No inference, retraining, recalibration, or template fitting was performed.', '',
             'Saved validation IDs: 25 per simulation task; 10 per real task. Policy successes: 19 original calibration + 20 original test episodes per task. Exact IDs are in `episode_selections.json`.', '',
             'All four models use K=4, L=4, score seed 0 and batch size 8. ODE: one RK4 integration step (four vector-field evaluations) per score.', '',
             'SD in the diagnostic exports is the weighted population SD. KS and Wasserstein distances are descriptive. `prob_policy_higher_half_ties` is a score-order effect size, not failure AUROC.', '',
             '## Files', '',
             '- `distribution_summary.csv`: all four cohorts and three weighting levels.',
             '- `episode_maximum_summary.csv`: episode-max median, Q1, Q3 and IQR for each cohort.',
             '- `comparisons.csv`: expert versus each policy cohort.',
             '- `density_histograms.csv`: shared bin edges, probability densities, and probability masses used in the plots.',
             '- `episodes.csv`: episode maxima, window counts, and wall durations.',
             '- `windows.csv`: every selected window score, with dataset cohort and frame IDs.',
             '- `results.json`, `provenance.json`, `episode_selections.json`: complete numeric results and verified sources.', '',
             '## Diagnostic distributions', '', PLOT_NOTE, '']
    for name in names:
        lines += [f'![{name}]({name}.png)', '', f'[PDF]({name}.pdf) · [SVG]({name}.svg)', '']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    html_table = '<table><thead><tr>'+''.join(f'<th>{html.escape(h)}</th>' for h in headers)+'</tr></thead><tbody>'
    html_table += ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    body = '<h1>Expert validation versus policy success: episode-maximum WM-NC</h1><p>'+html.escape(note)+'</p>'
    body += '<p><a href="report.md">Concise Markdown</a> · <a href="results.json">JSON</a> · <a href="distribution_summary.csv">Distributions CSV</a> · <a href="comparisons.csv">Comparisons CSV</a> · <a href="episodes.csv">Episodes CSV</a> · <a href="windows.csv">Windows CSV</a></p>'
    body += '<p><a href="episode_maximum_summary.csv">Episode-max median / IQR CSV</a> · <a href="density_histograms.csv">Histogram bins / densities CSV</a></p>'
    body += '<p>'+html.escape(quantile_note)+'</p>'+html_table
    body += '<h2>All cohorts</h2><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in cohort_headers)+'</tr></thead><tbody>'
    body += ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in cohort_rows)+'</tbody></table>'
    body += '<h2>Diagnostic distributions</h2><p>'+html.escape(PLOT_NOTE)+'</p>'
    body += ''.join(f'<h2>{html.escape(name)}</h2><img src="{name}.png" alt="{name}"><p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p>' for name in names)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Expert / policy score distributions</title><style>body{font:15px system-ui;margin:30px;color:#203040}p{max-width:1200px}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}img{max-width:1200px;width:100%}</style>'+body)


def refresh_plots(output):
    """Refresh presentation from exported numbers; retain all experiment data."""
    output = output.resolve()
    preserved = ['results.json', 'windows.csv', 'episodes.csv', 'distribution_summary.csv',
                 'comparisons.csv', 'episode_selections.json', 'provenance.json',
                 'experiment.yaml', 'labels_snapshot.yaml', 'state.json']
    hashes = {name: benchmark.digest(output/name) for name in preserved}
    if json.loads((output/'state.json').read_text())['status'] != 'complete':
        raise ValueError('Plot refresh requires a completed experiment')
    results = json.loads((output/'results.json').read_text())
    selections = json.loads((output/'episode_selections.json').read_text())
    provenance = json.loads((output/'provenance.json').read_text())
    groups = {}
    with (output/'windows.csv').open() as stream:
        for row in csv.DictReader(stream):
            key = (row['task'], row['view'], row['cohort'])
            groups.setdefault(key, {}).setdefault(int(row['episode_index']), {'score': []})['score'].append(float(row['score']))
    cases = []
    for task, view in dict.fromkeys((r['task'], r['view']) for r in results['summary']):
        curves = {cohort: groups[task, view, cohort] for cohort in COHORTS[:3]}
        if set(curves['policy_calibration']) & set(curves['policy_test']):
            raise ValueError('Calibration/test episode overlap in plot exports')
        curves['policy_success'] = curves['policy_calibration'] | curves['policy_test']
        case = {'task': task, 'view': view, 'selected': selections[task], 'distributions': {}}
        for cohort in COHORTS:
            if set(curves[cohort]) != set(selections[task][cohort]):
                raise ValueError('Exported windows do not match frozen episode selections')
            case['distributions'][cohort] = {}
            for level in LEVELS:
                dist = distribution(curves[cohort], selections[task][cohort], level)
                saved = next(r for r in results['summary'] if
                             (r['task'], r['view'], r['cohort'], r['level']) == (task, view, cohort, level))
                current = describe_distribution(*dist, quantile_method=saved.get('quantile_method', 'inverted_cdf'))
                if not all(np.isclose(v, saved[k], rtol=1e-12, atol=1e-12)
                           for k, v in current.items() if k != 'iqr' or k in saved):
                    raise ValueError('Exported scores do not reproduce saved statistics')
                case['distributions'][cohort][level] = dist
        cases.append(case)
    with tempfile.TemporaryDirectory(prefix='.density_plots_', dir=output.parent) as temporary:
        stage = Path(temporary)
        names = figures(stage, cases, results['episodes'])
        report(stage, results['summary'], results['comparisons'], provenance['sources'], names, note=results['note'])
        for name, expected in hashes.items():
            if benchmark.digest(output/name) != expected:
                raise ValueError(f'Experiment data changed during plot refresh: {name}')
        analysis.write_json(stage/'plot_provenance.json', {
            'created_utc': datetime.now(timezone.utc).isoformat(), 'plot_type': 'probability_density_histogram',
            'note': PLOT_NOTE, 'input_sha256': hashes, 'plot_code_sha256': benchmark.digest(Path(__file__)),
            'bin_rule': 'Shared expert+policy support; window FD bin count clipped to [12,40]; maxima ceil(sqrt(n)) clipped to [5,12]',
            'inference_performed': False})
        for path in stage.iterdir():
            path.replace(output/path.name)
    print(f'Updated density plots and HTML/Markdown: {output}/index.html; experiment data unchanged', flush=True)


def run(args):
    if getattr(args, 'refresh_plots', False):
        refresh_plots(args.output)
        return
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if config['schema_version'] != 1 or config['window_protocol'] != {'smoothing': 'trailing', 'window': 1, 'wm_frame_policy': 'all_endpoints'}:
        raise ValueError('Require schema 1 and raw all-endpoint scores')
    if len(set(config['tasks'])) != len(config['tasks']) or config['views'] != ['fix', 'wrist']:
        raise ValueError('Tasks must be unique; fixed/wrist views required')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a fresh experiment output folder: {output}')
    inputs = Inputs(REPO/config['source'])
    inputs.external(config_path, benchmark.digest(config_path))
    cases, selections, provenance, labels_path = load_cases(config, inputs)
    method = config.get('episode_quantile_method', 'inverted_cdf')
    summaries, comparisons, episodes, windows = tables(cases, episode_quantile_method=method)
    note = NOTE
    if config.get('dataset_validity'):
        note += (' The frozen dataset-validity mask is applied to policy scores only: Needle Pickup episode 5 '
                 'excludes raw frame 122 onward in both cameras. Expert episodes retain their complete eligible support.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.expert_policy_scores_', dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, rows in [('distribution_summary', summaries), ('comparisons', comparisons), ('episodes', episodes), ('windows', windows)]:
            analysis.write_csv(stage/f'{name}.csv', rows)
        analysis.write_csv(stage/'episode_maximum_summary.csv', [r for r in summaries if r['level'] == 'episode_maximum'])
        names = figures(stage, cases, episodes)
        analysis.write_json(stage/'results.json', {'note': note, 'summary': summaries, 'comparisons': comparisons, 'episodes': episodes,
            'episode_quantile_method': method, 'dataset_validity_applied': bool(config.get('dataset_validity'))})
        analysis.write_json(stage/'episode_selections.json', selections)
        (stage/'experiment.yaml').write_bytes(config_path.read_bytes())
        (stage/'labels_snapshot.yaml').write_bytes(labels_path.read_bytes())
        report(stage, summaries, comparisons, provenance, names, note=note)
        inputs.assert_unchanged()
        analysis.write_json(stage/'provenance.json', {
            'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
            'sources': provenance, 'verified_input_sha256': inputs.used,
            'code_sha256': {str(p.relative_to(REPO)): benchmark.digest(p) for p in
                            [Path(__file__), Path(analysis.__file__), Path(benchmark.__file__)]},
            'numpy_version': np.__version__, 'inference_performed': False,
            'note': note, 'episode_quantile_method': method, 'gpu_used': False,
            'elapsed_seconds': time.perf_counter()-started})
        analysis.write_json(stage/'state.json', {'status': 'complete', 'cases': len(cases), 'window_rows': len(windows),
                                                'episode_rows': len(episodes), 'elapsed_seconds': time.perf_counter()-started})
        output.mkdir(exist_ok=False)
        for path in stage.iterdir():
            path.replace(output/path.name)
    print(f'Complete: {output}/index.html ({time.perf_counter()-started:.1f}s; cached scores only)', flush=True)
