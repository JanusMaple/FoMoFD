#!/usr/bin/env python3
"""Archive benchmark evidence and build a standalone, CPU-only results explorer.

First build (also preserves existing reports/traces from logs):
    python3 lerobot/utils/results_inspector.py --archive-logs
Subsequent builds use only archived results under fomofd/tasks:
    python3 lerobot/utils/results_inspector.py
Add --figures to also export default all-task Matplotlib PDF/SVG/PNG plots.
Use --refresh-presentation to rebuild summaries/HTML from the existing result
snapshot without recalculating thresholds or touching archived scoring reports.
Every build also writes summary.md and summary.yaml, compact text companions.
Open fomofd/tasks/benchmark_results/index.html in a browser. No web server,
model loading, external JavaScript, or GPU is required. Add comparisons to
lerobot/configs/benchmark/results_catalog.yaml; intermediate reports are archived
but deliberately not pooled into the curated comparisons.

archive_result() is also called by the offline/online benchmark writers. Archives
are immutable, content-addressed copies. Original run directories stay intact.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import gzip
import json
import math
import re
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

import yaml
from model_paths import resolve_model_path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / 'fomofd/tasks/benchmark_results'
CATALOG = REPO / 'lerobot/configs/benchmark/results_catalog.yaml'
EXTENSIONS = {'.json', '.csv', '.yaml', '.yml', '.md', '.html'}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def archive_result(source, kind, root=ARCHIVE):
    """Copy and verify report artifacts; do not move logs or duplicate weights."""
    source, root = Path(source).resolve(), Path(root).resolve()
    if source.is_relative_to(root) and (source / 'archive_manifest.json').is_file():
        return source
    required = 'report.json' if kind == 'offline' else 'plan.json'
    if kind not in {'offline', 'online'} or not (source / required).is_file():
        raise ValueError(f'Not a {kind} result directory: {source}')
    files = [p for p in source.iterdir() if p.is_file() and p.suffix in EXTENSIONS]
    if kind == 'online':
        files.extend((source / 'trials').glob('*.json'))
    hashes = {str(p.relative_to(source)): sha256(p) for p in sorted(files)}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = root / kind / f'{source.name}_{identity[:12]}'
    manifest = {'schema_version': 1, 'kind': kind, 'source': relative(source),
                'sha256': identity, 'files': hashes}
    if destination.exists():
        old = json.loads((destination / 'archive_manifest.json').read_text())
        if old['files'] != hashes:
            raise ValueError(f'Archive identity conflict: {destination}')
        # Verify existing archives as well; never silently trust a truncated copy.
        for name, digest in hashes.items():
            if sha256(destination / name) != digest:
                raise ValueError(f'Archive damaged: {destination / name}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.archiving-', dir=destination.parent))
    try:
        for name, digest in hashes.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
            if sha256(target) != digest or sha256(source / name) != digest:
                raise ValueError(f'Result changed during archival: {source / name}')
        manifest['archived_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(staging / 'archive_manifest.json', manifest)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def archive_logs(root=ARCHIVE):
    count = {'offline': 0, 'online': 0}
    for path in sorted((REPO / 'logs/benchmarks').rglob('report.json')):
        report = json.loads(path.read_text())
        if 'score_caches' in report and 'summary_metrics' in report:
            archive_result(path.parent, 'offline', root)
            count['offline'] += 1
    for path in sorted((REPO / 'logs/benchmarks').rglob('plan.json')):
        if (path.parent / 'state.json').is_file() and any((path.parent / 'trials').glob('*.json')):
            state = json.loads((path.parent / 'state.json').read_text())
            if state.get('status') == 'running':
                print(f'Skipping active online run: {path.parent}', flush=True)
                continue
            archive_result(path.parent, 'online', root)
            count['online'] += 1
    print(f'Archived and SHA-256 verified: {count}', flush=True)


TEXT_COLUMNS = {'task', 'method', 'view', 'views', 'calibration', 'split', 'outcome',
                'failure_mode', 'failure_origin', 'onset_view', 'annotator_id',
                'annotator', 'copied_from_annotator', 'status', 'score_type'}


def csv_rows(path):
    def parse(key, value):
        if key in TEXT_COLUMNS:
            return value
        if value == '':
            return None
        if value in {'True', 'False'}:
            return value == 'True'
        try:
            number = float(value)
            if not math.isfinite(number):
                return value  # JSON has no infinity. JavaScript Number() restores it.
            return int(number) if number.is_integer() else number
        except ValueError:
            return value
    with path.open(newline='') as stream:
        return [{k: parse(k, v) for k, v in row.items()} for row in csv.DictReader(stream)]


def packed(rows):
    if not rows:
        return {'columns': [], 'rows': []}
    columns = list(dict.fromkeys(k for row in rows for k in row))
    return {'columns': columns, 'rows': [[r.get(k) for k in columns] for r in rows]}


def resolve_repo_path(path):
    """Relocate original machine paths, retaining the original string in reports."""
    path = str(path)
    marker = '/fomofd/tasks/'
    if marker in path:
        return resolve_model_path(REPO / 'fomofd/tasks' / path.split(marker, 1)[1])
    return resolve_model_path(path, REPO)


def trial_id(trial):
    if trial.get('trial_id'):
        return trial['trial_id']
    views = '_'.join(trial['input_views'])
    if trial.get('fusion'):
        views += '_out_' + trial['fusion']['output_view']
    elif trial.get('score_aggregation') == 'joint':
        views += '_out_joint'
    return f"{trial['variant']}_{trial['task']}_{trial['method']}_{views}_{trial['episode_index']:06d}"


def display_variant(row, aliases):
    """Rename a curated model without changing immutable trial IDs or evidence."""
    return {**row, 'variant': aliases.get(row['variant'], row['variant'])}


def online_coverage(runs, planned_by_source, completed):
    """Count unique recorded trials across recovery runs; retain attempt states."""
    completed = set(completed)
    planned = set().union(*planned_by_source.values())
    if completed - planned:
        raise ValueError('Online measurements have no current trial plan')
    for run in runs:
        ids = set(planned_by_source[run['source']])
        run.update(covered_trials=len(ids & completed), planned_trials=len(ids),
                   coverage_status='complete' if ids <= completed else 'incomplete')
    return {'status': 'complete' if planned <= completed else 'incomplete',
            'completed_trials': len(planned & completed), 'total_trials': len(planned),
            'missing_trials': len(planned - completed)}


def load_policy_success_rates(catalog_path):
    catalog_path = Path(catalog_path)
    reference = yaml.safe_load(catalog_path.read_text()).get('policy_success_rates')
    if not reference:
        return None
    source = catalog_path.parent / reference
    document = yaml.safe_load(source.read_text())
    if document.get('schema_version') != 1 or not document.get('tasks'):
        raise ValueError(f'Invalid policy success-rate document: {source}')
    rows = []
    for task, entry in document['tasks'].items():
        successes, trials = entry['successes'], entry['trials']
        if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials or trials <= 0:
            raise ValueError(f'Invalid policy success counts for {task}')
        rows.append({'task': task, **entry, 'success_rate': successes / trials})
    successes, trials = sum(r['successes'] for r in rows), sum(r['trials'] for r in rows)
    return {'source': relative(source), 'sha256': sha256(source), 'document': document,
            'rows': rows, 'overall': {'successes': successes, 'trials': trials, 'success_rate': successes / trials}}


def load_action_inventory(data):
    """Distinguish trained action variants from verified, cataloged measurements."""
    episodes = [dict(zip(data['episodes']['columns'], row)) for row in data['episodes']['rows']]
    rows = []
    for mode, label, flags in [('none', 'No action', (False, False)),
                               ('adaln', 'AdaLN only', (False, True)),
                               ('cross_attention', 'Cross-attention only', (True, False))]:
        variants = [r['id'] for r in data['runs'] if r['family'] == 'action' and r.get('action_mode') == mode]
        if len(variants) > 1:
            raise ValueError(f'Ambiguous action-ablation reference: {mode}')
        for task in ['needle_pickup', 'ring_over_post', 'tissue_retraction', 'shunt_insertion']:
            run = REPO/'fomofd/tasks'/task/'model/fm_wm'/f'fm_wm_vfmfvae_z8_both_state_off_k4_l4_action_{mode}_data100_seed0'
            path = run/'config.json'
            config = json.loads(path.read_text()) if path.is_file() else {}
            config = config.get('fm_wm', config)
            dc, ac = config.get('data', {}), config.get('action_conditioner', {})
            valid = ((run/'checkpoints/best.pt').is_file()
                     and dc.get('history_len') == dc.get('pred_horizon') == 4
                     and config.get('seed') == 0 and dc.get('train_episode_fraction', 1) == 1
                     and (ac.get('use_cross_attention'), ac.get('use_action_adaln')) == flags)
            benchmarked = any(r['variant'] in variants and r['task'] == task and r['method'] == 'wm_nc'
                              and r['calibration'] == 'episode' and r['split'] == 'test' for r in episodes)
            rows.append({'task': task, 'mode': mode, 'label': label, 'checkpoint_available': valid,
                         'benchmarked': benchmarked, 'variant': variants[0] if variants else None,
                         'checkpoint': relative(run/'checkpoints/best.pt')})
    return rows


def build_data(catalog_path=CATALOG, root=ARCHIVE):
    catalog = yaml.safe_load(Path(catalog_path).read_text())
    if catalog.get('schema_version') != 1:
        raise ValueError('Unknown results catalog schema')
    sources, inventory = {}, []
    for path in sorted(root.glob('*/*/archive_manifest.json')):
        manifest = json.loads(path.read_text())
        sources.setdefault(manifest['source'], []).append((path.parent, manifest))
        inventory.append({'kind': manifest['kind'], 'source': manifest['source'],
                          'path': str(path.parent.relative_to(root)), 'sha256': manifest['sha256'],
                          'files': len(manifest['files'])})

    def get(entry):
        matches = sources.get(entry['source'], [])
        if entry.get('sha256'):
            matches = [x for x in matches if x[1]['sha256'] == entry['sha256']]
        if len(matches) != 1:
            raise ValueError(f"Expected one archive for {entry['source']}; found {len(matches)}. "
                             'Run --archive-logs, or pin sha256 in the catalog if multiple snapshots exist.')
        directory, manifest = matches[0]
        for name, digest in manifest['files'].items():
            if sha256(directory / name) != digest:
                raise ValueError(f'Archived result failed verification: {directory / name}')
        return directory, manifest

    episodes, latencies, distributions, settings, runs, caches = [], [], [], [], [], {}
    label_map, annotators, identities = {}, {}, set()
    for entry in catalog['offline']:
        if entry['id'] in identities:
            raise ValueError(f"Duplicate variant: {entry['id']}")
        identities.add(entry['id'])
        directory, manifest = get(entry)
        report = json.loads((directory / 'report.json').read_text())
        labels = yaml.safe_load((directory / 'labels_snapshot.yaml').read_text())
        annotators.update({k: v['name'] for k, v in labels.get('annotators', {}).items()})
        for task, spec in labels['tasks'].items():
            for group in spec['groups']:
                for ep in range(group['episode_range'][0], group['episode_range'][1] + 1):
                    key = (task, ep)
                    label = {k: group[k] for k in ['split', 'outcome', 'failure_mode', 'failure_origin']}
                    if key in label_map and label_map[key] != label:
                        raise ValueError(f'Conflicting episode labels between runs: {key}')
                    label_map[key] = label
        path = str(directory.relative_to(root))
        runs.append({**entry, 'path': path, 'created': report['created_utc'],
                     'script_sha256': report['script_sha256'], 'labels_sha256': report['labels_sha256'],
                     'protocols': report['protocols'], 'alpha': report['alpha'],
                     'archive_sha256': manifest['sha256'], 'methods': report['methods']})
        for filename, target in [('episode_results', episodes), ('latency_episodes', latencies),
                                 ('score_distributions', distributions)]:
            rows = csv_rows(directory / f'{filename}.csv')
            for row in rows:
                row['variant'] = entry['id']
            target.extend(rows)
        for row in report['scoring_settings']:
            settings.append({**row, 'variant': entry['id']})
        for cache in report['score_caches']:
            actual = resolve_repo_path(cache['path'])
            if not actual.is_relative_to(REPO / 'fomofd/tasks'):
                raise ValueError(f'Score cache outside tasks: {actual}')
            if cache['path'] not in caches:
                if sha256(actual) != cache['sha256']:
                    raise ValueError(f'Score cache failed verification: {actual}')
                caches[cache['path']] = {'path': relative(actual), 'sha256': cache['sha256']}
    # Additional threshold protocols use the same detector identities, not new
    # model variants. Preserve the original archived constant/bin results.
    functional_runs = []
    for entry in catalog.get('functional_band_runs', []):
        directory = (root / entry['path']).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError('Functional reports must remain under tasks/benchmark_results')
        state_path = directory/'state.json'
        state = json.loads(state_path.read_text()) if state_path.is_file() else {'status': 'not_started'}
        summary = {'path': str(directory.relative_to(root)), 'state': state, 'variants': []}
        functional_runs.append(summary)
        if state.get('status') not in {'complete', 'partial'}:
            continue
        manifest = json.loads((directory/'artifact_manifest.json').read_text())
        def verified(path):
            key = str(path.relative_to(directory))
            if key not in manifest['files'] or sha256(path) != manifest['files'][key]:
                raise ValueError(f'Changed functional report: {path}')
            return path
        for run in runs:
            folder = directory/run.get('source_variant', run['id'])/'functional_band'
            if not folder.is_dir():
                continue
            report = json.loads(verified(folder/'report.json').read_text())
            labels = yaml.safe_load(verified(folder/'labels_snapshot.yaml').read_text())
            for task, specification in labels['tasks'].items():
                for group in specification['groups']:
                    for episode in range(group['episode_range'][0],group['episode_range'][1]+1):
                        if (task,episode) in label_map and any(label_map[(task,episode)][k]!=group[k] for k in ['split','outcome','failure_mode','failure_origin']):
                            raise ValueError('Functional comparison changes verified evaluation labels')
            annotators.update({k:v['name'] for k,v in labels.get('annotators',{}).items()})
            summary['variants'].append({'variant': run['id'], 'path': str(folder.relative_to(root)),
                                        'labels_sha256': report['labels_sha256'],
                                        'calibration': report['calibration']})
            for filename, target in [('episode_results',episodes),('latency_episodes',latencies),('score_distributions',distributions)]:
                rows = csv_rows(verified(folder/(filename+'.csv')))
                if any(r['calibration']!='functional_band' for r in rows):
                    raise ValueError('Extra protocol source contains a different calibration')
                target.extend({**r,'variant':run['id']} for r in rows)
    # The same episode must occur only once per comparison; never merge intermediate reports.
    keys = [(r['variant'], r['task'], r['method'], r['view'], r['calibration'], r['episode_index']) for r in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate offline episodes in curated catalog')

    # Preserve a complete cache inventory, including earlier/intermediate runs
    # excluded from comparisons. Their score files are already under tasks.
    for archive in inventory:
        if archive['kind'] != 'offline':
            continue
        report = json.loads((root / archive['path'] / 'report.json').read_text())
        for cache in report['score_caches']:
            if cache['path'] in caches:
                if caches[cache['path']]['sha256'] != cache['sha256']:
                    raise ValueError(f"Conflicting score cache hashes: {cache['path']}")
                continue
            actual = resolve_repo_path(cache['path'])
            if not actual.is_relative_to(REPO / 'fomofd/tasks') or sha256(actual) != cache['sha256']:
                raise ValueError(f'Score cache missing, changed, or outside tasks: {actual}')
            caches[cache['path']] = {'path': relative(actual), 'sha256': cache['sha256']}

    online, online_runs, plans, completed = [], [], {}, {}
    planned_by_source = {}
    for entry in catalog.get('online', []):
        aliases = entry.get('variant_aliases', {})
        directory, manifest = get(entry)
        plan = json.loads((directory / 'plan.json').read_text())
        state = json.loads((directory / 'state.json').read_text())
        online_runs.append({'path': str(directory.relative_to(root)), 'source': entry['source'],
                            'title': entry.get('title', entry['source'].split('/')[-1]),
                            'state': state, 'script_sha256': plan['script_sha256'],
                            'selection': plan['selection'], 'calibration': plan['calibration'],
                            'archive_sha256': manifest['sha256']})
        planned_by_source[entry['source']] = {trial_id(t) for t in plan['trials']}
        for trial in plan['trials']:
            tid = trial_id(trial)
            info = display_variant({k: trial[k] for k in ['variant', 'task', 'method', 'episode_index', 'input_views', 'requested_detector_views']}, aliases)
            info.update(trial_id=tid, calibration=plan['calibration'], selection=plan['selection'],
                        **label_map[(trial['task'], trial['episode_index'])])
            if tid in plans and plans[tid] != info:
                raise ValueError(f'Conflicting online plans: {tid}')
            plans[tid] = info
        for file in sorted((directory / 'trials').glob('*.json')):
            row = json.loads(file.read_text())
            tid = row['trial_id']
            if tid in completed:
                if sha256(file) != completed[tid]:
                    raise ValueError(f'Conflicting online measurements: {tid}')
                continue
            completed[tid] = sha256(file)
            events = row['events']
            # Keep individual timings for exact pooled quantiles after UI filtering.
            timings = [[e['processing_s'], e['source_age_s'], e['stages'].get('encoder_s'),
                        e['stages'].get('detector_s'),
                        sum(c['steps'] for c in e['stages'].get('ode_calls', [])),
                        sum(c['evaluations'] for c in e['stages'].get('ode_calls', []))]
                       for e in events]
            online.append({**plans[tid], 'path': str(file.relative_to(root)),
                           'latencies': [display_variant(r, aliases) for r in row['latencies']], 'timings': timings,
                           'processed': row['processed_endpoints'], 'skipped': row['skipped_endpoints'],
                           'elapsed_s': row['elapsed_s'], 'stop_reason': row['stop_reason'],
                           'hardware': row['hardware']})
    coverage = online_coverage(online_runs, planned_by_source, completed)
    history = []
    for entry in catalog.get('online_history', []):
        replacements = entry.get('superseded_by', [])
        if not entry.get('historical_only') and (not replacements or set(replacements) - set(planned_by_source)):
            raise ValueError('Historical online plan needs named current replacements')
        directory, manifest = get(entry)
        history.append({**entry, 'path': str(directory.relative_to(root)),
                        'title': entry.get('title', entry['source'].split('/')[-1]),
                        'state': json.loads((directory/'state.json').read_text()),
                        'archive_sha256': manifest['sha256']})
    return {'created': datetime.now(timezone.utc).isoformat(), 'runs': runs,
            'episodes': packed(episodes), 'latencies': packed(latencies),
            'distributions': packed(distributions), 'settings': settings,
            'online': online, 'online_runs': online_runs, 'plans': list(plans.values()),
            'online_coverage': coverage, 'historical_online_runs': history,
            'annotators': annotators, 'archives': inventory, 'caches': list(caches.values()),
            'functional_runs': functional_runs,
            'evaluation_protocol_path': catalog.get('evaluation_protocol'),
            'vlm_history': catalog.get('vlm_history', []),
            'summary_sources': catalog.get('summary_sources', {}),
            'supplementary_methods': catalog.get('supplementary_methods', []),
            'model_selection_notes': catalog.get('model_selection_notes', []),
            'model_selection_record': catalog.get('model_selection_record'),
            'training_setup': catalog.get('training_setup', {}),
            'historical_training_corrections': catalog.get('historical_training_corrections', []),
            'training_correction_evidence': catalog.get('training_correction_evidence', {}),
            'policy_success_rates': load_policy_success_rates(catalog_path)}


def export_figures(data, root):
    """Optional Matplotlib SVG/PDF/PNG copies of the default all-task comparisons."""
    import os
    import statistics
    from collections import defaultdict
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    directory = root / 'figures'
    if data.get('offline_protocol'):
        directory = root/data['offline_protocol']['path']/'figures'
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    colors = {'fix': '#176995', 'wrist': '#d37732', 'both': '#865aa6'}
    views = {'fix': 'Fixed', 'wrist': 'Wrist', 'both': 'Combined'}
    methods = {'wm_nc': 'WM-NC', 'wm_pe': 'WM-PE', 'wm_pu': 'WM-PU',
               'logpzo': 'logpZO · VAE', 'rnd': 'RND · VAE',
               'logpzo_native': 'logpZO · Policy', 'rnd_native': 'RND · Policy', 'ace': 'ACE',
               'act_ensemble_pos': 'AEU', 'act_ensemble_rot': 'AEU-Rot'}
    methods = {k:v for k,v in methods.items() if k not in data.get('supplementary_methods', [])}
    unpack = lambda p: [dict(zip(p['columns'], row)) for row in p['rows']]
    episodes = unpack(data['episodes'])
    annotator = next(iter(data['annotators']), None)

    def save(fig, name, title, source_values):
        paths = {}
        for ext in ['svg', 'pdf', 'png']:
            path = directory / f'{name}.{ext}'
            fig.savefig(path, dpi=180, bbox_inches='tight')
            paths[ext] = str(path.relative_to(root))
        plt.close(fig)
        entries.append({'name': title, 'paths': paths, 'values': source_values})

    with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'}):
        for axis in ['K', 'L']:
            variants = {r['id'] for r in data['runs'] if r['family'] == axis or r['id'] == 'main'}
            allowed = {'wm_nc'} if axis == 'K' else {'wm_nc', 'wm_pe'}
            groups = defaultdict(list)
            for r in episodes:
                if r['variant'] in variants and r['method'] in allowed and r['split'] == 'test' and r['calibration'] == 'episode':
                    groups[r['variant'], r['method'], r['view']].append(r)
            points = []
            for (variant, method, view), rows in groups.items():
                values = {s[axis] for s in data['settings'] if s['variant'] == variant and s['method'] == method}
                if len(values) != 1:
                    raise ValueError(f'Ambiguous {axis} settings for {variant}/{method}')
                pos = [r for r in rows if r['outcome'] == 'failure']
                neg = [r for r in rows if r['outcome'] == 'success']
                points.append({'variant': variant, 'method': method, 'view': view, 'x': values.pop(),
                               'FDR': sum(r['alarm'] for r in pos) / len(pos) if pos else None,
                               'FAR': sum(r['alarm'] for r in neg) / len(neg) if neg else None})
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 5.2))
            for (method, view) in sorted({(r['method'], r['view']) for r in points}):
                line = sorted([r for r in points if (r['method'], r['view']) == (method, view)], key=lambda r: r['x'])
                for ax, metric in zip(axes, ['FDR', 'FAR']):
                    ax.plot([r['x'] for r in line], [r[metric] for r in line],
                            color=colors[view], marker='s' if method == 'wm_pe' else 'o',
                            linestyle='--' if method == 'wm_pe' else '-', label=f'{methods[method]} · {views[view]}')
            far_max = min(1, max(.05, math.ceil(max((r['FAR'] or 0 for r in points), default=0)/.05)*.05))
            for ax, metric, limit in zip(axes, ['FDR', 'FAR'], [1, far_max]):
                ax.set_ylim(-.025*limit, 1.04*limit)
                ax.set_yticks([limit*i/4 for i in range(5)])
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.set_ylabel(metric)
                ax.grid(alpha=.25)
            axes[0].legend(loc='lower right', ncol=2, fontsize=8)
            axes[1].set_xticks(sorted({r['x'] for r in points}))
            axes[1].set_xlabel('Endpoint prediction horizon K' if axis == 'K' else 'Observation history length L')
            fig.suptitle(f'{axis} ablation · all tasks · full-episode calibration')
            fig.tight_layout()
            save(fig, f'{axis.lower()}_ablation', f'{axis} ablation', points)

        detector_sources = {method: data.get('summary_sources', {}).get(method, 'main') for method in methods}
        for source in ['online', 'offline']:
            records = [r for t in data['online'] if t['calibration'] == 'episode' for r in t['latencies']] if source == 'online' else unpack(data['latencies'])
            groups = defaultdict(list)
            for r in records:
                if r['variant'] == detector_sources.get(r['method']) and r['method'] in methods and r.get('calibration', 'episode') == 'episode' and r['annotator_id'] == annotator and r['status'] == 'detected':
                    value = r.get('delay_s' if source == 'online' else 'detection_delay_s')
                    if value is not None and math.isfinite(value) and (value >= 0 or r.get('detected_before_human_label')) and not (source == 'online' and r.get('late_alarm',False)):
                        groups[r['method'], r['view']].append(value)
            candidates = sorted({(r['method'], r['view']) for r in episodes if r['variant'] == detector_sources.get(r['method']) and r['method'] in methods}, key=lambda k: (list(methods).index(k[0]), list(views).index(k[1])))
            fig, ax = plt.subplots(figsize=(9.5, 6.8))
            summaries = []
            for i, (method, view) in enumerate(candidates):
                values = groups[method, view]
                average = statistics.mean(values) if values else None
                sd = statistics.stdev(values) if len(values) > 1 else None
                summaries.append({'method': method, 'view': view, 'n': len(values), 'mean_s': average, 'sd_s': sd})
                if values:
                    ax.errorbar(average, i, xerr=sd if sd is not None else 0, fmt='o', color=colors[view], capsize=3)
                ax.text(1.02, i, f'n={len(values)}' if values else 'No detections', transform=ax.get_yaxis_transform(), va='center', fontsize=9)
            ax.set_yticks(range(len(candidates)), [methods[m]+('' if v=='both' and m in ('ace','act_ensemble_pos','act_ensemble_rot') else ' · '+views[v]) for m, v in candidates])
            ax.invert_yaxis()
            ax.axvline(0, color='#8e9eaa', linestyle=':', linewidth=1)
            ax.grid(axis='x', alpha=.25)
            ax.set_xlabel('Onset-to-alarm latency (wall seconds), mean ± sample SD')
            ax.set_title(f'{source.capitalize()} · detected failures only · all tasks\nFull-episode calibration · {data["annotators"].get(annotator, "No annotator")}')
            fig.tight_layout()
            save(fig, f'latency_{source}', f'{source.capitalize()} latency', summaries)
        shared = data.get('shared_latency', {})
        for g in shared.get('groups', []):
            rows = g['summaries']
            fig, ax = plt.subplots(figsize=(8, 2.0 + .48 * len(rows)))
            for i, r in enumerate(rows):
                view = shared['detectors'][r['detector']]['view']
                ax.errorbar(r['mean_s'], i, xerr=r['sd_s'] or 0, fmt='o', color=colors[view], capsize=4)
            ax.set_yticks(range(len(rows)), [shared['detectors'][r['detector']]['label'] for r in rows])
            ax.invert_yaxis(); ax.axvline(0, color='#8e9eaa', linestyle=':', linewidth=1); ax.grid(axis='x', alpha=.25)
            ax.set_xlabel('Onset-to-alarm latency (s), mean ± sample SD')
            ax.set_title(f"Same {g['shared_n']} detected episodes · minimum {g['minimum']}\n{shared['annotator']} · original episode thresholds")
            fig.tight_layout()
            save(fig, 'shared_latency_' + g['id'], 'Shared latency · '+g['id'], g)
        pairwise = data.get('pairwise_latency', {})
        for view in ('wrist',):
            rows = [r for r in pairwise.get('comparisons', [])
                    if r['reference'] == f'main|wm_nc|{view}' and r['on_time_only'] and r['family'] != 'supplementary']
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(10, 2.0 + .55*len(rows)))
            for i, row in enumerate(rows):
                if row['eligible']:
                    for offset, color, item in [(-.13, '#158477', row['summaries'][0]), (.13, '#db8134', row['summaries'][1])]:
                        ax.errorbar(item['mean_s'], i+offset, xerr=item['sd_s'] or 0, fmt='o', color=color, capsize=3)
                else:
                    ax.text(0, i, 'Not measured online' if row['status']=='not_measured_online' else 'No shared detections', va='center', fontsize=8)
                ax.text(1.02, i, f"n={row['shared_n']}", transform=ax.get_yaxis_transform(), va='center', fontsize=8)
            ax.plot([], [], 'o', color='#158477', label='Main wrist WM-NC · recomputed for each pair')
            ax.plot([], [], 'o', color='#db8134', label='Comparator · same episodes')
            ax.set_yticks(range(len(rows)), [r['label'] for r in rows])
            ax.set_ylim(len(rows)-.45, -.65)
            ax.axvline(0, color='#8e9eaa', linestyle=':', linewidth=1); ax.grid(axis='x', alpha=.25)
            ax.legend(fontsize=8); ax.set_xlabel('Onset-to-alarm latency (s), mean ± sample SD')
            ax.set_title(f'Main WM-NC {view} vs each comparator · on-time shared detections\nEach detector uses its own onset; original operating points')
            fig.tight_layout()
            save(fig, 'pairwise_latency_'+view, f'Pairwise latency · {view}', rows)
    write_json(directory / 'figure_manifest.json', {'scope': 'All tasks; all failure types and origins; main methods and declared shared-detection groups; episode calibration',
               'annotator': data['annotators'].get(annotator), 'sd': 'Sample standard deviation (n-1); null if n<2',
               'latency': 'Detected post-onset episodes only; online conditional on offline-detected selection',
               'source_archives': {r['id']: r['archive_sha256'] for r in data['runs']}, 'figures': entries})
    return [{'name': e['name'], 'paths': e['paths']} for e in entries]


def write_markdown_summary(data, root, suffix=''):
    """Markdown/YAML summary with explicitly selected corrected detector sources."""
    from collections import defaultdict
    import statistics
    import benchmark_roc as roc
    from benchmark_studies import compact as compact_studies
    from zoneinfo import ZoneInfo

    methods = {'wm_nc': 'WM-NC', 'wm_pe': 'WM-PE', 'wm_pu': 'WM-PU',
               'logpzo': 'logpZO VAE', 'rnd': 'RND VAE',
               'logpzo_native': 'logpZO policy', 'rnd_native': 'RND policy', 'ace': 'ACE',
               'act_ensemble_pos': 'AEU', 'act_ensemble_rot': 'AEU-Rot'}
    views = {'fix': 'fixed', 'wrist': 'wrist', 'both': 'combined'}
    tasks = {'needle_pickup': 'Needle', 'ring_over_post': 'Ring',
             'tissue_retraction': 'Tissue', 'shunt_insertion': 'Shunt'}
    sources = {method: 'main' for method in methods}
    for method, variant in data.get('summary_sources', {}).items():
        if method not in methods or not any(r['id'] == variant and method in r['methods'] for r in data['runs']):
            raise ValueError(f'Unknown summary source: {method}/{variant}')
        sources[method] = variant
    unpack = lambda p: [dict(zip(p['columns'], r)) for r in p['rows']]
    episodes = unpack(data['episodes'])
    for r in episodes:
        if r['calibration'] == 'time_varying':
            r['calibration'] = 'wall_time_bin'
    test = [r for r in episodes if r['split'] == 'test']
    key = lambda r: (r['variant'], r['method'], r['view'], r['calibration'])
    groups = defaultdict(list)
    for r in test:
        groups[key(r)].append(r)
    analysis = roc.analyze(test, targets=(.05,), subsets=('all',))
    auc = {key(r): r['episode_AUROC'] for r in analysis['roc_summary'] if r['task'] == 'Overall'}
    matched = {key(r): r for r in analysis['matched_far'] if r['task'] == 'Overall'}
    pct = lambda v: '—' if v is None else f'{100*v:.1f}'
    def rates(rows):
        neg = [r for r in rows if r['outcome'] == 'success']
        pos = [r for r in rows if r['outcome'] == 'failure']
        return '/'.join(pct(sum(r['alarm'] for r in subset)/len(subset) if subset else None)
                        for subset in [neg, pos]) if rows else '—'
    def cell(variant, method, view, scheme, with_auc=False):
        k = (variant, method, view, scheme)
        result = rates(groups[k])
        value = auc.get(k)
        return result + '/' + (f'{value:.3f}' if value is not None else '—') if with_auc and groups[k] else result
    def matched_cell(method, view, scheme, variant=None):
        r = matched.get((variant or sources[method], method, view, scheme))
        return pct(r['FAR'])+'/'+pct(r['FDR']) if r else '—'
    tables = {}
    def table(headers, rows, section):
        tables[section] = {'columns': headers, 'rows': rows}
        escape = lambda value: str(value).replace('|', '\\|').replace('\n', ' ')
        return ['| '+' | '.join(map(escape, row))+' |' for row in [headers, ['---']*len(headers), *rows]]+['']
    all_candidates = sorted({(r['method'], r['view']) for r in test
                         if r['method'] in methods and r['variant'] == sources[r['method']]},
                        key=lambda k: (list(methods).index(k[0]), list(views).index(k[1])))
    candidates = [(m,v) for m,v in all_candidates if m not in data.get('supplementary_methods', [])]
    name = lambda method, view: methods[method]+('' if view=='both' and method in ('ace','act_ensemble_pos','act_ensemble_rot') else ' '+views[view])
    unique = {(r['task'], r['episode_index']): r for r in test}
    counts = {outcome: sum(r['outcome'] == outcome for r in unique.values()) for outcome in ['success', 'failure']}
    origins = {origin: sum(r['failure_origin'] == origin for r in unique.values()) for origin in ['natural', 'induced']}
    cal = defaultdict(set)
    for r in episodes:
        if r['split'] == 'calibration':
            cal[r['task']].add(r['episode_index'])
    cal_counts = sorted({len(ids) for ids in cal.values()})
    functional = '; '.join((f"{r['state']['status']} (cached-score recalibration, {len(r['variants'])} variants)" if data.get('offline_protocol') else f"{r['state']['status']} ({r['state'].get('completed_scores', r['state'].get('completed_jobs', 0))}/{r['state'].get('total_scores', r['state'].get('total_jobs', '?'))} scorers)")
                           for r in data.get('functional_runs', [])) or 'not available'
    timestamp = datetime.fromisoformat(data['created']).astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M %Z')
    smoothing = 'No smoothing.' if all(p.get('window') == 1 for r in data['runs'] for p in r['protocols'].values()) else 'Smoothing varies; see saved protocols.'
    alphas = ', '.join(str(v) for v in sorted({r['alpha'] for r in data['runs']}))
    lines = ['# FoMo-FD benchmark summary', '', f"Snapshot: {timestamp}. Functional band: {functional}.", '',
             f"{len(cal)} tasks; {counts['success']} successful test episodes; {counts['failure']} failures "
             f"({origins['natural']} natural, {origins['induced']} induced). Successful calibration episodes per task: {', '.join(map(str, cal_counts))}.", '',
             'Episode-level metrics. FAR/FDR pool episode counts; AUROC averages tasks equally. '+smoothing+' '
             'FDR requires an alarm at/after onset under the frozen protocol; '
             'early alarms are separate. Early-only episodes with complete post-onset evaluation count as missed failures; unknown streaming outcomes retain explicit bounds. '
             'Includes main detectors, K/L, data-size, seed and fused-view ablations, combined monitors, and action-ablation availability. '
             'Detailed failure groups and score curves remain in [HTML](index.html).', '',
             'Machine-readable companion: [summary.yaml](summary.yaml). '
             '[Reviewer evidence, exact latency comparisons and remaining gaps](reviewer_evidence.html).', '']
    if setup := data.get('training_setup'):
        lines += ['## Expert data and training setup', '',
                  f"Verified {setup['verified_date']}. **One person collected all four expert datasets**. "
                  'Counts are episodes; visual-VAE counts apply separately to each camera.', '']
        lines += table(['Task', 'Expert episodes', 'ACT train / val', 'Visual VAE train / val', 'Main WM train / val'],
                       [[r['label'], r['expert_episodes'],
                         f"{r['act_train_episodes']} / {r['act_val_episodes']}",
                         f"{r['visual_vae_train_episodes']} / {r['visual_vae_val_episodes']}",
                         f"{r['wm_train_episodes']} / {r['wm_val_episodes']}"] for r in setup['tasks']],
                       'training_populations')
        lines += [setup['act']['summary'], '', setup['visual_vae']['summary'], '',
                  setup['interpretation'], '',
                  '[Training setup and source records](protocol.md#expert-data-and-policy-training).', '']
    if data.get('historical_training_corrections'):
        lines += ['## Initial training corrections', '']
        lines += ['- '+item['summary'] for item in data['historical_training_corrections']]
        lines += ['', 'Confirmed from the October 2 checkpoint-config audit and October 4 completion logs. '
                  '[Training provenance and scope](protocol.md#training-provenance). '
                  'These corrections are separate from subsequent protocol changes and repeat-run selection.', '']
    coverage = data.get('online_coverage', {'status':'unknown', 'completed_trials':len(data['online']),
                                         'total_trials':len(data['plans'])})
    lines += [f"**Current online coverage: {coverage['completed_trials']}/{coverage['total_trials']} selected trials recorded.** "
              'Counts unique trials across completed and recovered runs; superseded plans remain in history.', '']
    if data.get('offline_protocol'):
        lines += ['**Offline/replay scope:** '+data['offline_protocol']['online_note'], '']
    elif any(s['id']=='populations' for s in data.get('reviewer_evidence',{}).get('sections',[])):
        lines += ['**Offline/replay operating points differ:** the main FAR/FDR table uses historical endpoint trimming; '
                  'replay uses all endpoints and separately calibrated thresholds. The WM-NC population discrepancy is fully reconciled '
                  'in the [population audit](analyses/2026-10-15/wm_nc_population_reconciliation/report.md). '
                  'Do not present the paper-table detection rate and replay latency as one operating point.', '']
    incomplete = [r for r in data['online_runs'] if r.get('coverage_status', r['state']['status']) != 'complete']
    if incomplete:
        latest = max(incomplete, key=lambda r: r['state'].get('finished', r['state'].get('started', '')))
        state = latest['state']
        finished = state.get('finished', state.get('started'))
        when = datetime.fromisoformat(finished).astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M %Z') if finished else 'unknown time'
        lines += [f"**Online completion warning:** {latest['title']}: {state['status']}, "
                  f"{state['completed_trials']}/{state['total_trials']} trials recorded ({when}). "
                  'Missing timings are unavailable; completed offline results remain usable.', '',
                  'Reason: '+str(state.get('error', 'See run state.')), '']
    source_notes = []
    for method, variant in sources.items():
        if variant == 'main' or method in data.get('supplementary_methods', []):
            continue
        run = next(r for r in data['runs'] if r['id'] == variant)
        measured_online = any(t['variant']==variant and t['method']==method for t in data['online'])
        source_notes.append(f"[{methods[method]} source report]({run['path']}/report.md). "
                            + ('Its revised episode-calibrated online results are included. ' if measured_online else 'Its revised online latency is unavailable. ')
                            + 'Unmeasured protocols remain unavailable; historical values are not substituted.')
    source_notes += [
        'AEU (Action Ensemble Uncertainty) is the primary positional variant: the signed change in positional disagreement among three ACT policies, adapted from Thompson et al. AEU is our descriptive abbreviation, not the paper’s method name. AEU-Rot is the supplementary rotational variant, retained under [All comparisons](index.html#all) and the supplementary table below. ACE, AEU and AEU-Rot each use fixed and wrist inputs jointly.',
        'ACE is an ACT-based adaptation of FIPER’s action branch. WM-PU is a Ward-inspired score on FoMo-FD; neither is a full reproduction.',
        'WM-PU retains its existing one-RK4-step, 16-sample results. Numerical convergence is not claimed.']
    source_notes.extend(data.get('model_selection_notes', []))
    lines += [' '.join(source_notes), '']
    if data.get('model_selection_record'):
        lines += [f"[Seed-1 checkpoint selection, archived paths and verification]({data['model_selection_record']}).", '']
    policy_success = data.get('policy_success_rates')
    if policy_success:
        policy_rows = [[tasks.get(r['task'], r['task']), r['environment'],
                        f"{r['successes']}/{r['trials']}", pct(r['success_rate'])] for r in policy_success['rows']]
        overall = policy_success['overall']
        policy_rows.append(['Overall', 'Pooled', f"{overall['successes']}/{overall['trials']}", f"{100*overall['success_rate']:.2f}"])
        lines += ['## Natural policy success rates', '',
                  'User-reported ACT task-completion counts during natural execution (no deliberately induced failures). '
                  'Separate from detector FAR/FDR and the curated PRL split. These aggregate counts were not recomputed from episode records. '
                  '[Source YAML](policy_success_rates.yaml).', '']
        lines += table(['Task', 'Environment', 'Successes / trials', 'Success rate (%)'], policy_rows, 'policy_success_rates')
    categories = data.get('failure_categories')
    if categories:
        from benchmark_failure_categories import format_rate
        lines += ['## Failure-category FDR', '', categories['definition'], '',
                  'Failure pools: '+', '.join(f"{c['title']}: {c['total']}" for c in categories['categories'])+'.', '',
                  f"[All variants/tasks and exact counts]({categories['report_path']}) · "
                  f"[CSV]({categories['csv_path']}) · [JSON]({categories['json_path']}).", '']
        for scheme, title in [('episode','Constant episode thresholds'),('functional_band','Functional bands')]:
            selected = [r for r in categories['main_records'] if r['calibration']==scheme or
                        scheme=='episode' and r['measurement']=='recorded_streaming']
            detectors = sorted({(r['variant'],r['method'],r['view'],r['calibration']) for r in selected})
            lookup = {(r['variant'],r['method'],r['view'],r['calibration'],r['category']):r for r in selected}
            rows = [[methods.get(m,m)+('' if v=='both' and m in ('ace','act_ensemble_pos','act_ensemble_rot') else ' '+views[v])+(' (streaming)' if calibration=='binary_anytime' else ''),
                     *[format_rate(lookup.get((variant,m,v,calibration,c['id']))) for c in categories['categories']]]
                    for variant,m,v,calibration in detectors]
            if rows:
                lines += ['### '+title, '']
                lines += table(['Detector',*[c['title'] for c in categories['categories']]], rows,
                               'failure_categories_'+scheme)
    lines += ['## Main methods', '',
             'First three result columns: **FAR% / FDR% / AUROC**. Matched columns: **achieved FAR% / FDR%** at a 5% FAR budget, '
             'using success-test thresholds separately per task (retrospective ROC, not deployment calibration).', '',
             'Episode = constant threshold for continuous-score methods. Qwen v9 appears in the first column as an uncalibrated '
             '**onset-aware streaming baseline**, with censored outcomes shown as bounds and late returned alarms identified separately; it has no continuous AUROC or matched-FAR result. '
             'Functional = expert-validation mean + trajectory-wide multiplier from all 19 calibration successes, '
             'using scaled residuals. Bins = diagnostic wall-time thresholds, without simultaneous episode coverage. Alpha='+alphas+'.', '']
    main_rows = [[name(m, v), *[cell(sources[m], m, v, s, True) for s in ['episode', 'functional_band', 'wall_time_bin']],
                  matched_cell(m, v, 'episode'), matched_cell(m, v, 'functional_band')] for m, v in candidates]
    for vlm in data.get('vlm_runs', []):
        aid = next(iter(data['annotators']))
        for view in ('fix', 'wrist'):
            records = [r for r in vlm['episodes'] if r['view']==view]
            neg = [r for r in records if r['outcome']=='success']
            pos = [r for r in records if r['outcome']=='failure']
            alarms = [r.get('onset_assessments',{}).get(aid,{}).get('post_alarm') for r in pos]
            n, unknown = sum(v is True for v in alarms), sum(v is None for v in alarms)
            fdr = (pct(n/len(pos)) if not unknown else
                   pct(n/len(pos))+'–'+pct((n+unknown)/len(pos))+f' ({unknown} unknown)') if pos else '—'
            far = pct(sum(r['alarm'] for r in neg)/len(neg)) if neg else '—'
            main_rows.append([f'Qwen v9 {views[view]} · onset-aware',far+'/'+fdr+'/—','—','—','—','—'])
    lines += table(['Detector', 'Episode / VLM onset-aware', 'Functional', 'Bins', 'Matched episode', 'Matched functional'], main_rows, 'main_methods')
    supplemental = [(m,v) for m,v in all_candidates if m in data.get('supplementary_methods', [])]
    if supplemental:
        lines += ['### Supplementary detectors', '',
                  'Retained for inspection; excluded from the main-method charts and tables.', '']
        lines += table(['Detector','Episode FAR% / FDR% / AUROC'],
                       [[name(m,v),cell(sources[m],m,v,'episode',True)] for m,v in supplemental], 'supplementary_detectors')
    if data.get('onset_analysis'):
        aid = next(iter(data['annotators']))
        def onset_counts(rows, field):
            failures = [r for r in rows if r['outcome'] == 'failure']
            if not failures:
                return '—'
            values = [r.get('onset_assessments', {}).get(aid, {}).get(field) for r in failures]
            n, unknown = sum(v is True for v in values), sum(v is None for v in values)
            return f'{n}/{len(values)}' + (f' (+{unknown} unknown)' if unknown else f' ({100*n/len(values):.1f}%)')
        onset_rows = [[name(m, v), *[onset_counts(groups[sources[m], m, v, scheme], field)
                                  for scheme in ('episode', 'functional_band') for field in ('post_alarm', 'early_alarm', 'sustained_pre_onset_alarm')]]
                      for m, v in candidates]
        for vlm in data.get('vlm_runs', []):
            for view in ('fix', 'wrist'):
                selected = [r for r in vlm['episodes'] if r['view'] == view]
                onset_rows.append([f'Qwen v9 {views[view]} · streaming',
                                   onset_counts(selected, 'post_alarm'), onset_counts(selected, 'early_alarm'), onset_counts(selected, 'sustained_pre_onset_alarm'), '—', '—', '—'])
        lines += ['## Detection relative to updated onsets', '',
                  f"Annotator: {data['annotators'][aid]}. [Current label snapshot](analyses/onset_detection/labels_snapshot.yaml). "
                  'Any strict crossing at/after onset counts. Other early alarms and sustained pre-onset runs are separate; episode counts may overlap when different runs occur. '
                  'Fixed/wrist use their own labels; combined detectors use the earlier view. '
                  'Qwen uses recorded source frames; an early-stopped trial without later evidence is unknown, not a miss. '
                  'Onsets are frozen. FAR, thresholds and whole-episode AUROC do not use onset labels. '
                  'Original run reports retain their historical annotation snapshots.', '']
        lines += table(['Detector', 'Episode post-onset', 'Episode other early', 'Episode sustained pre-onset', 'Functional post-onset', 'Functional other early', 'Functional sustained pre-onset'],
                       onset_rows, 'main_onset_detection')
    lines += ['## Main methods by task', '', '**FAR% / FDR%**. Needle/ring are simulation; tissue/shunt are real.', '']
    for scheme, title in [('episode', 'Episode'), ('functional_band', 'Functional band')]:
        lines += [title+'.', '']
        task_rows = [[name(m, v), *[rates([r for r in groups[sources[m], m, v, scheme] if r['task'] == task])
                                   for task in tasks]] for m, v in candidates]
        if scheme == 'episode':
            for vlm in data.get('vlm_runs', []):
                aid = next(iter(data['annotators']))
                def vlm_rates(task, view):
                    rows = [r for r in vlm['episodes'] if r['task']==task and r['view']==view]
                    neg = [r for r in rows if r['outcome']=='success']
                    pos = [r.get('onset_assessments',{}).get(aid,{}).get('post_alarm') for r in rows if r['outcome']=='failure']
                    n, unknown = sum(v is True for v in pos), sum(v is None for v in pos)
                    value = pct(n/len(pos)) if not unknown else pct(n/len(pos))+'–'+pct((n+unknown)/len(pos))+f' ({unknown} unknown)'
                    return pct(sum(r['alarm'] for r in neg)/len(neg))+'/'+value
                for view in ['fix','wrist']:
                    task_rows.append([f'Qwen v9 {views[view]} · onset-aware',*[vlm_rates(t,view) for t in tasks]])
        lines += table(['Detector', *tasks.values()], task_rows,
                       'main_methods_by_task' if scheme == 'functional_band' else 'main_methods_by_task_episode')
    lines += ['## K/L ablations', '', '**FAR% / FDR%**, pooled over all four tasks. K sweep holds L=4; L sweep holds K=4. '
              'Shared K4/L4 reference is reused. Unavailable calibration protocols are shown as —.', '']
    ablations = []
    ablation_headers = ['Setting', 'Method', 'Fixed episode', 'Fixed functional', 'Wrist episode', 'Wrist functional']
    for axis in ['K', 'L']:
        axis_rows = []
        for method in (['wm_nc'] if axis == 'K' else ['wm_nc', 'wm_pe']):
            points = []
            for run in data['runs']:
                if run['family'] != axis and run['id'] != 'main':
                    continue
                values = {r[axis] for r in data['settings'] if r['variant'] == run['id'] and r['method'] == method and r[axis] is not None}
                if len(values) != 1:
                    continue
                points.append((values.pop(), run['id']))
            for value, variant in sorted(points):
                axis_rows.append([f'{axis}={value}', methods[method], *[cell(variant, method, view, scheme)
                                  for view in ['fix', 'wrist'] for scheme in ['episode', 'functional_band']]])
        lines += [f'### {axis} ablation', '']
        lines += table(ablation_headers, axis_rows, axis.lower()+'_ablation')
        ablations.extend(axis_rows)
    # Keep the existing machine-readable combined table for downstream readers.
    tables['k_l_ablations'] = {'columns': ablation_headers, 'rows': ablations}

    lines += ['## Action-conditioning ablation', '',
              'K=L=4, full expert data, seed 0. Compare No action, AdaLN only, cross-attention only and both. '
              '**FAR% / FDR%** uses episode calibration. The HTML provides per-failure comparisons, including execution mismatch. '
              'Unscored variants remain pending; historical K3 results are excluded.', '']
    action_rows = []
    action_inventory = data.get('action_inventory') or load_action_inventory(data)
    for mode in ['none', 'adaln', 'cross_attention']:
        entries = [r for r in action_inventory if r['mode'] == mode]
        variant = entries[0]['variant']
        action_rows.append([entries[0]['label'], ', '.join(tasks[r['task']] for r in entries if r['checkpoint_available']) or 'None',
                            f"{sum(r['benchmarked'] for r in entries)}/4 tasks scored",
                            *[cell(variant, 'wm_nc', view, 'episode') if variant else '—' for view in ['fix', 'wrist']]])
    action_rows.append(['Cross-attention + AdaLN', 'Main reference', '4/4 tasks scored',
                        cell('main', 'wm_nc', 'fix', 'episode'), cell('main', 'wm_nc', 'wrist', 'episode')])
    lines += table(['Conditioning', 'K4/L4 checkpoints available', 'WM-NC benchmark status', 'Fixed episode', 'Wrist episode'],
                   action_rows, 'action_ablation')
    lines += ['[Action comparison and per-failure results](index.html#action).', '']

    lines += ['## Data-size and seed ablations', '',
              'Episode calibration: **FAR% / FDR% / AUROC**; matched columns use the same 5% success-test FAR budget. '
              'Data-size subsets use 75/150/225 training episodes per simulation task and 30/60/90 per real task. '
              'K=L=4; the full-data seed-0 reference is shared. Functional bands are unavailable for the new data-size/seed models.', '']
    variant_rows = []
    for run in data['runs']:
        if run['family'] not in {'data_size', 'seed', 'seed_repeat'} and run['id'] != 'main':
            continue
        for method in ['wm_nc', 'wm_pe']:
            if not any(groups[run['id'], method, v, 'episode'] for v in ['fix', 'wrist']):
                continue
            variant_rows.append([run['title'], methods[method],
                                 *[cell(run['id'], method, v, 'episode', True) for v in ['fix', 'wrist']],
                                 *[matched_cell(method, v, 'episode', run['id']) for v in ['fix', 'wrist']]])
    lines += table(['Model', 'Score', 'Fixed episode', 'Wrist episode', 'Fixed matched', 'Wrist matched'], variant_rows, 'data_size_and_seed')
    if any(r['family']=='seed_repeat' for r in data['runs']):
        lines += ['Local seed-2/3 repeats are additional realizations of existing seeds, scored with WM-NC only. '
                  'They do not replace the active models or enter the eight-seed mean/SD. '
                  'See the per-task repeat comparison and training audit below.', '']
    historical = [r for r in data['runs'] if r['family'] == 'seed_history']
    if historical:
        lines += ['Archived seed realizations (excluded from the active seed comparison). '
                  'Retained so the post-evaluation replacement is visible.', '']
        lines += table(['Model', 'Score', 'Fixed episode', 'Wrist episode', 'Fixed matched', 'Wrist matched'],
                       [[r['title'], methods[m], *[cell(r['id'],m,v,'episode',True) for v in ['fix','wrist']],
                         *[matched_cell(m,v,'episode',r['id']) for v in ['fix','wrist']]]
                        for r in historical for m in ['wm_nc','wm_pe']], 'historical_seeds')
    if data.get('training_runs_report'):
        lines += [f"[Seed-training loss curves, checkpoint paths, and completion audit]({data['training_runs_report']}).", '']

    for run in data['runs']:
        if run['family'] != 'view_fusion':
            continue
        joint = any(groups[run['id'], m, 'both', 'episode'] for m in ['wm_nc','wm_pe','wm_pu'])
        fused_views = ['both'] if joint else ['fix', 'wrist']
        lines += ['## Fused-view model', '',
                  ('One WM over concatenated fixed/wrist VAE channels; one joint score and one calibrated threshold. '
                   'NC/PE sum the coordinate scores; PU averages coordinate standard deviations across both views. '
                   if joint else 'Historical component diagnostics from the joint WM; camera-specific scores. ')
                  + 'No extra OR/AND alarms. '
                  'Episode cells: **FAR% / FDR% / AUROC**. '
                  + ('Functional-band results are unavailable. ' if not any(groups[run['id'], m, v, 'functional_band'] for m in ['wm_nc','wm_pe','wm_pu'] for v in fused_views) else '')
                  + 'Online availability is listed below.', '']
        lines += table(['Detector', 'Main fixed', 'Main wrist', 'Fused model', 'Fused matched'],
                       [[name(m, v), cell('main', m, 'fix', 'episode', True), cell('main', m, 'wrist', 'episode', True),
                         cell(run['id'], m, v, 'episode', True), matched_cell(m, v, 'episode', run['id'])]
                        for m in ['wm_nc', 'wm_pe', 'wm_pu'] for v in fused_views], 'fused_view_'+run['id'])
        lines += ['Fused model by task: **FAR% / FDR%**, episode calibration.', '']
        lines += table(['Detector', *tasks.values()],
                       [[name(m, v), *[rates([r for r in groups[run['id'], m, v, 'episode'] if r['task'] == task]) for task in tasks]]
                        for m in ['wm_nc', 'wm_pe', 'wm_pu'] for v in fused_views], 'fused_view_by_task_'+run['id'])

    fusion_names = {'wm_nc_or': 'Fixed OR wrist', 'wm_nc_and': 'Fixed AND wrist',
                    'wm_nc_logpzo_native_or': 'WM-NC OR policy logpZO',
                    'wm_nc_logpzo_native_and': 'WM-NC AND policy logpZO'}
    fusion_detectors = [(m, v) for m in fusion_names for v in ['both', 'fix', 'wrist'] if groups['main', m, v, 'episode']]
    fusion_levels = data.get('fusion_level_comparison')
    if fusion_levels:
        lines += ['## Combined monitors · episode and endpoint levels', '', fusion_levels['note'], '',
                  '[Both levels in HTML](index.html#fusion) · '
                  f"[Exact episode scores]({fusion_levels['path']}/episode_scores.csv) · "
                  f"[Per-task/per-mode metrics]({fusion_levels['path']}/metrics.csv)", '',
                  'Rates below use frozen component thresholds. Matched rates are a separate retrospective 5% FAR comparison; AUROC uses whole-episode maxima.', '']
        overall = [r for r in fusion_levels['metrics'] if r['task']=='Overall' and r['subset']=='all']
        lines += table(['Monitor','View','Level','False alarms / successes','Detected / failures','FAR% / FDR% / AUROC','Matched FAR% / FDR%'],
                       [[r['label'],views[r['view']],r['level'],f"{r['false_alarm_n']}/{r['success_n']}",
                         f"{r['detected_n']}/{r['failure_n']}", f"{pct(r['FAR'])}/{pct(r['FDR'])}/{r['episode_AUROC']:.3f}",
                         f"{pct(r['matched_FAR'])}/{pct(r['matched_FDR'])}"] for r in overall], 'combined_monitors')
        lookup = {(r['method'],r['view'],r['level'],r['task']):r for r in fusion_levels['metrics'] if r['subset']=='all'}
        lines += table(['Monitor / view / level',*tasks.values()],
                       [[r['label']+' / '+views[r['view']]+' / '+r['level'],
                         *[pct(lookup[r['method'],r['view'],r['level'],t]['FAR'])+'/'+pct(lookup[r['method'],r['view'],r['level'],t]['FDR']) for t in tasks]]
                        for r in overall], 'combined_monitors_by_task')
    elif fusion_detectors:
        lines += ['## Combined monitors', '',
                  'Original separate-branch WM-NC; component thresholds and scales fitted on calibration successes. '
                  'OR=max and AND=min of signed margins at the same raw frame. Only common eligible endpoints are scored; '
                  'native-only alarms outside WM endpoints do not enter fusion. OR does not retain a joint 5% FAR guarantee. '
                  'Episode: **FAR% / FDR% / AUROC**. Functional-band fusion is not defined; online availability is listed below.', '']
        lines += table(['Monitor', 'View', 'Episode', 'Matched episode', 'Natural detected/n', 'Induced detected/n'],
                       [[fusion_names[m], views[v], cell('main', m, v, 'episode', True), matched_cell(m, v, 'episode', 'main'),
                         *[f"{sum(r['alarm'] for r in rows)}/{len(rows)}" for origin in ['natural', 'induced']
                           for rows in [[r for r in groups['main', m, v, 'episode'] if r['outcome']=='failure' and r['failure_origin']==origin]]]]
                        for m, v in fusion_detectors], 'combined_monitors')
        lines += table(['Monitor / view', *tasks.values()],
                       [[fusion_names[m]+' '+views[v], *[rates([r for r in groups['main', m, v, 'episode'] if r['task']==task]) for task in tasks]]
                        for m, v in fusion_detectors], 'combined_monitors_by_task')
    annotator = next(iter(data['annotators']), None)
    delays = defaultdict(list)
    for trial in data['online']:
        if trial['variant'] != sources.get(trial['method']) or trial['calibration'] != 'episode':
            continue
        for r in trial['latencies']:
            value = r.get('delay_s')
            if (r['annotator_id'] == annotator and r['status'] == 'detected'
                    and value is not None and math.isfinite(value) and (value >= 0 or r.get('detected_before_human_label')) and not r.get('late_alarm',False)):
                delays[r['method'], r['view']].append(value)
    lines += ['## Online latency', '',
              f"{len(data['online'])}/{len(data['plans'])} planned trials recorded. Episode calibration; annotator: {data['annotators'].get(annotator, 'none')}. "
              'Wall seconds to the first completed post-onset-source alarm; sample SD; late decisions excluded. Selection includes only offline post-onset detections. '
              'Sustained pre-onset alarms are reported separately from other early alarms; neither receives negative latency. Methods detect different subsets. '
              'Main-baseline replay uses the same corrected episode thresholds as the offline report. '
              'Functional-band online latency has not been measured. '
              '[Current latency records and provenance]('+data.get('interpreted_results',{}).get('path','analyses/2026-10-16_post_onset_v2')+'/report.md). '
              'Linked source-run reports below preserve their original v1 interpretation.', '']
    lines += table(['Run', 'Attempt state', 'This attempt', 'Coverage across runs', 'Report'],
                   [[r['title'], r['state']['status'], f"{r['state']['completed_trials']}/{r['state']['total_trials']}",
                     f"{r.get('covered_trials',r['state']['completed_trials'])}/{r.get('planned_trials',r['state']['total_trials'])}",
                     f"[details]({r['path']}/report.html)"] for r in data['online_runs']], 'online_run_status')
    for old in data.get('historical_online_runs', []):
        lines += [f"Historical attempt: [{old['title']}]({old['path']}/report.html) — "
                  f"{old['state']['status']}, {old['state']['completed_trials']}/{old['state']['total_trials']} recorded. "
                  + old['reason'] + ' Excluded from current coverage.', '']
    lines += table(['Detector', 'Mean ± SD (s)', 'On-time detected n', 'Sustained pre-onset n'],
                   [[name(m, v), (f'{statistics.mean(a):.2f} ± '+(f'{statistics.stdev(a):.2f}' if len(a)>1 else '—')) if a else '—',
                     len(a), sum(bool(r.get('sustained_pre_onset_alarm')) for t in data['online']
                       if t['variant']==sources[m] and t['method']==m for r in t['latencies']
                       if r['view']==v and r['annotator_id']==annotator)]
                    for m, v in candidates for a in [delays[m, v]]], 'online_latency')
    extra = defaultdict(list)
    for trial in data['online']:
        if trial['calibration']!='episode' or trial['variant']==sources.get(trial['method']):
            continue
        for r in trial['latencies']:
            if r['annotator_id']==annotator:
                extra[trial['variant'],trial['method'],r['view']].append(r)
    if extra:
        titles = {r['id']:r['title'] for r in data['runs']}
        rows = []
        for (variant,method,view), records in sorted(extra.items()):
            values = [r['delay_s'] for r in records if r['status']=='detected' and r.get('delay_s') is not None
                      and math.isfinite(r['delay_s']) and (r['delay_s']>=0 or r.get('detected_before_human_label')) and not r.get('late_alarm',False)]
            delay = (f'{statistics.mean(values):.2f} ± '+(f'{statistics.stdev(values):.2f}' if len(values)>1 else '—')) if values else '—'
            rows.append([titles.get(variant,variant), methods.get(method,fusion_names.get(method,method))+' '+views[view],
                         len(records),len(values),sum(bool(r.get('early_alarm',r['status']=='pre_onset_evidence')) for r in records),
                         sum(not r['alarm'] for r in records),sum(r['status']=='unannotated' for r in records),delay])
        lines += ['Online ablations and combined monitors. Same annotator; combined outputs use the earlier fixed/wrist onset. Selected = replayed detector-view episodes. '
                  'Missing-onset episodes are excluded from delay/early classification; misses count every absent alarm.', '']
        lines += table(['Model','Detector','Selected','Post-onset n','Early','Misses','Missing onset','Mean ± SD (s)'],rows,'online_ablation_and_fusion_latency')
    shared = data.get('shared_latency')
    if data.get('pairwise_latency'):
        lines += ['## Main wrist WM-NC versus each method', '',
                  'Pairwise latency uses main wrist WM-NC as the sole reference, including main fixed WM-NC among the comparators. Each pair uses its shared post-onset detections and the same annotator; each detector keeps its own view-specific onset, which need not match. '
                  'Each pair has its own episode set and its own recomputed wrist-method mean. Cross-view differences can reflect different onset times as well as alarm times. '
                  'Primary plots require on-time alarms from both methods; late alarms can be included for both in the HTML. '
                  'This is conditional timing at replay-specific all-endpoint operating points, not matched FAR or an overall method ranking. '
                  'Action variants without online traces remain explicitly unavailable. '
                  '[Interactive chart](index.html#pairwise) · [Pairwise results](pairwise_latency/report.md) · '
                  '[Episode IDs and paired delays](pairwise_latency/episodes.csv).', '']
    if shared:
        lines += ['## Online latency on shared detections', '',
                  'Every detector in a group uses the same episodes, annotator and onset timestamps. '
                  'Minimum counts are floors; all shared episodes are retained. Groups maximize detector count, '
                  'then shared count, without using latency to choose methods. Current main revisions plus joint models; '
                  'original episode thresholds remain unchanged. These are conditional comparisons, not matched-FAR latency. '
                  'Compare means within a group; different groups can contain different failures. '
                  '[Episode IDs, paired differences and full results](shared_latency/report.md).', '']
        rows = []
        for g in shared['groups']:
            for r in g['summaries']:
                rows.append([g['id'], g['minimum'], g['shared_n'], shared['detectors'][r['detector']]['label'],
                             f"{r['mean_s']:.2f} ± {r['sd_s']:.2f}", r['own_post_onset_n']])
        lines += table(['Group','Minimum n','Shared n','Detector','Mean ± SD (s)','Own post-onset n'], rows, 'shared_online_latency')
    for vlm in data.get('vlm_runs', []):
        lines += ['## '+vlm['title'], '', vlm['note'], '',
                  f"Completed {vlm['complete']}/{vlm['planned']} camera/episode replays; "
                  f"indeterminate {vlm['indeterminate']}; unattempted {vlm['unattempted']}. "
                  f"[Full report and raw-result protocol]({vlm['path']}/index.html).", '']
        rows = []
        for r in vlm['summary']+vlm['overall']:
            rate = lambda metric, timing: f"{r[metric+'_'+timing+'_count']}/{r[metric+'_denominator']} ({pct(r[metric+'_'+timing])}%)"
            n = r['conditional_delay_n']
            delay = f"{r['conditional_delay_mean_s']:.2f} ± {r['conditional_delay_sd_s']:.2f} (n={n})" if n else '—'
            rows.append([tasks.get(r['task'],r['task']), views[r['view']], rate('FAR','any'), rate('FDR','any'),
                         rate('FAR','on_time'), rate('FDR','on_time'), r['pre_onset_alarms'], r['late_alarms'], delay])
        lines += table(['Task','View','Any-time FAR','Post-onset FDR incl. late','On-time FAR','On-time post-onset FDR',
                        'Other early alarms','Late alarms','Post-onset delay mean ± population SD (s)'], rows, 'vlm_'+vlm['id'])
    studies = data.get('studies', [])
    if studies:
        lines += ['## Additional studies', '', '[Browse studies inside the main HTML](index.html#studies). '
                  'These are separate descriptive analyses; primary benchmark metrics above retain their original protocols.', '']
    for study in studies:
        lines += [f'### {study["title"]}', '', f'{study["date"]} · {study["status"]}. '+study.get('scope', ''), '']
        lines += [text+'\n' for text in study.get('highlights', [])]
        if study.get('plot_note'):
            lines += [study['plot_note'], '']
        lines += [' · '.join(f'[{label}]({path})' for label, path in study.get('links', {}).items()), '']
        if study['status'] != 'complete':
            continue
        if study['kind'] == 'score_distributions':
            lookup = {(r['task'], r['view'], r['cohort'], r['level']): r for r in study['summary']}
            rows = []
            for task, view in sorted({(r['task'], r['view']) for r in study['summary']}):
                values = [lookup[task, view, cohort, 'episode_maximum'] for cohort in ('expert_validation', 'policy_success')]
                rows.append([tasks[task], views[view], f'{values[0]["n_episodes"]} / {values[1]["n_episodes"]}',
                             *[f'{r["median"]:.2f} ({r["q75"]-r["q25"]:.2f})' for r in values]])
            lines += table(['Task', 'View', 'Expert / policy episodes', 'Expert episode-max median (IQR)',
                            'Policy-success episode-max median (IQR)'],
                           rows, study['id'])
            lines += ['One maximum per episode; IQR = Q3 − Q1. Quantile method: '+study.get('episode_quantile_method','inverted_cdf')+'. '
                      'Calibration (19), success test (20), and all successes (39) are also reported separately in the linked cohort table. '
                      'Maxima use all eligible valid endpoints; episode length and phase can affect them. '
                      'Expert validation was used for checkpoint selection. This does not establish exchangeability.', '']
        elif study.get('overall'):
            rows = []
            for r in study['overall']:
                values = [f'{100*r[m+"_mean"]:.1f} ± {100*r[m+"_sd"]:.1f} [{100*r[m+"_p05"]:.1f}, {100*r[m+"_p95"]:.1f}]'
                          for m in ('FAR', 'FDR', 'episode_AUROC')]
                rows.append([views[r['view']], r['calibration'], *values])
            lines += table(['View', 'Protocol', 'FAR mean ± SD [5–95%] %', 'FDR mean ± SD [5–95%] %', 'AUROC mean ± SD [5–95%] %'],
                           rows, study['id'])
    if data.get('reviewer_evidence'):
        from benchmark_review_results import markdown_sections
        lines += ['## Reviewer evidence and coverage', '',
                  'The following tables make the reviewer-facing details available without HTML filters. '
                  '[Chart-first supplement](reviewer_evidence.html) · [Standalone Markdown](reviewer_evidence.md).', '']
        lines += markdown_sections(data['reviewer_evidence']['sections'], heading=3)
    ode = []
    for method in ['wm_nc', 'wm_pe', 'wm_pu']:
        specs = {(r['ode_solver'], r['ode_steps_per_solve'], r['ode_solves_per_score'], r['vector_field_evals_per_score'])
                 for r in data['settings'] if r['variant'] == sources[method] and r['method'] == method}
        ode.extend(f'{methods[method]}: {solver}, {steps} step/solve × {solves} solves = {evals} field evaluations/score'
                   for solver, steps, solves, evals in sorted(specs, key=str))
    lines += ['ODE settings: '+'; '.join(ode)+'.', '',
              '“—” means unavailable, not zero. This is a fixed all-task summary; HTML filters do not alter it.', '']
    if data.get('evaluation_protocol', {}).get('detection', {}).get('onset_reference'):
        missing = data.get('interpreted_results', {}).get('offline_detected_without_replay', [])
        lines += ['Onset reference: mean wall-clock annotation time per camera (three annotators); combined detectors use the earlier view mean. '
                  'Copied simulated-induced labels are not independent ratings. Individual labels and previous reports are preserved.', '',
                  f'{len(missing)} offline-detected main-detector episode/view result(s) have no saved replay under the updated onset reference; latency remains unmeasured. '
                  '[Coverage details]('+data['interpreted_results']['path']+'/missing_replays.json).', '']
    document='\n'.join(lines)
    if data.get('offline_protocol'):
        p=data['offline_protocol']
        document='# Offline protocol: '+p['id']+'\n\n'+p['note']+'\n\n'+p['online_note']+'\n\n'+document
        document=document.replace('Main FAR/FDR uses historical endpoint trimming; replay uses all endpoints and separately calibrated thresholds.',p['online_note'])
        document=document.replace('historical endpoint trimming','the historical endpoint-trimming protocol (archived only)')
    if suffix:
        for ext in ('html','md','yaml'):
            document=document.replace('(reviewer_evidence.'+ext+')','(reviewer_evidence'+suffix+'.'+ext+')')
        document=document.replace('(summary.yaml)','(summary'+suffix+'.yaml)')
    temporary = root/('.summary'+suffix+'.md.tmp')
    temporary.write_text(document)
    temporary.replace(root/('summary'+suffix+'.md'))
    # Keep tables compact and identical to Markdown, rather than duplicating
    # all per-episode records. Full-precision metrics remain in source reports.
    class TableRow(list):
        pass
    class SummaryDumper(yaml.SafeDumper):
        pass
    SummaryDumper.add_representer(TableRow, lambda dumper, row:
        dumper.represent_sequence('tag:yaml.org,2002:seq', row, flow_style=True))
    payload = {
        'schema_version': 1, 'created_utc': data['created'], 'offline_protocol': data.get('offline_protocol'), 'evaluation_protocol': data.get('evaluation_protocol'),
        'scope': {'tasks': len(cal), 'success_test_episodes': counts['success'], 'failure_test_episodes': counts['failure'],
                  'natural_failures': origins['natural'], 'induced_failures': origins['induced'],
                  'calibration_successes_by_task': {t: len(cal[t]) for t in tasks}},
        'summary_sources': sources,
        'supplementary_methods': data.get('supplementary_methods', []),
        'online_coverage': coverage,
        'historical_online_runs': data.get('historical_online_runs', []),
        'source_reports': {r['id']: {'path': r['path']+'/report.json', 'archive_sha256': r['archive_sha256']} for r in data['runs']},
        'online_runs': [{'source':r['source'], 'path':r['path'], 'status':r['state']['status'],
                         'completed_trials':r['state']['completed_trials'], 'total_trials':r['state']['total_trials'],
                         'coverage_status':r.get('coverage_status',r['state']['status']),
                         'covered_trials':r.get('covered_trials',r['state']['completed_trials']),
                         'planned_trials':r.get('planned_trials',r['state']['total_trials']),
                         'finished':r['state'].get('finished'), 'error':r['state'].get('error')}
                        for r in data['online_runs']],
        'definitions': {'main_metrics': 'FAR% / FDR% / task-macro episode AUROC',
                        'task_and_ablation_metrics': 'FAR% / FDR%',
                        'matched_metrics': 'achieved FAR% / FDR%; 5% success-test FAR budget; retrospective ROC',
                        'missing': '— means unavailable, never zero or a historical substitute',
                        'online_latency': 'seconds to first completed post-onset-source alarm; first annotator; corrected episode thresholds; on-time delays only',
                        'FDR': 'any alarm at or after onset; early alarms reported separately; frozen protocol',
                        'combined_onset': 'earlier fixed/wrist onset; both annotations required',
                        'aggregation': 'FAR/FDR pool counts; AUROC averages tasks equally',
                        'alpha': sorted({r['alpha'] for r in data['runs']}), 'smoothing': smoothing},
        'source_notes': source_notes, 'ode_settings': ode,
        'onset_analysis': data.get('onset_analysis'),
        'pairwise_latency': {'report': 'pairwise_latency/report.md', 'data': 'pairwise_latency/analysis.json',
                             'definition': data['pairwise_latency']['definition']} if data.get('pairwise_latency') else None,
        'action_checkpoint_inventory': action_inventory,
        'policy_success_rates': policy_success,
        'vlm_runs': [{k:v for k,v in r.items() if k != 'episodes'} for r in data.get('vlm_runs', [])],
        'model_selection_notes': data.get('model_selection_notes', []),
        'model_selection_record': data.get('model_selection_record'),
        'training_setup': data.get('training_setup', {}),
        'historical_training_corrections': data.get('historical_training_corrections', []),
        'training_correction_evidence': data.get('training_correction_evidence', {}),
        'additional_studies': compact_studies(studies),
        'reviewer_evidence': data.get('reviewer_evidence'),
        'failure_categories': data.get('failure_categories'),
        'interpreted_results': data.get('interpreted_results'),
        'fusion_level_comparison': {k:v for k,v in data.get('fusion_level_comparison',{}).items() if k!='rows'},
        'tables': {k: {'columns': TableRow(v['columns']), 'rows': [TableRow(row) for row in v['rows']]}
                   for k, v in tables.items()}}
    temporary = root/('.summary'+suffix+'.yaml.tmp')
    temporary.write_text(yaml.dump(payload, Dumper=SummaryDumper, sort_keys=False, allow_unicode=True, width=160))
    temporary.replace(root/('summary'+suffix+'.yaml'))


def write_viewer(data, root):
    """Atomically publish the standalone viewer without touching score reports."""
    root = Path(root)
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    encoded = encoded.replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    output = root/'index.html'
    temporary = root/'.index.html.tmp'
    temporary.write_text(HTML.replace('__RESULT_DATA__', base64.b64encode(gzip.compress(encoded.encode(), mtime=0)).decode()))
    temporary.replace(output)
    return output


def read_viewer_snapshot(path):
    text = Path(path).read_text()
    match = re.search(r'<script id="result-data"[^>]*>([^<]+)</script>', text)
    if not match:
        raise ValueError('Existing viewer has no result snapshot; run a normal build first')
    return json.loads(gzip.decompress(base64.b64decode(match[1], validate=True)))


def publish_protocol(catalog, root):
    """Keep local Markdown links valid when copying the protocol to results."""
    from os.path import relpath
    source = Path(catalog).parent/'PROTOCOL.md'
    def relocate(match):
        target = match[1]
        if target.startswith(('#', '/', 'https://', 'http://')):
            return match[0]
        path, separator, fragment = target.partition('#')
        return ']('+relpath(source.parent/path, root)+(separator+fragment if separator else '')+')'
    (Path(root)/'protocol.md').write_text(re.sub(r'\]\(([^)]+)\)', relocate, source.read_text()))


def refresh_studies(catalog=CATALOG, root=ARCHIVE):
    """Update auxiliary score studies while preserving every primary result."""
    from benchmark_studies import load_studies
    root, catalog = Path(root), Path(catalog)
    data = read_viewer_snapshot(root/'index.html')
    entries = [e for e in yaml.safe_load(catalog.read_text()).get('studies', []) if e['kind'] == 'score_distributions']
    others = [s for s in data.get('studies', []) if s['kind'] != 'score_distributions']
    data['studies'] = load_studies(catalog, root, entries=entries) + others
    data['auxiliary_studies_refreshed_utc'] = datetime.now(timezone.utc).isoformat()
    write_markdown_summary(data, root)
    write_viewer(data, root)
    print(f'Auxiliary score studies refreshed; primary results unchanged. {root/"index.html"}')


def refresh_online_records(data, catalog, root, spec):
    """Load archived replays while retaining the displayed offline experiment.

    In particular, never rewrite the score reports and saved sensitivity inputs
    just to add a completed replay. Archive/source checks still use build_data.
    """
    from benchmark_consensus_onsets import derive_labels
    from benchmark_onset_results import RawClocks, refresh_online_latencies
    from benchmark_frozen_protocol import update_recorded_online, export_interpreted
    fresh = build_data(catalog, root)
    if {r['id'] for r in fresh['runs']} != {r['id'] for r in data['runs']}:
        raise ValueError('Offline variants changed; use a full report build')
    for name in ('online', 'online_runs', 'plans', 'online_coverage', 'historical_online_runs'):
        data[name] = fresh[name]
    archives = {r['path']: r for r in data['archives']}
    archives.update({r['path']:r for r in fresh['archives'] if r['kind']=='online'})
    data['archives'] = list(archives.values())
    labels_path = resolve_repo_path(spec['frozen_inputs']['onsets']['path'])
    if sha256(labels_path) != spec['frozen_inputs']['onsets']['sha256']:
        raise ValueError('Onsets changed from the frozen protocol')
    labels = derive_labels(yaml.safe_load(labels_path.read_text()), spec, resolve_repo_path)
    if set(labels['annotators']) != set(data['annotators']):
        raise ValueError('Annotator roster changed; use a full report build')
    clocks = RawClocks(resolve_repo_path)
    for trial in data['online']:
        saved = json.loads((root/trial['path']).read_text())
        wall = clocks(saved['trial']['dataset'], trial['episode_index'])
        trial['latencies'] = refresh_online_latencies(trial, saved, labels, wall)
    # VLM traces are already interpreted in this snapshot. Importing additional
    # continuous-score replays must not overwrite their historical audit fields.
    update_recorded_online({**data, 'vlm_runs': []}, root, resolve_repo_path, spec)
    data['created'] = datetime.now(timezone.utc).isoformat()
    data['onset_analysis']['online_trials_reannotated'] = len(data['online'])
    data['interpreted_results'] = export_interpreted(data, root)
    if data.get('offline_protocol'):
        data['offline_protocol']['online_note'] = (
            'Current replay uses corrected all-endpoint episode thresholds and offline post-onset selection. '
            'Conditional latency/retention only; Qwen separately replays all test episodes. '
            'Included replay groups: '+', '.join(r['title'] for r in data['online_runs'])+'. '
            'Historical ablation replay is excluded.')
    print(f'Refreshed {len(data["online"])} archived online trials; offline scores/splits preserved.', flush=True)


def refresh_presentation(catalog=CATALOG, root=ARCHIVE, figures=False, online=False):
    """Reuse displayed results; never recompute offline protocols/sensitivity."""
    from benchmark_failure_categories import write_report as write_categories
    from benchmark_frozen_protocol import load_protocol
    from benchmark_review_results import write_review_evidence
    from benchmark_shared_latency import write_pairwise_report, write_report as write_shared_report
    root, catalog = Path(root), Path(catalog)
    data = read_viewer_snapshot(root/'index.html')
    config = yaml.safe_load(catalog.read_text())
    spec = load_protocol(config['evaluation_protocol'])
    if data.get('evaluation_protocol') != spec:
        raise ValueError('Displayed snapshot does not match the current frozen protocol')
    if online:
        refresh_online_records(data, catalog, root, spec)
    data['presentation_refreshed_utc'] = datetime.now(timezone.utc).isoformat()
    data['supplementary_methods'] = config.get('supplementary_methods', [])
    data['training_setup'] = config.get('training_setup', {})
    data['historical_training_corrections'] = config.get('historical_training_corrections', [])
    data['training_correction_evidence'] = config.get('training_correction_evidence', {})
    titles = {r['id']: r['title'] for r in config.get('offline', [])}
    for run in data['runs']:
        if run['id'] in titles:
            run['title'] = titles[run['id']]
    online_titles = {r['source']: r['title'] for r in config.get('online', []) if r.get('title')}
    for run in data['online_runs']:
        if run['source'] in online_titles:
            run['title'] = online_titles[run['source']]
    data['shared_latency'] = write_shared_report(data, root)
    data['pairwise_latency'] = write_pairwise_report(data, root)
    if figures:
        data['figure_exports'] = export_figures(data, root)
    data['failure_categories'] = write_categories(data, root)
    if config.get('reviewer_evidence'):
        data['reviewer_evidence'] = write_review_evidence(
            data, root, catalog.parent/config['reviewer_evidence'], resolve_repo_path)
    from benchmark_fusion_levels import build as build_fusion_levels
    data['fusion_level_comparison'] = build_fusion_levels(data, root, resolve_repo_path)
    write_markdown_summary(data, root)
    publish_protocol(catalog, root)
    write_viewer(data, root)
    print(f'Presentation refreshed from {data["created"]}; no score/calibration recomputation. {root/"index.html"}')
    return data['failure_categories']


def build(catalog=CATALOG, root=ARCHIVE, figures=False):
    root.mkdir(parents=True, exist_ok=True)
    data = build_data(catalog, root)
    from benchmark_vlm_results import load_runs as load_vlm_runs
    data['vlm_runs'] = load_vlm_runs(yaml.safe_load(Path(catalog).read_text()).get('vlm_online', []), root)
    label_lookup = {(r['task'],r['episode_index']):r for r in
                    (dict(zip(data['episodes']['columns'], row)) for row in data['episodes']['rows'])}
    for run in data['vlm_runs']:
        for row in run['episodes']:
            reference = label_lookup[row['task'],row['episode_index']]
            if any((row[k] or None) != (reference[k] or None) for k in ['split','outcome','failure_mode','failure_origin']):
                raise ValueError('VLM and main-method evaluation labels differ')
    if data.get('policy_success_rates'):
        (root/'policy_success_rates.yaml').write_text(yaml.safe_dump(
            data['policy_success_rates']['document'], sort_keys=False, allow_unicode=True))
    from benchmark_studies import load_studies
    data['studies'] = load_studies(catalog, root)
    from benchmark_onset_results import augment
    frozen_path = yaml.safe_load(Path(catalog).read_text()).get('evaluation_protocol')
    from benchmark_frozen_protocol import load_protocol
    spec = load_protocol(frozen_path) if frozen_path else None
    augment(data, root, resolve_repo_path, packed, spec=spec)
    if frozen_path:
        from benchmark_frozen_protocol import load_protocol, update_recorded_online
        spec = load_protocol(frozen_path)
        data['evaluation_protocol'] = spec
        update_recorded_online(data, root, resolve_repo_path, spec)
        data['onset_analysis']['latency_rule'] = 'First completed post-onset-source alarm; sustained pre-onset runs classified separately'
        (root/'evaluation_protocol.yaml').write_bytes(resolve_repo_path(frozen_path).read_bytes())
        publish_protocol(catalog, root)
    from benchmark_shared_latency import write_report as write_shared_latency
    data['shared_latency'] = write_shared_latency(data, root)
    from benchmark_shared_latency import write_pairwise_report
    data['pairwise_latency'] = write_pairwise_report(data, root)
    training_report = root/'analyses/2026-10-14/seed_convergence/index.html'
    if not training_report.is_file():
        training_report = root/'analyses/2026-10-11/seed_convergence/index.html'
    data['training_runs_report'] = str(training_report.relative_to(root)) if training_report.is_file() else None
    data['action_inventory'] = load_action_inventory(data)
    action_audit = root/'analyses/2026-10-14/action_protocol_audit/report.md'
    data['action_protocol_audit'] = str(action_audit.relative_to(root)) if action_audit.is_file() else None
    review_config = yaml.safe_load(Path(catalog).read_text()).get('reviewer_evidence')
    if yaml.safe_load(Path(catalog).read_text()).get('offline_endpoint_options'):
        from benchmark_endpoint_protocol import build_options
        options = build_options(data, root, catalog, resolve_repo_path, packed)
    else:
        options = {'uncorrected': {}}
    for option, patch in options.items():
        selected = {**data, **patch}
        from benchmark_frozen_protocol import export_interpreted
        selected['interpreted_results'] = export_interpreted(selected, root)
        suffix = '' if frozen_path else '_visual_corrected' if option=='visual_corrected' else ''
        selected['figure_exports'] = export_figures(selected, root) if figures else []
        selected['report_links'] = {'markdown':'summary'+suffix+'.md','yaml':'summary'+suffix+'.yaml',
                                    'evidence':'reviewer_evidence'+suffix+'.html'}
        if frozen_path:
            from benchmark_failure_categories import write_report as write_categories
            selected['failure_categories'] = write_categories(selected, root, suffix=suffix)
        if review_config:
            from benchmark_review_results import write_review_evidence
            selected['reviewer_evidence'] = write_review_evidence(
                selected, root, Path(catalog).parent/review_config, resolve_repo_path, suffix=suffix)
        if frozen_path:
            from benchmark_fusion_levels import build as build_fusion_levels
            selected['fusion_level_comparison'] = build_fusion_levels(selected, root, resolve_repo_path)
        write_markdown_summary(selected, root, suffix=suffix)
        patch.update({k:selected[k] for k in ['figure_exports','report_links','reviewer_evidence','failure_categories','interpreted_results','fusion_level_comparison'] if k in selected})
    data.update(options['visual_corrected' if frozen_path else 'uncorrected'])
    if data.get('offline_protocol'):
        if not frozen_path:
            data['offline_options'] = options
        write_json(root/'offline_options_provenance.json', {key:{'protocol':p['offline_protocol'],
            'runs':[{'variant':r['id'],'report':r['path']+'/report.json','report_sha256':sha256(root/r['path']/'report.json')} for r in p['runs']]}
            for key,p in options.items()})
    data['roc_analyses'] = [{'path': str(p.parent.relative_to(root)),
                            'created': json.loads(p.read_text()).get('created_utc')}
                           for p in sorted(root.glob('analyses/**/roc_report.json'))]
    output = write_viewer(data, root)
    write_json(root / 'catalog_snapshot.json', yaml.safe_load(Path(catalog).read_text()))
    write_json(root / 'score_cache_inventory.json', data['caches'])
    write_json(root / 'archive_index.json', data['archives'])
    from benchmark_studies import compact as compact_studies
    write_json(root / 'study_inventory.json', compact_studies(data['studies']))
    (root / 'README.md').write_text(
        '# Benchmark results\n\nOpen [index.html](index.html) in a browser. This is a self-contained, '
        'read-only explorer; no server, network or inference is needed.\n\n'
        '[summary.md](summary.md) and [summary.yaml](summary.yaml) are concise companions for sharing with ChatGPT. '
        'The [reviewer evidence supplement](reviewer_evidence.html) maps comments to results and remaining gaps, '
        'with full numeric latency comparisons also embedded in both summary files. '
        'They are regenerated with the HTML and summarize all tasks using frozen post-onset FDR, independently of browser filters. '
        'The HTML global FDR control switches to any crossing at/after onset and separately reports early alarms. '
        'FAR and whole-episode AUROC remain unchanged. Combined detectors automatically use the earlier camera onset. '
        'Derived onset results and provenance are in analyses/onset_detection/. '
        'Use **Hyperparameter ablation** for K and L charts together; these are sensitivity comparisons, not held-out hyperparameter selection. Tables remain available in expandable sections. '
        'Use **Calibration split sensitivity** in the main navigation for the 500-split WM-NC study and the 200-split multi-method comparison. **Training seeds** shows repeated training runs; **Data sensitivity** shows expert/policy probability-density plots. **Action conditioning** shows measured results or explicit pending status; **Per-failure results** includes NoAct once scored. '
        'The 500-split/200-split calibration-sensitivity experiments have their own task/view '
        'selectors and are also summarized in the Markdown/YAML. Study figures are embedded in the HTML. '
        'Qwen v9 is included in **Main methods** using post-onset FDR with explicit unknowns; **Qwen VLM** retains on-time rates, late alarms and conditional delays. '
        'The training-seed comparison uses the selected desktop seed 1; archived PACE seed 1 remains under All comparisons. '
        'The catalog’s `summary_sources` selects corrected detector revisions for these reports; '
        'historical comparisons remain available in the HTML.\n\n'
        'Rebuild from the repository root with `python3 lerobot/utils/results_inspector.py`. '
        'Add `--figures` for Matplotlib SVG/PDF/PNG copies of the default all-task comparisons. '
        'Add `--archive-logs` to collect existing completed/failed run artifacts from logs. '
        'Future benchmark writers automatically archive finished reports here.\n\n'
        'Curate comparisons in `lerobot/configs/benchmark/results_catalog.yaml`. '
        'Source paths there are provenance identifiers, resolved through archived manifests; '
        'logs are not needed to build or view this dashboard. Pin a manifest sha256 when '
        'multiple versions of one source exist.\n\n'
        '`offline/` preserves original reports, per-episode scores, calibration, distributions, '
        'configuration and annotation snapshots. `online/` preserves plans, runtime environment, '
        'validation and every measured event in trial JSON. Each archive has a SHA-256 manifest. '
        'Original logs are left intact. Raw NPZ score caches remain under each task’s '
        '`prl_data/score_result/`; see `score_cache_inventory.json`. Weights and datasets are '
        'not duplicated. Absolute paths inside original files are historical provenance.\n\n'
        'Success-test negatives are retained when filtering failure origins/types. Offline '
        'AUROC uses one maximum calibrated margin per episode; overall AUROC is a task macro mean. '
        'Offline alarm metrics use the paper endpoint policy; offline latency uses all endpoints. '
        'Online results describe only selected offline-detected failures, not overall online FAR/FDR. '
        'Copied annotations are identified and annotators are never pooled.\n')
    with (root/'README.md').open('a') as stream:
        stream.write('\nThe ROC & matched FAR tab uses success-TEST episodes to select per-task '
                     'thresholds at FAR budgets 0%, 5%, 10% and 20%. This is retrospective '
                     'ROC analysis, separate from deployment calibration and original latency. '
                     'Time-varying scores can also be compared. Full AUROC is threshold-independent.\n')
        stream.write('\nShared-episode latency: [report](shared_latency/report.md), '
                     '[episode IDs](shared_latency/episodes.csv), and [paired differences](shared_latency/paired_differences.csv). '
                     'The Shared latency comparison requires a common post-onset detection set and identical '
                     'onset timestamps. Minimum counts are floors. Original thresholds remain fixed; '
                     'group search uses detection overlap, never latency.\n')
    print(f"Viewer: {output}\n{len(data['runs'])} offline comparisons; "
          f"{len(data['online'])}/{len(data['plans'])} planned online trials recorded; "
          f"{len(data['caches'])} score caches verified. No inference performed.")
    return data


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FoMo-FD · Results explorer</title>
<style>
:root{color-scheme:light;--ink:#193142;--muted:#5f7484;--line:#dbe4ea;--blue:#176995;--teal:#138075;--bg:#f3f6f8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}
header{padding:24px 30px;background:#143344;color:white;display:flex;justify-content:space-between;align-items:center;gap:20px}
h1{font-size:25px;margin:0;letter-spacing:-.6px}header p{margin:4px 0 0;color:#bcd7e6}h2{font-size:19px;margin:0 0 12px}h3{font-size:15px}
main{max-width:1800px;margin:auto;padding:20px 26px}.filters,.panel,.card{background:white;border:1px solid var(--line);border-radius:12px}
.filters{padding:16px;display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin-bottom:12px}label{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--muted)}
select,button{font:inherit;padding:8px 10px;border:1px solid #cad7df;border-radius:6px;background:white;color:var(--ink)}select{max-width:260px}button{cursor:pointer}button:hover{border-color:var(--blue);color:var(--blue)}
.nav{display:flex;gap:6px;flex-wrap:wrap;margin:16px 0}.nav a{padding:7px 10px}.nav details{padding:8px 10px}.nav button.active{background:var(--blue);border-color:var(--blue);color:white}.panel{padding:20px;margin:14px 0;overflow:hidden}
.comparisons{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:10px}.comparisons a{display:block;padding:10px 14px;border:1px solid var(--line);border-radius:10px;background:white;text-decoration:none;color:var(--ink)}.comparisons a:hover{border-color:var(--blue)}.comparisons a.active{background:#e8f3fa;border:2px solid var(--blue);padding:9px 13px}.comparisons strong{display:block;font-size:16px}.comparisons span{display:block;font-size:12px;color:var(--muted);margin-top:3px}.comparisons a:focus-visible{outline:3px solid var(--teal);outline-offset:3px}
.study-shortcuts a{border:1px solid var(--line);border-radius:6px;text-decoration:none}.study-shortcuts a.active{background:#e8f3fa;border-color:var(--blue);font-weight:600}
.secondary-nav{padding:6px 0;margin:6px 0 10px}.metric-notes{margin:8px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{padding:8px 14px}.card strong{font-size:18px;display:block}.muted,.note{color:var(--muted)}.note{font-size:13px;max-width:1200px}.tag{padding:3px 8px;border-radius:12px;background:#e5f3ee;color:#246251;white-space:nowrap;font-size:12px}
.warning{border-left:3px solid #cd9230;background:#fff8e8;padding:10px 14px;font-size:13px}.scroll{overflow:auto;max-height:610px}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{border-bottom:1px solid #e4ebef;text-align:left;padding:8px 10px}th{position:sticky;top:0;background:#eef4f7;z-index:1}tr:hover td{background:#f6fafc}td.num{text-align:right;font-variant-numeric:tabular-nums}a{color:var(--blue)}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;justify-content:space-between;margin-bottom:14px}.inline{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.checks{display:flex;flex-wrap:wrap;gap:7px 18px;padding:10px 0}.checks label{flex-direction:row;align-items:center;font-size:13px;color:var(--ink)}details summary{cursor:pointer;color:var(--blue)}
svg{display:block;width:100%;height:auto;font:12px system-ui,sans-serif}.chart{overflow:auto}.small{font-size:12px}.empty{padding:24px;color:var(--muted)}.page{display:none}.page.active{display:block}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#f4f7f9;padding:12px;border-radius:6px}
.figure{max-width:1100px;margin:0 auto}.figure svg{font-size:13px}.details-body{padding-top:16px}[hidden]{display:none!important}
#sharedGroups td:nth-child(3){white-space:normal;min-width:280px;max-width:780px}#sharedGroups button{white-space:nowrap}
.study-figure{margin:20px 0}.study-figure img{display:block;width:100%;max-width:1200px;height:auto}.study-figure h3{margin-bottom:4px}
@media(max-width:800px){header{padding:18px;display:block}main{padding:12px}.cards,.comparisons{grid-template-columns:repeat(2,1fr)}.filters{gap:8px}select{max-width:185px}.panel{padding:12px}}
@media print{.filters,.nav,button,details{display:none}.page{display:block}.scroll,.chart{max-height:none;overflow:visible}header{background:white;color:var(--ink)}.panel{break-inside:avoid}}
</style></head><body>
<header><div><h1>FoMo-FD <span style="font-weight:400;color:#bed7e3">/ Results explorer</span></h1><p>Failure detection, ablations & recorded-wall-time latency</p></div><div><span class="tag">Read-only · Local results</span><p class="small" id="built"></p><p><a href="summary.md" download style="color:#d5efff">Download concise Markdown</a></p></div></header>
<main>
<div class="panel"><strong>Reviewer evidence and completeness</strong><p>Onset latency with explicit denominators, paired comparisons, uncertainty, per-failure action results, episode durations, and remaining requests. <a href="reviewer_evidence.html">Open chart-first evidence report</a> · <a href="summary.md">Complete Markdown</a> · <a href="summary.yaml">Complete YAML</a></p></div>
<div class="panel" id="offlineProtocolPanel" hidden><strong>Frozen data/scoring · mean-onset reporting v3</strong><p>All valid endpoints; dataset visual correction applied; no smoothing; FDR requires a crossing at/after the mean annotated onset. <a href="protocol.md">Protocol and paper wording</a> · <a href="evaluation_protocol.yaml">Frozen specification</a></p><p class="note" id="offlineProtocolNote"></p><p id="offlineProtocolDownloads"></p></div>
<details class="panel" id="trainingSetup" hidden><summary>Expert data and training setup · operator, ACT/VAE splits, frozen DINOv2</summary><div class="details-body" id="trainingSetupItems"></div><p><a href="protocol.md#expert-data-and-policy-training">Verified training setup and source records</a></p></details>
<details class="panel" id="trainingCorrections" hidden><summary>Initial training corrections · ring action horizon and shunt expert baselines</summary><div class="details-body" id="trainingCorrectionItems"></div><p><a href="protocol.md#training-provenance">Training provenance and scope</a></p></details>
<nav id="comparisons" class="comparisons" aria-label="Reviewer questions">
<a href="#main" data-comparison="main"><strong>Main methods</strong><span data-comparison-summary="main">Native features, uncertainty, ACE &amp; Qwen</span></a>
<a href="#main/roc" data-comparison="main/roc"><strong>Matched FAR &amp; AUROC</strong><span>Compare at the same false-alarm budget</span></a>
<a href="#study/main_calibration_sensitivity_500" data-comparison="study/main_calibration_sensitivity_500" data-study-kind="calibration_sensitivity"><strong>Calibration split sensitivity</strong><span>500 splits · main WM-NC / 200 splits · multi-method</span></a>
<a href="#pairwise" data-comparison="pairwise"><strong>Detection timing</strong><span>Wrist WM-NC vs each method on shared detections</span></a>
<a href="#action" data-comparison="action"><strong>Action conditioning</strong><span>NoAct, AdaLN, cross-attention &amp; both</span></a>
<a href="#hyper" data-comparison="hyper"><strong>Hyperparameter ablation</strong><span>K and L sweeps together · shared K4/L4 reference</span></a>
<a href="#seed" data-comparison="seed"><strong>Training seeds</strong><span data-comparison-summary="seed">Repeated training runs</span></a>
<a href="#main/failure" data-comparison="main/failure"><strong>Failure categories &amp; origins</strong><span>Four categories · pooled FDR · natural / induced</span></a>
<a href="#data_size" data-comparison="data_size"><strong>Data sensitivity</strong><span>Training quantity &amp; expert–policy shift</span></a>
</nav>
<details id="moreComparisons" class="secondary-nav"><summary>View fusion, VLM details and all experiments</summary><nav class="nav" id="extraComparisons" aria-label="More experiments">
<a href="#K" data-comparison="K">K only</a><span data-comparison-summary="K" hidden></span>
<a href="#L" data-comparison="L">L only</a><span data-comparison-summary="L" hidden></span>
<a href="#fusion" data-comparison="fusion">OR/AND monitors</a>
<a href="#fused" data-comparison="fused">Joint-view WM</a>
<a href="#vlm" data-comparison="vlm">Qwen details</a>
<a href="#shared" data-comparison="shared">Shared latency · multi-method groups</a>
<a href="#studies" data-comparison="studies">Calibration &amp; score distributions</a>
<a href="#all" data-comparison="all">All comparisons / supplementary / historical results</a>
</nav><p class="note">Available evidence is mapped to reviewer comments; this is not a claim that every concern is resolved. Longer-horizon / ex vivo experiments, irreversible-failure timing and targeted recovery are not established by these benchmarks.</p></details>
<p class="note" id="comparisonContext"></p>
<p class="note" id="modelSelectionNotes" hidden></p>
<div class="panel" id="annotationControls"><div class="inline">
<label>Frozen FDR definition<select id="detectionRule" disabled><option value="post">Any alarm at/after onset</option></select></label>
<label>Annotator<select id="annotator"></select></label></div>
<p class="note" id="detectionNote"></p></div>
<div id="benchmarkControls">

<select id="preset" hidden aria-label="Comparison"><option value="main">Main methods</option><option value="action">Action conditioning</option><option value="hyper">Hyperparameter ablation · K and L</option><option value="K">K ablation · WM-NC</option><option value="L">L ablation · WM-NC / WM-PE</option><option value="fusion">Combined monitors</option><option value="fused">Fused-view model</option><option value="data_size">Training data size</option><option value="seed">Training seed</option><option value="all">All scored detectors &amp; fusions</option></select>
<div class="filters">
<label>Task<select id="task"><option value="all">All tasks</option></select></label>
<label>Failure origin<select id="origin"><option value="all">Natural + induced</option><option value="natural">Natural only</option><option value="induced">Induced only</option></select></label>
<label>Failure type<select id="failure"><option value="all">All failure types</option></select></label>
<label>Calibration / ROC score transform<select id="scheme"><option value="episode">Episode calibration / onset-aware VLM</option><option value="functional_band">Functional simultaneous band</option><optgroup label="Legacy diagnostic"><option value="time_varying">Wall-time bins</option></optgroup></select></label>
<label>Detector view<select id="view"><option value="all">All views</option><option value="fix">Fixed</option><option value="wrist">Wrist</option><option value="both">Combined</option></select></label>
<button id="reset">Reset</button></div>

<details class="panel" style="padding:12px 18px"><summary id="methodSummary">Choose detectors</summary><div class="inline"><button id="selectAll">Select all available</button><button id="selectNone">Clear selection</button></div><div id="methods" class="checks"></div></details>
<details class="metric-notes"><summary>Metric definitions and comparison limits</summary><p class="note">Failure filters apply to positive episodes. <b>Successful test episodes always remain in FAR and AUROC.</b> Calibration episodes never enter test metrics. All-task FAR/FDR use pooled counts; AUROC averages the eligible per-task AUROCs.</p>
<p class="note" id="qwenProtocolNote">Qwen v9 is an uncalibrated streaming baseline. Primary FDR requires post-onset evidence; censored early-stopped traces are unknown, and late returned decisions are identified separately. AUROC, ROC sweeps and functional/bin thresholds are unavailable for its binary output. The Qwen VLM panel retains on-time rates and late-alarm counts.</p></details>
<p class="note" id="operatingPointNote"><strong>Offline/replay operating points differ.</strong> Historical reports trim endpoints; replay uses all endpoints and separately calibrated thresholds. Read the <a href="reviewer_evidence.html#populations">population reconciliation</a> before pairing a detection rate with online latency.</p>
<p class="note" id="functionalStatus"></p><div id="cards" class="cards"></div>
</div>
<nav class="nav" id="tabs"><button data-page="compare" class="active">FAR / FDR / AUROC</button><button data-page="roc">Matched FAR &amp; ROC</button><button data-page="failure">Per-failure results</button><button data-page="latency">Latency &amp; coverage</button><details><summary>Diagnostics &amp; source records</summary><div class="nav"><button data-page="distribution">Score diagnostics</button><button data-page="episodes">Episode audit</button><button data-page="provenance">Runs &amp; ODE settings</button></div></details></nav>
<section class="page" id="page-vlm"><div class="panel"><h2>VLM full-test streaming benchmark</h2>
<p id="vlmNote"></p><p id="vlmStatus"></p><div id="vlmLinks"></div>
<div class="inline"><label>Task<select id="vlmTask"><option value="all">All tasks + pooled overall</option></select></label><label>View<select id="vlmView"><option value="all">Both separate runs</option><option value="fix">Fixed</option><option value="wrist">Wrist</option></select></label><button data-export="vlm">Export rows</button></div></div>
<div class="panel"><h2>Alarms and detection</h2><div id="vlmTable" class="scroll"></div><details><summary>Recorded any-time / on-time decisions · updated onset labels</summary><div id="vlmOriginalTable" class="scroll"></div></details></div>
<div class="panel"><h2>Conditional delay and completion</h2><p class="note">Delays use on-time alarms supported by post-onset observations. These are different detection sets from other monitors. SD follows the source report’s population convention.</p><div id="vlmLatencyTable" class="scroll"></div></div></section>
<section class="page" id="page-studies">
<div class="panel"><h2 id="studyHeading">Additional experiments</h2><p>Each study retains its original episode selection and protocol. Use the task and view selectors below to explore its plots.</p>
<nav id="studyShortcuts" class="nav study-shortcuts" aria-label="Calibration sensitivity studies" hidden>
<a href="#study/main_calibration_sensitivity_500" data-comparison="study/main_calibration_sensitivity_500">500 splits · Main WM-NC</a>
<a href="#study/calibration_sensitivity_200" data-comparison="study/calibration_sensitivity_200">200 splits · Multi-method comparison</a>
</nav>
<div class="inline"><label>Experiment<select id="studySelect" style="max-width:520px"></select></label><label>Task<select id="studyTask"></select></label><label>View<select id="studyView"></select></label></div>
<h3 id="studyTitle"></h3><p id="studyScope"></p><div id="studyHighlights"></div><p class="note" id="studyPlotNote"></p><div id="studyDownloads"></div>
<details><summary>Study definitions and limits</summary><p id="studyNote"></p></details></div>
<div class="panel" id="studyFigures"></div><details class="panel"><summary>Study values and CSV export</summary><button data-export="study">Export selected rows</button><div id="studyTable" class="scroll"></div></details></section>
<section class="page active" id="page-compare">
<div class="panel" id="questionLinks" hidden></div>
<div class="panel" id="fusionLevelPanel" hidden><h2>Episode versus endpoint alarm fusion</h2>
<p class="note" id="fusionLevelNote"></p><p class="note">Episode AND is agreement within the post-onset episode interval, not simultaneous confirmation. This comparison has no new online latency measurement.</p>
<div class="toolbar"><div class="inline"><label>Fusion level<select id="fusionLevel"><option value="both">Both levels</option><option value="episode">Episode alarms</option><option value="endpoint">Same endpoint</option></select></label><label>Metric<select id="fusionMetric"><option value="FDR">FDR</option><option value="FAR">FAR</option><option value="AUROC">Episode AUROC</option></select></label></div><div class="inline"><button data-svg="fusionLevelChart">Download SVG</button><button data-export="fusion_levels">Export metrics</button><button data-export="fusion_scores">Export episode scores</button></div></div>
<div class="figure" id="fusionLevelChart"></div><p id="fusionLevelDownloads"></p><details><summary>Exact counts and results by task</summary><div class="scroll" id="fusionLevelTable"></div></details>
<p class="note">Other tabs retain the original endpoint-level diagnostics. Both levels’ per-failure results and scores are available in the CSV downloads.</p></div>
<section id="hyperPanel" hidden><p class="note">WM-NC hyperparameter ablation. Shared K4/L4 reference; identical task, failure-origin, view, calibration and FDR filters. No hyperparameter is selected from these test curves.</p><div class="panel"><div class="toolbar"><h2 id="hyperKTitle"></h2><div class="inline"><button data-svg="hyperKChart">Download SVG</button><button data-export="hyperK">Export plotted values</button></div></div><div class="figure" id="hyperKChart"></div></div><div class="panel"><div class="toolbar"><h2 id="hyperLTitle"></h2><div class="inline"><button data-svg="hyperLChart">Download SVG</button><button data-export="hyperL">Export plotted values</button></div></div><div class="figure" id="hyperLChart"></div></div></section>
<div class="panel" id="ablationPanel" hidden><div class="toolbar"><h2 id="ablationTitle">Horizon ablation</h2><div class="inline"><button data-svg="ablationChart">Download SVG</button><button data-export="ablation">Export plotted values</button></div></div>
<p class="note">FDR above, FAR below. Fixed and wrist views keep the same colors in both panels; solid lines are WM-NC and dashed lines are WM-PE. Points show measured episode proportions. FAR has its own labeled scale. Hover over a point for its numerator and denominator.</p>
<div id="ablationChart" class="figure"></div></div>
<div class="panel" id="singleMetricPanel"><div class="toolbar"><h2>Detection performance</h2><div class="inline"><label>Chart metric<select id="metric"><option value="FDR">FDR ↑</option><option value="FAR">FAR ↓</option><option value="AUROC">Episode AUROC ↑</option></select></label><button data-svg="comparisonChart">Download SVG</button></div></div><div id="comparisonChart" class="chart"></div></div>
<details class="panel" id="actionAvailability" hidden><summary>Action checkpoint availability · K4 / L4</summary><p class="note">NoAct is compared with the main model per failure mode (R3-3). Availability is shown explicitly; unscored checkpoints are not zero-performing detectors.</p><div id="actionInventory" class="scroll"></div></details>
<details class="panel" id="compactMetricsPanel"><summary>Exact counts, rates and annotation coverage</summary><p class="note">FDR follows the global definition above; early alarms are reported separately. Qwen includes late decisions. Whole-episode AUROC is unchanged and averages tasks equally. Matched FAR compares discrimination at a common false-alarm budget.</p><div id="compactMetrics" class="scroll"></div></details>

<details class="panel" id="metricsDetailPanel"><summary>Detailed metrics by task and detector</summary><div class="details-body"><div class="toolbar"><p class="note">Counts, episode AUROC and unsupported episodes.</p><button data-export="metrics">Export CSV</button></div><div id="metricsTable" class="scroll"></div></div></details></section>
<section class="page" id="page-roc">
<div class="warning">Empirical test-set analysis: thresholds are selected from successful TEST episodes, separately per task and detector. These results describe discrimination at a FAR budget; they are not a deployment calibration or held-out FAR guarantee. Existing latency and alarm tables retain their original calibrated thresholds.</div>
<div class="toolbar"><h2>Matched-FAR comparison</h2><div class="inline"><label>FAR budget<select id="rocFar"><option value="0">0%</option><option value="0.05" selected>5%</option><option value="0.1">10%</option><option value="0.2">20%</option></select></label><button data-svg="matchedChart">Download SVG</button><button data-export="matched_far">Export metrics & thresholds</button></div></div>
<section id="matchedHyperPanel" hidden><p class="note">WM-NC hyperparameter ablation. Shared K4/L4 reference; identical task, failure-origin, view, calibration and FDR filters. No hyperparameter is selected from these test curves.</p><div class="panel"><div class="toolbar"><h2 id="matchedHyperKTitle"></h2><div class="inline"><button data-svg="matchedHyperKChart">Download SVG</button><button data-export="matchedHyperK">Export plotted values</button></div></div><div class="figure" id="matchedHyperKChart"></div></div><div class="panel"><div class="toolbar"><h2 id="matchedHyperLTitle"></h2><div class="inline"><button data-svg="matchedHyperLChart">Download SVG</button><button data-export="matchedHyperL">Export plotted values</button></div></div><div class="figure" id="matchedHyperLChart"></div></div></section><div class="panel" id="matchedSinglePanel"><h2 id="matchedTitle">FDR at a common FAR budget</h2>
<p class="note">Plots use achievable deterministic thresholds with FAR ≤ the budget, and pool episode counts across tasks. With 20 test successes/task, FAR moves in 5% steps. Tied scores stay together, so achieved FAR can be lower. The optional interpolated FDR in the table is a descriptive ROC value, not a measured operating point. Unsupported episodes remain tied at −∞ and are counted below.</p><div id="matchedChart" class="figure"></div></div>
<details class="panel"><summary>Full ROC curves and definitions</summary><div class="toolbar"><h2>Episode-level ROC</h2><div class="inline"><label>FAR axis<select id="rocRange"><option value="0.2">0–20%</option><option value="1">0–100%</option></select></label><button data-svg="rocChart">Download SVG</button><button data-export="roc_points">Export ROC points</button></div></div>
<p class="note">Whole-episode discrimination diagnostic (not onset-restricted FDR). One peak standardized margin per episode, with failures positive. The matched-FAR table uses post-onset failure maxima. All-task curves average per-task ROC curves equally; raw scores are never pooled across tasks. AUROC is over the full 0–100% FAR range, independent of the selected FAR budget. Wall-bin and fusion scores retain their original definitions. Functional-band ROC uses max((raw score − expert mean)/g), sweeping h with the template frozen.</p><div id="rocChart" class="figure"></div><p class="note" id="rocDownloads"></p></details>
<details class="panel"><summary>Matched-FAR metrics, thresholds and interpolation by task</summary><div class="details-body"><div id="matchedTable" class="scroll"></div></div></details></section>
<section class="page" id="page-failure"><section class="panel" id="policySuccessPanel" hidden><h2>Natural policy success rates</h2><p class="note">User-reported ACT task-completion counts during natural execution, without deliberately induced failures. Separate from detector FAR/FDR and the curated PRL split; unaffected by the detector filters below. <a href="policy_success_rates.yaml">Source YAML</a></p><div id="policySuccessTable" class="scroll"></div></section>
<div class="panel" id="categoryPanel"><div class="toolbar"><h2>Failure-category FDR</h2><button data-export="categories">Export counts / FDR</button></div><p class="note">Pool detected / total episodes within each category, separately for each detector/view and threshold protocol. Both lateral-grasp-drift modes are sensing/actuation anomalies. Origin is a separate filter. Unknown outcomes retain their denominators with bounds. Continuous methods use dense offline scores; Qwen uses recorded streaming decisions, including late detections. <a href="failure_categories/report.md">Full report</a> · <a href="failure_categories/metrics.csv">All variants CSV</a> · <a href="failure_categories/mapping.yaml">Mapping</a></p><div id="categoryCharts"></div><details><summary>Exact category counts and rates</summary><div id="categoryTable" class="scroll"></div></details></div>
<div class="panel"><div class="toolbar"><h2>Paper-style failure table</h2><button data-export="failures">Export CSV</button></div><p class="note">No failure: false alarms / successful test episodes. Failure rows: detected / total. A dash means the detector was not scored on that task.</p><div id="failureTable" class="scroll"></div></div></section>
<section class="page" id="page-distribution"><div class="panel"><h2>Expert versus policy score densities</h2><p><a href="#study/expert_policy_score_distributions">Open window-level and episode-maximum probability-density plots</a> for expert validation versus successful policy rollouts, with task/view selection.</p></div><div class="panel"><h2>Episode-maximum score distributions</h2><div class="inline"><label>Detector<select id="distDetector"></select></label><label>Task<select id="distTask"></select></label><button data-svg="distributionChart">Download SVG</button></div><p class="note">ECDF of maximum signed calibrated margin per episode (alarm above zero). For functional bands, the axis is the maximum template residual R, and the deployed threshold is h. Success calibration, success test, and selected failure groups are separate. Non-finite scores are counted below the plot and excluded from its finite axis.</p><div id="distributionChart"></div></div><details class="panel"><summary>Dense score quantiles by failure type</summary><div class="toolbar"><h2>Score diagnostics</h2><div class="inline"><label>Score space<select id="scoreType"><option value="margin">Calibrated margin</option><option value="raw">Raw NC score (if available)</option></select></label><button data-export="distributions">Export CSV</button></div></div><p class="note">Each row is the saved frame-level distribution for one label group, not an average of episode quantiles. No cross-task pooling of raw scores. Unsupported scores are reported separately.</p><div id="distributionTable" class="scroll"></div></details></section>
<section class="page" id="page-latency">
<details class="panel"><summary>Recorded online coverage and run history</summary><div id="onlineRunStatus" role="status"></div></details>
<div class="panel" id="pairwiseLatencyPanel"><div class="toolbar"><h2>Main wrist WM-NC versus each method</h2><div class="inline">
<span class="tag">Reference: wrist WM-NC · K4 L4</span>
<label>Comparisons<select id="pairFamily"><option value="baseline">Main baselines</option><option value="all">All comparisons</option><option value="K">K ablation</option><option value="L">L ablation</option><option value="action">Action ablation</option><option value="supplementary">Supplementary detectors</option></select></label>
<label>Alarm timing<select id="pairTiming"><option value="on_time">On-time alarms only</option><option value="all">Include late alarms</option></select></label>
<label>Minimum shared n<input id="pairMinimum" type="number" min="1" step="1" value="1" style="width:85px;padding:8px"></label>
<label>Plot<select id="pairPlot"><option value="means">Both means ± SD</option><option value="difference">Paired difference ± SD</option></select></label>
<button data-svg="pairwiseChart">Download SVG</button><button data-export="pairwise">Export statistics</button><button data-export="pairwise_episodes">Export episode IDs &amp; delays</button>
</div></div>
<p class="note">Each row uses <b>all episodes detected by both methods after their respective onsets</b>, for the same annotator. Green is main wrist WM-NC, orange is the comparator; the wrist mean is recomputed for every row. Compare within rows: different rows can contain different failures. Thresholds retain their original operating points; FAR is not matched here. SD describes variability, not a confidence interval.</p>
<p class="note">Fixed, wrist and joint comparators are included. Each detector uses its own view-specific onset (joint outputs use the earlier camera onset); onset times need not match. Cross-view delay differences can therefore reflect different onset times as well as alarm times. Uses the task, failure origin/type and annotator filters above. Late alarms are included or excluded consistently for both methods. Early-only alarms and unmeasured trials are excluded; missing action-ablation timings stay unavailable.</p>
<p class="note" id="pairwiseStatus"></p><div id="pairwiseChart" class="figure"></div>
<details><summary>Counts, exclusions and paired differences</summary><div id="pairwiseTable" class="scroll"></div></details>
<p class="note"><a href="pairwise_latency/report.md">Fixed report</a> · <a href="pairwise_latency/summary.csv">CSV</a> · <a href="pairwise_latency/analysis.json">JSON</a> · <a href="#shared" data-comparison="shared">Compare a common multi-method group</a></p></div>
<div class="panel" id="sharedLatencyPanel"><div class="toolbar"><h2>Online latency on the same detected episodes</h2><div class="inline">
<label>Minimum shared episodes<input id="sharedMinimum" type="number" min="1" step="1" value="150" style="width:100px;padding:8px"></label>
<button id="shared150">150 episodes</button><button id="shared50">50 episodes</button>
<button id="sharedSearch">Find largest eligible groups</button><button id="sharedPool">Restore comparison pool</button>
</div></div>
<p class="note">Choose detectors above or apply a suggested group. Every method uses <b>all episodes in the same intersection</b>, with the same annotator and onset time. Early alarms, misses and missing measurements are excluded; different camera onset times do not pair. Minimum n is a floor, not a subsampling target. Thresholds remain at their original calibration; this does not impose matched FAR.</p>
<p id="sharedStatus" class="warning" role="status"></p><div id="sharedGroups" class="scroll"></div>
<p id="sharedComposition" class="note"></p>
<div class="toolbar"><h3>Paired latency distribution</h3><div class="inline"><label>Display<select id="latencyStyle"><option value="mean_sd">Mean ± standard deviation</option><option value="box">Box plot + mean</option></select></label><button data-svg="sharedChart">Download SVG</button><button data-export="shared_latency">Export statistics</button><button data-export="shared_episodes">Export shared episode IDs & delays</button><button data-export="shared_pairs">Export paired differences</button></div></div>
<p class="note" id="sharedPlotNote"></p><div id="sharedChart" class="figure"></div><div id="sharedTable" class="scroll"></div>
<details><summary>Paired delay differences (A − B; negative means A is earlier)</summary><div id="sharedPairs" class="scroll"></div></details>
<p class="note">Group search maximizes the number of detectors, then orders ties by shared count and detector ID. It never uses latency. Search considers the selected pool (up to 20 individually eligible detectors); manual intersections work for any selection. Different groups can contain different tasks and failures, so compare delays within each group. These conditional results supplement detection coverage/FDR. <a href="shared_latency/report.md">Fixed report</a> · <a href="shared_latency/analysis.json">JSON</a></p></div>
<details class="panel"><summary>Individual detection sets · unpaired latency (coverage context)</summary><div class="toolbar"><h2>Individual detection sets · unpaired latency</h2><div class="inline">
<label>Measurement<select id="latencySource"><option value="online">Online · end-to-end alarm</option><option value="offline">Offline · dense score alarm</option></select></label>

<button data-svg="latencyChart">Download SVG</button><button data-export="latency_plot">Export plotted values</button></div></div>
<p class="note" id="latencyPlotNote"></p><div id="latencyChart" class="figure"></div>
<p class="note">One nonnegative delay per detected episode: onset to the first completed alarm based on a post-onset observation. Sustained pre-onset alarms are a separate class; other early alarms and missing/censored observations do not receive latency credit. All-task plots pool individual detected episodes; they do not average task means. Methods can detect different subsets, so compare the displayed n alongside latency.</p></details>
<div class="warning">Online trials cover only failures detected by the offline all-endpoint protocol. Online alarm retention is conditional on this selection; it is not overall FDR or FAR. Unmeasured variants have no latency estimate.</div>
<details class="panel"><summary>Online completion, alarm retention and misses</summary><div class="details-body"><div class="toolbar"><h2>Online coverage</h2><button data-export="online">Export CSV</button></div><div id="onlineTable" class="scroll"></div></div></details>
<details class="panel"><summary>Offline latency by task, including annotation coverage</summary><div class="details-body"><div class="toolbar"><h2>Offline onset-to-alarm delay</h2><button data-export="latency">Export CSV</button></div><p class="note">Dense offline delays use the selected all-endpoint/correction protocol and exclude inference runtime. Recorded online timings retain their original thresholds. Annotators are never pooled.</p><div id="latencyTable" class="scroll"></div></div></details>
<details class="panel"><summary>Online computation, skipped windows and ODE work</summary><div class="details-body"><div class="toolbar"><h2>Measured online computation</h2><button data-export="runtime">Export CSV</button></div><p class="note">One row per input stream; joint fixed/wrist computation is counted once. Per-window quantiles are recomputed from all selected measured events. Includes visual encoding and scoring; preloading, model loading and warm-up are excluded. Skipped counts cover eligible windows through the stopping point.</p><div id="runtimeTable" class="scroll"></div></div></details></section>
<section class="page" id="page-episodes"><div class="panel"><div class="toolbar"><h2>Episode audit</h2><div class="inline"><label>Table page<select id="episodePage"></select></label><button data-export="episodes">Export all filtered rows</button></div></div><p class="note">Paper alarm and dense offline delay both use all valid endpoints and the frozen visual correction. Online trial links contain the full event trace. Missing online rows were either not selected or not run; check coverage.</p><div id="episodeTable" class="scroll"></div></div></section>
<section class="page" id="page-provenance"><div class="panel"><h2>Protocol & provenance</h2><div id="figureDownloads"></div><p class="note">Original reports, label snapshots, thresholds and trial traces are archived under <code>fomofd/tasks/benchmark_results</code>. Raw score NPZs remain under each task’s <code>prl_data/score_result</code>. <a href="score_cache_inventory.json">Verified cache inventory</a> · <a href="catalog_snapshot.json">Comparison catalog</a>. Original files retain historical absolute paths. Archived endpoint fusion uses same-frame min/max standardized margins. The supplemental episode-level comparison combines component maxima on identical support; episode AND does not require simultaneous detections. OR has no unchanged-alpha familywise guarantee. Wall-time-bin calibration is diagnostic, without simultaneous episode coverage. Functional-band calibration uses one maximum signed template residual per trajectory and a finite-sample corrected multiplier. Window 1 disables smoothing.</p><div id="runTable" class="scroll"></div><details><summary>Exact recorded protocols and source hashes</summary><pre id="protocols"></pre></details></div><div class="panel"><div class="toolbar"><h2>Scoring settings, including ODE work</h2><button data-export="settings">Export CSV</button></div><div id="settingsTable" class="scroll"></div></div><div class="panel"><h2>All archived reports and online traces</h2><p class="note">Only curated comparisons enter the metrics. Intermediate reports are preserved without being counted again. Historical and intermediate comparisons remain available here.</p><div id="archiveTable" class="scroll"></div></div></section>
<p class="note" id="status" role="status">Loading saved results…</p></main>
<script id="result-data" type="application/octet-stream">__RESULT_DATA__</script>
<script>window.addEventListener('error',e=>{document.body.dataset.error=e.message;document.getElementById('status').textContent='Viewer error: '+e.message});</script>
<script>
(async()=>{
'use strict';
const compressed=Uint8Array.from(atob(document.getElementById('result-data').textContent),c=>c.charCodeAt(0));
const D=JSON.parse(await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
const unpack=p=>p.rows.map(a=>Object.fromEntries(p.columns.map((k,i)=>[k,a[i]])));
D.episodes=unpack(D.episodes);
const VLM_EPISODES=(D.vlm_runs||[]).flatMap(v=>v.episodes||[]);
const comparisonEpisodes=()=>D.episodes.concat(VLM_EPISODES);
const postMode=()=>$('detectionRule').value==='post';
function assessmentRow(r){
 const a=r.onset_assessments?.[$('annotator').value];
 return {...r,any_time_alarm:r.any_time_alarm??r.alarm,alarm:postMode()&&r.outcome==='failure'?(a?.post_alarm??null):r.alarm,
  post_score_max:a?.post_score_max??null,early_score_max:a?.early_score_max??null,early_alarm:a?.early_alarm??null,sustained_pre_onset_alarm:a?.sustained_pre_onset_alarm??null,any_pre_onset_alarm:a?.any_pre_onset_alarm??null,onset_status:a?.status,
  detection_rule:$('detectionRule').value,annotator_id:$('annotator').value};
}
const unknownCount=a=>a.filter(r=>r.outcome==='failure'&&r.alarm===null).length;
const earlyCount=a=>a.filter(r=>r.outcome==='failure'&&r.early_alarm===true).length;
const coverageText=r=>r.unknown?`${r.unknown} unknown; FDR ${pct(r.tp/r.pos)}–${pct((r.tp+r.unknown)/r.pos)}`:'Complete';

D.latencies=unpack(D.latencies);D.distributions=unpack(D.distributions);
// time_varying is a legacy storage alias only; the UI explicitly labels it as diagnostic bins.
for(const rows of [D.episodes,D.latencies,D.distributions])for(const r of rows)if(r.calibration==='wall_time_bin')r.calibration='time_varying';
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
if(D.training_setup?.tasks?.length){
 const s=D.training_setup;
 $('trainingSetup').hidden=false;
 const rows=s.tasks.map(r=>[r.label,r.expert_episodes,`${r.act_train_episodes} / ${r.act_val_episodes}`,`${r.visual_vae_train_episodes} / ${r.visual_vae_val_episodes}`,`${r.wm_train_episodes} / ${r.wm_val_episodes}`]);
 $('trainingSetupItems').innerHTML=`<p><strong>One person collected all four expert datasets.</strong> Author-confirmed; verified ${esc(s.verified_date)}. Counts are episodes; visual-VAE counts apply to each camera.</p><div class="scroll"><table><thead><tr>${['Task','Expert episodes','ACT train / val','Visual VAE train / val','Main WM train / val'].map(v=>`<th>${esc(v)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(v=>`<td>${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p>${esc(s.act.summary)}</p><p>${esc(s.visual_vae.summary)}</p><p class="note">${esc(s.interpretation)}</p>`;
}
if(D.historical_training_corrections?.length){$('trainingCorrections').hidden=false;$('trainingCorrectionItems').innerHTML=D.historical_training_corrections.map(r=>`<p>${esc(r.summary)}</p>`).join('');}
const TASKS={needle_pickup:'Needle pickup · Sim.',ring_over_post:'Ring over post · Sim.',tissue_retraction:'Tissue retraction · Real',shunt_insertion:'Shunt insertion · Real'};
const METHODS={wm_nc:'WM-NC',wm_pe:'WM-PE',wm_pu:'WM-PU',logpzo:'logpZO · VAE',rnd:'RND · VAE',logpzo_native:'logpZO · Policy',rnd_native:'RND · Policy',ace:'ACE',act_ensemble_pos:'AEU',act_ensemble_rot:'AEU-Rot',qwen_v9:'Qwen v9 · onset-aware',wm_nc_and:'WM-NC · Fixed AND wrist',wm_nc_or:'WM-NC · Fixed OR wrist'};
const VIEWS={fix:'Fixed',wrist:'Wrist',both:'Combined'},runs=new Map(D.runs.map(r=>[r.id,r]));
for(const v of D.vlm_runs||[])runs.set('vlm_'+v.id,{id:'vlm_'+v.id,title:'Qwen v9',family:'vlm'});
function updateOfflineProtocol(){
 if(!D.offline_protocol)return;
 $('offlineProtocolPanel').hidden=false;
 $('offlineProtocolNote').textContent=D.offline_protocol.note;
 $('operatingPointNote').textContent=D.offline_protocol.online_note+' Online latency plots below are recorded measurements; the correction does not rerun them.';
 const links=D.report_links;
 $('offlineProtocolDownloads').innerHTML=`<a href="${links.markdown}">Selected Markdown</a> · <a href="${links.yaml}">Selected YAML</a> · <a href="${links.evidence}">Selected reviewer evidence</a> · <a href="${D.offline_protocol.legacy_report}">Archived legacy summary</a>`;
 for(const a of document.querySelectorAll('a[data-report-link]'))a.href=links[a.dataset.reportLink]+(a.dataset.reportAnchor||'');
 document.body.dataset.offlineProtocol=D.offline_protocol.id;
}
function selectOfflineProtocol(value){
 const p=D.offline_options?.[value];if(!p)return;
 Object.assign(D,p);
 for(const name of ['episodes','latencies','distributions']){
  D[name]=unpack(p[name]);for(const r of D[name])if(r.calibration==='wall_time_bin')r.calibration='time_varying';
 }
 runs.clear();for(const r of D.runs)runs.set(r.id,r);
 for(const v of D.vlm_runs||[])runs.set('vlm_'+v.id,{id:'vlm_'+v.id,title:'Qwen v9',family:'vlm'});
 updateOfflineProtocol();exports={};sharedSearchResult=null;refresh();
}
const number=v=>v===null||v===undefined||v===''?null:v==='-inf'?-Infinity:v==='inf'?Infinity:Number(v);
const fmt=(v,n=3)=>v===null||v===undefined||Number.isNaN(v)?'—':!Number.isFinite(Number(v))?String(v):Number(v).toFixed(n);
const pct=v=>v===null?'—':(100*v).toFixed(1)+'%';
const key=r=>[r.variant,r.method,r.view].join('|'), group=(a,f)=>{const m=new Map();for(const r of a){const k=f(r);if(!m.has(k))m.set(k,[]);m.get(k).push(r)}return m};
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
const quantile=(a,q)=>{a=a.filter(x=>x!==null&&Number.isFinite(x)).sort((a,b)=>a-b);if(!a.length)return null;const x=(a.length-1)*q,i=Math.floor(x);return a[i]+(a[Math.ceil(x)]-a[i])*(x-i)};
const methodName=m=>METHODS[m]||m.replace('wm_nc_','WM-NC + ').replace('_native',' policy').replace(/_/g,' ').replace(/\b(and|or)\b/g,s=>s.toUpperCase());
const detectorName=r=>{
 const current=r.variant===(D.summary_sources?.[r.method]||'main');
 const joint=r.view==='both'&&['ace','act_ensemble_pos','act_ensemble_rot'].includes(r.method);
 return methodName(r.method)+(joint?'':' · '+VIEWS[r.view])+(current||r.score_kind==='binary_decision'||r.method==='qwen_v9'?'':' · '+(runs.get(r.variant)?.title||r.variant));
};
const taskName=t=>TASKS[t]||t;
const palette=['#176995','#db8134','#158477','#a65891','#6375ba','#916a32','#b74e52','#44894b'];
let selected=new Set(), activePage='compare', exports={}, visibleEpisodes=[], revision=0;
function option(select,value,text){const o=document.createElement('option');o.value=value;o.textContent=text;select.appendChild(o)}
function fill(select,items){const old=select.value;select.replaceChildren();for(const [v,t] of items)option(select,v,t);if(items.some(x=>x[0]===old))select.value=old}
function eligibleComparison(r,p){
 if(p==='pairwise')return !!D.pairwise_latency?.detectors[key(r)];
 if(['studies','vlm'].includes(p))p='main';
 if(['main','shared'].includes(p)&&(D.supplementary_methods||[]).includes(r.method))return false;
 if(p==='hyper')return r.method==='wm_nc'&&(r.variant==='main'||['K','L'].includes(runs.get(r.variant)?.family));
 if(p==='action')return r.method==='wm_nc'&&(r.variant==='main'||runs.get(r.variant)?.family==='action');
 if(p==='main'&&r.method==='wm_nc'&&runs.get(r.variant)?.family==='action'&&runs.get(r.variant)?.action_mode==='none')return true;
 if(r.score_kind==='binary_decision')return ['main','all'].includes(p);
 if(p==='shared')return (r.method in METHODS&&!['wm_nc_and','wm_nc_or'].includes(r.method)&&r.variant===(D.summary_sources?.[r.method]||'main'))||(['wm_nc','wm_pe','wm_pu'].includes(r.method)&&runs.get(r.variant)?.family==='view_fusion');
 if(['data_size','seed'].includes(p))return ['wm_nc','wm_pe'].includes(r.method)&&(r.variant==='main'||runs.get(r.variant)?.family===p||(p==='seed'&&runs.get(r.variant)?.family==='seed_repeat'));
 if(p==='fusion')return r.variant==='main'&&['wm_nc','logpzo_native','wm_nc_and','wm_nc_or','wm_nc_logpzo_native_and','wm_nc_logpzo_native_or'].includes(r.method);
 if(p==='fused')return ['wm_nc','wm_pe','wm_pu'].includes(r.method)&&(r.variant==='main'||runs.get(r.variant)?.family==='view_fusion');
 return p==='all'||(p==='main'?r.variant===(D.summary_sources?.[r.method]||'main')&&r.method in METHODS&&!['wm_nc_and','wm_nc_or'].includes(r.method):p==='K'?r.method==='wm_nc'&&(r.variant==='main'||runs.get(r.variant)?.family==='K'):['wm_nc','wm_pe'].includes(r.method)&&(r.variant==='main'||runs.get(r.variant)?.family==='L'))
}
function eligiblePreset(r){return eligibleComparison(r,$('preset').value)}
function updateComparisonNavigation(){
 const preset=$('preset').value;
 const study=preset==='studies'?(D.studies||[]).find(s=>s.id===$('studySelect').value):null;
 const route=study?'study/'+study.id:(activePage==='roc'||activePage==='failure'?preset+'/'+activePage:preset);
 for(const a of document.querySelectorAll('[data-comparison]')){const active=a.dataset.studyKind?study?.kind===a.dataset.studyKind:a.dataset.comparison===route;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')}
 for(const axis of ['K','L']){
  const other=axis==='K'?'L':'K',settings=D.settings.filter(r=>eligibleComparison(r,axis));
  const values=key=>[...new Set(settings.map(r=>r[key]).filter(v=>v!==null&&v!==undefined&&Number.isFinite(Number(v))))].sort((a,b)=>a-b).join(', ');
  document.querySelector(`[data-comparison-summary="${axis}"]`).textContent=`${axis} = ${values(axis) || 'not scored'} · ${other} = ${values(other) || '—'}`;
 }
 const seeds=[0,...D.runs.filter(r=>r.family==='seed').map(r=>Number(r.id.match(/seed(\d+)/)?.[1])).filter(Number.isFinite)];
 const seedText=[...new Set(seeds)].sort((a,b)=>a-b).join(', ');
 document.querySelector('[data-comparison-summary="seed"]').textContent='K4 L4 · seeds '+seedText;
 $('comparisonContext').textContent=preset==='K'?'K ablation · endpoint horizon varies; history length L is held fixed. WM-NC FDR/FAR trends include the shared main-model reference.':preset==='L'?'L ablation · history length varies; endpoint horizon K is held fixed. WM-NC and WM-PE FDR/FAR trends include the shared main-model reference.':preset==='main'?'Main methods · compare the detectors, then use ROC & matched FAR, failure breakdowns, or latency below.':'All scored comparisons · includes main methods, horizon ablations, and detector fusions.';
 if(preset==='hyper')$('comparisonContext').textContent='Hyperparameter ablation: K varies with L=4; L varies with K=4. Both plots reuse the same K4/L4 reference and filters. These are test-set sensitivity results, not validation-based hyperparameter selection.';
 if(preset==='action')$('comparisonContext').textContent='Action conditioning at K4/L4, seed 0, full expert data. Compare both pathways with NoAct, AdaLN only and cross-attention only. Use Per-failure results for execution mismatch and each other failure mode; trained but unscored variants stay pending.';
 if(preset==='fusion')$('comparisonContext').textContent='Existing separate-branch WM-NC: compare endpoint fusion with episode-level OR/AND on identical eligible endpoints. Episode AND does not require simultaneous detections. Overview compares both levels; other tabs retain endpoint diagnostics. OR can increase FAR. Functional-band fusion is not defined.';
 if(preset==='fused')$('comparisonContext').textContent='Fused-view architecture: one WM over concatenated fixed/wrist latent channels, one joint score and one calibrated threshold. NC/PE sum coordinate scores; PU averages coordinate standard deviations. Main fixed/wrist detectors are references. One RK4 step and 16 samples for WM-PU.';
 if(preset==='main')$('comparisonContext').textContent='Main detectors and policy-native baselines. AEU = Action Ensemble Uncertainty: positional disagreement change among three ACT policies, adapted from Thompson et al. AEU-Rot is retained under All comparisons. ACE and AEU each use both cameras. ACE adapts FIPER; WM-PU is Ward-inspired. Qwen uses post-onset evidence; censored outcomes and late decisions are reported separately.';
 if(preset==='data_size')$('comparisonContext').textContent='Training data-size sensitivity: K4/L4, seed 0. Simulation: 75/150/225 training episodes; real tasks: 30/60/90. Full-data reference is shared. Choose a task to inspect task-level differences.';
 if(preset==='seed')$('comparisonContext').textContent='Training-seed sensitivity: full expert data, K4/L4, seeds '+seedText+'. Seed 0 is the shared main reference. The episode split and calibration protocol are unchanged.';
 if(preset==='seed'&&D.training_runs_report)$('comparisonContext').innerHTML+=' '+link(D.training_runs_report,'Open training curves and checkpoint locations');
 if(preset==='shared')$('comparisonContext').textContent='Shared online latency · current main detectors and joint-view models. Choose a minimum overlap, then compare the same episodes for every selected detector. Original FAR/FDR and thresholds remain visible in the other tabs.';
 if(preset==='pairwise')$('comparisonContext').textContent='Pairwise online latency · main wrist WM-NC is the sole reference, including fixed WM-NC as a comparator. Each pair uses shared detections with each detector’s own onset. Ablation traces are historical and excluded from current latency comparisons.';
 if(preset==='studies')$('comparisonContext').textContent='Additional studies · use the experiment, task and view selectors below. These descriptive studies supplement the primary benchmark comparisons.';
 if(study?.kind==='calibration_sensitivity')$('comparisonContext').textContent='Calibration split sensitivity · 19 calibration / 20 test successes per split, with the failure set held fixed. Compare FAR/FDR variability against the original split. Resampling describes sensitivity, not independent experiments or confidence intervals.';
 if(preset==='vlm')$('comparisonContext').textContent='Qwen v9 · all test episodes replayed; separate fixed/wrist cameras. This panel uses its own task/view filters and does not apply continuous-score calibration or ROC controls.';
}
function showPage(page){activePage=page;$('benchmarkControls').hidden=['studies','vlm'].includes(page);for(const b of $('tabs').querySelectorAll('[data-page]'))b.classList.toggle('active',b.dataset.page===page);for(const p of document.querySelectorAll('.page'))p.classList.toggle('active',p.id==='page-'+page);$('annotationControls').hidden=false;
 const paired=page==='latency'&&$('preset').value==='pairwise';
 for(const id of ['view','scheme','detectionRule'])$(id).parentElement.hidden=paired;
 for(const id of ['cards','functionalStatus','detectionNote'])$(id).hidden=paired;
 $('methodSummary').closest('details').hidden=paired;
}
function chooseComparison(value,updateURL=true){
 const route=value,studyId=value.startsWith('study/')?value.slice(6):null;
 let page=null;
 if(studyId){fill($('studySelect'),(D.studies||[]).map(s=>[s.id,s.title]));if((D.studies||[]).some(s=>s.id===studyId))$('studySelect').value=studyId;value='studies';}
 else if(value.includes('/')){[value,page]=value.split('/');if(!['compare','roc','failure','latency','distribution','episodes','provenance'].includes(page))return;}
 if(![...$('preset').options].some(o=>o.value===value))return;
 $('preset').value=value;
 if(value==='fused')$('view').value='all';
 if(['pairwise','fusion'].includes(value))$('scheme').value='episode';
 if(['K','L','hyper','action'].includes(value)&&$('view').value==='both')$('view').value='all';
 setMethods();showPage(['studies','vlm'].includes(value)?value:page||(['shared','pairwise'].includes(value)?'latency':'compare'));
 sharedSearchResult=null;
 if(updateURL)history.pushState(null,'','#'+route);
 refresh();
 if(value==='shared')findSharedGroups();
}
function setMethods(){sharedSearchResult=null;const entries=[...new Map(comparisonEpisodes().filter(eligiblePreset).map(r=>[key(r),r])).entries()].sort((a,b)=>detectorName(a[1]).localeCompare(detectorName(b[1])));selected=new Set(entries.map(e=>e[0]));$('methods').innerHTML=entries.map(([k,r])=>`<label><input type="checkbox" value="${esc(k)}" checked>${esc(detectorName(r))}</label>`).join('');$('methodSummary').textContent=`Choose detectors (${entries.length} available)`}
function base(r){return selected.has(key(r))&&($('task').value==='all'||r.task===$('task').value)&&($('view').value==='all'||r.view===$('view').value)&&(!r.calibration||r.calibration===$('scheme').value||(r.calibration==='binary_anytime'&&$('scheme').value==='episode'))}
function failureMatch(r){return ($('origin').value==='all'||r.failure_origin===$('origin').value)&&($('failure').value==='all'||r.failure_mode===$('failure').value)}
function outcomeMatch(r){return r.outcome==='success'||failureMatch(r)}
function annotationMatch(r){return r.annotator_id===$('annotator').value}
function auc(rows){const pos=rows.filter(r=>r.outcome==='failure').map(r=>number(r.score_max)),neg=rows.filter(r=>r.outcome==='success').map(r=>number(r.score_max));if(!pos.length||!neg.length||[...pos,...neg].some(x=>x===null||Number.isNaN(x)))return null;let n=0;for(const p of pos)for(const q of neg)n+=p>q?1:p===q?.5:0;return n/(pos.length*neg.length)}
function metrics(rows){const result=[];for(const items of group(rows,r=>r.task+'|'+key(r)).values()){const first=items[0],neg=items.filter(r=>r.outcome==='success'),pos=items.filter(r=>r.outcome==='failure'),fa=neg.filter(r=>r.alarm).length,tp=pos.filter(r=>r.alarm).length;result.push({task:first.task,variant:first.variant,method:first.method,view:first.view,calibration:first.calibration,detection_rule:$('detectionRule').value,annotator_id:$('annotator').value,score_kind:first.score_kind,neg:neg.length,pos:pos.length,fa,tp,late_alarms:first.score_kind==='binary_decision'?items.filter(r=>r.late_alarm).length:null,FAR:neg.length?fa/neg.length:null,unknown:unknownCount(items),early:earlyCount(items),sustained:items.filter(r=>r.sustained_pre_onset_alarm===true).length,FDR:pos.length&&!unknownCount(items)?tp/pos.length:null,AUROC:auc(items),auc_tasks:auc(items)!==null?1:0,unsupported:items.filter(r=>r.n_supported_scores===0).length})}for(const items of group(result,key).values()){const first=items[0],sum=f=>items.reduce((s,r)=>s+r[f],0),neg=sum('neg'),pos=sum('pos'),fa=sum('fa'),tp=sum('tp'),aucs=items.map(r=>r.AUROC).filter(x=>x!==null);result.push({...first,task:'Overall',neg,pos,fa,tp,late_alarms:first.score_kind==='binary_decision'?sum('late_alarms'):null,FAR:neg?fa/neg:null,unknown:sum('unknown'),early:sum('early'),sustained:sum('sustained'),FDR:pos&&!sum('unknown')?tp/pos:null,AUROC:mean(aucs),auc_tasks:aucs.length,unsupported:sum('unsupported')})}return result}
function rocCurve(rows){
 const pos=rows.filter(r=>r.outcome==='failure').length,neg=rows.length-pos;
 if(!pos||!neg)return [];
 let fp=0,tp=0;const points=[{threshold:null,comparison:'never',FAR:0,FDR:0,fp,tp}];
 const buckets=[...group(rows,r=>number(r.score_max))].sort((a,b)=>a[0]===b[0]?0:a[0]>b[0]?-1:1);
 for(const [threshold,a]of buckets){fp+=a.filter(r=>r.outcome==='success').length;tp+=a.filter(r=>r.outcome==='failure').length;points.push({threshold,comparison:'>=',FAR:fp/neg,FDR:tp/pos,fp,tp})}
 return points;
}
function rocAt(points,target,side='right'){
 if(!points.length)return null;
 let index=0;while(index+1<points.length&&points[index+1].FAR<=target)index++;
 if(points[index].FAR===target){if(side==='left')while(index>0&&points[index-1].FAR===target)index--;return points[index].FDR}
 const a=points[index],b=points[index+1];return a.FDR+(b.FDR-a.FDR)*(target-a.FAR)/(b.FAR-a.FAR);
}
function matchedMetrics(rows,target){
 const result=[];
 for(const a of group(rows,r=>r.task+'|'+key(r)).values()){
  const neg=a.filter(r=>r.outcome==='success').map(r=>number(r.score_max)).sort((a,b)=>a===b?0:a>b?-1:1),pos=a.filter(r=>r.outcome==='failure').map(r=>postMode()?(r.post_score_max===null?null:number(r.post_score_max)):number(r.score_max)),allowed=Math.floor(target*neg.length+1e-10),threshold=neg.length?(allowed<neg.length?neg[allowed]:-Infinity):null;
  const alarm=v=>neg.length>0&&(allowed===neg.length||v>threshold),fa=neg.filter(alarm).length,tp=pos.filter(v=>v!==null&&alarm(v)).length,first=a[0];
  result.push({task:first.task,variant:first.variant,method:first.method,view:first.view,calibration:first.calibration,detection_rule:$('detectionRule').value,annotator_id:$('annotator').value,target_FAR:target,threshold,comparison:!neg.length?'unavailable':allowed===neg.length?'always':'>',neg:neg.length,pos:pos.length,fa,tp,late_alarms:first.score_kind==='binary_decision'?a.filter(r=>r.late_alarm).length:null,FAR:neg.length?fa/neg.length:null,unknown:pos.filter(v=>v===null).length,early:a.filter(r=>r.outcome==='failure'&&r.early_score_max!==null&&alarm(number(r.early_score_max))).length,FDR:neg.length&&pos.length&&!pos.includes(null)?tp/pos.length:null,AUROC:auc(a),auc_tasks:neg.length&&pos.length?1:0,unsupported:a.filter(r=>r.n_supported_scores===0).length,interpolated_FDR:postMode()?null:rocAt(rocCurve(a),target)});
 }
 for(const a of group(result,key).values()){
  const sum=k=>a.reduce((s,r)=>s+r[k],0),neg=sum('neg'),pos=sum('pos'),fa=sum('fa'),tp=sum('tp'),aucs=a.map(r=>r.AUROC).filter(x=>x!==null);
  result.push({...a[0],task:'Overall',neg,pos,fa,tp,threshold:null,comparison:'per-task thresholds',FAR:neg?fa/neg:null,unknown:sum('unknown'),early:sum('early'),sustained:sum('sustained'),FDR:pos&&!sum('unknown')?tp/pos:null,AUROC:mean(aucs),auc_tasks:aucs.length,unsupported:sum('unsupported'),interpolated_FDR:!postMode()&&pos?a.reduce((s,r)=>s+(r.interpolated_FDR??0)*r.pos,0)/pos:null});
 }
 return result;
}
function drawROC(rows){
 const xmax=Number($('rocRange').value),series=[];exports.roc_points=[];
 for(const a of group(rows,key).values()){
  const tasks=[...group(a,r=>r.task).values()].map(items=>({task:items[0].task,points:rocCurve(items),auc:auc(items)})).filter(t=>t.points.length),first=a[0];
  for(const t of tasks)exports.roc_points.push(...t.points.map(p=>({variant:first.variant,task:t.task,method:first.method,view:first.view,calibration:first.calibration,failure_origin_filter:$('origin').value,failure_mode_filter:$('failure').value,...p})));
  if(!tasks.length)continue;
  // Preserve the left and right values at every vertical ROC segment. Collapsing
  // repeated FAR values before interpolation would inflate the averaged AUC.
  const grid=[...new Set([0,xmax,...tasks.flatMap(t=>t.points.map(p=>p.FAR).filter(v=>v<=xmax))])].sort((a,b)=>a-b),points=grid.flatMap(FAR=>['left','right'].map(side=>({FAR,FDR:mean(tasks.map(t=>rocAt(t.points,FAR,side)))})));
  series.push({...first,points,AUROC:mean(tasks.map(t=>t.auc)),auc_tasks:tasks.length});
 }
 series.sort(detectorOrder);
 if(!series.length){$('rocChart').innerHTML='<p class="empty">ROC requires successful test episodes and selected failures.</p>';return}
 const h=445+Math.ceil(series.length/2)*24,x=v=>85+870*v/xmax,y=v=>355-285*v;
 let svg=`<svg viewBox="0 0 1060 ${h}" xmlns="http://www.w3.org/2000/svg"><rect width="1060" height="${h}" fill="white"/><text x="85" y="24" fill="#536e7e">${esc(plotContext())}</text><text x="85" y="48">Whole-episode true-positive rate (%) ↑</text>`;
 for(let i=0;i<=4;i++){const v=i/4;svg+=`<line x1="85" x2="955" y1="${y(v)}" y2="${y(v)}" stroke="#dde5eb"/><text x="70" y="${y(v)+4}" text-anchor="end">${pct(v)}</text><line x1="${x(v*xmax)}" x2="${x(v*xmax)}" y1="70" y2="355" stroke="#edf1f4"/><text x="${x(v*xmax)}" y="380" text-anchor="middle">${pct(v*xmax)}</text>`}
 svg+=`<path d="M${x(0)},${y(0)} L${x(xmax)},${y(xmax)}" stroke="#9aabb5" stroke-dasharray="3 4" fill="none"/><text x="520" y="407" text-anchor="middle">Episode FAR (%) →</text>`;
 const colors=new Map();
 series.forEach((s,i)=>{
  const k=s.variant+'|'+s.method;if(!colors.has(k))colors.set(k,palette[colors.size%palette.length]);const color=colors.get(k),dash=s.view==='wrist'?'7 4':s.view==='both'?'2 3':'',xx=85+(i%2)*475,yy=438+Math.floor(i/2)*24;
  svg+=`<path d="${s.points.map((p,j)=>(j?'L':'M')+x(p.FAR)+','+y(p.FDR)).join(' ')}" stroke="${color}" stroke-dasharray="${dash}" stroke-width="2.2" fill="none"><title>${esc(detectorName(s))} · AUROC ${fmt(s.AUROC)} · ${s.auc_tasks} task(s)</title></path><line x1="${xx}" x2="${xx+26}" y1="${yy-4}" y2="${yy-4}" stroke="${color}" stroke-dasharray="${dash}" stroke-width="2.5"/><text x="${xx+34}" y="${yy}" font-size="11">${esc(detectorName(s))} · AUC ${fmt(s.AUROC)}</text>`;
 });
 $('rocChart').innerHTML=svg+'</svg>';
}
function renderROC(rows){
 rows=rows.filter(r=>r.score_kind!=='binary_decision');
 $('rocDownloads').innerHTML=(D.roc_analyses||[]).map(a=>'Saved analysis: '+link(a.path+'/report.html','HTML')+' · '+link(a.path+'/report.md','Markdown')+' · '+link(a.path+'/matched_far.csv','CSV')).join(' · ');
 const target=Number($('rocFar').value),stats=matchedMetrics(rows,target).sort((a,b)=>a.task.localeCompare(b.task)||detectorOrder(a,b));
 exports.matched_far=stats.map(r=>({...r,failure_origin_filter:$('origin').value,failure_mode_filter:$('failure').value}));
 drawROC(rows);const overall=stats.filter(r=>r.task==='Overall');
 $('matchedHyperPanel').hidden=$('preset').value!=='hyper';
 $('matchedSinglePanel').hidden=$('preset').value==='hyper';
 document.querySelector('[data-svg="matchedChart"]').hidden=$('preset').value==='hyper';
 if($('preset').value==='hyper')drawHyperparameters(overall,true);
 else if(['K','L','action'].includes($('preset').value))drawAblation(overall,'matchedChart',true);
 else{$('matchedTitle').textContent='FDR at '+pct(target)+' FAR budget';drawBars(overall,'matchedChart','FDR')}
 plainTable('matchedTable',['Task','Detector','FAR budget','Achieved FAR','False alarms / success','FDR at cap','Detected / failure','Coverage / bounds','Interpolated any-time FDR','Full AUROC','AUC tasks','Threshold','Comparison','Unsupported'],stats.map(r=>[taskName(r.task),detectorName(r),pct(target),pct(r.FAR),`${r.fa}/${r.neg}`,pct(r.FDR),`${r.tp}/${r.pos}`,coverageText(r),pct(r.interpolated_FDR),fmt(r.AUROC),r.auc_tasks,r.threshold===null?'—':Number.isFinite(r.threshold)?r.threshold.toPrecision(8):String(r.threshold),r.comparison,r.unsupported]));
}
function table(id,headers,rows){$(id).innerHTML=rows.length?`<table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(v=>`<td>${v??'—'}</td>`).join('')}</tr>`).join('')}</tbody></table>`:'<div class="empty">No results for this selection.</div>'}
function plainTable(id,headers,rows){table(id,headers,rows.map(r=>r.map(v=>esc(v??'—'))))}
function link(path,text){return `<a href="${esc(path)}" target="_blank" rel="noopener">${esc(text)}</a>`}
function ablationValue(r,axis){
 if(axis==='action')return r.variant==='main'?3:({none:0,cross_attention:1,adaln:2}[runs.get(r.variant)?.action_mode]??null);
 const values=[...new Set(D.settings.filter(s=>s.variant===r.variant&&s.method===r.method&&s[axis]!==null).map(s=>s[axis]))];
 return values.length===1?values[0]:null;
}
function detectorOrder(a,b){
 const axis=$('preset').value;
 if(['K','L','action'].includes(axis)){const difference=ablationValue(a,axis)-ablationValue(b,axis);if(difference)return difference}
 const order=['wm_nc','wm_pe','wm_pu','logpzo','rnd','logpzo_native','rnd_native','ace','act_ensemble_pos','act_ensemble_rot','qwen_v9'];
 const rank=r=>order.includes(r.method)?order.indexOf(r.method):20;
 return rank(a)-rank(b)||a.method.localeCompare(b.method)||(Number(a.variant!=='main')-Number(b.variant!=='main')||a.variant.localeCompare(b.variant))||['fix','wrist','both'].indexOf(a.view)-['fix','wrist','both'].indexOf(b.view);
}
const viewColors={fix:'#176995',wrist:'#d37732',both:'#865aa6'};
function plotContext(){
 return [$('task').selectedOptions[0].textContent,$('origin').selectedOptions[0].textContent,
         $('failure').value==='all'?'All failure types':$('failure').value,$('scheme').selectedOptions[0].textContent,$('detectionRule').selectedOptions[0].textContent].join(' · ');
}
function drawAblation(rows,target='ablationChart',matched=false,axis=$('preset').value,titleId=null,exportId=null){
 const points=rows.map(r=>({...r,x:ablationValue(r,axis)})).filter(r=>r.x!==null);
 const ticks=axis==='action'?[0,1,2,3]:[...new Set(points.map(r=>r.x))].sort((a,b)=>a-b);
 exports[exportId||(matched?'matched_ablation':'ablation')]=points.map(r=>({task:$('task').value,variant:r.variant,method:r.method,view:r.view,
   axis,value:r.x,calibration:$('scheme').value,failure_origin:$('origin').value,failure_type:$('failure').value,
   detection_rule:$('detectionRule').value,annotator_id:$('annotator').value,unknown:r.unknown,early:r.early,
   detected:r.tp,failures:r.pos,false_alarms:r.fa,successes:r.neg,FDR:r.FDR,FAR:r.FAR}));
 $(titleId||(matched?'matchedTitle':'ablationTitle')).textContent=(axis==='action'?'Action conditioning':axis==='K'?'Endpoint horizon K':'History length L')+' · FDR and FAR'+(matched?' at '+pct(Number($('rocFar').value))+' FAR budget':'');
 if(!ticks.length){$(target).innerHTML='<p class="empty">No ablation points selected.</p>';return}
 const x=v=>ticks.length===1?540:110+860*(v-ticks[0])/(ticks.at(-1)-ticks[0]);
 const series=[...group(points,r=>r.method+'|'+r.view).values()];
 // Keep FDR at 0–100%; scale FAR independently from zero so small changes remain visible.
 const farMax=Math.min(1,Math.max(.05,Math.ceil(Math.max(0,...points.map(r=>r.FAR||0))/.05)*.05));
 let svg='<svg viewBox="0 0 1060 625" xmlns="http://www.w3.org/2000/svg" style="font:13px sans-serif"><rect width="1060" height="625" fill="white"/>';
 svg+=`<text x="110" y="25" fill="#536e7e">${esc(plotContext())}${matched?' · empirical FAR ≤ '+pct(Number($('rocFar').value)):''}</text>`;
 series.forEach((a,i)=>{
   const xx=110+i*218,color=viewColors[a[0].view],dash=a[0].method==='wm_pe'?'7 4':'';
   svg+=`<line x1="${xx}" x2="${xx+28}" y1="53" y2="53" stroke="${color}" stroke-width="2.5" stroke-dasharray="${dash}"/><text x="${xx+36}" y="57">${esc(methodName(a[0].method)+' · '+VIEWS[a[0].view])}</text>`;
 });
 for(const [metric,top,limit] of [['FDR',90,1],['FAR',350,farMax]]){
   const y=v=>top+200-200*v/limit;
   svg+=`<text x="110" y="${top-12}" font-weight="bold">${metric} (%) ${metric==='FDR'?'↑':'↓'}</text>`;
   for(let i=0;i<=4;i++){
     const v=limit*i/4,yy=y(v);
     svg+=`<line x1="110" x2="970" y1="${yy}" y2="${yy}" stroke="#dde5eb"/><text x="94" y="${yy+4}" text-anchor="end">${fmt(100*v,metric==='FDR'?0:2)}</text>`;
   }
   for(const v of ticks)svg+=`<line x1="${x(v)}" x2="${x(v)}" y1="${top}" y2="${top+200}" stroke="#edf1f4"/><text x="${x(v)}" y="${top+222}" text-anchor="middle">${axis==='action'?['No action','Cross-attention','AdaLN','Both'][v]:v}</text>`;
   for(const a of series){
     const color=viewColors[a[0].view],pe=a[0].method==='wm_pe',sorted=[...a].sort((a,b)=>a.x-b.x);
     let path='',connected=false;
     for(const r of sorted){
       if(r[metric]===null){connected=false;continue}
       path+=`${connected?'L':'M'}${x(r.x)},${y(r[metric])} `;connected=true;
     }
     svg+=`<path d="${axis==='action'?'':path}" fill="none" stroke="${color}" stroke-width="2.7" stroke-dasharray="${pe?'7 4':''}"/>`;
     for(const r of sorted){
       if(r[metric]===null)continue;
       const title=esc(`${detectorName(r)}, ${axis}=${r.x}: ${metric} ${pct(r[metric])} (${metric==='FDR'?r.tp:r.fa}/${metric==='FDR'?r.pos:r.neg})`);
       svg+=pe?`<rect x="${x(r.x)-4}" y="${y(r[metric])-4}" width="8" height="8" fill="white" stroke="${color}" stroke-width="2"><title>${title}</title></rect>`:
               `<circle cx="${x(r.x)}" cy="${y(r[metric])}" r="4.5" fill="${color}"><title>${title}</title></circle>`;
     }
   }
 }
 svg+=`<text x="540" y="610" text-anchor="middle">${axis==='action'?'Action-conditioning pathways (K = L = 4)':axis==='K'?'Endpoint prediction horizon K':'Observation history length L'}</text></svg>`;
 $(target).innerHTML=svg;
}
function delaySummary(values){
 const a=values.filter(v=>v!==null&&Number.isFinite(v)&&v>=0).sort((a,b)=>a-b),average=mean(a);
 const sd=a.length>1?Math.sqrt(a.reduce((s,v)=>s+(v-average)**2,0)/(a.length-1)):null;
 const q1=quantile(a,.25),median=quantile(a,.5),q3=quantile(a,.75),iqr=q3-q1;
 const inside=a.filter(v=>v>=q1-1.5*iqr&&v<=q3+1.5*iqr);
 return {detected_n:a.length,mean_s:average,sd_s:sd,median_s:median,q1_s:q1,q3_s:q3,
   whisker_low_s:inside[0]??null,whisker_high_s:inside.at(-1)??null,
   outliers:a.filter(v=>v<q1-1.5*iqr||v>q3+1.5*iqr),values:a};
}
function latencyPlotRows(){
 const online=$('latencySource').value==='online';
 const records=online?[...D.online,...VLM_EPISODES.filter(r=>r.outcome==='failure')].flatMap(t=>t.latencies.map(r=>({...r,calibration:t.calibration,
   failure_origin:t.failure_origin,failure_mode:t.failure_mode}))):D.latencies;
 const matching=records.filter(r=>base(r)&&failureMatch(r)&&annotationMatch(r));
 const grouped=group(matching,key);
 const candidates=[...new Map(comparisonEpisodes().filter(r=>r.split==='test'&&r.outcome==='failure'&&base(r)&&failureMatch(r)).map(r=>[key(r),r])).values()];
 return candidates.sort(detectorOrder).map(r=>{
   const a=grouped.get(key(r))||[],stats=delaySummary(a.filter(r=>r.status==='detected'&&!(online&&r.late_alarm)).map(r=>number(online?r.delay_s:r.detection_delay_s)));
   return {variant:r.variant,method:r.method,view:r.view,source:online?'online':'offline',
     task_filter:$('task').value,calibration:$('scheme').value,annotator_id:$('annotator').value,
     onset_view:r.view==='both'?'earlier_view':r.view,failure_origin_filter:$('origin').value,
     failure_mode_filter:$('failure').value,measured_n:a.length,excluded_n:a.length-stats.detected_n,
     early_n:a.filter(r=>r.early_alarm??(r.status===(online?'pre_onset_evidence':'pre_onset_alarm'))).length,
     missed_n:a.filter(r=>r.status==='missed').length,copied_n:a.filter(r=>r.copied_from_annotator).length,...stats};
 });
}
let sharedSearchResult=null;
function sharedContext(){return ['task','origin','failure','scheme','view','annotator','detectionRule'].map(id=>$(id).value).join('|')}
function sharedMinimum(){const n=Number($('sharedMinimum').value);return Number.isInteger(n)&&n>0?n:null}
function sharedCandidates(pool=selected){return [...new Map(D.episodes.filter(r=>pool.has(key(r))&&r.split==='test'&&r.outcome==='failure'&&failureMatch(r)&&($('task').value==='all'||r.task===$('task').value)&&($('view').value==='all'||r.view===$('view').value)&&r.calibration===$('scheme').value).map(r=>[key(r),r])).values()].sort(detectorOrder)}
function sharedIndex(pool,includeVLM=false,onTimeOnly=false){
 const index=new Map([...pool].map(id=>[id,{records:new Map(),detected:new Map()}]));
 const trials=includeVLM?[...D.online,...VLM_EPISODES.filter(r=>r.outcome==='failure').map(r=>({...r,path:r.trace_path}))]:D.online;
 for(const t of trials){
  if((t.calibration!==$('scheme').value&&!(includeVLM&&t.calibration==='binary_anytime'&&$('scheme').value==='episode'))||($('task').value!=='all'&&t.task!==$('task').value)||!failureMatch(t))continue;
  for(const row of t.latencies){
   if(!pool.has(key(row))||!annotationMatch(row))continue;
   const bucket=index.get(key(row)),episode=JSON.stringify([t.task,t.episode_index]);
   if(bucket.records.has(episode))throw Error('Duplicate shared-latency detector/episode');
   const r={...row,failure_origin:t.failure_origin,failure_mode:t.failure_mode,trace:t.path};bucket.records.set(episode,r);
   if(r.status==='detected'&&r.delay_s!==null&&Number.isFinite(r.delay_s)&&(r.delay_s>=0||r.detected_before_human_label)&&r.onset_time_s!==null&&Number.isFinite(r.onset_time_s)&&!(onTimeOnly&&r.late_alarm))bucket.detected.set(JSON.stringify([t.task,t.episode_index,r.onset_time_s]),r);
  }
 }
 return index;
}
function sharedIntersection(index,ids){let common=null;for(const id of ids){const s=new Set(index.get(id)?.detected.keys()||[]);common=common===null?s:new Set([...common].filter(k=>s.has(k)))}return [...(common||[])].sort()}
function renderPairwiseLatency(){
 const catalog=D.pairwise_latency?.detectors||{},reference='main|wm_nc|wrist',minimum=Number($('pairMinimum').value),onTime=$('pairTiming').value==='on_time';
 exports.pairwise=[];exports.pairwise_episodes=[];
 if($('scheme').value!=='episode'||!Number.isInteger(minimum)||minimum<1){$('pairwiseStatus').textContent='Choose full-episode calibration and a positive minimum shared count.';$('pairwiseChart').replaceChildren();$('pairwiseTable').replaceChildren();return}
 const candidates=Object.entries(catalog).filter(([id,r])=>id!==reference&&($('pairFamily').value==='all'||r.family===$('pairFamily').value));
 const index=sharedIndex(new Set([reference,...candidates.map(([id])=>id)]),true,onTime),rows=[];
 // Pair task/episode identities, retaining each detector's own onset-relative delay.
 for(const bucket of index.values())bucket.detected=new Map([...bucket.detected.values()].map(r=>[JSON.stringify([r.task,r.episode_index]),r]));
 for(const [id,meta]of candidates){
  const a=index.get(reference),b=index.get(id),common=sharedIntersection(index,[reference,id]);
  const lhs=delaySummary(common.map(k=>a.detected.get(k).delay_s)),rhs=delaySummary(common.map(k=>b.detected.get(k).delay_s));
  const diffs=common.map(k=>a.detected.get(k).delay_s-b.detected.get(k).delay_s),average=mean(diffs),sd=diffs.length>1?Math.sqrt(diffs.reduce((n,v)=>n+(v-average)**2,0)/(diffs.length-1)):null;
  const eligible=common.length>=minimum,status=!b.records.size?'Not measured online':!common.length?'No shared detections':!eligible?'Below minimum':'Available';
  const row={reference,comparator:id,label:meta.label,family:meta.family,task_filter:$('task').value,failure_origin_filter:$('origin').value,failure_mode_filter:$('failure').value,annotator_id:$('annotator').value,on_time_only:onTime,minimum_shared_n:minimum,shared_n:common.length,status,eligible,
   reference_mean_s:lhs.mean_s,reference_sd_s:lhs.sd_s,comparator_mean_s:rhs.mean_s,comparator_sd_s:rhs.sd_s,mean_difference_s:average,sd_difference_s:sd,reference_faster_n:diffs.filter(v=>v<0).length,
   reference_own_n:a.detected.size,comparator_own_n:b.detected.size,reference_measured_n:a.records.size,comparator_measured_n:b.records.size,
   reference_late_n:[...a.records.values()].filter(r=>r.alarm&&r.late_alarm).length,comparator_late_n:[...b.records.values()].filter(r=>r.alarm&&r.late_alarm).length,
   different_onset_excluded_n:0,different_onset_included_n:common.filter(k=>{const left=a.detected.get(k),right=b.detected.get(k);return left.onset_time_s!==right.onset_time_s||left.onset_raw_frame!==right.onset_raw_frame}).length};
  rows.push(row);
  for(const k of common){const left=a.detected.get(k),right=b.detected.get(k);exports.pairwise_episodes.push({reference,comparator:id,task:left.task,episode_index:left.episode_index,annotator_id:$('annotator').value,reference_onset_time_s:left.onset_time_s,comparator_onset_time_s:right.onset_time_s,reference_onset_raw_frame:left.onset_raw_frame,comparator_onset_raw_frame:right.onset_raw_frame,on_time_only:onTime,reference_delay_s:left.delay_s,comparator_delay_s:right.delay_s,difference_s:left.delay_s-right.delay_s,failure_origin:left.failure_origin,failure_mode:left.failure_mode,reference_trace:left.trace,comparator_trace:right.trace})}
 }
 exports.pairwise=rows;
 $('pairwiseStatus').textContent=`${rows.filter(r=>r.eligible).length}/${rows.length} comparisons meet n ≥ ${minimum}. ${rows.filter(r=>r.status==='Not measured online').length} have no online measurements in this selection. ${onTime?'Alarms must return before the recorded episode ends.':'Late alarms are included for both methods.'} Shared counts are conditional; they do not replace FDR or FAR.`;
 plainTable('pairwiseTable',['Comparator','Shared n','Wrist / comparator own detections','Wrist / comparator measured','Wrist / comparator late alarms','Different onsets included','Mean wrist − other (s)','Wrist shorter delay / shared','Status'],rows.map(r=>[r.label,r.shared_n,`${r.reference_own_n}/${r.comparator_own_n}`,`${r.reference_measured_n}/${r.comparator_measured_n}`,`${r.reference_late_n}/${r.comparator_late_n}`,r.different_onset_included_n,fmt(r.mean_difference_s),`${r.reference_faster_n}/${r.shared_n}`,r.status]));
 drawPairwiseLatency(rows);
}
function drawPairwiseLatency(rows){
 const difference=$('pairPlot').value==='difference',values=rows.filter(r=>r.eligible).flatMap(r=>difference?[r.mean_difference_s-(r.sd_difference_s||0),r.mean_difference_s+(r.sd_difference_s||0)]:[r.reference_mean_s-(r.reference_sd_s||0),r.reference_mean_s+(r.reference_sd_s||0),r.comparator_mean_s-(r.comparator_sd_s||0),r.comparator_mean_s+(r.comparator_sd_s||0)]);
 let lo=Math.min(0,...values),hi=Math.max(0,...values);if(lo===hi)hi=lo+1;const pad=(hi-lo)*.04;lo-=pad;hi+=pad;
 const left=340,width=475,height=140+rows.length*57,x=v=>left+(v-lo)/(hi-lo)*width;
 let svg=`<svg viewBox="0 0 1240 ${height}" xmlns="http://www.w3.org/2000/svg"><rect width="1240" height="${height}" fill="white"/><text x="20" y="23" fill="#536e7e">${esc($('task').value==='all'?'All tasks':taskName($('task').value))} · ${esc($('origin').selectedOptions[0].textContent)} · ${esc($('failure').selectedOptions[0].textContent)}</text><text x="20" y="45">Main wrist WM-NC · K4 L4 · ${esc($('annotator').selectedOptions[0].textContent)} · ${$('pairTiming').value==='on_time'?'on-time':'including late'} alarms</text><text x="20" y="65" fill="#158477">${difference?'Wrist − comparator: negative means shorter wrist delay':'Green: main wrist WM-NC · Orange: comparator · own-onset mean ± sample SD'}</text>`;
 for(let i=0;i<=5;i++){const v=lo+(hi-lo)*i/5,xx=x(v);svg+=`<line x1="${xx}" x2="${xx}" y1="82" y2="${height-48}" stroke="#e5edf2"/><text x="${xx}" y="${height-26}" text-anchor="middle">${fmt(v,1)}</text>`}
 svg+=`<line x1="${x(0)}" x2="${x(0)}" y1="82" y2="${height-48}" stroke="#879ca9" stroke-dasharray="4 3"/>`;
 const point=(v,sd,y,color)=>`<path d="M${x(v-(sd||0))},${y} H${x(v+(sd||0))}" stroke="${color}" stroke-width="2"/><circle cx="${x(v)}" cy="${y}" r="4" fill="${color}"/>`;
 rows.forEach((r,i)=>{const y=100+i*57;svg+=`<text x="${left-12}" y="${y+4}" text-anchor="end">${esc(r.label)}</text><text x="1220" y="${y+4}" text-anchor="end">n=${r.shared_n}</text>`;
  if(!r.eligible){svg+=`<text x="${left+10}" y="${y+4}" fill="#758794">${esc(r.status)}</text>`;return}
  if(difference){svg+=point(r.mean_difference_s,r.sd_difference_s,y,'#176995')+`<text x="840" y="${y+4}">${fmt(r.mean_difference_s,2)} ± ${fmt(r.sd_difference_s,2)} s</text>`}
  else {svg+=`<path d="M${x(r.reference_mean_s)},${y-8} L${x(r.comparator_mean_s)},${y+8}" stroke="#bdcbd3"/>`+point(r.reference_mean_s,r.reference_sd_s,y-8,'#158477')+point(r.comparator_mean_s,r.comparator_sd_s,y+8,'#db8134')+`<text x="840" y="${y-4}" fill="#158477">Main ${fmt(r.reference_mean_s,2)} ± ${fmt(r.reference_sd_s,2)} s</text><text x="840" y="${y+14}" fill="#a65a1c">Other ${fmt(r.comparator_mean_s,2)} ± ${fmt(r.comparator_sd_s,2)} s</text>`}
 });
 svg+=`<text x="580" y="${height-4}" text-anchor="middle">${difference?'Paired latency difference':'Onset-to-alarm latency'} (wall seconds)</text></svg>`;
 $('pairwiseChart').innerHTML=svg;
}
function largestSharedGroups(index,ids,minimum){
 const candidates=[...new Set(ids)].filter(id=>(index.get(id)?.detected.size||0)>=minimum).sort();
 if(candidates.length>20)return {complete:false,groups:[],candidate_count:candidates.length,reason:'Select at most 20 individually eligible detectors for exact group search. Manual shared comparisons have no such limit.'};
 let bestSize=1,best=[];
 const visit=(chosen,common,start)=>{
  if(chosen.length>=2){if(chosen.length>bestSize){bestSize=chosen.length;best=[]}if(chosen.length===bestSize)best.push({detectors:[...chosen],shared_n:common.length})}
  if(chosen.length+candidates.length-start<bestSize)return;
  for(let i=start;i<candidates.length;i++){const support=index.get(candidates[i]).detected,next=common===null?[...support.keys()]:common.filter(k=>support.has(k));if(next.length>=minimum)visit([...chosen,candidates[i]],next,i+1)}
 };
 visit([],null,0);best.sort((a,b)=>b.shared_n-a.shared_n||a.detectors.join('|').localeCompare(b.detectors.join('|')));
 return {complete:true,groups:best,maximum_detectors:best.length?bestSize:0,candidate_count:candidates.length};
}
function findSharedGroups(){
 const minimum=sharedMinimum();if(minimum===null){$('sharedStatus').textContent='Enter a positive whole-number minimum.';return}
 const pool=sharedSearchResult?.applied&&sharedSearchResult.context===sharedContext()?new Set(sharedSearchResult.pool):new Set(selected);
 const candidates=sharedCandidates(pool),ids=candidates.map(key),index=sharedIndex(new Set(ids));
 sharedSearchResult={...largestSharedGroups(index,ids,minimum),minimum,pool:[...pool],context:sharedContext(),applied:false};
 renderSharedLatency();
}
function applySharedGroup(position){
 const g=sharedSearchResult?.groups[position];if(!g)return;
 selected=new Set(g.detectors);sharedSearchResult.applied=true;
 for(const c of $('methods').querySelectorAll('input'))c.checked=selected.has(c.value);
 refresh();
}
function renderSharedLatency(){
 const candidates=sharedCandidates(),ids=candidates.map(key),minimum=sharedMinimum(),index=sharedIndex(new Set(ids)),common=sharedIntersection(index,ids);
 if(sharedSearchResult&&sharedSearchResult.context!==sharedContext())sharedSearchResult=null;
 const search=sharedSearchResult;
 let searchText='';
 if(search){
  searchText=search.complete?`Search pool: ${search.pool.length} detectors. Minimum ${search.minimum}: largest eligible groups contain ${search.maximum_detectors} detectors; ${search.groups.length} tied group(s). Showing up to 12, ordered by shared count, never latency.`:search.reason;
  const lookup=new Map(sharedCandidates(new Set(search.pool)).map(r=>[key(r),r]));
  table('sharedGroups',['Detectors','Shared n','Methods','Apply'],search.groups.slice(0,12).map((g,i)=>[g.detectors.length,g.shared_n,esc(g.detectors.map(d=>detectorName(lookup.get(d))).join('; ')),`<button data-shared-group="${i}">Compare group</button>`]));
 }else $('sharedGroups').replaceChildren();
 const eligible=ids.length>=2&&minimum!==null&&common.length>=minimum;
 let status=minimum===null?'Enter a positive whole-number minimum.':ids.length<2?'Select at least two detectors.':`${ids.length} selected detectors share ${common.length} post-onset detections with identical onset times; minimum ${minimum}. ${eligible?'All shared episodes are used.':'Below the minimum: no latency ranking is shown. Remove detectors or find eligible groups.'}`;
 if($('scheme').value!=='episode')status='Online latency has not been measured for this calibration protocol.';
 $('sharedStatus').textContent=status+' '+searchText;
 const composition=new Map(),origins=new Map(),failures=new Map();
 for(const k of common){const r=index.get(ids[0]).detected.get(k);for(const [map,name]of [[composition,r.task],[origins,r.failure_origin],[failures,r.task+' / '+r.failure_mode]])map.set(name,(map.get(name)||0)+1)}
 $('sharedComposition').textContent=common.length?'Shared-set composition — '+[...composition].map(([k,v])=>`${taskName(k)}: ${v}`).join('; ')+'. '+[...origins].map(([k,v])=>`${k}: ${v}`).join('; ')+'. Failures: '+[...failures].map(([k,v])=>`${k}: ${v}`).join('; '):'No common post-onset detections in this selection.';
 exports.shared_episodes=[];exports.shared_pairs=[];
 const rows=eligible?candidates.map(r=>{const bucket=index.get(key(r)),a=common.map(k=>bucket.detected.get(k)),s=delaySummary(a.map(r=>r.delay_s));return {...r,...s,measured_n:bucket.records.size,own_post_onset_n:bucket.detected.size,excluded_n:bucket.detected.size-common.length,minimum_shared_n:minimum,annotator_id:$('annotator').value,source:'online_shared',task_filter:$('task').value,onset_view:r.view==='both'?'earlier_view':r.view}}):[];
 drawLatency(rows,'sharedChart','sharedPlotNote','shared_latency',`Online shared episodes · n=${common.length} · minimum ${minimum}`);
 plainTable('sharedTable',['Detector','Mean ± SD (s)','Median (s)','Shared / own detections','Other own detections excluded'],rows.map(r=>[detectorName(r),`${fmt(r.mean_s)} ± ${fmt(r.sd_s)}`,fmt(r.median_s),`${r.detected_n}/${r.own_post_onset_n}`,r.excluded_n]));
 if(eligible){
  for(const k of common)for(const id of ids){const r=index.get(id).detected.get(k);exports.shared_episodes.push({detector:id,task:r.task,episode_index:r.episode_index,annotator:r.annotator,annotator_id:r.annotator_id,onset_view:r.onset_view,onset_time_s:r.onset_time_s,delay_s:r.delay_s,failure_mode:r.failure_mode,failure_origin:r.failure_origin,minimum_shared_n:minimum,shared_n:common.length,trace:r.trace})}
  for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++){
   const diffs=common.map(k=>index.get(ids[i]).detected.get(k).delay_s-index.get(ids[j]).detected.get(k).delay_s),average=mean(diffs);
   exports.shared_pairs.push({detector_a:ids[i],detector_b:ids[j],n:diffs.length,mean_difference_s:average,sd_difference_s:diffs.length>1?Math.sqrt(diffs.reduce((s,v)=>s+(v-average)**2,0)/(diffs.length-1)):null,median_difference_s:quantile(diffs,.5),a_faster_n:diffs.filter(v=>v<0).length,tie_n:diffs.filter(v=>v===0).length});
  }
 }
 const labels=new Map(candidates.map(r=>[key(r),detectorName(r)]));
 plainTable('sharedPairs',['A','B','Shared n','Mean A − B (s)','SD of paired difference (s)','A earlier / n'],exports.shared_pairs.map(r=>[labels.get(r.detector_a),labels.get(r.detector_b),r.n,fmt(r.mean_difference_s),fmt(r.sd_difference_s),`${r.a_faster_n}/${r.n}`]));
}
function drawLatency(rows=latencyPlotRows(),target='latencyChart',note='latencyPlotNote',exportName='latency_plot',subtitle=null){
 const box=$('latencyStyle').value==='box';
 exports[exportName]=rows.map(({values,outliers,...r})=>({...r,outlier_n:outliers.length}));
 $(note).textContent=box?
   'Boxes: 25th–75th percentiles; line: median; diamond: mean. Whiskers extend to observed values within 1.5 × IQR; dots show outliers. n is the number of post-onset detections.':
   'Dots: mean. Whiskers: ±1 sample standard deviation (n−1), not a confidence interval. SD is unavailable for n=1. Negative whisker bounds are a statistical spread, not negative observed delays.';
 if(!rows.length){$(target).innerHTML='<p class="empty">No eligible latency comparison for this selection.</p>';return}
 const limits=rows.filter(r=>r.detected_n).flatMap(r=>box?[r.values[0],r.values.at(-1)]:[r.mean_s-(r.sd_s||0),r.mean_s+(r.sd_s||0)]);
 let lo=Math.min(0,...limits),hi=Math.max(0,...limits);if(lo===hi)hi=1;
 const margin=(hi-lo)*.04;lo-=lo<0?margin:0;hi+=margin;
 const left=310,width=480,height=135+rows.length*35,x=v=>left+(v-lo)/(hi-lo)*width;
 let svg=`<svg viewBox="0 0 1100 ${height}" xmlns="http://www.w3.org/2000/svg" style="font:13px sans-serif"><rect width="1100" height="${height}" fill="white"/>`;
 svg+=`<text x="20" y="23" fill="#536e7e">${esc(plotContext())}</text><text x="20" y="44" fill="#536e7e">${esc(subtitle||($('latencySource').value==='online'?'Online alarm-ready latency':'Offline dense-score latency'))} · Annotator: ${esc($('annotator').selectedOptions[0]?.textContent||'None')}</text>`;
 svg+='<text x="820" y="72" font-weight="bold">Mean ± SD (s)</text><text x="1045" y="72" text-anchor="end" font-weight="bold">n</text>';
 for(let i=0;i<=4;i++){
   const v=lo+(hi-lo)*i/4,xx=x(v);
   svg+=`<line x1="${xx}" x2="${xx}" y1="80" y2="${height-53}" stroke="#e1e9ee"/><text x="${xx}" y="${height-30}" text-anchor="middle">${fmt(v,2)}</text>`;
 }
 if(lo<0)svg+=`<line x1="${x(0)}" x2="${x(0)}" y1="80" y2="${height-53}" stroke="#91a3ae" stroke-dasharray="4 4"/>`;
 rows.forEach((r,i)=>{
   const yy=97+35*i,color=viewColors[r.view],summary=`${detectorName(r)}: mean ${fmt(r.mean_s)} s, SD ${fmt(r.sd_s)} s, detected n=${r.detected_n}; excluded ${r.excluded_n}`;
   svg+=`<text x="${left-15}" y="${yy+4}" text-anchor="end">${esc(target==='sharedChart'&&runs.get(r.variant)?.family==='view_fusion'?'Joint-view '+methodName(r.method):detectorName(r))}</text>`;
   if(!r.detected_n){svg+=`<text x="${left+12}" y="${yy+4}" fill="#758794">${r.measured_n?'No post-onset detections':$('latencySource').value==='online'?'No online measurements':'No detections'}</text><text x="1045" y="${yy+4}" text-anchor="end">0</text>`;return}
   const low=box?r.whisker_low_s:r.mean_s-(r.sd_s||0),high=box?r.whisker_high_s:r.mean_s+(r.sd_s||0);
   svg+=`<g><title>${esc(summary)}</title><path d="M${x(low)},${yy} H${x(high)} M${x(low)},${yy-6} V${yy+6} M${x(high)},${yy-6} V${yy+6}" stroke="${color}" stroke-width="2" fill="none"/>`;
   if(box){
     svg+=`<rect x="${x(r.q1_s)}" y="${yy-9}" width="${Math.max(.7,x(r.q3_s)-x(r.q1_s))}" height="18" fill="${color}" fill-opacity=".18" stroke="${color}"/><path d="M${x(r.median_s)},${yy-9} V${yy+9}" stroke="${color}" stroke-width="2"/>`;
     r.outliers.forEach(v=>{svg+=`<circle cx="${x(v)}" cy="${yy}" r="2" fill="${color}" fill-opacity=".5"/>`});
     svg+=`<path d="M${x(r.mean_s)},${yy-5} l5,5 l-5,5 l-5,-5 Z" fill="${color}"/>`;
   }else svg+=`<circle cx="${x(r.mean_s)}" cy="${yy}" r="4.5" fill="${color}"/>`;
   svg+=`</g><text x="820" y="${yy+4}">${fmt(r.mean_s,2)} ± ${fmt(r.sd_s,2)}</text><text x="1045" y="${yy+4}" text-anchor="end">${r.detected_n}</text>`;
 });
 svg+=`<text x="${left+width/2}" y="${height-6}" text-anchor="middle">Onset-to-alarm latency (wall seconds) ↓</text></svg>`;
 $(target).innerHTML=svg;
}
function drawBars(rows,target='comparisonChart',m=$('metric').value){const w=1040,left=360,right=80,h=Math.max(110,rows.length*30+55);let svg=`<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg"><rect width="${w}" height="${h}" fill="white"/>`;for(let t=0;t<=1;t+=.25){const x=left+t*(w-left-right);svg+=`<line x1="${x}" x2="${x}" y1="20" y2="${h-25}" stroke="#e1e9ee"/><text x="${x}" y="${h-6}" text-anchor="middle" fill="#617888">${Math.round(100*t)}%</text>`}rows.forEach((r,i)=>{const y=20+i*30,v=r[m];svg+=`<text x="${left-12}" y="${y+16}" text-anchor="end" fill="#193142">${esc(detectorName(r))}</text>`;if(v!==null)svg+=`<rect x="${left}" y="${y}" width="${Math.max(0,v)*(w-left-right)}" height="21" rx="3" fill="${m==='FAR'?'#cf8350':palette[i%palette.length]}"/>`;svg+=`<text x="${left+(v||0)*(w-left-right)+7}" y="${y+16}" fill="#193142">${pct(v)}</text>`});$(target).innerHTML=svg+'</svg>'}
function drawHyperparameters(overall,matched=false){
 for(const axis of ['K','L']){
  const stem=(matched?'matchedHyper':'hyper')+axis;
  drawAblation(overall.filter(r=>eligibleComparison(r,axis)),stem+'Chart',matched,axis,stem+'Title',stem);
 }
}
function renderFusionLevels(){
 const study=D.fusion_level_comparison;
 if(!study)return;
 $('fusionLevelNote').textContent=study.note;
 $('fusionLevelDownloads').innerHTML=link(study.path+'/report.md','Definitions and summary')+' · '+link(study.path+'/metrics.csv','Per-task / per-failure CSV')+' · '+link(study.path+'/episode_scores.csv','Both levels’ scores')+' · '+link(study.path+'/component_calibration.csv','Frozen component calibration');
 const level=$('fusionLevel').value;
 const rows=study.rows.filter(r=>r.split==='test'&&base(r)&&outcomeMatch(r)&&(level==='both'||r.level===level)).map(r=>({...assessmentRow({...r,alarm:r.any_time_alarm}),variant:'fusion_'+r.level}));
 for(const level of ['endpoint','episode'])runs.set('fusion_'+level,{id:'fusion_'+level,title:level==='endpoint'?'Endpoint':'Episode',family:'alarm_fusion'});
 const stats=metrics(rows).map(r=>({...r,level:r.variant.slice(7)}));
 exports.fusion_levels=stats.map(r=>({...r,failure_origin_filter:$('origin').value,failure_mode_filter:$('failure').value}));
 exports.fusion_scores=rows.map(r=>Object.fromEntries(['task','method','view','level','episode_index','split','outcome','failure_mode','failure_origin','score_max','post_score_max','alarm','any_time_alarm','annotator_id','n_supported_scores','left_max','right_max'].map(k=>[k,r[k]])));
 plainTable('fusionLevelTable',['Task','Monitor','View','Level','False alarms / success','FAR','Detected / failure','FDR','Unknown','Whole-episode AUROC'],stats.map(r=>[taskName(r.task),methodName(r.method),VIEWS[r.view],r.level,`${r.fa}/${r.neg}`,pct(r.FAR),`${r.tp}/${r.pos}`,pct(r.FDR),r.unknown,fmt(r.AUROC)]));
 const metric=$('fusionMetric').value,overall=stats.filter(r=>r.task==='Overall'),groups=[...group(overall,r=>r.method+'|'+r.view).values()];
 if(!groups.length){$('fusionLevelChart').innerHTML='<p class="empty">No fusion results for this selection. This comparison uses episode calibration; choose at least one OR/AND detector.</p>';return;}
 const max=metric==='FAR'?Math.min(1,Math.max(.05,...overall.map(r=>r.FAR??0))*1.1):1;
 const x=v=>365+590*v/max,h=105+groups.length*76;
 let svg=`<svg viewBox="0 0 1060 ${h}" xmlns="http://www.w3.org/2000/svg"><rect width="1060" height="${h}" fill="white"/><text x="365" y="22">${esc(metric)} · ${esc($('task').selectedOptions[0].textContent)}</text><text x="365" y="44" fill="#176995">■ Same endpoint</text><text x="565" y="44" fill="#db8134">■ Episode alarms</text>`;
 for(let i=0;i<=4;i++){const v=max*i/4;svg+=`<line x1="${x(v)}" x2="${x(v)}" y1="63" y2="${h-35}" stroke="#e1e8ed"/><text x="${x(v)}" y="${h-15}" text-anchor="middle">${metric==='AUROC'?fmt(v,2):pct(v)}</text>`;}
 groups.forEach((items,i)=>{
  const first=items[0],label=study.rows.find(r=>r.method===first.method&&r.view===first.view)?.label||methodName(first.method),y=80+i*76;
  svg+=`<text x="350" y="${y+18}" text-anchor="end">${esc(label)}${first.view==='both'?'':' · '+VIEWS[first.view]}</text>`;
  for(const r of items){const value=r[metric],yy=y+(r.level==='episode'?25:0),color=r.level==='endpoint'?'#176995':'#db8134',count=metric==='FDR'?`${r.tp}/${r.pos}`:metric==='FAR'?`${r.fa}/${r.neg}`:'';
   if(value===null){svg+=`<text x="370" y="${yy+13}">Unavailable (${r.unknown} unknown)</text>`;continue;}
   svg+=`<rect x="365" y="${yy}" width="${590*value/max}" height="17" fill="${color}"><title>${esc(label)} · ${r.level}: ${metric==='AUROC'?fmt(value):pct(value)} ${count}</title></rect><text x="${x(value)+7}" y="${yy+13}">${metric==='AUROC'?fmt(value):pct(value)}</text>`;
  }
 });
 $('fusionLevelChart').innerHTML=svg+'</svg>';
}
function renderOverview(rows){
 const preset=$('preset').value,stats=metrics(rows).sort((a,b)=>a.task.localeCompare(b.task)||detectorOrder(a,b)),overall=stats.filter(r=>r.task==='Overall');
 exports.metrics=stats.map(r=>({...r,calibration:r.calibration||$('scheme').value,failure_origin_filter:$('origin').value,failure_mode_filter:$('failure').value}));
 const action=preset==='action',hyper=preset==='hyper',ablation=['K','L','action'].includes(preset),fusion=preset==='fusion'&&!!D.fusion_level_comparison;
 $('fusionLevelPanel').hidden=!fusion;
 $('compactMetricsPanel').hidden=fusion;
 $('metricsDetailPanel').hidden=fusion;
 $('hyperPanel').hidden=!hyper;
 $('ablationPanel').hidden=!ablation||(action&&!overall.some(r=>runs.get(r.variant)?.family==='action'));
 $('singleMetricPanel').hidden=ablation||hyper||fusion;
 if(fusion)renderFusionLevels();else if(hyper)drawHyperparameters(overall);else if(ablation)drawAblation(overall);else drawBars(overall);
 plainTable('compactMetrics',['Detector','False alarms / success','FAR','Detected / failure','FDR','Other early alarms','Sustained pre-onset','Coverage / bounds','Whole-episode AUROC'],overall.map(r=>[detectorName(r),`${r.fa}/${r.neg}`,pct(r.FAR),`${r.tp}/${r.pos}`,pct(r.FDR),r.early,r.sustained,coverageText(r),fmt(r.AUROC)]));
 plainTable('metricsTable',['Task','Detector','False alarms / success','FAR','Detected / failure','FDR','Early alarms','Coverage / bounds','Late alarms','Whole-episode AUROC','AUROC tasks','Unsupported episodes'],stats.map(r=>[taskName(r.task),detectorName(r),`${r.fa}/${r.neg}`,pct(r.FAR),`${r.tp}/${r.pos}`,pct(r.FDR),r.early,coverageText(r),r.late_alarms,fmt(r.AUROC),r.auc_tasks,r.unsupported]));
 $('actionAvailability').hidden=!action;
 if(action){
  const inventory=(D.action_inventory||[]).filter(r=>$('task').value==='all'||r.task===$('task').value);
  const modeRows=[...group(inventory,r=>r.mode).values()].map(a=>[a[0].label,`${a.filter(r=>r.checkpoint_available).length}/${a.length} tasks`,`${a.filter(r=>r.benchmarked).length}/${a.length} tasks`,a.filter(r=>!r.benchmarked).map(r=>taskName(r.task)).join('; ')||'None']);
  modeRows.push(['Both · main reference','Available','Scored','None']);
  plainTable('actionInventory',['Conditioning','Trained checkpoints','Scored tasks','Pending scoring'],modeRows);
 }
 const links={
  main:'Calibration stability: <a href="#study/main_calibration_sensitivity_500">500 splits · main WM-NC</a> · <a href="#study/calibration_sensitivity_200">200 splits · multi-method comparison</a>. View FAR/FDR distributions and the original split by task and camera.',
  hyper:'Both sweeps below use WM-NC, with the same K4/L4 reference. Open <a href="#K">K only</a> or <a href="#L">L only (including WM-PE)</a> for individual views.',
  K:'<a href="#K">K sweep</a> · <a href="#L">L sweep</a> · <a href="#K/latency">Latency across K</a>. These are sensitivity results on the existing evaluation set, not independent validation-set hyperparameter selection.',
  L:'<a href="#K">K sweep</a> · <a href="#L">L sweep</a>. The K4/L4 main checkpoint is reused.',
  action:'<a href="#action/failure">All variants by failure mode</a> · <button id="actionMismatch">Inspect execution mismatch</button>. The NoAct variant also appears with main baselines once scored. FAR and FDR use the same calibration / test episodes.',
  seed:'<a href="#study/main_calibration_sensitivity_500">500 calibration splits · main WM-NC</a> · <a href="#study/calibration_sensitivity_200">200 splits · frozen multi-method study</a>. Resampling describes sensitivity, not independent experiments or confidence intervals.',
  data_size:'<a href="#study/expert_policy_score_distributions">Expert-validation versus successful-policy score densities</a>. Training-quantity experiments do not separately isolate operator diversity.'
 };
 if(preset==='action'&&D.action_protocol_audit)links.action+=' '+link(D.action_protocol_audit,'Why results differ from the original paper: smoothing and retraining audit')+'. Current results remain unsmoothed.';
 $('questionLinks').hidden=!links[preset];$('questionLinks').innerHTML=links[preset]?'<p class="note">'+links[preset]+'</p>':'';
 const mismatch=$('actionMismatch');if(mismatch)mismatch.onclick=()=>{$('task').value='needle_pickup';$('origin').value='all';$('scheme').value='episode';updateFailureOptions();$('failure').value='Execution Mismatch';chooseComparison('action/failure')};
}
function renderFailures(rows){const cols=[...new Map(rows.map(r=>[key(r),r])).values()].sort(detectorOrder),lines=[],out=[];for(const [k,items] of group(rows,r=>[r.task,r.outcome==='success'?'No failure':r.failure_mode,r.failure_origin].join('|'))){const [task,mode,origin]=k.split('|'),cells=[taskName(task),mode,origin];for(const c of cols){const a=items.filter(r=>key(r)===key(c));cells.push(a.length?`${a.filter(r=>r.alarm).length}/${a.length}${unknownCount(a)?' ('+unknownCount(a)+' unknown)':''}${a[0].outcome==='failure'?'; early '+earlyCount(a):''}`:'—')}lines.push(cells);out.push(Object.fromEntries(['Task','Failure mode','Origin',...cols.map(detectorName)].map((h,i)=>[h,cells[i]])))}exports.failures=out.map(r=>({...r,detection_rule:$('detectionRule').value,annotator_id:$('annotator').value}));plainTable('failureTable',['Task','Failure mode','Origin',...cols.map(detectorName)],lines)}
function renderCategories(rows){
 const categories=D.failure_categories?.categories||[];
 $('categoryPanel').hidden=!categories.length;exports.categories=[];
 if(!categories.length)return;
 const detectors=[...new Map(rows.map(r=>[key(r),r])).values()].sort(detectorOrder),tableRows=[];
 $('categoryCharts').innerHTML=categories.map(c=>`<h3>${esc(c.title)}</h3><div id="category_${esc(c.id)}"></div>`).join('');
 for(const c of categories){
  const modes=c.modes.filter(m=>($('task').value==='all'||m.task===$('task').value)&&failureMatch(m));
  const allowed=new Set(modes.map(m=>m.task+'|'+m.failure_mode));
  const total=modes.reduce((n,m)=>n+m.episodes,0),chart=[];
  for(const detector of detectors){
   const items=rows.filter(r=>r.outcome==='failure'&&key(r)===key(detector)&&allowed.has(r.task+'|'+r.failure_mode));
   if(!total)continue;
   const detected=items.filter(r=>r.alarm===true).length,unknown=unknownCount(items),missing=total-items.length;
   const result={...detector,category:c.id,category_title:c.title,task:$('task').value,detected,total,recorded:items.length,unknown,missing,
    FDR:unknown+missing?null:detected/total,FDR_lower:detected/total,FDR_upper:(detected+unknown+missing)/total,
    origin_filter:$('origin').value,failure_mode_filter:$('failure').value,annotator_id:$('annotator').value,detection_rule:$('detectionRule').value};
   chart.push(result);
   exports.categories.push(Object.fromEntries(['variant','method','view','calibration','task','category','category_title','detected','total','recorded','unknown','missing','FDR','FDR_lower','FDR_upper','origin_filter','failure_mode_filter','annotator_id','detection_rule'].map(k=>[k,result[k]])));
   tableRows.push([c.title,detectorName(detector),`${detected}/${total}`,pct(result.FDR),unknown,missing,unknown+missing?`${pct(result.FDR_lower)}–${pct(result.FDR_upper)}`:'—']);
  }
  if(chart.length)drawBars(chart,'category_'+c.id,'FDR');
  else $('category_'+c.id).innerHTML='<p class="empty">No episodes in this category for the current selection.</p>';
 }
 plainTable('categoryTable',['Category','Detector','Detected / total','FDR','Unknown','Missing','Bounds'],tableRows);
}
function renderDistribution(){const options=[...new Map(D.episodes.filter(r=>base(r)&&outcomeMatch(r)).map(r=>[key(r),r])).entries()];fill($('distDetector'),options.map(([k,r])=>[k,detectorName(r)]));fill($('distTask'),Object.keys(TASKS).filter(t=>($('task').value==='all'||t===$('task').value)&&D.episodes.some(r=>r.task===t&&key(r)===$('distDetector').value)).map(t=>[t,taskName(t)]));const chosen=r=>base(r)&&outcomeMatch(r)&&key(r)===$('distDetector').value&&r.task===$('distTask').value;const eps=D.episodes.filter(chosen),groups=group(eps,r=>r.outcome==='success'?'Success · '+r.split:r.failure_mode+' · '+r.failure_origin);const all=eps.map(r=>number(r.score_max)).filter(Number.isFinite);if(!all.length){$('distributionChart').innerHTML='<p class="empty">No finite scores.</p>'}else{const bands=(D.functional_runs||[]).flatMap(r=>r.variants||[]).filter(v=>v.variant===eps[0]?.variant).flatMap(v=>v.calibration||[]),entry=bands.find(r=>r.task===$('distTask').value&&r.method===eps[0]?.method&&r.view===eps[0]?.view&&r.protocol==='paper'),boundary=$('scheme').value==='functional_band'&&entry?number(entry.h):0;let lo=Math.min(Number.isFinite(boundary)?boundary:0,...all),hi=Math.max(Number.isFinite(boundary)?boundary:0,...all);if(lo===hi){lo-=1;hi+=1}const pad=(hi-lo)*.04;lo-=pad;hi+=pad;const x=v=>70+(v-lo)/(hi-lo)*850,y=v=>280-v*230;let svg='<svg viewBox="0 0 1060 '+(360+groups.size*22)+'" xmlns="http://www.w3.org/2000/svg"><rect width="1060" height="100%" fill="white"/>';for(let i=0;i<=4;i++){const v=i/4;svg+=`<line x1="70" x2="920" y1="${y(v)}" y2="${y(v)}" stroke="#e2e9ed"/><text x="55" y="${y(v)+4}" text-anchor="end">${v}</text>`;const t=lo+(hi-lo)*v;svg+=`<text x="${x(t)}" y="305" text-anchor="middle">${fmt(t,2)}</text>`}svg+=`<line x1="${x(boundary)}" x2="${x(boundary)}" y1="40" y2="280" stroke="#698292" stroke-dasharray="5 4"/><text x="${x(boundary)+6}" y="30">${$('scheme').value==='functional_band'?'Calibrated h = '+fmt(boundary):'Threshold 0'}</text>`;let i=0;for(const [name,a]of groups){const vals=a.map(r=>number(r.score_max)),finite=vals.filter(Number.isFinite).sort((a,b)=>a-b),bad=vals.length-finite.length;let path=`M${x(lo)},${y(0)}`;finite.forEach((v,j)=>{path+=` H${x(v)} V${y((j+1)/finite.length)}`});path+=` H${x(hi)}`;svg+=`<path d="${path}" stroke="${palette[i%palette.length]}" fill="none" stroke-width="2"/><text x="70" y="${348+i*22}" fill="${palette[i%palette.length]}">${esc(name)} · n=${a.length}, non-finite=${bad}</text>`;i++}$('distributionChart').innerHTML=svg+'</svg>'}const dist=D.distributions.filter(r=>chosen(r)&&r.score_type===$('scoreType').value);exports.distributions=dist;plainTable('distributionTable',['Split','Group','Origin','Episodes','Scores','Unsupported','Mean','SD','p05','Median','p95','Max'],dist.map(r=>[r.split,r.failure_mode||'Success',r.failure_origin,r.n_episodes,r.n_scores,r.n_unsupported_scores,...['mean','std','p05','median','p95','max'].map(k=>fmt(number(r[k])))]))}
function latencyStats(a,online=false){const status=s=>a.filter(r=>r.status===s).length,delays=a.filter(r=>r.status==='detected'&&!(online&&r.late_alarm)).map(r=>number(online?r.delay_s:r.detection_delay_s));return{n:a.length,detected:status('detected'),missed:status('missed'),early:status(online?'pre_onset_evidence':'pre_onset_alarm'),other:a.filter(r=>!['detected','missed',online?'pre_onset_evidence':'pre_onset_alarm'].includes(r.status)).length,copied:a.filter(r=>r.copied_from_annotator).length,median:quantile(delays,.5),p95:quantile(delays,.95)}}
function onlinePlans(){return D.plans.flatMap(r=>r.requested_detector_views.map(view=>({...r,view}))).filter(r=>base(r)&&failureMatch(r))}
function renderLatency(){
 $('sharedLatencyPanel').hidden=$('preset').value==='pairwise';$('pairwiseLatencyPanel').hidden=$('preset').value==='shared';
 renderPairwiseLatency();renderSharedLatency();drawLatency();const lat=D.latencies.filter(r=>base(r)&&failureMatch(r)&&annotationMatch(r)),off=[];for(const a of group(lat,r=>r.task+'|'+key(r)).values())off.push({...a[0],...latencyStats(a)});exports.latency=off.map(r=>({task:r.task,variant:r.variant,method:r.method,view:r.view,calibration:r.calibration,annotator:r.annotator,onset_view:r.onset_view,...latencyStats(lat.filter(x=>x.task===r.task&&key(x)===key(r)))}));plainTable('latencyTable',['Task','Detector','Onset view','Annotator','Failure episodes','Detected','Early alarm','Missed','Other / missing label','Copied labels','Median (s)','p95 (s)'],off.map(r=>[taskName(r.task),detectorName(r),VIEWS[r.onset_view],r.annotator,r.n,r.detected,r.early,r.missed,r.other,r.copied,fmt(r.median),fmt(r.p95)]));
 const completed=new Map(D.online.map(t=>[t.trial_id,t])),planned=onlinePlans(),coverage=[];const candidates=new Map(D.episodes.filter(r=>base(r)&&r.split==='test'&&r.outcome==='failure'&&failureMatch(r)).map(r=>[r.task+'|'+key(r),r]));for(const [k,r]of candidates){const p=planned.filter(t=>t.task+'|'+key(t)===k),done=p.filter(t=>completed.has(t.trial_id)),latencies=done.flatMap(t=>completed.get(t.trial_id).latencies.filter(l=>l.view===r.view&&annotationMatch(l)));const s=latencyStats(latencies,true),alarms=latencies.filter(l=>l.alarm).length;coverage.push({task:r.task,variant:r.variant,method:r.method,view:r.view,planned:p.length,completed:done.length,alarms,misses:latencies.filter(l=>!l.alarm).length,retention:latencies.length?alarms/latencies.length:null,...s,status:$('scheme').value!=='episode'?'Not measured for this calibration':p.length?done.length===p.length?'Complete':'Not measured / partial':r.method==='ace'?'No selected failures':'Not scheduled / no selected failures'})}exports.online=coverage;plainTable('onlineTable',['Task','Detector','Status','Completed / selected','Online alarms','Online misses','Alarm retention','Early evidence','Other / missing label','Copied labels','Median delay (s)','p95 (s)'],coverage.map(r=>[taskName(r.task),detectorName(r),r.status,`${r.completed}/${r.planned}`,r.completed?r.alarms:'—',r.completed?r.misses:'—',pct(r.retention),r.completed?r.early:'—',r.completed?r.other:'—',r.completed?r.copied:'—',fmt(r.median),fmt(r.p95)]));
 const selectedTrials=new Set(planned.map(r=>r.trial_id)),trials=D.online.filter(r=>selectedTrials.has(r.trial_id)),runtime=[];for(const a of group(trials,r=>[r.task,r.variant,r.method,r.input_views.join('+')].join('|')).values()){const t=a[0],events=a.flatMap(t=>t.timings);runtime.push({task:t.task,variant:t.variant,method:t.method,input_views:t.input_views.join('+'),trials:a.length,processed:a.reduce((s,t)=>s+t.processed,0),skipped:a.reduce((s,t)=>s+t.skipped,0),compute_median_s:quantile(events.map(e=>e[0]),.5),compute_p95_s:quantile(events.map(e=>e[0]),.95),source_age_p95_s:quantile(events.map(e=>e[1]),.95),encoder_median_s:quantile(events.map(e=>e[2]),.5),detector_median_s:quantile(events.map(e=>e[3]),.5),ode_steps_mean:mean(events.map(e=>e[4])),ode_evaluations_mean:mean(events.map(e=>e[5]))})}exports.runtime=runtime;plainTable('runtimeTable',['Task','Variant','Method','Input views','Trials','Processed','Skipped','Compute median (s)','Compute p95 (s)','Source age p95 (s)','Encoder median (s)','Detector median (s)','ODE steps / score','ODE evals / score'],runtime.map(r=>[taskName(r.task),runs.get(r.variant)?.title,methodName(r.method),r.input_views,r.trials,r.processed,r.skipped,...['compute_median_s','compute_p95_s','source_age_p95_s','encoder_median_s','detector_median_s','ode_steps_mean','ode_evaluations_mean'].map(k=>fmt(r[k]))]))}
function renderEpisodes(rows){const latencyIndex=new Map(D.latencies.filter(r=>base(r)&&annotationMatch(r)).map(r=>[[r.task,key(r),r.episode_index].join('|'),r])),onlineIndex=new Map(D.online.flatMap(t=>t.latencies.filter(annotationMatch).map(l=>[[l.task,key(l),l.episode_index].join('|'),{...l,path:t.path}])));visibleEpisodes=rows.map(r=>{const k=[r.task,key(r),r.episode_index].join('|'),l=latencyIndex.get(k),o=$('scheme').value==='episode'?onlineIndex.get(k):null;return{...r,annotator:$('annotator').selectedOptions[0]?.textContent,offline_latency_status:l?.status,offline_delay_s:l?.detection_delay_s,onset_raw_frame:l?.onset_raw_frame,copied_from_annotator:l?.copied_from_annotator,online_status:o?.status,online_delay_s:o?.delay_s,online_path:o?.path||r.trace_path}});exports.episodes=visibleEpisodes;fill($('episodePage'),Array.from({length:Math.max(1,Math.ceil(rows.length/150))},(_,i)=>[String(i),`${i+1} / ${Math.max(1,Math.ceil(rows.length/150))}`]));renderEpisodePage()}
function renderEpisodePage(){const start=Number($('episodePage').value)*150;table('episodeTable',['Task','Detector','Episode','Outcome','Failure mode','Origin','Episode alarm','Episode score','Causal latency status','Onset frame','Copied label','Offline delay (s)','Online status','Online delay (s)','Trace'],visibleEpisodes.slice(start,start+150).map(r=>[esc(taskName(r.task)),esc(detectorName(r)),r.episode_index,esc(r.outcome),esc(r.failure_mode||'Success'),esc(r.failure_origin),r.alarm===null?'Unknown':r.alarm?'Yes':'No',fmt(number(r.score_max)),esc(r.offline_latency_status||'—'),r.onset_raw_frame??'—',r.copied_from_annotator?'Yes':'No',fmt(r.offline_delay_s),esc(r.online_status||'—'),fmt(r.online_delay_s),r.online_path?link(r.online_path,'JSON'):'—']))}
function renderProvenance(){$('figureDownloads').innerHTML=(D.figure_exports||[]).length?'<p class="note">Matplotlib figures for the default all-task, full-episode comparison (use the plot buttons for current filters): '+D.figure_exports.map(f=>esc(f.name)+' '+Object.entries(f.paths).map(([ext,path])=>link(path,ext.toUpperCase())).join(' / ')).join(' · ')+'</p>':'';table('runTable',['Variant','Created (UTC)','Alpha','Label snapshot hash','Original report','Calibration','Labels'],D.runs.map(r=>[esc(r.title),esc(r.created),r.alpha,esc(r.labels_sha256.slice(0,12)),link(r.path+'/report.html','HTML'),link(r.path+'/calibration.csv','CSV'),link(r.path+'/labels_snapshot.yaml','YAML')]));$('protocols').textContent=JSON.stringify({offline:D.runs,online:D.online_runs,functional:D.functional_runs},null,2);const detectors=D.episodes.filter(base);const rows=D.settings.filter(r=>($('task').value==='all'||r.task===$('task').value)&&detectors.some(d=>d.variant===r.variant&&d.task===r.task&&(d.method===r.method||(d.method.startsWith('wm_nc_')&&(r.method==='wm_nc'||d.method.includes(r.method))))&&(d.view==='both'||r.views==='both'||r.views.split(',').includes(d.view))));exports.settings=rows;plainTable('settingsTable',['Task','Variant','Method','Views','K','L','Solver','Steps / solve','Solves / score','Steps / score','Field evals / score','Measured steps mean','Measured evals mean','Samples','Batch','Seed'],rows.map(r=>[taskName(r.task),runs.get(r.variant)?.title,methodName(r.method),r.views,r.K,r.L,r.ode_solver,r.ode_steps_per_solve,r.ode_solves_per_score,r.ode_steps_per_score,r.vector_field_evals_per_score,r.measured_steps_mean,r.measured_evaluations_mean,r.samples_per_score,r.batch_size,r.seed]));table('archiveTable',['Kind','Original source','Files','Archive SHA-256','Files & manifest'],D.archives.map(r=>[esc(r.kind),esc(r.source),r.files,esc(r.sha256.slice(0,16)),link(r.path+'/archive_manifest.json','Manifest')]))}
function renderVLM(){
 const runs=D.vlm_runs||[],match=r=>($('vlmTask').value==='all'||r.task===$('vlmTask').value)&&($('vlmView').value==='all'||r.view===$('vlmView').value);
 const rows=runs.flatMap(v=>[...v.summary,...v.overall].map(r=>({...r,protocol:v.protocol_id}))).filter(match);
 $('vlmNote').textContent=runs.map(v=>v.note).join(' ');
 $('vlmStatus').textContent=runs.map(v=>`${v.title}: ${v.complete}/${v.planned} complete; ${v.indeterminate} indeterminate; ${v.unattempted} unattempted.`).join(' ');
 $('vlmLinks').innerHTML=runs.map(v=>link(v.path+'/index.html','Original report')+' · '+link(v.path+'/report.json','Source v1 JSON')+' · '+link(v.path+'/summary.csv','CSV')+' · '+link(v.path+'/plan.json','Frozen source protocol')+' · '+link((D.interpreted_results?.path||'analyses/2026-10-16_post_onset_v2')+'/vlm_summary.csv','Current onset-aware CSV')).join('<br>');
 const rate=(r,m,t)=>`${r[m+'_'+t+'_count']}/${r[m+'_denominator']} (${pct(r[m+'_'+t])})`;
 $('vlmLinks').innerHTML+=' · '+link('reviewer_evidence.html#vlm_protocols','V9 / V12 / V13 prompt comparison');
 const selectedRows=VLM_EPISODES.filter(match).map(assessmentRow),selectedStats=metrics(selectedRows);
 exports.vlm=selectedStats;
 plainTable('vlmTable',['Task','View','FAR','Detected / failure','FDR','Early alarms','Coverage / bounds'],selectedStats.map(r=>[taskName(r.task),VIEWS[r.view],pct(r.FAR),`${r.tp}/${r.pos}`,pct(r.FDR),r.early,coverageText(r)]));
 plainTable('vlmOriginalTable',['Task','View','Any-time FAR','Post-onset FDR incl. late','On-time FAR','On-time post-onset FDR','Other early alarms','Any pre-onset evidence','Sustained pre-onset','Late alarms'],rows.map(r=>[taskName(r.task),VIEWS[r.view],rate(r,'FAR','any'),rate(r,'FDR','any'),rate(r,'FAR','on_time'),rate(r,'FDR','on_time'),r.pre_onset_alarms,r.pre_onset_evidence,r.sustained_pre_onset_count,r.late_alarms]));
 plainTable('vlmLatencyTable',['Task','View','Post-onset n','Mean ± SD (s)','Detected by 5s / annotated failures','On-time misses / failures','Complete / planned','Indeterminate','Unattempted'],rows.map(r=>[taskName(r.task),VIEWS[r.view],r.conditional_delay_n,fmt(r.conditional_delay_mean_s,2)+' ± '+fmt(r.conditional_delay_sd_s,2),`${r.detected_by_5s_count}/${r.detected_by_5s_denominator}`,`${r.on_time_misses}/${r.FDR_denominator}`,`${r.complete}/${r.planned}`,r.indeterminate,r.unattempted]));
}
function renderStudies(){
 const studies=D.studies||[];fill($('studySelect'),studies.map(s=>[s.id,s.title]));
 const s=studies.find(s=>s.id===$('studySelect').value);
 if(!s){$('studyScope').textContent='No studies are registered in the results catalog.';return}
 const calibration=s.kind==='calibration_sensitivity';
 $('studyHeading').textContent=calibration?'Calibration split sensitivity':'Score distributions';
 $('studyShortcuts').hidden=!calibration;
 updateComparisonNavigation();
 const tasks=[...new Set(s.summary.map(r=>r.task))],views=[...new Set(s.summary.map(r=>r.view))];
 fill($('studyTask'),[...Object.entries(TASKS).filter(([t])=>tasks.includes(t)),['all','All tasks']]);
 fill($('studyView'),[...Object.entries(VIEWS).filter(([v])=>views.includes(v)),['all','All views']]);
 $('studyTitle').textContent=s.title;$('studyScope').textContent=s.date+' · '+s.status+' · '+(s.scope||'')+' FDR follows the protocol documented for this study; historical studies remain explicitly labeled.';
 $('studyHighlights').innerHTML='<ul>'+s.highlights.map(t=>'<li>'+esc(t)+'</li>').join('')+'</ul>';
 $('studyPlotNote').textContent=s.plot_note||'';$('studyNote').textContent=s.note||'';
 $('studyDownloads').innerHTML=Object.entries(s.links).map(([label,path])=>link(path,label)).join(' · ');
 const match=r=>($('studyTask').value==='all'||r.task===$('studyTask').value)&&($('studyView').value==='all'||r.view===$('studyView').value);
 const rows=s.summary.filter(match);exports.study=rows.map(r=>({study:s.id,...r}));
 if(s.kind==='score_distributions'){
   const maxima=rows.filter(r=>r.level==='episode_maximum');exports.study=maxima.map(r=>({study:s.id,...r,iqr:r.q75-r.q25}));
   plainTable('studyTable',['Task','View','Cohort','Episodes','Median episode maximum','Q1','Q3','IQR'],maxima.map(r=>[taskName(r.task),VIEWS[r.view],r.cohort.replace(/_/g,' '),r.n_episodes,fmt(r.median,2),fmt(r.q25,2),fmt(r.q75,2),fmt(r.q75-r.q25,2)]));
 }
 else {const range=(r,m)=>r[m+'_mean']==null?'—':`${(100*r[m+'_mean']).toFixed(1)} ± ${(100*r[m+'_sd']).toFixed(1)} [${(100*r[m+'_p05']).toFixed(1)}, ${(100*r[m+'_p95']).toFixed(1)}]`;
 plainTable('studyTable',['Task','Detector / variant','View','Protocol','Splits','Original FAR / FDR','FAR mean ± SD [5–95%] %','FDR mean ± SD [5–95%] %','AUROC mean ± SD [5–95%] %'],rows.map(r=>[taskName(r.task),methodName(r.method)+' / '+(runs.get(r.variant)?.title||r.variant),VIEWS[r.view],r.calibration,r.n_splits,pct(r.original_FAR)+' / '+pct(r.original_FDR),range(r,'FAR'),range(r,'FDR'),range(r,'episode_AUROC')]));}
 const figures=s.figures.filter(match);
 $('studyFigures').innerHTML=figures.length?figures.map(f=>'<div class="study-figure"><h3>'+esc(taskName(f.task)+' · '+VIEWS[f.view]+' · '+f.kind.replace(/_/g,' '))+'</h3><p>'+Object.entries(f.downloads).map(([label,path])=>link(path,label)).join(' · ')+'</p><img loading="lazy" src="'+f.image+'" alt="'+esc(s.title+' '+f.task+' '+f.view+' '+f.kind)+'"></div>').join(''):'<p class="empty">No figure for this selection. See the selected numeric results or the full study report.</p>';
}
function updateFailureOptions(){const candidates=D.episodes.filter(r=>r.outcome==='failure'&&($('task').value==='all'||r.task===$('task').value)&&($('origin').value==='all'||r.failure_origin===$('origin').value));fill($('failure'),[['all','All failure types'],...[...new Set(candidates.map(r=>r.failure_mode))].sort().map(x=>[x,x])])}
function refresh(){
 $('detectionNote').textContent=(postMode()?'FDR: any crossing at/after onset counts, including after an early alarm. Unknown episodes stay in the denominator; no point FDR is shown when any are unknown. ':'FDR: any alarm during the scored episode. ')+
 'Early and post-onset counts may overlap. Onsets follow the selected annotator: fixed → fixed, wrist → wrist, combined → earlier view (both labels required). FAR and whole-episode AUROC remain trajectory-wide. Latency always uses the first completed post-onset-source alarm. Sustained pre-onset runs are separate from other early alarms, except for simulated induced failures, whose pre-onset alarms remain early. Early-stopped streaming trials cannot establish later detection. Saved study and original VLM-detail tables retain their explicitly named definitions.';

 updateComparisonNavigation();
 $('modelSelectionNotes').hidden=!['seed','all'].includes($('preset').value)&&activePage!=='provenance';
 $('qwenProtocolNote').hidden=!['main','all'].includes($('preset').value);
 const incompleteOnline=D.online_runs.filter(r=>(r.coverage_status||r.state.status)!=='complete');
 const latestIncomplete=[...incompleteOnline].sort((a,b)=>(b.state.finished||b.state.started||'').localeCompare(a.state.finished||a.state.started||''))[0];
 $('onlineRunStatus').hidden=false;
 if(latestIncomplete){const r=latestIncomplete,s=r.state;$('onlineRunStatus').innerHTML='<strong>Online benchmark incomplete</strong><p>'+esc(r.title||r.source)+': '+esc(s.status)+' · '+s.completed_trials+' / '+s.total_trials+' trials recorded. Offline results below are available; missing online measurements are not zero latency. '+link(r.path+'/report.html','Run report')+'</p><details><summary>Recorded reason and other incomplete runs</summary><p>'+esc(s.error||'See saved run state')+'</p><ul>'+incompleteOnline.map(x=>'<li>'+esc(x.title||x.source)+': '+esc(x.state.status)+' · '+x.state.completed_trials+' / '+x.state.total_trials+'</li>').join('')+'</ul></details>'}
 else {const c=D.online_coverage||{completed_trials:D.online.length,total_trials:D.plans.length};$('onlineRunStatus').innerHTML='<strong>Previously scheduled online trials complete</strong><p>'+c.completed_trials+' / '+c.total_trials+' selected trials recorded across original and recovery runs. Successes and offline misses were excluded. This does not imply online coverage for newly added offline-only variants.</p>';const historical=[...D.online_runs.filter(r=>r.state.status!=='complete'),...(D.historical_online_runs||[])];if(historical.length)$('onlineRunStatus').innerHTML+='<details><summary>Historical attempts</summary><ul>'+historical.map(r=>'<li>'+esc(r.title||r.source)+': '+esc(r.state.status)+' · '+r.state.completed_trials+' / '+r.state.total_trials+' in that attempt. '+esc(r.reason||'Its planned trials are now covered by subsequent runs.')+' '+link(r.path+'/report.html','Original report')+'</li>').join('')+'</ul></details>';}
 const functional=$('scheme').value==='functional_band';
 const pending=(D.functional_runs||[]).filter(r=>!['complete','partial'].includes(r.state.status));
 $('functionalStatus').innerHTML=functional?('Expert-derived functional bands. '+(D.functional_runs||[]).map(r=>esc(r.state.status)+(['complete','partial'].includes(r.state.status)?' · '+link(r.path+'/report.html','Comparison report'):' · '+(r.state.completed_jobs??0)+' / '+(r.state.total_jobs??80)+' scorer jobs')).join(' · ')+' Calibrated with all 19 original successful rollouts; no online timings have been rerun for this protocol.'):(pending.length?'Functional-band comparison is still running: '+pending.map(r=>(r.state.completed_jobs??0)+' / '+(r.state.total_jobs??80)+' scorer jobs').join(' · ')+'. Status is a snapshot at page build time; the comparison will appear after completion and a page reload.':'');
 const rows=comparisonEpisodes().filter(r=>r.split==='test'&&base(r)&&outcomeMatch(r)).map(assessmentRow),unique=[...new Map(rows.map(r=>[r.task+'|'+r.episode_index,r])).values()],p=unique.filter(r=>r.outcome==='failure').length,n=unique.length-p,plans=onlinePlans(),done=new Set(D.online.map(t=>t.trial_id)),ids=new Set(plans.map(t=>t.trial_id)),complete=[...ids].filter(i=>done.has(i)).length;const card=(v,label)=>`<div class="card"><strong>${esc(v)}</strong><span class="muted">${esc(label)}</span></div>`;$('cards').innerHTML=card(p,'Selected failure episodes')+card(n,'Successful test episodes retained')+card(new Set(rows.map(key)).size,'Detector / view comparisons')+card(`${complete} / ${ids.size}`,'Online trials completed / selected');
 if(activePage==='roc')renderROC(rows);if(activePage==='compare')renderOverview(rows);if(activePage==='failure'){renderCategories(rows);renderFailures(rows)}if(activePage==='distribution')renderDistribution();if(activePage==='latency')renderLatency();if(activePage==='episodes')renderEpisodes(rows);if(activePage==='provenance')renderProvenance();if(activePage==='studies')renderStudies();if(activePage==='vlm')renderVLM();$('status').textContent=`${rows.length.toLocaleString()} test-episode records in this selection · ${D.archives.length} preserved archives · built ${new Date(D.created).toLocaleString()}`;document.body.dataset.ready='true';document.body.dataset.revision=String(++revision)}
function download(filename,text,type){const u=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=u;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(u),5000)}
function exportCSV(name){const rows=(exports[name]||[]).map(r=>({offline_protocol:D.offline_protocol?.id||'legacy',...r}));if(!rows.length)return;const cols=Object.keys(rows[0]),quote=v=>{let s=String(v??'');if(/^[=+@]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"'};download('fomofd_'+(D.offline_protocol?.id||'legacy')+'_'+name+'.csv',[cols,...rows.map(r=>cols.map(k=>r[k]))].map(r=>r.map(quote).join(',')).join('\r\n'),'text/csv;charset=utf-8')}
for(const [k,v]of Object.entries(TASKS))option($('task'),k,v);for(const [k,v]of Object.entries(D.annotators))option($('annotator'),k,v);
$('built').textContent='Built '+new Date(D.created).toLocaleString();
for(const a of document.querySelectorAll('a[href]')){
 const [file,anchor]=a.getAttribute('href').split('#');
 const kind={'summary.md':'markdown','summary.yaml':'yaml','reviewer_evidence.html':'evidence'}[file];
 if(kind){a.dataset.reportLink=kind;a.dataset.reportAnchor=anchor?'#'+anchor:'';}
}
updateOfflineProtocol();
for(const id of ['task','origin','failure','scheme','view','annotator','detectionRule','metric','latencySource','latencyStyle','distDetector','distTask','scoreType','rocFar','rocRange'])$(id).addEventListener('change',()=>{if(id==='task'||id==='origin')updateFailureOptions();refresh()});
$('preset').addEventListener('change',()=>{setMethods();if(['K','L','hyper','action'].includes($('preset').value)&&activePage!=='roc')document.querySelector('[data-page="compare"]').click();else refresh()});$('methods').addEventListener('change',e=>{sharedSearchResult=null;if(e.target.checked)selected.add(e.target.value);else selected.delete(e.target.value);refresh()});
$('selectAll').onclick=()=>{sharedSearchResult=null;for(const c of $('methods').querySelectorAll('input')){c.checked=true;selected.add(c.value)}refresh()};$('selectNone').onclick=()=>{sharedSearchResult=null;selected.clear();for(const c of $('methods').querySelectorAll('input'))c.checked=false;refresh()};
$('tabs').onclick=e=>{const page=e.target.dataset.page;if(!page)return;if(['studies','vlm'].includes(page)){chooseComparison(page);return}if(['studies','vlm'].includes($('preset').value)){$('preset').value='main';setMethods();history.replaceState(null,'','#main')}showPage(page);history.replaceState(null,'','#'+$('preset').value+'/'+page);refresh()};$('episodePage').onchange=renderEpisodePage;
document.addEventListener('click',e=>{const a=e.target.closest('[data-comparison]');if(!a||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey)return;e.preventDefault();chooseComparison(a.dataset.comparison)});
window.addEventListener('hashchange',()=>chooseComparison(location.hash.slice(1)||'main',false));
$('reset').onclick=()=>{for(const id of ['preset','task','origin','scheme','view','annotator','detectionRule','latencySource','latencyStyle'])$(id).selectedIndex=0;$('rocFar').value='0.05';$('rocRange').value='0.2';updateFailureOptions();$('failure').value='all';setMethods();history.replaceState(null,'','#main');refresh()};
document.addEventListener('click',e=>{const t=e.target;if(t.dataset.export)exportCSV(t.dataset.export);if(t.dataset.svg){const svg=$(t.dataset.svg).querySelector('svg');if(svg)download('fomofd_'+t.dataset.svg+'.svg',svg.outerHTML,'image/svg+xml')}});
option($('preset'),'shared','Shared online latency');
option($('preset'),'pairwise','Pairwise online latency');
for(const id of ['pairFamily','pairTiming','pairMinimum','pairPlot'])$(id).onchange=renderPairwiseLatency;
for(const id of ['fusionLevel','fusionMetric'])$(id).onchange=renderFusionLevels;
option($('preset'),'studies','Additional studies');
option($('preset'),'vlm','Qwen VLM');
for(const [k,v] of Object.entries(TASKS))option($('vlmTask'),k,v);
for(const id of ['vlmTask','vlmView'])$(id).onchange=renderVLM;
$('modelSelectionNotes').textContent=(D.model_selection_notes||[]).join(' ');
if(D.model_selection_record)$('modelSelectionNotes').innerHTML+=' '+link(D.model_selection_record,'Selection record and archive paths');
$('studySelect').onchange=()=>{history.replaceState(null,'','#study/'+$('studySelect').value);renderStudies()};
for(const id of ['studyTask','studyView'])$(id).onchange=renderStudies;
$('sharedSearch').onclick=findSharedGroups;
$('sharedMinimum').onchange=findSharedGroups;
for(const n of [150,50])$('shared'+n).onclick=()=>{$('sharedMinimum').value=n;findSharedGroups()};
$('sharedPool').onclick=()=>{setMethods();refresh();findSharedGroups()};
$('sharedGroups').onclick=e=>{const b=e.target.closest('[data-shared-group]');if(b)applySharedGroup(Number(b.dataset.sharedGroup))};
if([...$('preset').options].some(o=>o.value===location.hash.slice(1)))$('preset').value=location.hash.slice(1);
if(D.policy_success_rates){const p=D.policy_success_rates,o=p.overall;$('policySuccessPanel').hidden=false;plainTable('policySuccessTable',['Task','Environment','Successes / trials','Success rate (%)'],[...p.rows.map(r=>[taskName(r.task),r.environment,`${r.successes}/${r.trials}`,fmt(100*r.success_rate,1)]),['Overall','Pooled',`${o.successes}/${o.trials}`,fmt(100*o.success_rate,2)]])}
setMethods();updateFailureOptions();chooseComparison(location.hash.slice(1)||'main',false);
})().catch(error=>{document.body.dataset.error=String(error);document.getElementById('status').textContent='Unable to load results: '+error});
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archive-logs', action='store_true', help='Copy and verify existing reports/traces; originals remain intact')
    parser.add_argument('--catalog', type=Path, default=CATALOG)
    parser.add_argument('--archive-root', type=Path, default=ARCHIVE)
    parser.add_argument('--figures', action='store_true', help='Also export default all-task plots with Matplotlib as SVG/PDF/PNG')
    parser.add_argument('--refresh-presentation', action='store_true', help='Reuse the existing viewer snapshot; update summaries, pairwise comparisons and HTML. Add --figures to refresh exported plots.')
    parser.add_argument('--refresh-online', action='store_true', help='Load the catalogued replay archives into the existing snapshot and refresh reports without rewriting offline scores/sensitivity studies.')
    parser.add_argument('--refresh-studies', action='store_true', help='Refresh auxiliary score-distribution studies only; preserve all primary results')
    args = parser.parse_args()
    root = args.archive_root.resolve()
    if args.refresh_studies:
        if args.archive_logs or args.refresh_presentation or args.refresh_online or args.figures:
            parser.error('--refresh-studies cannot be combined with other refresh/build options')
        refresh_studies(args.catalog.resolve(), root)
        return
    if args.refresh_presentation or args.refresh_online:
        if args.archive_logs:
            parser.error('--refresh-presentation does not archive runs')
        refresh_presentation(args.catalog.resolve(), root, figures=args.figures, online=args.refresh_online)
        return
    if args.archive_logs:
        archive_logs(root)
    build(args.catalog.resolve(), root, figures=args.figures)


if __name__ == '__main__':
    main()
