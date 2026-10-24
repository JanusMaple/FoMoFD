"""Read completed auxiliary studies for the main dashboard without recomputing.

Figures are embedded so the main HTML can be viewed without a local web server.
Studies remain separate from primary alarm metrics and from one another.
"""
import base64
import hashlib
import json
from pathlib import Path

import yaml


def load_studies(catalog, root, entries=None):
    root = Path(root).resolve()
    if entries is None:
        entries = yaml.safe_load(Path(catalog).read_text()).get('studies', [])
    result = []
    if len({e['id'] for e in entries}) != len(entries):
        raise ValueError('Duplicate study IDs in results catalog')
    for entry in entries:
        directory = (root/entry['path']).resolve()
        if not directory.is_relative_to(root):
            raise ValueError('Study path must be inside benchmark_results')
        item = dict(entry, status='unavailable', links={}, figures=[], summary=[], overall=[], highlights=[], source_sha256={})
        def read(name):
            path = directory/name
            payload = path.read_bytes()
            item['source_sha256'][str(path.relative_to(root))] = hashlib.sha256(payload).hexdigest()
            return json.loads(payload)
        def link(label, name):
            if (directory/name).is_file():
                item['links'][label] = str((directory/name).relative_to(root))
        if not (directory/'state.json').is_file():
            item['note'] = 'No completed local output is available.'
            result.append(item)
            continue
        item['state'] = read('state.json')
        item['status'] = item['state']['status']
        if item['status'] != 'complete':
            item['note'] = 'The study is incomplete; no statistics are merged.'
            result.append(item)
            continue
        if entry['kind'] == 'score_distributions':
            data, selections = read('results.json'), read('episode_selections.json')
            item['summary'], item['note'] = data['summary'], data['note']
            item['dataset_validity_applied'] = data.get('dataset_validity_applied', False)
            item['episode_quantile_method'] = data.get('episode_quantile_method', 'inverted_cdf')
            item['episode_counts'] = {task: {cohort: len(ids) for cohort, ids in cohorts.items()}
                                      for task, cohorts in selections.items()}
            item['scope'] = 'Main K4/L4 WM-NC · four tasks · fixed and wrist · raw window scores and episode maxima'
            item['plot_note'] = 'Probability-density histograms: shared bin edges and unit area per cohort. No KDE. Blue = expert validation; green = policy successes. Fourth panel shows episode length versus maximum.'
            expert = {(r['task'], r['view']): r for r in data['summary'] if r['cohort'] == 'expert_validation' and r['level'] == 'episode_maximum'}
            policy = {(r['task'], r['view']): r for r in data['summary'] if r['cohort'] == 'policy_success' and r['level'] == 'episode_maximum'}
            if set(expert) != set(policy):
                raise ValueError('Study lacks paired expert/policy summaries')
            lower = sum(policy[k]['median'] < e['median'] for k, e in expert.items())
            item['highlights'] = [f'Policy successes have lower median episode maxima in {lower}/{len(expert)} task/view comparisons.',
                                  'Expert validation: 25 episodes per simulation task, 10 per real task. Policy success: 39 per task (19 calibration + 20 test).']
            if item['dataset_validity_applied']:
                item['highlights'].insert(0, 'Episode-maximum WM-NC scores: median and IQR; linear quantiles; frozen visual-validity correction applied.')
            elif entry.get('historical'):
                item['highlights'].insert(0, 'Historical distribution study — uncorrected episode 5; preserved separately.')
            for label, name in [('HTML', 'index.html'), ('Markdown', 'report.md'), ('JSON', 'results.json'),
                                ('Episode median / IQR CSV', 'episode_maximum_summary.csv'),
                                ('Distributions CSV', 'distribution_summary.csv'), ('Bins CSV', 'density_histograms.csv'),
                                ('Episode maxima CSV', 'episodes.csv'),
                                ('Episode IDs', 'episode_selections.json')]:
                link(label, name)
            suffix, folder = '_density', ''
        elif entry['kind'] == 'calibration_sensitivity':
            meta = read('manifest.json')
            item['summary'] = read('summary.json')
            item['overall'] = read('overall_summary.json') if (directory/'overall_summary.json').is_file() else []
            item['note'], item['n_splits'] = meta['note'], meta['n_splits']
            item['scope'] = f'{meta["case_count"]} checkpoint/view cases · {meta["n_splits"]} shared 19/20 splits per task · episode and functional-band thresholds'
            item['plot_note'] = 'Each split reuses cached scores and frozen templates. Ranges describe overlapping resamples, not confidence intervals or independent experiments. A mean FAR near 5% follows rank symmetry and does not validate calibration.'
            item['highlights'] = [f'{meta["n_splits"]} reproducible splits retained; {meta["protocol_case_count"]} original protocol/case checks passed.']
            for r in item['overall']:
                if r['view'] == 'wrist' and r['calibration'] == 'episode':
                    item['highlights'].append(f'Wrist WM-NC episode FDR across splits: {100*r["FDR_mean"]:.1f}% ± {100*r["FDR_sd"]:.1f} percentage points.')
            if entry.get('context'):
                item['highlights'].append(entry['context'])
            for label, name in [('HTML', 'report.html'), ('Markdown', 'summary.md'), ('Summary JSON', 'summary.json'),
                                ('Summary CSV', 'summary.csv'), ('Split IDs', 'splits.json'),
                                ('Offline metrics', 'offline_metrics.html'), ('Offline metrics Markdown', 'offline_metrics.md')]:
                link(label, name)
            suffix, folder = '_sensitivity', 'figures'
        else:
            raise ValueError(f'Unsupported study kind: {entry["kind"]}')
        pairs = sorted({(r['task'], r['view']) for r in item['summary']})
        for task, view in pairs:
            for ending in [suffix] + (['_paired'] if entry['kind'] == 'calibration_sensitivity' else []):
                stem = directory/folder/f'{task}_{view}{ending}'
                image = stem.with_suffix('.png')
                if not image.is_file():
                    if ending == '_paired' or view == 'both':
                        continue
                    raise FileNotFoundError(f'Completed study figure missing: {image}')
                payload = image.read_bytes()
                path = str(image.relative_to(root))
                item['source_sha256'][path] = hashlib.sha256(payload).hexdigest()
                item['figures'].append({'task': task, 'view': view, 'kind': ending.lstrip('_'), 'path': path,
                                        'image': 'data:image/png;base64,'+base64.b64encode(payload).decode(),
                                        'downloads': {ext.upper(): str(stem.with_suffix('.'+ext).relative_to(root))
                                                      for ext in ('png', 'pdf', 'svg') if stem.with_suffix('.'+ext).is_file()}})
        result.append(item)
    return result


def compact(studies):
    """No embedded images or per-case metric duplication in the YAML overview."""
    fields = ('id', 'title', 'date', 'kind', 'status', 'path', 'scope', 'links', 'highlights', 'plot_note', 'context',
              'historical', 'dataset_validity_applied', 'episode_quantile_method', 'source_sha256')
    return [{k: s[k] for k in fields if k in s} for s in studies]
