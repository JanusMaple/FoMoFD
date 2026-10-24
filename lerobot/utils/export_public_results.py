#!/usr/bin/env python3
"""Export current result tables with exact counts and documented protocols."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

from public_results import current_rows, current_result

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / 'logs/benchmarks/result_tables/details'
SITE = ROOT / 'fomofd/tasks/review_site'
PUBLIC = ROOT / 'docs/results'


def load_site(site=SITE):
    data = json.loads((site / 'data/report.js').read_text().removeprefix('window.REVIEW_DATA=').removesuffix(';\n'))
    for path in sorted((site / 'data').glob('chunk_*.js')):
        match = re.fullmatch(r'window\.REVIEW_DATA\.(\w+)\.push\(\.\.\.(.*)\);\n', path.read_text(), re.S)
        if not match:
            raise ValueError(f'Unexpected data asset: {path.name}')
        data[match[1]].extend(json.loads(match[2]))
    for key in ('offline', 'online', 'pairs'):
        fields = data['pair_fields'] if key == 'pairs' else data['row_fields']
        data[key] = [dict(zip(fields, row)) for row in data[key]]
    return data


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty public table: {path.name}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_private(name):
    with (PRIVATE / name).open() as f:
        return list(csv.DictReader(f))


def export(site=SITE, output=PUBLIC):
    data = load_site(site)
    output.mkdir(parents=True, exist_ok=True)
    details = output / 'details'
    details.mkdir(exist_ok=True)
    tables = {}
    for key in ('offline', 'online'):
        tables[f'{key}_by_failure.csv'] = current_rows(read_private(f'{key}_by_failure.csv'), online=key == 'online')
    tables['pairwise_latency_summary.csv'] = data['pairs']
    safe_episode_fields = ('variant task method view calibration episode_index split outcome failure_mode failure_origin '
                           'status alarm post_alarm late_alarm any_pre_onset_alarm sustained_pre_onset_alarm '
                           'detection_delay_s onset_time_s detection_time evaluation_protocol replay_scope').split()
    tables['online_episode_outcomes.csv'] = [
        {k: r[k] for k in safe_episode_fields} for r in read_private('online_episode_outcomes.csv')
        if current_result(r['variant'], r['method'])]
    allowed_pairs = {r['comparator'] for r in data['pairs']}
    tables['pairwise_latency_episodes.csv'] = [
        {k: v for k, v in r.items() if k not in {'reference_trace', 'comparator_trace', 'annotator_id'}}
        for r in read_private('pairwise_latency_episodes.csv')
        if r['on_time_only'] == 'True' and r['comparator'] in allowed_pairs]
    tables['fusion_metrics.csv'] = data['fusion']
    for study in data['studies']:
        tables[study['id'] + '.csv'] = study['rows']
    tables['failure_categories.csv'] = [r for r in read_private('failure_categories.csv')
                                       if current_result(r['variant'], r['method'])]
    for name, rows in tables.items():
        write_csv(details / name, rows)
    manifest = {'schema_version': 1, 'protocol_id': data['protocol_id'],
                'files': {name: {'rows': len(rows), 'sha256': hashlib.sha256((details / name).read_bytes()).hexdigest()}
                          for name, rows in tables.items()}}
    (details / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copyfile(site / 'downloads/protocol.md', output / 'protocol.md')
    lines = ['# Results', '', '[Evaluation protocol](protocol.md)', '',
             'Current results use the three-annotation mean onset and the same frozen',
             'evaluation population as the revised paper. Rates pool episode counts.',
             'The website includes supplementary VAE-feature baselines, labeled separately',
             'from the ACT-native Policy baselines used in the paper.', '',
             '| Data | Contents |', '| --- | --- |']
    descriptions = {'offline_by_failure.csv': 'Detection counts, calibration and 5% FAR budget results',
                    'online_by_failure.csv': 'Streaming outcomes and conditional replay summaries',
                    'online_episode_outcomes.csv': 'Individual measured online outcomes',
                    'pairwise_latency_summary.csv': 'On-time paired latency comparisons',
                    'pairwise_latency_episodes.csv': 'Shared recordings underlying the paired comparisons',
                    'fusion_metrics.csv': 'Explicitly labeled endpoint and episode fusion',
                    'main_calibration_sensitivity_500.csv': 'All 500 calibration splits, including pooled summaries',
                    'calibration_matched_far.csv': '500-split retrospective 5% FAR budget sensitivity',
                    'expert_policy_score_distributions.csv': 'Episode-maximum score medians and interquartile ranges',
                    'failure_categories.csv': 'Detection counts by task and failure category'}
    for name in tables:
        lines.append(f'| [{name}](details/{name}) | {descriptions[name]} |')
    lines += ['', 'The public tables retain exact measurements. Blank values are unavailable.',
              'Current seed IDs identify the four runs used in the paper; archived and',
              'development runs are excluded. The protocol records the distinction between',
              'calibrated thresholds, retrospective thresholds, and conditional online latency.', '']
    (output / 'README.md').write_text('\n'.join(lines))
    # Remove only the known earlier export's files after making a local backup.
    retired = ('summary.md', 'summary.yaml', 'frozen_protocol.md', 'fusion_protocol.md',
               'evaluation_protocol.yaml', 'archived_experiments.yaml')
    retired_details = ('scoring_settings.csv', 'fusion_component_calibration.csv', 'fusion_episode_scores.csv',
                       'expert_policy_episode_maxima.csv', 'expert_policy_episode_maxima_summary.csv',
                       'calibration_matched_far_summary.csv')
    backup = ROOT / 'logs/benchmarks/retired_public_reports'
    for path in [*(output / name for name in retired), *(details / name for name in retired_details)]:
        if path.is_file():
            target = backup / path.relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
            path.unlink()
    print(f'Public results: {len(tables)} tables in {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=SITE)
    parser.add_argument('--output', type=Path, default=PUBLIC)
    args = parser.parse_args()
    export(args.site, args.output)
