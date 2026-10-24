"""Count-pooled failure-category reporting; no scoring or calibration changes."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'lerobot/configs/benchmark/failure_categories.yaml'


def load_categories(path=CONFIG):
    path = Path(path)
    doc = yaml.safe_load(path.read_text())
    if doc.get('schema_version') != 1 or len(doc['categories']) != 4:
        raise ValueError('Expected four failure categories, schema 1')
    label_path = REPO / doc['labels']
    labels = yaml.safe_load(label_path.read_text())
    available = {(task, g['failure_mode']): g for task, t in labels['tasks'].items()
                 for g in t['groups'] if g['outcome'] == 'failure'}
    mapping, categories = {}, []
    for category, entry in doc['categories'].items():
        modes, count = [], 0
        for task, names in entry['modes'].items():
            for mode in names:
                key = (task, mode)
                if key not in available or key in mapping:
                    raise ValueError(f'Unknown or duplicate failure assignment: {key}')
                g = available[key]
                if g['split'] != 'test':
                    raise ValueError('Failure category must reference test episodes')
                lo, hi = g['episode_range']
                count += hi - lo + 1
                mapping[key] = category
                modes.append(dict(task=task, failure_mode=mode, episode_range=[lo, hi],
                                  failure_origin=g['failure_origin'], episodes=hi-lo+1))
        if count != entry['expected_failure_episodes']:
            raise ValueError(f'Category count mismatch: {category}')
        categories.append(dict(id=category, title=entry['title'], total=count, modes=modes))
    if set(mapping) != set(available):
        raise ValueError(f'Unassigned failure modes: {set(available)-set(mapping)}')
    if sum(c['total'] for c in categories) != doc['expected_failure_episodes']:
        raise ValueError('Total failure-category count mismatch')
    return dict(classification_id=doc['classification_id'], categories=categories,
                mapping_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                labels_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(), notes=doc['notes'])


def pool_categories(rows, taxonomy, annotator):
    """Pool distinct test episodes per detector, view and threshold protocol.

    Missing rows and unknown post-onset outcomes cannot silently become misses.
    Category totals are fixed by the mapping; partial coverage receives bounds.
    """
    labels, expected = {}, defaultdict(set)
    for category in taxonomy['categories']:
        for mode in category['modes']:
            task = mode['task']
            for ep in range(mode['episode_range'][0], mode['episode_range'][1]+1):
                identity = (task, ep)
                if identity in labels:
                    raise ValueError('Overlapping category episode ranges')
                labels[identity] = (category['id'], mode)
                expected[category['id'], task].add(identity)
                expected[category['id'], 'Overall'].add(identity)
    groups, seen = defaultdict(list), set()
    for r in rows:
        if r['split'] != 'test' or r['outcome'] != 'failure':
            continue
        identity = (r['task'], r['episode_index'])
        if identity not in labels:
            raise ValueError(f'Unmapped failure episode: {identity}')
        category, label = labels[identity]
        if any(r[k] != label[k] for k in ('failure_mode', 'failure_origin')):
            raise ValueError(f'Failure label mismatch: {identity}')
        detector = tuple(r[k] for k in ('variant', 'method', 'view', 'calibration'))
        key = (*detector, *identity)
        if key in seen:
            raise ValueError(f'Duplicate detector episode: {key}')
        seen.add(key)
        assessment = r.get('onset_assessments', {}).get(annotator)
        if assessment is None:
            raise ValueError(f'Missing explicit post-onset assessment: {key}')
        alarm = assessment['post_alarm']
        if alarm is not None and type(alarm) is not bool:
            raise ValueError('Post-onset alarm must be true, false or unknown')
        item = dict(detected=alarm, origin=r['failure_origin'],
                    late=bool(r.get('late_alarm')),
                    measurement='recorded_streaming' if r.get('score_kind') == 'binary_decision' else 'dense_offline')
        groups[(*detector, category, r['task'])].append(item)
        groups[(*detector, category, 'Overall')].append(item)
    result = []
    titles = {c['id']: c['title'] for c in taxonomy['categories']}
    for key, items in sorted(groups.items()):
        variant, method, view, calibration, category, task = key
        measurements = {r['measurement'] for r in items}
        if len(measurements) != 1:
            raise ValueError('Cannot pool offline and streaming measurements')
        n, total = len(items), len(expected[category, task])
        detected = sum(r['detected'] is True for r in items)
        unknown = sum(r['detected'] is None for r in items)
        missing = total-n
        if missing < 0:
            raise ValueError('Category denominator exceeds labeled cohort')
        result.append(dict(variant=variant, method=method, view=view, calibration=calibration,
            category=category, category_title=titles[category], task=task,
            measurement=next(iter(measurements)), annotator_id=annotator,
            detected=detected, total=total, recorded=n, unknown=unknown, missing=missing,
            misses=sum(r['detected'] is False for r in items),
            natural=sum(r['origin']=='natural' for r in items),
            induced=sum(r['origin']=='induced' for r in items),
            late_detections=sum(r['detected'] is True and r['late'] for r in items),
            FDR=detected/total if not unknown+missing else None,
            FDR_lower=detected/total, FDR_upper=(detected+unknown+missing)/total))
    return result


def format_rate(row):
    if row is None:
        return '—'
    if row['FDR'] is None:
        return (f"{row['detected']}/{row['total']}; {row['unknown']} unknown, {row['missing']} missing "
                f"[{100*row['FDR_lower']:.1f}–{100*row['FDR_upper']:.1f}%]")
    return f"{row['detected']}/{row['total']} ({100*row['FDR']:.1f}%)"


def write_report(data, root, path=CONFIG, suffix=''):
    taxonomy = load_categories(path)
    spec = data.get('evaluation_protocol', {})
    annotator = spec.get('detection', {}).get('primary_annotator')
    if not annotator or taxonomy['labels_sha256'] != spec['frozen_inputs']['onsets']['sha256']:
        raise ValueError('Category report requires matching frozen onset labels')
    rows = [dict(zip(data['episodes']['columns'], r)) for r in data['episodes']['rows']]
    rows += [r for run in data.get('vlm_runs', []) for r in run['episodes']]
    records = pool_categories(rows, taxonomy, annotator)
    sources = data.get('summary_sources', {})
    main_methods = {'wm_nc','wm_pe','wm_pu','logpzo','rnd','logpzo_native','rnd_native','ace',
                    'act_ensemble_pos','act_ensemble_rot'} - set(data.get('supplementary_methods', []))
    main = [r for r in records if r['task']=='Overall' and
            (r['method'] in main_methods and r['variant']==sources.get(r['method'],'main')
             or r['measurement']=='recorded_streaming')]
    folder = Path(root)/('failure_categories'+suffix)
    folder.mkdir(exist_ok=True)
    note = ('FDR = sum of post-onset detected episodes / sum of labeled failure episodes; '
            'never an average of task or mode percentages. Cameras, checkpoints and calibration '
            'protocols remain separate. Natural/induced origin is independent of category. '
            'Unknown or missing outcomes retain denominators with bounds, not zero imputation. '
            'VLM rows are recorded streaming results, including late detections (listed separately); '
            'historical early-stopped traces can be unknown. Continuous-score rows are dense offline results. '
            'No category-specific thresholds are fitted; successful episodes have no failure category.')
    payload = dict(**taxonomy, source_snapshot_utc=data['created'], evaluation_protocol=spec['protocol_id'],
                   definition=note, records=records, main_records=main,
                   report_path=folder.name+'/report.md', csv_path=folder.name+'/metrics.csv',
                   json_path=folder.name+'/results.json')
    (folder/'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    with (folder/'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else [])
        writer.writeheader(); writer.writerows(records)
    (folder/'mapping.yaml').write_bytes(Path(path).read_bytes())
    lines = ['# Failure-category FDR', '', note, '',
             f"Source result snapshot: {data['created']}. Mapping: {taxonomy['classification_id']}.", '',
             '| Category | Failure episodes |', '|---|---|']
    lines += [f"| {c['title']} | {c['total']} |" for c in taxonomy['categories']]
    lines += ['', '| Method | View | Calibration | Category | Detected / total (FDR) |', '|---|---|---|---|---|']
    method_labels = {'act_ensemble_pos': 'AEU', 'act_ensemble_rot': 'AEU-Rot'}
    lines += [f"| {method_labels.get(r['method'], r['method'])} | {r['view']} | {r['calibration']} | {r['category_title']} | {format_rate(r)} |" for r in main]
    lines += ['', '[All task/variant results (CSV)](metrics.csv) · [Structured results](results.json) · [Mapping](mapping.yaml)', '']
    (folder/'report.md').write_text('\n'.join(lines))
    return payload
