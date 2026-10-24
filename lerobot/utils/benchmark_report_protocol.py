"""Attach frozen evaluation metadata to saved scoring reports.

The scorer applies the frozen protocol to rows, but older report writers omit
the top-level protocol declaration. Verify the actual saved configuration and
row semantics before publishing that declaration. Never infer it from a title.
This separate export step also preserves byte-identical inference sources for
cached-score provenance and already validated replay environments.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil

from benchmark_frozen_protocol import validate_config


def verified_protocol(report, config, episodes, latencies, label_bytes):
    spec = validate_config(config)
    label_hash = hashlib.sha256(label_bytes).hexdigest()
    if label_hash != spec['frozen_inputs']['onsets']['sha256'] or report['labels_sha256'] != label_hash:
        raise ValueError('Report labels do not match the frozen onset snapshot')
    if report['alpha'] != config['alpha'] or report['protocols'] != {
            k: config[k] for k in ('paper_protocol', 'latency_protocol')}:
        raise ValueError('Report configuration differs from its saved frozen configuration')
    if not episodes or not latencies:
        raise ValueError('Missing evaluation records; cannot certify the protocol')
    failures = [r for r in episodes if r['outcome'] == 'failure']
    primary = spec['detection']['primary_annotator']
    if not failures or any(r.get('detection_definition') != 'at_or_after_onset' or
                           r.get('primary_annotator') != primary for r in failures):
        raise ValueError('Failure records are not frozen post-onset evaluations')
    annotated = [r for r in latencies if r.get('onset_raw_frame') not in ('', None)]
    if not annotated or any(r.get('evaluation_protocol') != spec['protocol_id'] for r in annotated):
        raise ValueError('Annotated latency records do not use the frozen protocol')
    if report.get('evaluation_protocol') not in (None, spec):
        raise ValueError('Refusing to replace a different declared protocol')
    return spec


def publish(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a fresh destination; existing reports are preserved')
    read_rows = lambda name: list(csv.DictReader((source/name).open()))
    report = json.loads((source/'report.json').read_text())
    config = json.loads((source/'benchmark_config.json').read_text())
    spec = verified_protocol(report, config, read_rows('episode_results.csv'),
                             read_rows('latency_episodes.csv'), (source/'labels_snapshot.yaml').read_bytes())
    # Validate score-cache content too; publication never invokes a model.
    from unified_benchmark import digest, rooted, REPO, write_json
    for cache in report['score_caches']:
        if digest(Path(cache['path'])) != cache['sha256']:
            raise ValueError(f'Cached scores changed: {cache["path"]}')
    protocol_path = rooted(config['evaluation_protocol'], REPO)
    report.update(evaluation_protocol=spec, evaluation_protocol_sha256=digest(protocol_path),
        protocol_export={'source': str(source), 'source_report_sha256': digest(source/'report.json'),
                         'export_script_sha256': digest(Path(__file__)),
                         'numerical_results': 'unchanged; saved config, row semantics and score hashes verified'})
    shutil.copytree(source, destination)
    write_json(destination/'report.json', report)
    from results_inspector import archive_result
    archived = archive_result(destination, 'offline')
    print(f'Published verified frozen-protocol report: {destination}\nArchive: {archived}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    publish(args.source, args.output)
