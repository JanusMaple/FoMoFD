#!/usr/bin/env python3
"""Build an anonymous source + website directory, without local history or logs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import yaml

from export_public_results import export as export_results
from public_results import current_result, STUDIES, TITLES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'public_release'
TEXT_SUFFIXES = {'.py', '.md', '.yaml', '.yml', '.json', '.csv', '.txt', '.toml', '.xml', '.html', '.js', '.css', '.sh', '.cfg', '.ini', '.rviz'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_text(text, path, terms):
    # Personal contact metadata is separate from license/copyright notices.
    text = re.sub(r'[\w.+-]+@[\w.-]+', lambda m: 'maintainer@example.invalid'
                  if any(t in m[0].lower() for t in terms) else m[0], text)
    for term in sorted(terms, key=len, reverse=True):
        text = re.sub(re.escape(term), 'anonymous', text, flags=re.I)
    return text


def curate_catalog(path):
    data = yaml.safe_load(path.read_text())
    for key in ('reviewer_evidence', 'historical_training_corrections', 'training_correction_evidence',
                'online_history', 'vlm_history', 'archived_experiments', 'model_selection_record'):
        data.pop(key, None)
    data['offline'] = [r for r in data['offline'] if any(current_result(r['id'], method)
                       for method in ('wm_nc', 'ace', 'act_ensemble_pos'))]
    for row in data['offline']:
        row['title'] = TITLES.get(row['id'], row['title'])
    data['studies'] = [r for r in data['studies'] if r['id'] in STUDIES]
    setup = data['training_setup']
    for key in ('reviewer_comments', 'source_paths', 'sources', 'evidence'):
        setup.pop(key, None)
    setup['demonstration_operators'].pop('source', None)
    # Retain methodological selection facts, without local workstation names.
    data['model_selection_notes'] = [
        'The reported seed-1 run was selected after benchmark comparison. '
        'The four-seed study uses main seed 0, that seed-1 run, and the recorded seed-2/3 repeats.']
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def scan_release(folder, terms):
    errors = []
    for path in folder.rglob('*'):
        if path.is_symlink():
            errors.append(f'Symlink: {path.relative_to(folder)}')
        if path.is_dir() and path.name in {'.git', 'logs', '__pycache__', '.pytest_cache', '.aws', '.codex', '.agents'}:
            errors.append(f'Private/cache directory: {path.relative_to(folder)}')
        if not path.is_file():
            continue
        if path.name in {'.env', '.git', 'id_rsa', 'id_ed25519'} or path.suffix in {'.pem', '.key', '.pt', '.npz'}:
            errors.append(f'Private asset: {path.relative_to(folder)}')
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text().lower()
        if any(term in text for term in terms):
            errors.append(f'Identity: {path.relative_to(folder)}')
        if re.search(r'/home/(?!person/)[a-z0-9_.-]+/|/storage/project/[a-z0-9_.-]+/|-----begin (?:rsa |openssh )?private key', text):
            errors.append(f'Local path or key: {path.relative_to(folder)}')
    if errors:
        raise ValueError('\n'.join(errors))


def build(output=DEFAULT):
    output = output.resolve()
    if output.exists():
        raise ValueError('Choose a fresh release directory; existing releases are not overwritten.')
    if output == ROOT or not output.is_relative_to(ROOT):
        raise ValueError('Choose a new directory inside the repository.')
    export_results()
    private_terms = ROOT / 'logs/benchmarks/private_release_terms.json'
    terms = json.loads(private_terms.read_text()) if private_terms.exists() else []
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    output.mkdir(parents=True)
    for name in sorted(set(paths)):
        source = ROOT / name
        if not name or not source.is_file() or source.is_symlink():
            continue
        top = name.split('/')[0]
        if top not in {'.gitignore', 'README.md', 'docs', 'dvrk_ws', 'fomofd', 'lerobot'} and not top.startswith(('LICENSE', 'COPYING', 'NOTICE')):
            continue
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if source.suffix in TEXT_SUFFIXES:
            target.write_text(release_text(source.read_text(), source, terms))
    catalog = output / 'lerobot/configs/benchmark/results_catalog.yaml'
    curate_catalog(catalog)
    # Annotation values and IDs stay intact; remove annotators' real names only.
    labels = output / 'lerobot/configs/benchmark/prl_episode_labels.yaml'
    annotation = yaml.safe_load(labels.read_text())
    for index, entry in enumerate(annotation['annotators'].values(), 1):
        entry['name'] = f'Annotator {index}'
    labels.write_text(yaml.safe_dump(annotation, sort_keys=False, allow_unicode=True))
    # The exported labels have new bytes; keep their frozen-input hashes valid.
    for spec in (output / 'lerobot/configs/benchmark').glob('evaluation_protocol*.yaml'):
        data = yaml.safe_load(spec.read_text())
        for record in data.get('frozen_inputs', {}).values():
            source = output / record['path']
            if source.is_file():
                record['sha256'] = digest(source)
        spec.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    site = output / 'fomofd/tasks/review_site'
    shutil.copytree(ROOT / 'fomofd/tasks/review_site', site)
    # Include the static companion when the release is imported into a fresh repo.
    ignore = output / '.gitignore'
    ignore.write_text(ignore.read_text().replace('/fomofd/tasks/review_site/\n', ''))
    scan_release(output, terms)
    files = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}
    (output / 'release_manifest.json').write_text(json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n')
    print(f'Release ready: {output} ({len(files)} files; no Git history)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    build(parser.parse_args().output)
