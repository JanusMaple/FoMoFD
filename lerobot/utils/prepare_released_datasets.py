#!/usr/bin/env python3
"""Regenerate FoMo-FD VAE/ACT-token datasets from the raw Hugging Face release.

Dry-run is the default. --execute runs the existing converter; --resume skips
only outputs whose recorded inputs, conversion code, and saved stats match.
"""
from pathlib import Path
import argparse
from functools import lru_cache
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

TASKS = ('needle_pickup', 'ring_over_post', 'tissue_retraction', 'shunt_insertion')
MODES = {'vae': 'processed_le_v21_vfmf_vae_fix_wrist', 'native': 'processed_le_v21_tokens'}
SAVED = ('stats.json', 'gripper_transform.json', 'latent_stats.npz',
         'latent_stats_provenance.json', 'preparation_splits.json')


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=None)
def hash_file(path, size, mtime_ns):
    path = Path(path)
    before = path.stat()
    if (before.st_size, before.st_mtime_ns) != (size, mtime_ns):
        raise ValueError(f'Input changed: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (size, mtime_ns):
        raise ValueError(f'Input changed while reading: {path}')
    return digest.hexdigest()


def sha256(path):
    path = Path(path)
    st = path.stat()
    return hash_file(str(path), st.st_size, st.st_mtime_ns)


def make_job(repo, task, population, mode):
    task_root = repo / 'fomofd/tasks' / task
    name = MODES[mode]
    reference = task_root / population / 'release_metadata' / name / 'meta'
    preparation = read(reference / 'preparation.json')
    info = read(reference / 'info.json')
    raw = task_root / population / 'raw'
    mapping = preparation['episode_mapping']
    files = sorted(raw.glob('episode_*.npz'), key=lambda p: int(p.stem.split('_')[-1]))
    if [p.name for p in files] != [r['raw_file'] for r in mapping]:
        raise ValueError(f'Raw episode IDs/order differ from release: {raw}')
    shift = int(not (population == 'prl_data' and task in ('tissue_retraction', 'shunt_insertion')))
    if preparation['action_shift'] != shift:
        raise ValueError(f'Action alignment differs from released recording convention: {reference}')
    if info['total_episodes'] != len(mapping) or info['total_frames'] != sum(r['prepared_frames'] for r in mapping):
        raise ValueError(f'Reference frame/episode counts inconsistent: {reference}')
    manifest = read(task_root / population / 'release_metadata/raw_manifest.json')
    byname = {r['path']: r for r in manifest}
    if set(byname) != {p.name for p in files}:
        raise ValueError(f'Raw manifest differs from episode mapping: {raw}')
    checks = []
    for path in files:
        row = byname[path.name]
        if path.stat().st_size != row['bytes']:
            raise ValueError(f'Raw file size mismatch: {path}')
        checks.append((path, row['sha256']))
    target = task_root / population / name
    command = [sys.executable, str(repo / 'lerobot/utils/npz_to_le_v21.py'),
               '--task', task, '--raw_dir', str(raw), '--output_dir', str(target),
               '--fps', 'auto', '--including_gripper', '--gripper_transform', 'std_scale',
               '--gripper_std_scale', '10.0']
    if shift:
        command.append('--shift_actions_to_next_obs')
    if population == 'prl_data':
        command.append('--prl')
    if mode == 'vae':
        command += ['--extract_vfmf_vae_fix_wrist', '--vae_deterministic']
        checkpoints = []
        for view in ('fix', 'wrist'):
            checkpoint = task_root / f'model/vfmf_vae/dino_vfmf_vae_{view}_L8_3072/checkpoints/best.pt'
            checkpoints.append(checkpoint)
            command += [f'--vae_ckpt_{view}', str(checkpoint)]
        if not (reference / 'latent_stats.npz').is_file():
            raise ValueError(f'Missing frozen latent statistics: {reference}')
    else:
        act = task_root / 'model/act_dinov2/checkpoints/200000/pretrained_model'
        command += ['--extract_camera_tokens', '--act_ckpt_dir', str(act)]
        checkpoints = [act / 'model.safetensors', act / 'config.json']
    for checkpoint in checkpoints:
        suffix = checkpoint.relative_to(repo).as_posix()
        matches = [v for k, v in preparation['checkpoint_sha256'].items() if k.endswith(suffix)]
        if not checkpoint.is_file() or len(matches) != 1:
            raise ValueError(f'Missing or unidentifiable released checkpoint: {checkpoint}')
        checks.append((checkpoint, matches[0]))
    if population == 'data':
        split = read(reference / 'preparation_splits.json')
        train, val = split['train_episode_ids'], split['val_episode_ids']
        if set(train) & set(val) or sorted(train + val) != list(range(len(files))):
            raise ValueError(f'Invalid expert episode split: {reference}')
    metadata_hashes = {p.name: sha256(p) for p in reference.iterdir() if p.is_file()}
    identity = {'task': task, 'population': population, 'representation': mode,
                'action_shift': shift, 'metadata_sha256': metadata_hashes,
                'inputs_sha256': {str(p.relative_to(repo)): h for p, h in checks},
                'converter_sha256': sha256(repo / 'lerobot/utils/npz_to_le_v21.py'),
                'helper_sha256': sha256(Path(__file__))}
    if mode == 'native':
        tokenizer = repo / 'lerobot/policies/act_dinov2/src/lerobot_policy_act_dinov2/dino_visual_tokenizer.py'
        identity['tokenizer_sha256'] = sha256(tokenizer)
    return {'reference': reference, 'target': target, 'command': command,
            'info': info, 'preparation': preparation, 'checks': checks, 'identity': identity}


def validate_output(folder, job):
    info = read(folder / 'meta/info.json')
    for key in ('total_frames', 'total_episodes', 'fps'):
        if info[key] != job['info'][key]:
            raise ValueError(f'Converted {key} differs from release: {folder}')
    for key, expected in job['info']['features'].items():
        observed = info['features'].get(key, {})
        if observed.get('shape') != expected.get('shape') or observed.get('dtype') != expected.get('dtype'):
            raise ValueError(f'Converted schema differs for {key}: {folder}')
    if not any((folder / 'data').rglob('*.parquet')):
        raise ValueError(f'No converted data: {folder}')
    gripper = read(folder / 'meta/gripper_transform.json')
    if bool(gripper['action_aligned_to_next_observation']) != bool(job['identity']['action_shift']):
        raise ValueError(f'Converted action alignment differs: {folder}')


def run_job(repo, job, resume=False):
    for path, expected in job['checks']:
        if sha256(path) != expected:
            raise ValueError(f'Input checksum mismatch: {path}')
    target = job['target']
    if target.exists():
        marker = target / 'meta/release_preparation.json'
        if not resume or not marker.is_file() or read(marker)['inputs'] != job['identity']:
            raise FileExistsError(f'Existing output preserved: {target}; use --resume only for a matching completed release conversion.')
        validate_output(target, job)
        for name in SAVED:
            source = job['reference'] / name
            if source.exists() and sha256(target / 'meta' / name) != sha256(source):
                raise ValueError(f'Saved metadata changed: {target / "meta" / name}')
        print(f'Already complete: {target}', flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + target.name + '-', dir=target.parent) as directory:
        stage = Path(directory) / 'dataset'
        command = list(job['command'])
        command[command.index('--output_dir') + 1] = str(stage)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(repo / 'lerobot') + os.pathsep + env.get('PYTHONPATH', '')
        subprocess.run(command, cwd=repo / 'lerobot', env=env, check=True)
        validate_output(stage, job)
        meta = stage / 'meta'
        # Retain freshly generated info/episode-shard metadata, which can depend
        # on the installed LeRobot version. Restore only fixed stats and splits.
        shutil.copy2(meta / 'stats.json', meta / 'stats_regenerated.json')
        for name in SAVED:
            source = job['reference'] / name
            if source.is_file():
                shutil.copy2(source, meta / name)
        shutil.copy2(job['reference'] / 'preparation.json', meta / 'preparation_reference.json')
        preparation = dict(job['preparation'])
        preparation['job'] = dict(preparation.get('job', {}), output=str(target), command=job['command'])
        preparation['checkpoint_sha256'] = {str(p): h for p, h in job['checks'] if '/model/' in p.as_posix()}
        preparation.pop('previous_metadata_backup', None)
        if (meta / 'latent_stats.npz').is_file():
            preparation['normalization'] = {'source': str(job['reference'] / 'latent_stats.npz'),
                                            'source_sha256': sha256(meta / 'latent_stats.npz')}
        (meta / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
        (meta / 'release_preparation.json').write_text(json.dumps({'inputs': job['identity'],
            'normalization': 'released_saved_statistics', 'original_reference': str(job['reference'])}, indent=2) + '\n')
        if target.exists():
            raise FileExistsError(f'Output appeared during conversion: {target}')
        stage.rename(target)
    print(f'Completed: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--task', choices=(*TASKS, 'all'), default='all')
    parser.add_argument('--representations', nargs='+', choices=tuple(MODES), default=list(MODES))
    parser.add_argument('--splits', nargs='+', choices=('data', 'prl_data'), default=['data', 'prl_data'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    tasks = TASKS if args.task == 'all' else (args.task,)
    jobs = [make_job(repo, task, pop, mode) for task in tasks
            for pop in dict.fromkeys(args.splits) for mode in dict.fromkeys(args.representations)]
    for job in jobs:
        print(shlex.join(job['command']), flush=True)
    print(f'{len(jobs)} jobs; saved normalization; no raw-frame truncation during conversion.', flush=True)
    if args.execute:
        for job in jobs:
            run_job(repo, job, resume=args.resume)
    else:
        print('Dry-run only. Add --execute to extract features (GPU recommended).')


if __name__ == '__main__':
    main()
