#!/usr/bin/env python3
"""Validate sampled detectors on expert normals, then freeze settings.

Stages: prepare, ace, wm, freeze. PRL failure features/scores are never read by
these stages. GPU work is explicit (--device cuda); ACE also supports CPU.
Validation artifacts live under tasks/benchmark_results, separate from primary
reports. See configs/benchmark/sampled_score_validation.yaml for fixed rules.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import yaml
import unified_benchmark as b
import sampled_score_protocol as p


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return b.digest(Path(path))


def prepare(root, config_path):
    if (root/'plan.json').exists():
        raise FileExistsError('Existing validation plan is immutable')
    rules = yaml.safe_load(config_path.read_text())
    benchmark_path = b.REPO/rules['benchmark_config']
    cfg = yaml.safe_load(benchmark_path.read_text())
    tasks = list(cfg['tasks'])
    jobs = b.make_jobs(cfg, b.REPO, tasks, ['ace', 'wm_pu'],
        SimpleNamespace(config=benchmark_path, dry_run=False, batch_size=None, device='cuda'))
    plan = {'schema_version': 1, 'created_utc': b.utc_now(), 'rules': rules,
            'benchmark_config': cfg, 'reproduction_boundaries': p.BOUNDARIES,
            'orientation': 'higher_is_more_abnormal; never selected from test labels',
            'tasks': {}, 'frozen_inputs': {str(path): sha(path) for path in
                [config_path, benchmark_path, Path(__file__), Path(p.__file__)]}}
    for job in jobs:
        task, method = job.task, job.method
        expert = b.REPO/'fomofd/tasks'/task/'data'/job.dataset.name
        if not expert.is_dir():
            raise FileNotFoundError(expert)
        split_path = expert/'meta/preparation_splits.json' if method == 'ace' else job.checkpoint.parent.parent/'splits.json'
        split = json.loads(split_path.read_text())
        val = sorted(split['val_episode_ids'])
        if set(val) & set(split['train_episode_ids']):
            raise ValueError('Expert split overlaps')
        merged = deepcopy(job.merged)
        merged['data']['dataset_root'] = str(expert)
        for key in ('vfmf_vae_latent_stats_path', 'action_stats_path', 'state_stats_path'):
            if merged['data'].get(key):
                merged['data'][key] = str(b.rooted(merged['data'][key], b.REPO/'lerobot'))
        spec = {'task': task, 'method': method, 'dataset': str(expert),
                'checkpoint': str(job.checkpoint), 'checkpoint_sha256': sha(job.checkpoint),
                'merged': merged, 'seed': job.seed, 'expert_val_ids': val,
                'split_manifest': str(split_path), 'split_sha256': sha(split_path)}
        if method == 'ace':
            b.validate_native_provenance(replace(job, dataset=expert))
            fit, check = p.normal_partition(val, rules['selection_seed'], rules['ace']['fit_fraction'])
            spec.update(fit_episode_ids=fit, check_episode_ids=check)
        else:
            rng = np.random.default_rng(rules['selection_seed'])
            spec['probe_episode_ids'] = sorted(rng.choice(val, min(len(val), rules['wm_pu']['normal_episodes']), replace=False).tolist())
        plan['tasks'].setdefault(task, {})[method] = spec
        files = [job.checkpoint, job.train_config, split_path, expert/'meta/preparation.json']
        files += list((expert/'data').rglob('*.parquet'))
        for name in ('vfmf_vae_latent_stats_path', 'action_stats_path', 'state_stats_path'):
            if merged['data'].get(name): files.append(Path(merged['data'][name]))
        if method == 'ace':
            files += list(job.checkpoint.parent.glob('*processor*'))
        else:
            files += [b.REPO/'lerobot/models/flow_matching_wm'/n for n in ('model.py','config.py','dataset.py','score.py')]
        # Exact weights and metadata; expert parquet metadata guards avoid a
        # second bulk read while another GPU job is consuming the same disk.
        for path in files:
            if path.suffix == '.parquet':
                stat = path.stat()
                plan.setdefault('expert_file_stats', {})[str(path)] = [stat.st_size, stat.st_mtime_ns]
            else:
                plan['frozen_inputs'][str(path)] = sha(path)
    root.mkdir(parents=True, exist_ok=True)
    labels = b.REPO/cfg['labels']
    (root/'labels_snapshot.yaml').write_bytes(labels.read_bytes())
    plan['frozen_inputs'][str(root/'labels_snapshot.yaml')] = sha(root/'labels_snapshot.yaml')
    save(root/'plan.json', plan)
    (root/'validation_rules.yaml').write_text(config_path.read_text())
    print('Prepared normal-only validation plan:', root, flush=True)


def check_plan(root):
    plan = json.loads((root/'plan.json').read_text())
    for path, expected in plan['frozen_inputs'].items():
        if sha(path) != expected: raise ValueError('Frozen validation input changed: '+path)
    for path, expected in plan['expert_file_stats'].items():
        stat = Path(path).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError('Expert feature dataset changed: '+path)
    return plan


def expert_rows(spec, keys, columns):
    import pyarrow.dataset as ds
    path = Path(spec['dataset'])
    if path.parent.name != 'data':
        raise ValueError('Validation accepts expert data only')
    expression = None
    for ep, frames in keys.items():
        clause = (ds.field('episode_index') == ep) & ds.field('frame_index').isin(frames)
        expression = clause if expression is None else expression | clause
    table = ds.dataset(path/'data', format='parquet').to_table(
        columns=list(dict.fromkeys(['episode_index', 'frame_index']+columns)), filter=expression, use_threads=False)
    rows = table.to_pylist()
    lookup = {(int(r['episode_index']), int(r['frame_index'])):r for r in rows}
    if set(lookup) != {(e, f) for e, frames in keys.items() for f in frames}:
        raise ValueError('Missing/duplicate requested normal frames')
    return [lookup[key] for key in sorted(lookup)]


def configure_torch(device):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; no implicit CPU fallback')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    return torch, torch.device(device)


def ace_validation(root, plan, task, device):
    spec = plan['tasks'][task]['ace']; rules = plan['rules']['ace']
    output = root/task/'ace_validation.json'
    if output.exists():
        return json.loads(output.read_text())
    torch, device = configure_torch(device)
    meta = json.loads((Path(spec['dataset'])/'meta/preparation.json').read_text())
    lengths = {m['dataset_episode_index']: m['prepared_frames'] for m in meta['episode_mapping']}
    keys = {e:np.unique(np.rint(np.linspace(0, lengths[e]-1, rules['frames_per_episode'])).astype(int)).tolist()
            for e in spec['expert_val_ids']}
    sample_path = root/task/'ace_normal_positions.npz'
    if sample_path.exists():
        with np.load(sample_path) as z:
            positions, episodes, frames = z['positions'], z['episodes'], z['frames']
    else:
        checkpoint = Path(spec['checkpoint'])
        policy = b.load_act_decoder(checkpoint, device)
        prep, eps = b.processor_stats(checkpoint, 'policy_preprocessor.json', 'normalizer_processor')
        post, _ = b.processor_stats(checkpoint, 'policy_postprocessor.json', 'unnormalizer_processor')
        rows = expert_rows(spec, keys, ['observation.state', 'observation.tokens.fix_camera', 'observation.tokens.wrist_camera'])
        groups=[]; episodes=[]; frames=[]
        for start in range(0, len(rows), 4):
            chunk = rows[start:start+4]
            ep = [r['episode_index'] for r in chunk]; fr = [r['frame_index'] for r in chunk]
            noise = b.frame_noise(spec, ep, fr, rules['num_samples'], (policy.config.latent_dim,))
            batch = b.ace_inputs(chunk, prep, eps, device)
            actions = b.sample_ace_batch(policy, batch, noise, 4, policy.config.chunk_size)
            actions = actions*post['action.std']+post['action.mean']
            groups.append(actions[..., :3].cumsum(2)); episodes.extend(ep); frames.extend(fr)
            print(f'ACE normal {task}: {start+len(chunk)}/{len(rows)} frames', flush=True)
        positions=np.concatenate(groups);episodes=np.array(episodes);frames=np.array(frames)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, positions=positions, episodes=episodes, frames=frames)
        sample_path.with_suffix('.tmp').replace(sample_path)
        del policy
        if device.type == 'cuda': torch.cuda.empty_cache()
    result = p.fit_ace_resolution(positions[np.isin(episodes, spec['fit_episode_ids'])],
        positions[np.isin(episodes, spec['check_episode_ids'])], rules)
    result.update(task=task, source_role='expert_validation', checkpoint_sha256=spec['checkpoint_sha256'],
        fit_episode_ids=spec['fit_episode_ids'], check_episode_ids=spec['check_episode_ids'],
        expert_val_ids=spec['expert_val_ids'], num_samples=rules['num_samples'], horizon=int(positions.shape[2]),
        normal_samples_sha256=sha(sample_path), device=str(device), created_utc=b.utc_now())
    save(output, result)
    print(task, 'ACE validation:', result['reason'], flush=True)
    return result


def wm_validation(root, plan, task, device):
    import torch
    from models.flow_matching_wm.score import build_config, load_model_from_checkpoint
    from models.flow_matching_wm.dataset import EndpointWindowDataset, default_model_inputs_from_batch
    spec=plan['tasks'][task]['wm_pu'];rules=plan['rules']['wm_pu']
    output=root/task/'wm_validation.json'
    if output.exists() and json.loads(output.read_text()).get('finished'):
        return json.loads(output.read_text())
    torch,device=configure_torch(device)
    torch.set_float32_matmul_precision(spec['merged']['sampling']['matmul_precision'])
    cfg=build_config(spec['merged'])
    if cfg.fm_head.ode_solver != 'rk4':raise ValueError('Expected saved RK4 configuration')
    meta=json.loads((Path(spec['dataset'])/'meta/preparation.json').read_text())
    lengths={m['dataset_episode_index']:m['prepared_frames'] for m in meta['episode_mapping']}
    keys={e:list(range(lengths[e])) for e in spec['probe_episode_ids']}
    columns=[cfg.data.fix_key,cfg.data.wrist_key,cfg.data.action_key]
    if cfg.data.use_state:columns.append(cfg.data.state_key)
    rows=expert_rows(spec,keys,columns)
    index={e:[i for i,r in enumerate(rows) if r['episode_index']==e] for e in keys}
    ds=EndpointWindowDataset(rows,index,list(keys),cfg)
    selected=[]
    for ep in spec['probe_episode_ids']:
        indices=[i for i,s in enumerate(ds.samples) if s.episode_id==ep]
        selected.extend(indices[round(q*(len(indices)-1))] for q in rules['phase_fractions'])
    batches=[torch.utils.data.default_collate([ds[i]]) for i in selected]
    identities=[{'episode':int(x['episode_index'][0]),'frame':int(x['anchor_frame_index'][0])+cfg.data.pred_horizon} for x in batches]
    noises=[b.frame_noise(spec,[x['episode']],[x['frame']],rules['num_samples'],(2,16,16,8)) for x in identities]
    noise_hash=hashlib.sha256(b''.join(n.tobytes() for n in noises)).hexdigest()
    model=load_model_from_checkpoint(cfg,spec['checkpoint'],device)
    result={'task':task,'source_role':'expert_validation','checkpoint_sha256':spec['checkpoint_sha256'],
            'windows':identities,'fixed_noise_sha256':noise_hash,'num_samples':rules['num_samples'],
            'refinements':[],'steps':[],'passed':False,'finished':False,'created_utc':b.utc_now(),
            'adaptation':p.BOUNDARIES['wm_pu'],'device':str(device),
            'matmul_precision':spec['merged']['sampling']['matmul_precision']}
    previous=None;previous_steps=None
    for steps in rules['step_grid']:
        path=root/task/f'wm_normal_endpoints_{steps:03d}.npz'
        start=time.monotonic()
        if path.exists():
            with np.load(path) as z:
                if str(z['noise_sha256'].item())!=noise_hash:raise ValueError('Random samples changed')
                endpoints=z['endpoints']
        else:
            parts=[]
            for i,(batch,noise) in enumerate(zip(batches,noises)):
                inputs=default_model_inputs_from_batch({k:v.to(device) for k,v in batch.items()})
                parts.append(torch.cat([g.cpu() for g in p.sample_wm_endpoints(model,inputs,noise,
                    rules['sample_batch_size'],'float32',steps)],dim=1).numpy())
                print(f'WM normal {task}: RK4={steps}, window {i+1}/{len(batches)}',flush=True)
            endpoints=np.concatenate(parts)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,endpoints=endpoints,noise_sha256=np.array(noise_hash))
            path.with_suffix('.tmp').replace(path)
        result['steps'].append({'steps':steps,'samples_sha256':sha(path),'seconds':time.monotonic()-start})
        if previous is not None:
            result['refinements'].append({'coarse_steps':previous_steps,'fine_steps':steps,
                                          'error':p.refinement_error(previous,endpoints)})
            chosen=p.select_converged_steps(result['refinements'],rules['criteria'])
            if chosen is not None:
                result.update(passed=True,finished=True,ode_steps=chosen,
                    reason='Both views pass two successive fixed-sample refinements')
                save(output,result);print(task,'WM convergence selected RK4',chosen,flush=True)
                return result
        previous=endpoints;previous_steps=steps
        save(output,result)
    result.update(finished=True,reason='Convergence not established by the declared cap; no failure evaluation')
    save(output,result)
    return result


def freeze(root, plan):
    target=root/'frozen_sampling.json'
    if target.exists(): raise FileExistsError('Frozen settings are immutable')
    tasks={}
    for task in plan['tasks']:
        tasks[task]={}
        for method,name in [('ace','ace_validation.json'),('wm_pu','wm_validation.json')]:
            result=json.loads((root/task/name).read_text())
            if not result['passed']:raise ValueError(f'{task}/{method} did not pass normal-data validation')
            tasks[task][method]=result
    value={'schema_version':1,'status':'frozen_before_policy_calibration_and_test',
        'created_utc':b.utc_now(),'plan_sha256':sha(root/'plan.json'),'tasks':tasks,
        'rules':plan['rules'],'boundaries':p.BOUNDARIES,'score_orientation':'higher_is_more_abnormal',
        'policy_calibration_episode_ids':list(range(19)),'success_test_episode_ids':list(range(19,39))}
    save(target,value)
    save(root/'freeze_receipt.json',{'sha256':sha(target),'created_utc':b.utc_now()})
    print('FROZEN before PRL evaluation:',target,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['prepare','ace','wm','freeze'])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=b.REPO/'lerobot/configs/benchmark/sampled_score_validation.yaml')
    ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    ap.add_argument('--task')
    args=ap.parse_args();root=args.output.resolve()
    if args.stage=='prepare':prepare(root,args.config.resolve());return
    plan=check_plan(root)
    tasks=[args.task] if args.task else list(plan['tasks'])
    if args.stage=='freeze':freeze(root,plan);return
    for task in tasks:
        result=(ace_validation if args.stage=='ace' else wm_validation)(root,plan,task,args.device)
        if not result['passed']:raise RuntimeError(f'{task}/{args.stage}: {result["reason"]}')


if __name__=='__main__':
    sys.path.insert(0,str(b.REPO/'lerobot'))
    sys.path.insert(0,str(b.REPO/'lerobot/policies/act_dinov2/src'))
    main()
