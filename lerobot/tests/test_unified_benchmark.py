"""CPU-only benchmark protocol tests; no real checkpoints or CUDA are loaded."""
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

PATH = Path(__file__).resolve().parents[1] / 'utils/unified_benchmark.py'
sys.path.insert(0, str(PATH.parent))
spec = importlib.util.spec_from_file_location('unified_benchmark_tested', PATH)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


def curve(values, raw=None, wall=None):
    n = len(values)
    return dict(score=np.array(values, dtype=float), frame=np.arange(n),
                raw=np.arange(n) if raw is None else np.array(raw),
                wall=np.arange(n, dtype=float) if wall is None else np.array(wall, dtype=float))


def toy_suite(root):
    groups = [dict(episode_range=[0,18], split='calibration', outcome='success', failure_mode=None, failure_origin=None),
              dict(episode_range=[19,20], split='test', outcome='success', failure_mode=None, failure_origin=None),
              dict(episode_range=[21,22], split='test', outcome='failure', failure_mode='Drop', failure_origin='natural')]
    labels = dict(annotators={'a': {'name': 'Annotator'}}, tasks={'toy': dict(groups=groups,
        failure_onsets={21: {'a': {'frame_index': 2}}}, expected_episode_count=23)})
    config = dict(alpha=.05, tasks={'toy': {'title': 'Toy'}},
        paper_protocol=dict(smoothing='centered', window=1, wm_frame_policy='all_endpoints'),
        latency_protocol=dict(smoothing='trailing', window=1, wm_frame_policy='all_endpoints'),
        time_varying={'bin_seconds': 1}, fusion={'wm_views': True, 'conditional_marginal': True})
    clocks = {e: {'frame': np.arange(4), 'raw': np.arange(4), 'wall': np.arange(4.) + e*100} for e in range(23)}
    jobs = []
    for method in ['wm_nc','logpzo','ace']:
        views = ('fix','wrist') if method=='wm_nc' else (('both',) if method=='ace' else ('fix',))
        selections = [views] if method!='logpzo' else [('fix',),('wrist',)]
        for selected in selections:
            job = b.Job('toy',method,selected,root/'best.pt',root/'config.json',root/'scores.yaml',{},
                        {'data': {'history_len':1,'pred_horizon':1}},root, 'unused','unused',
                        'reactive' if method=='wm_nc' else None,1,0,'cpu')
            es,fs,ws,ss = [],[],[],[]
            for e in range(23):
                for f in range(4):
                    es.append(e);fs.append(f);ws.append(clocks[e]['wall'][f])
                    value = 1 if e<21 or f<2 else 10
                    ss.append([value]*len(selected))
            arrays=dict(episode_index=np.array(es),frame_index=np.array(fs), raw_frame_index=np.array(fs),
                        wall_timestamp=np.array(ws),scores=np.array(ss,dtype=float))
            jobs.append((job,arrays,clocks))
    return jobs, labels, config


class CalibrationTests(unittest.TestCase):
    def test_joint_wm_reduces_endpoints_before_calibration_and_keeps_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            scored, labels, config = toy_suite(Path(directory))
            job, arrays, timeline = scored[0]
            job.merged['backbone'] = {'view_fusion':'concat_channels'}
            # Separate camera peaks occur at DIFFERENT endpoints. Joint max=10,
            # never sum(max(fixed), max(wrist))=20.
            arrays['scores'][:] = np.tile([[10.,0.],[0.,10.],[10.,0.],[0.,10.]],(23,1))
            original = arrays['scores'].copy()
            config.update(wm_score_aggregation='joint', fusion={'wm_views':False,'conditional_marginal':False})
            report = b.evaluate_suite([(job,arrays,timeline)],labels,config,[],['wm_nc'],['episode'])
            self.assertEqual({r['view'] for r in report['episodes']},{'both'})
            self.assertEqual({r['threshold'] for r in report['calibration']},{10.})
            self.assertFalse(any(r['alarm'] for r in report['episodes']))  # strict >
            np.testing.assert_array_equal(arrays['scores'],original)
            np.testing.assert_array_equal(b.aggregate_wm_scores('wm_pe',original,'joint'),np.full((92,1),10.))
            np.testing.assert_array_equal(b.aggregate_wm_scores('wm_pu',original,'joint'),np.full((92,1),5.))
            job.merged['backbone']['view_fusion']='separate'
            with self.assertRaisesRegex(ValueError,'concat_channels'):
                b.evaluate_suite([(job,arrays,timeline)],labels,config,[],['wm_nc'],['episode'])

    def test_offline_batch_precedence_and_validation(self):
        config = {'score_batch_size': 2, 'score_batch_sizes': {'logpzo': 128},
                  'tasks': {'toy': {'score_batch_sizes': {'logpzo': 1}}}}
        self.assertEqual(b.score_batch_size(config, 'toy', 'rnd'), 2)
        self.assertEqual(b.score_batch_size(config, 'toy', 'logpzo'), 1)
        self.assertEqual(b.score_batch_size(config, 'toy', 'logpzo', 8), 8)
        del config['tasks']['toy']['score_batch_sizes']
        self.assertEqual(b.score_batch_size(config, 'toy', 'logpzo'), 128)
        for invalid in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError): b.score_batch_size(config, 'toy', 'rnd', invalid)
        config['score_batch_sizes']['unknown'] = 4
        with self.assertRaises(ValueError): b.score_batch_size(config, 'toy', 'rnd')

    def test_sample_noise_is_independent_of_batch_partition_and_order(self):
        spec = dict(seed=0, task='toy', method='wm_pu')
        noise = b.frame_noise(spec, [1,1,2], [7,8,7], 7, (2,3))
        separate = np.concatenate([b.frame_noise(spec, [ep], [fr], 7, (2,3))
                                   for ep,fr in [(1,7),(1,8),(2,7)]])
        np.testing.assert_array_equal(noise, separate)
        reversed_order = b.frame_noise(spec, [2,1,1], [7,8,7], 7, (2,3))
        np.testing.assert_array_equal(noise, reversed_order[::-1])
        self.assertFalse(np.array_equal(noise[0], noise[1]))
        other = b.frame_noise({**spec, 'task':'other'}, [1], [7], 7, (2,3))
        self.assertFalse(np.array_equal(noise[:1], other))

    def test_wm_sample_and_window_batches_preserve_pairing_and_tail(self):
        import torch
        class Field(torch.nn.Module):
            def set_context(self, fix, wrist, state, actions):
                self.factor = actions[:,0,0].reshape(-1,1,1,1,1)
            def forward(self, t, x):
                return x * self.factor
        model = SimpleNamespace(odefunc=Field(), _num_views=lambda:2,
            config=SimpleNamespace(fm_head=SimpleNamespace(base_std=1., ode_rtol=1e-4, ode_atol=1e-5, ode_solver='rk4')))
        inputs = dict(fix_hist=None, wrist_hist=None, state_hist=None,
                      action_chunk=torch.arange(1,6,dtype=torch.float32).reshape(5,1,1)/10)
        noise = b.frame_noise(dict(seed=0,task='toy',method='wm_pu'), [0]*5, list(range(5)), 7, (2,2,2,1))
        batched = b.score_wm_pu_batch(model, inputs, noise, 3, 'float32')
        serial = torch.cat([b.score_wm_pu_batch(model, {**inputs, 'action_chunk':inputs['action_chunk'][i:i+1]},
                                              noise[i:i+1], 1, 'float32') for i in range(5)])
        torch.testing.assert_close(batched, serial)
        self.assertEqual(tuple(batched.shape), (5,2))

    def test_ace_frame_sample_pairing_and_partial_sample_batch(self):
        import torch
        class Decoder:
            def decode_with_given_latent(self, batch, latent):
                return (batch['observation.state'] + latent[:, :2])[:,None,:].expand(-1,3,-1)
        policy = SimpleNamespace(model=Decoder())
        inputs = {'observation.state':torch.arange(10,dtype=torch.float32).reshape(5,2)}
        noise = b.frame_noise(dict(seed=0,task='toy',method='ace'), [0]*5, list(range(5)), 7, (2,))
        batched = b.sample_ace_batch(policy, inputs, noise, 3, 2)
        serial = np.concatenate([b.sample_ace_batch(policy, {'observation.state':inputs['observation.state'][i:i+1]},
                                                   noise[i:i+1], 1, 2) for i in range(5)])
        np.testing.assert_array_equal(batched, serial)
        self.assertEqual(batched.shape, (5,7,2,2))

    def test_json_scientific_notation_keeps_numeric_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'
            path.write_text('{"optim":{"min_lr":1e-06},"fm_head":{"ode_atol":1e-05}}')
            loaded=b.load_node(path,'fm_wm')
            self.assertEqual(loaded['optim']['min_lr'],1e-6)
            self.assertIsInstance(loaded['fm_head']['ode_atol'],float)

    def test_finite_sample_rank_and_ties(self):
        self.assertEqual(b.conformal_threshold(np.arange(19),.05),18)
        self.assertTrue(math.isinf(b.conformal_threshold(np.arange(18),.05)))
        self.assertIsNone(b.first_alarm(curve([18,18]),18))
        self.assertEqual(b.first_alarm(curve([18,19]),18)['raw'],1)

    def test_time_bins_use_wall_clock_and_never_test_scores(self):
        source={e:curve([1,2],wall=[100*e,100*e+1]) for e in range(20)}
        clocks={e:{'wall':np.array([100*e])} for e in source}
        source[19]=curve([100,100,100],wall=[1900,1901,1902])
        result,records=b.calibrate_curves(source,list(range(19)),clocks,.05,'time_varying',{'bin_seconds':1})
        self.assertEqual([r['threshold'] for r in records],[1,2])
        self.assertTrue(np.isneginf(result[19]['score'][2]))
        source[19]['score'][:]=1e9
        _,again=b.calibrate_curves(source,list(range(19)),clocks,.05,'time_varying',{'bin_seconds':1})
        self.assertEqual(records,again)
        source[18]=curve([1],wall=[1800])
        result,records=b.calibrate_curves(source,list(range(19)),clocks,.05,'time_varying',{'bin_seconds':1})
        self.assertEqual(records[1]['n_calibration'],18)
        self.assertEqual(records[1]['threshold'],'+inf')

    def test_origin_is_recording_start_not_first_wm_endpoint(self):
        source={e:curve([1,2],wall=[100*e+4,100*e+5]) for e in range(19)}
        clocks={e:{'wall':np.array([100*e])} for e in source}
        _,records=b.calibrate_curves(source,list(source),clocks,.05,'time_varying',{'bin_seconds':1})
        self.assertEqual([r['bin_index'] for r in records],[4,5])

    def test_and_is_same_frame_not_latched(self):
        lhs={0:curve([1,-1],raw=[4,5],wall=[1,2])}
        rhs={0:curve([-1,1],raw=[4,5],wall=[1,2])}
        self.assertIsNone(b.first_alarm(b.combine_curves(lhs,rhs,'and')[0],0))
        self.assertIsNotNone(b.first_alarm(b.combine_curves(lhs,rhs,'or')[0],0))
        rhs[0]['wall'][0]=9
        with self.assertRaises(ValueError): b.combine_curves(lhs,rhs,'or')

    def test_endpoint_marginal_join_uses_raw_frame(self):
        wm={0:curve([4,8],raw=[5,6],wall=[5,6])}
        obs={0:curve([9,2,3],raw=[4,5,6],wall=[4,5,6])}
        result=b.combine_curves(wm,obs,'and')[0]
        np.testing.assert_array_equal(result['raw'],[5,6])
        np.testing.assert_array_equal(result['score'],[2,3])

    def test_episode_auc_ties_and_missing_class(self):
        self.assertEqual(b.episode_auroc([False,True],[0,1]),1)
        self.assertEqual(b.episode_auroc([False,True],[1,0]),0)
        self.assertEqual(b.episode_auroc([False,True],[1,1]),.5)
        self.assertEqual(b.episode_auroc([False,True],[-np.inf,-np.inf]),.5)
        self.assertIsNone(b.episode_auroc([True],[1]))

    def test_ace_entropy_is_joint_xyz_and_sums_horizon(self):
        samples=np.zeros((4,2,3))
        self.assertEqual(b.ace_histogram_entropy(samples,np.ones(3)),0)
        samples[2:]=10
        self.assertEqual(b.ace_histogram_entropy(samples,np.ones(3)),2)
        self.assertEqual(b.ace_histogram_entropy(samples+32,np.ones(3)),2)
        with self.assertRaises(ValueError): b.ace_histogram_entropy(samples,np.zeros(3))

    def test_suite_latency_auc_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            scored,labels,config=toy_suite(root)
            reports=b.evaluate_suite(scored,labels,config,['a'],['wm_nc','logpzo','ace'],['episode','time_varying'])
            metrics=b.aggregate_metrics(reports['episodes'])
            for row in metrics:
                self.assertEqual(row['FAR'],0)
                self.assertEqual(row['FDR'],1)
                self.assertEqual(row['episode_AUROC'],1)
            latency=b.latency_summary(reports['latencies'])
            for row in latency:
                self.assertEqual(row['n_detected'],1)
                self.assertEqual(row['n_unannotated'],1)
                self.assertEqual(row['delay_mean_s'],0)
                self.assertFalse(row['annotation_complete'])
            for scheme in ['episode','time_varying']:
                h,t=b.paper_table([r for r in reports['episodes'] if r['calibration']==scheme],config,labels,['toy'],['wm_nc','logpzo','ace'])
                self.assertEqual(len(h),13)
                self.assertTrue(all(len(row)==len(h) for row in t))
            b.write_json(root/'report.json',{'metrics':metrics,'latency':latency,'calibration':reports['calibration']})
            for key in reports: b.write_csv(root/f'{key}.csv',reports[key],[])

    def test_pre_onset_is_not_zero_latency_and_labels_do_not_affect_auc(self):
        with tempfile.TemporaryDirectory() as tmp:
            scored,labels,config=toy_suite(Path(tmp))
            original=b.evaluate_suite(scored,labels,config,['a'],['wm_nc','logpzo','ace'],['episode'])
            labels['tasks']['toy']['failure_onsets'][21]['a']['frame_index']=3
            later=b.evaluate_suite(scored,labels,config,['a'],['wm_nc','logpzo','ace'],['episode'])
            self.assertEqual(b.aggregate_metrics(original['episodes']),b.aggregate_metrics(later['episodes']))
            annotated=[r for r in later['latencies'] if r['episode_index']==21]
            self.assertTrue(all(r['status']=='pre_onset_alarm' and r['signed_first_alarm_delay_s']==-1
                                and r['detection_delay_s'] is None for r in annotated))

    def test_auroc_ignores_calibration_and_macros_across_tasks(self):
        rows=[]
        for task,scores in [('a',[0,1]),('b',[101,100])]:
            for e,(outcome,score) in enumerate(zip(['success','failure'],scores)):
                rows.append(dict(task=task,method='rnd',view='fix',calibration='episode',split='test',
                                 outcome=outcome,score_max=score,alarm=False,n_supported_scores=1))
            rows.append(dict(rows[-1],split='calibration',score_max=-1e9))
        result=b.aggregate_metrics(rows)
        self.assertEqual(result[-1]['episode_AUROC'],.5)
        self.assertEqual(result[-1]['failure_episodes'],2)

    def test_missing_cache_never_invokes_inference_without_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            scored,_,_=toy_suite(Path(tmp))
            job,arrays,timeline=scored[0]
            args=SimpleNamespace(cache_only=True,mode='normal',allow_inference=False)
            with patch.object(b,'score_job',side_effect=AssertionError('model loading forbidden')):
                with self.assertRaises(FileNotFoundError): b.acquire_scores(job,b.REPO,timeline,{},Path(tmp)/'missing.npz',args)
                args.cache_only=False
                with self.assertRaises(ValueError): b.acquire_scores(job,b.REPO,timeline,{},Path(tmp)/'missing.npz',args)

    def test_cli_report_writer_from_cached_scores_and_fresh_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            scored,labels,config=toy_suite(root)
            labels.update(episode_id_space='prepared_prl_episode_index',onset_frame_id_space='raw_recording_frame_index',onset_frame_index_base=0)
            config.update(schema_version=1,score_seed=0,score_batch_size=1,labels=str(root/'labels.yaml'))
            (root/'labels.yaml').write_text(yaml.safe_dump(labels))
            (root/'benchmark.yaml').write_text(yaml.safe_dump(config))
            (root/'meta').mkdir()
            (root/'meta/preparation.json').write_text(json.dumps({'episode_mapping':[{'dataset_episode_index':e} for e in range(23)]}))
            report=root/'report'
            identity={'test':'fixture'}
            digest=b.hashlib.sha256(b.canonical(identity).encode()).hexdigest()
            for job,_,_ in scored:
                job.checkpoint.write_bytes(b'fixture')
                job.cache_dir.mkdir(parents=True,exist_ok=True)
                np.savez(job.cache_dir/f'{digest}.npz',provenance_json=np.asarray('{}'))
            def cached(job,*args):
                return next(arrays for j,arrays,_ in scored if j==job)
            argv=['benchmark','--repo-root',str(root),'--config',str(root/'benchmark.yaml'),
                  '--methods','wm_nc','logpzo','ace','--cache-only','--report-dir',str(report)]
            with patch.object(sys,'argv',argv), patch.object(b,'make_jobs',return_value=[j for j,_,_ in scored]), \
                 patch.object(b,'fingerprint',return_value=identity), patch.object(b,'validate_native_provenance'), \
                 patch.object(b,'load_timeline',return_value=scored[0][2]), patch.object(b,'acquire_scores',side_effect=cached), \
                 patch('results_inspector.archive_result',return_value=root/'test_archive'), \
                 patch.object(b,'score_job',side_effect=AssertionError('No inference')), patch('sys.stdout',new=io.StringIO()):
                b.main()
            saved=json.loads((report/'report.json').read_text())
            self.assertEqual(saved['calibration_schemes'],['episode','wall_time_bin'])
            self.assertTrue((report/'paper_table_wall_time_bin.csv').is_file())
            self.assertEqual((report/'labels_snapshot.yaml').read_text(),(root/'labels.yaml').read_text())
            self.assertTrue(all(row['episode_AUROC']==1 for row in saved['summary_metrics']))
            self.assertTrue((report/'scoring_settings.csv').is_file())
            self.assertIn('Steps/solve', (report/'report.md').read_text())
            self.assertIn('Episode AUROC', (report/'report.html').read_text())
            self.assertIn('filter', (report/'report.html').read_text())
            self.assertEqual(len(saved['scoring_settings']), len(scored))

    def test_ode_steps_are_distinct_from_evaluations_and_sample_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            scored,_,_=toy_suite(Path(tmp));job=scored[0][0]
            job.merged['fm_head']={'ode_solver':'rk4','score_transport_mode':'ode','per_chunk_wm_samples':16}
            row=b.scoring_settings(job)
            self.assertEqual(row['ode_steps_per_solve'],1)
            self.assertEqual(row['vector_field_evals_per_solve'],4)
            self.assertEqual(row['ode_solves_per_score'],1)
            self.assertEqual(row['ode_direction'],'1 -> 0')
            job.method='logpzo_native';row=b.scoring_settings(job)
            self.assertEqual(row['ode_steps_per_score'],1)
            self.assertEqual(row['vector_field_evals_per_score'],4)
            job.method='wm_pu';row=b.scoring_settings(job)
            self.assertEqual(row['ode_steps_per_score'],16)
            self.assertEqual(row['vector_field_evals_per_score'],64)
            self.assertEqual(row['ode_direction'],'0 -> 1')
            job.method='wm_pe';row=b.scoring_settings(job)
            self.assertEqual(row['ode_solves_per_score'],1)
            job.merged['fm_head']['ode_solver']='dopri5';row=b.scoring_settings(job)
            self.assertIsNone(row['ode_steps_per_solve'])
            self.assertIsNone(row['vector_field_evals_per_score'])
            job.method='wm_nc';job.merged['fm_head']['score_transport_mode']='one_step_t0'
            row=b.scoring_settings(job)
            self.assertEqual(row['ode_steps_per_score'],0)
            self.assertEqual(row['vector_field_evals_per_score'],1)

    def test_markdown_cells_escape_pipes(self):
        text=b.markdown_table(['Name','Score'],[['Alice | Bob','1\n2']])
        self.assertIn('Alice \\| Bob',text)
        self.assertIn('1<br>2',text)

    def test_native_readiness_waits_for_final_checkpoint_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            scored,_,_=toy_suite(root)
            run=root/'run'; (run/'checkpoints').mkdir(parents=True)
            checkpoint=run/'checkpoints/best.pt';checkpoint.write_bytes(b'fixture')
            job=b.Job(**{**scored[0][0].__dict__, 'method':'logpzo_native', 'checkpoint':checkpoint,
                         'merged':{'runtime':{'max_steps':100}}})
            history=run/'history.jsonl'
            history.write_text(json.dumps({'step':100,'split':'train'})+'\n')
            self.assertIsNotNone(b.job_readiness(job))
            history.write_text(json.dumps({'step':100,'split':'val'})+'\n')
            last=run/'checkpoints/last.pt';last.write_bytes(b'incomplete')
            self.assertIsNotNone(b.job_readiness(job))
            with b.zipfile.ZipFile(last,'w') as archive: archive.writestr('last/data.pkl',b'fixture')
            self.assertIsNone(b.job_readiness(job))

    def test_camera_specific_latency_and_missing_view_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            scored,labels,config=toy_suite(Path(tmp))
            labels['tasks']['toy']['failure_onsets'][21]['a']={
                'mode':'separate','views':{'fix':{'frame_index':3},'wrist':{'frame_index':1}}}
            reports=b.evaluate_suite(scored,labels,config,['a'],['wm_nc','logpzo','ace'],['episode'])
            entries=[r for r in reports['latencies'] if r['episode_index']==21]
            for row in entries:
                if row['onset_view']=='fix':
                    self.assertEqual(row['status'],'pre_onset_alarm')
                    self.assertEqual(row['signed_first_alarm_delay_s'],-1)
                else:
                    self.assertEqual(row['status'],'detected')
                    self.assertEqual(row['detection_delay_s'],1)
            self.assertEqual({r['onset_view'] for r in entries if r['method']=='ace'},{'fix','wrist'})
            del labels['tasks']['toy']['failure_onsets'][21]['a']['views']['fix']
            reports=b.evaluate_suite(scored,labels,config,['a'],['wm_nc','logpzo','ace'],['episode'])
            fixed=[r for r in reports['latencies'] if r['episode_index']==21 and r['onset_view']=='fix']
            self.assertTrue(all(r['status']=='unannotated' and r['onset_raw_frame'] is None for r in fixed))
            summary=b.latency_summary(reports['latencies'])
            self.assertEqual({r['onset_view'] for r in summary if r['method']=='ace'},{'fix','wrist'})

    def test_success_onsets_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,labels,_=toy_suite(Path(tmp))
            labels.update(episode_id_space='prepared_prl_episode_index',onset_frame_id_space='raw_recording_frame_index',onset_frame_index_base=0)
            labels['tasks']['toy']['failure_onsets'][19]={'a':{'frame_index':0}}
            p=Path(tmp)/'labels.yaml';p.write_text(yaml.safe_dump(labels))
            with self.assertRaises(ValueError): b.load_labels(p,['toy'])


class ScorerBootstrapTests(unittest.TestCase):
    def test_wm_subprocess_resolves_sibling_helpers_without_pythonpath(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('Torch is required for the real scorer entry point')
        with tempfile.TemporaryDirectory() as directory:
            metadata=Path(directory)/'environment.json'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                     PYTHONDONTWRITEBYTECODE='1')
            env.pop('PYTHONPATH',None)
            result=subprocess.run([sys.executable,'-c',b.SCORER_BOOTSTRAP,str(PATH.parents[1]),
                                   'models.flow_matching_wm.score','0','cpu',str(metadata),'--help'],
                                  cwd=directory,env=env,capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--checkpoint',result.stdout)
            self.assertEqual(json.loads(metadata.read_text())['device'],'cpu')


if __name__=='__main__': unittest.main()
