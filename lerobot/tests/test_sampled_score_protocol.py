"""Normal-only fitting, fixed-noise integration and frozen-setting invariants."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
import sampled_score_protocol as p


class SampledProtocolTests(unittest.TestCase):
    def test_episode_partition_is_disjoint_and_reproducible(self):
        a, b = p.normal_partition(list(range(25)), 19)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a+b), set(range(25)))
        self.assertEqual((a,b), p.normal_partition(list(reversed(range(25))),19))

    def test_fits_small_physical_spread_without_saturating(self):
        rng = np.random.default_rng(3)
        samples = rng.normal(size=(60,32,4,3))*2e-6
        cfg = dict(min_width_m=1e-8,width_multipliers=[8,4,2,1,.5,.25],criteria=dict(
            max_zero_fraction=.25,max_saturated_fraction=.1,median_entropy_min=.5,
            median_entropy_max=4,min_unique_scores=10))
        result = p.fit_ace_resolution(samples[:40],samples[40:],cfg)
        self.assertTrue(result['passed'])
        self.assertLess(max(result['widths_m']),1e-4)
        self.assertEqual(p.histogram_health(samples,np.ones(3)*.002)['zero_fraction'],1)

    def test_degenerate_samples_fail_instead_of_forcing_useful_score(self):
        cfg = dict(min_width_m=1e-8,width_multipliers=[1,.5],criteria=dict(
            max_zero_fraction=.25,max_saturated_fraction=.1,median_entropy_min=.5,
            median_entropy_max=4,min_unique_scores=10))
        self.assertFalse(p.fit_ace_resolution(np.zeros((20,32,4,3)),np.zeros((20,32,4,3)),cfg)['passed'])

    def test_holdout_does_not_select_widths(self):
        rng = np.random.default_rng(1)
        fit = rng.normal(size=(40,32,4,3))*1e-5
        cfg = dict(min_width_m=1e-8,width_multipliers=[4,2,1,.5],criteria=dict(
            max_zero_fraction=.25,max_saturated_fraction=.1,median_entropy_min=.5,
            median_entropy_max=4,min_unique_scores=10))
        a=p.fit_ace_resolution(fit,fit[:20],cfg)
        b=p.fit_ace_resolution(fit,np.zeros_like(fit[:20]),cfg)
        self.assertEqual(a['widths_m'],b['widths_m'])
        self.assertFalse(b['passed'])

    def test_two_refinements_required_and_cap_is_not_success(self):
        criteria={'score_relative_p95':.05}
        good={'score_relative_p95':[.01,.02]}; bad={'score_relative_p95':[.1,.01]}
        rec=[dict(coarse_steps=1,fine_steps=2,error=bad),dict(coarse_steps=2,fine_steps=4,error=good)]
        self.assertIsNone(p.select_converged_steps(rec,criteria))
        rec.append(dict(coarse_steps=4,fine_steps=8,error=good))
        self.assertEqual(p.select_converged_steps(rec,criteria),2)

    def test_refinement_retains_camera_axis(self):
        rng=np.random.default_rng(0);x=rng.normal(size=(5,16,2,2,2,3));y=x.copy();y[:,:,1]*=2
        result=p.refinement_error(x,y)
        self.assertEqual(result['score_relative_p95'][0],0)
        self.assertAlmostEqual(result['score_relative_p95'][1],.5)

    def test_fixed_noise_integration_converges_and_one_step_is_legacy(self):
        import torch
        import unified_benchmark as old
        class Field(torch.nn.Module):
            def set_context(self,*args):pass
            def forward(self,t,x):return x
        model=SimpleNamespace(odefunc=Field(),_num_views=lambda:2,
            config=SimpleNamespace(fm_head=SimpleNamespace(ode_solver='rk4',base_std=1.,ode_atol=1e-5,ode_rtol=1e-4)))
        inputs=dict(fix_hist=None,wrist_hist=None,state_hist=None,action_chunk=torch.zeros(2,4,10))
        noise=old.frame_noise(dict(task='toy',method='wm_pu',seed=0),[0,1],[7,7],16,(2,2,2,1))
        base=old.score_wm_pu_batch(model,inputs,noise,4,'float32')
        actual=p.score_wm_pu_batch(model,inputs,noise,4,'float32',1)
        np.testing.assert_array_equal(base,actual)
        # Independent RK4 reference for x'=x, h=1: 1+1+1/2+1/6+1/24.
        expected_one_step=noise.std(1).reshape(2,2,-1).mean(-1)*(65/24)
        np.testing.assert_allclose(actual.numpy(),expected_one_step,rtol=3e-7)
        fine=p.score_wm_pu_batch(model,inputs,noise,4,'float32',8).numpy()
        target=noise.std(1).reshape(2,2,-1).mean(-1)*np.e
        self.assertLess(np.max(np.abs(fine-target)),np.max(np.abs(actual.numpy()-target)))

    def test_frozen_manifest_rejects_unfrozen_wrong_model_and_wrong_role(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'frozen.json'
            data={'status':'frozen_before_policy_calibration_and_test','tasks':{'toy':{'ace':{
                'passed':True,'checkpoint_sha256':'abc','source_role':'expert_validation'}}}}
            path.write_text(json.dumps(data))
            self.assertTrue(p.load_frozen_sampling(path,d,'toy','ace','abc')['passed'])
            with self.assertRaises(ValueError):p.load_frozen_sampling(path,d,'toy','ace','wrong')
            data['status']='draft';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):p.load_frozen_sampling(path,d,'toy','ace','abc')

    def test_prefix_maxima_are_monotone_and_phase_is_only_diagnostic(self):
        r=p.phase_length_summary({0:{'score':np.array([1.,3.,2.,5.]),'wall':np.arange(4.)}})
        values=[x['maximum'] for x in r['prefixes']]
        self.assertTrue(np.all(np.diff(values)>=0))
        self.assertIn('never online',r['phase_note'])

    def test_ode_count_validation(self):
        for invalid in [True,0,-1,2.5,1025]:
            with self.assertRaises(ValueError):p.validate_ode_steps(invalid)


if __name__=='__main__':unittest.main()
