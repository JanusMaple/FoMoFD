"""CPU-only tests of functional calibration, data roles and finite-sample rules."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'utils'))
import functional_conformal as f
import unified_benchmark as b


def trace(values,times=None):
    n=len(values)
    return {'score':np.asarray(values,dtype=float),'wall':np.asarray(times if times is not None else range(n),dtype=float),'raw':np.arange(n),'frame':np.arange(n)}

def clocks(source):return {e:{'wall':np.array([0.])} for e in source}

def constant_template(mu=0,g=1):
    return {'grid_s':[0.,1.],'mu':[mu,mu],'g':[g,g],'extrapolation':'hold_boundary_values'}


class FunctionalBandTests(unittest.TestCase):
    def test_rank_19_is_maximum(self):
        self.assertEqual(f.order_statistic(list(range(19)),.05),(18.,19))
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(f.order_statistic(list(range(18)),.05),(math.inf,19))
            self.assertEqual(len(w),1)
        with self.assertWarns(RuntimeWarning):self.assertEqual(f.order_statistic([],.05),(math.inf,1))

    def test_constant_template_recovers_episode_max_and_strict_ties(self):
        source={e:trace([0,e,0]) for e in range(21)}
        source[19]=trace([18.,18.,18.])
        timeline=clocks(source)
        band=f.calibrate_band(source,timeline,list(range(19)),constant_template(),.05)
        curves=f.apply_band(source,timeline,constant_template(),band)
        self.assertEqual(band['h'],b.conformal_threshold([source[e]['score'].max() for e in range(19)],.05))
        self.assertFalse((curves[19]['score']>0).any())
        self.assertTrue((curves[20]['score']>0).any())

    def test_signed_upper_residual_not_absolute(self):
        source={e:trace([-100,2]) for e in range(19)}
        band=f.calibrate_band(source,clocks(source),list(source),constant_template(1),.05)
        self.assertEqual(band['h'],1)

    def test_crossing_equals_trajectory_residual_above_h(self):
        rng=np.random.default_rng(321)
        source={e:trace(rng.normal(size=3+e),np.linspace(.2,20,3+e)) for e in range(29)}
        template={'grid_s':[0,2,5],'mu':[1,2,-1],'g':[.5,1,2],'extrapolation':'hold_boundary_values'}
        band=f.calibrate_band(source,clocks(source),list(range(19)),template,.05)
        for curve in f.apply_band(source,clocks(source),template,band).values():
            self.assertEqual(bool((curve['score']>0).any()),bool(curve['roc_score'].max()>band['h']))
            self.assertEqual(bool((curve['score']>0).any()),bool((curve['raw_score']>curve['threshold']).any()))

    def test_variable_lengths_do_not_require_19_per_bin(self):
        source={e:trace([e,e],times=[0,1 if e else 100]) for e in range(20)}
        template=constant_template()
        band=f.calibrate_band(source,clocks(source),list(range(19)),template,.05)
        out=f.apply_band(source,clocks(source),template,band)
        self.assertTrue(np.isfinite(out[0]['threshold']).all())
        np.testing.assert_array_equal(out[0]['support'],[19,19])
        self.assertEqual(out[19]['threshold'][-1],18)

    def test_template_available_support_and_boundary_hold(self):
        source={100:trace([1,3],[1,3]),101:trace([3,3],[1,2])}
        template=f.fit_template(source,clocks(source),list(source),{'role':'expert_validation'},{'grid_seconds':1})
        # At t=2: expert A interpolates to 2, expert B supplies 3.
        mu,g=f.template_values(template,np.array([0,1,2,3,100.]))
        np.testing.assert_array_equal(mu,[2,2,2.5,3,3]);np.testing.assert_array_equal(g,np.ones(5))
        self.assertEqual(template['contributors'][-1],1)

    def test_extra_test_episode_cannot_enter_template(self):
        source={100:trace([1,2]),999:trace([999,999])}
        with self.assertRaisesRegex(ValueError,'expert-validation'):
            f.fit_template(source,clocks(source),[100],{'role':'expert_validation'})
        with self.assertRaises(ValueError):f.fit_template(source,clocks(source),[100,999],{'role':'policy_test'})

    def test_calibration_does_not_see_evaluation_values(self):
        source={e:trace([e,e+1]) for e in range(20)};timeline=clocks(source)
        one=f.calibrate_band(source,timeline,list(range(19)),constant_template(),.05)
        source[19]=trace([-1e9,1e9])
        self.assertEqual(one,f.calibrate_band(source,timeline,list(range(19)),constant_template(),.05))

    def test_nonfinite_empty_invalid_time_are_explicit_errors(self):
        for curve in [trace([]),trace([np.nan]),trace([np.inf]),trace([-np.inf]),trace([1,2],[1,0]),trace([1],[-1])]:
            with self.assertRaisesRegex(ValueError,'Episode 7'):
                f.valid_trace(curve,{'wall':[0]},episode=7)
        for g in [0,-1,np.nan]:
            with self.assertRaises(ValueError):f.template_values(constant_template(g=g),[0])

    def test_optional_scale_is_positive_and_expert_only(self):
        source={e:trace([1,1]) for e in range(3)}
        template=f.fit_template(source,clocks(source),list(source),{'role':'expert_validation'}, {'modulation':'robust_mad','scale_floor':.25})
        np.testing.assert_array_equal(template['g'],[.25,.25])

    def test_alias_is_identical_to_historical_bins(self):
        source={e:trace([e,e+1]) for e in range(20)}
        a,ar=b.calibrate_curves(source,list(range(19)),clocks(source),.05,'time_varying',{})
        c,cr=b.calibrate_curves(source,list(range(19)),clocks(source),.05,'wall_time_bin',{})
        self.assertEqual(ar,cr)
        for e in source:np.testing.assert_array_equal(a[e]['score'],c[e]['score'])

    def test_unified_evaluation_uses_residual_for_roc(self):
        import test_unified_benchmark as old
        with tempfile.TemporaryDirectory() as tmp:
            scored,labels,config=old.toy_suite(Path(tmp))
            labels['tasks']['toy']['failure_onsets'][21]['a']['frame_index']=3
            labels['tasks']['toy']['failure_onsets'][22]={'a':{'frame_index':1}}
            templates={(j.task,j.method,v,p):constant_template(1,2) for j,_,_ in scored for v in j.views for p in ['paper','latency']}
            result=b.evaluate_suite(scored,labels,config,list(labels['annotators']),['wm_nc','logpzo','ace'],['functional_band'],templates)
            self.assertTrue(all(r['method'] in ['wm_nc','logpzo','ace'] for r in result['episodes']))
            self.assertEqual(max(r['score_max'] for r in result['episodes']),4.5)
            self.assertTrue(all(r['n_cal']==19 and r['rank']==19 for r in result['calibration']))
            # Tailored curve crossing/latency conventions are the existing helper.
            self.assertIn('pre_onset_alarm',{r['status'] for r in result['latencies']})


if __name__=='__main__':unittest.main()
