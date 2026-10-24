"""CPU-only tests of causal fusion, component costs, and provenance guards."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
import online_benchmark as online


class Clock:
    def __init__(self): self.t = 0.
    def __call__(self): return self.t
    def sleep(self, seconds): self.t += seconds


class FusionTests(unittest.TestCase):
    def fixture(self, operation='and', native=False):
        clock, calls = Clock(), []
        left = dict(method='wm_nc', view='fix', dataset='vae', offline_score_cache='wm',
                    K=1, L=1, threshold=1., scale=2.)
        right = dict(left, view='wrist')
        if native:
            right.update(method='logpzo_native', view='fix', dataset='tokens', offline_score_cache='native', K=None, L=None)
        trial = {'fusion': {'components': [left, right], 'operation': operation,
                            'output_view': 'fix' if native else 'both'}}
        class Fake:
            def __init__(self, spec, device): self.method=spec['method'];self.last_stage_times={}
            def reset_episode(self): pass
            def score(self, payload, window):
                calls.append((self.method, window.raw_endpoint, window.history))
                clock.t += .16
                self.last_stage_times={'encoder_s':.10, 'detector_s':.06,
                                       'ode_calls':[{'steps':1,'evaluations':4}]}
                # Different-frame alarms must NOT satisfy AND.
                return {'fix': 2. if window.raw_endpoint % 2 else 0.,
                        'wrist': 0. if window.raw_endpoint % 2 else 2.}
        adapter = online.FusionAdapter(trial, 'cpu', factory=Fake)
        base = {'raw': np.arange(6), 'elapsed': np.arange(6) * .1}
        payload = dict(base, components={'vae':dict(base), 'tokens':dict(base)})
        return adapter, payload, clock, calls

    def test_and_requires_same_frame_not_latched_component_alarms(self):
        adapter, payload, clock, calls = self.fixture()
        result = online.replay_episode(payload, adapter, lambda v,t:0., ['both'], 'wm_nc_and',
                                       1, 1, clock=clock, sleep=clock.sleep)
        self.assertEqual(result['alarms'], {})
        self.assertEqual(len(calls), result['processed_endpoints'])  # one WM pass
        self.assertTrue(result['skipped_endpoints'] > 0)

    def test_or_alarms_and_includes_encoder_and_detector_time(self):
        adapter, payload, clock, calls = self.fixture('or')
        result = online.replay_episode(payload, adapter, lambda v,t:0., ['both'], 'wm_nc_or',
                                       1, 1, clock=clock, sleep=clock.sleep)
        self.assertEqual(result['alarms']['both']['raw_frame'], 1)
        self.assertAlmostEqual(result['alarms']['both']['alarm_time_s'], .26)
        self.assertEqual(len(calls), 1)

    def test_native_fusion_scores_same_raw_endpoint_and_charges_both(self):
        adapter, payload, clock, calls = self.fixture('or', native=True)
        result = online.replay_episode(payload, adapter, lambda v,t:0., ['fix'], 'wm_nc_logpzo_native_or',
                                       1, 1, clock=clock, sleep=clock.sleep)
        self.assertEqual(calls, [('wm_nc',1,(0,)), ('logpzo_native',1,())])
        self.assertAlmostEqual(result['alarms']['fix']['alarm_time_s'], .42)
        self.assertEqual(len(adapter.last_stage_times['ode_calls']), 2)
        self.assertAlmostEqual(adapter.last_stage_times['encoder_s'], .20)

    def test_strict_threshold_and_sign_preserving_scales(self):
        specs=[dict(threshold=10.,scale=100.),dict(threshold=-4.,scale=.1)]
        self.assertEqual(online.fusion_margin([10.,-4.], specs, 'or'), 0.)
        self.assertLess(online.fusion_margin([11.,-5.], specs, 'and'),0)
        self.assertGreater(online.fusion_margin([11.,-5.], specs, 'or'),0)
        with self.assertRaises(ValueError):
            online.fusion_margin([11.,-5.], [dict(threshold=0,scale=-1), specs[1]], 'or')

    def test_reject_missing_raw_endpoint_or_changed_wall_time(self):
        adapter,payload,clock,calls=self.fixture(native=True)
        window=online.input_window(1,payload['raw'],payload['elapsed'],'wm_nc',1,1)
        payload['components']['tokens']['raw']=np.array([0,2,3,4,5])
        with self.assertRaisesRegex(ValueError,'matching raw'):adapter.score(payload,window)
        payload['components']['tokens']['raw']=np.arange(6)
        payload['components']['tokens']['elapsed']=payload['elapsed']+1
        with self.assertRaisesRegex(ValueError,'wall time'):adapter.score(payload,window)

    def test_private_frozen_package_checks_every_module(self):
        original=online.REPO/'lerobot/models/flow_matching_wm'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);identity={'sha256':{}}
            for name in ['__init__.py','config.py','model.py','dataset.py','score.py']:
                (root/name).write_text('# frozen\n')
                identity['sha256'][str(original/name)]=hashlib.sha256((root/name).read_bytes()).hexdigest()
            self.assertEqual(len(online.frozen_model_package(identity, root)),5)
            (root/'model.py').write_text('# changed\n')
            with self.assertRaisesRegex(ValueError,'differs'):online.frozen_model_package(identity,root)

    def test_output_view_is_part_of_trial_identity(self):
        trial={'input_views':['fix','wrist'],'fusion':{'output_view':'fix'}}
        other=copy.deepcopy(trial);other['fusion']['output_view']='wrist'
        self.assertNotEqual(online.trial_viewtag(trial),online.trial_viewtag(other))


if __name__ == '__main__': unittest.main()
