import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from act_ensemble_score import pose_dispersion, rotations_from_6d, difference_scores
from online_benchmark import replay_episode, frozen_latencies


class EnsembleTests(unittest.TestCase):
    def actions(self):
        a = np.zeros((2, 3, 10))
        a[..., 3] = 1; a[..., 7] = 1
        return a

    def test_zero_disagreement_and_ignored_gripper(self):
        a = self.actions(); a[..., -1] = np.arange(3)
        np.testing.assert_allclose(pose_dispersion(a), 0, atol=1e-8)

    def test_position_population_variance_in_physical_units(self):
        a = self.actions(); a[..., 0] = [-.001, 0, .001]
        np.testing.assert_allclose(pose_dispersion(a)[:, 0], 2e-6/3)

    def test_rotation_columns_geodesic_angle_and_not_squared_angle(self):
        a = self.actions(); angles = np.array([-.2, 0., .2])
        a[..., 3] = np.cos(angles); a[..., 4] = np.sin(angles)
        a[..., 6] = -np.sin(angles); a[..., 7] = np.cos(angles)
        rotation = rotations_from_6d(a[..., 3:9])
        np.testing.assert_allclose(np.linalg.det(rotation), 1.)
        np.testing.assert_allclose(pose_dispersion(a)[:, 1], .4/3, atol=1e-8)

    def test_no_cross_episode_difference_and_signed_orientation(self):
        ep, fr, values = difference_scores([0,0,0,1,1], [0,1,2,0,1],
                                            [[1,1],[2,3],[1,0],[9,9],[8,8]])
        np.testing.assert_array_equal(ep, [0,0,1]); np.testing.assert_array_equal(fr,[1,2,1])
        np.testing.assert_allclose(values, [[1,2],[-1,-3],[-1,-1]])
        with self.assertRaises(ValueError): difference_scores([0,0], [0,2], [[1,1],[2,2]])

    def test_rotation_degeneracy_is_explicit(self):
        with self.assertRaises(ValueError): rotations_from_6d(np.zeros((3,6)))


class FrozenReplayTests(unittest.TestCase):
    def replay(self, scores, method='rnd'):
        class Clock:
            now = 0.
            def __call__(self): return self.now
            def sleep(self, seconds): self.now += seconds
        clock = Clock()
        class Adapter:
            def reset_episode(self): pass
            def score(self, payload, window):
                clock.now += .01
                return {'both': scores[window.endpoint]}
        payload = {'raw':np.arange(len(scores)), 'elapsed':np.arange(len(scores),dtype=float),
                   'raw_wall':np.arange(len(scores),dtype=float)}
        result = replay_episode(payload, Adapter(), lambda v,t: 1., ['both'], method,
                                clock=clock, sleep=clock.sleep, stop_at_first_alarm=False)
        return result, payload

    def label(self, result, payload, task='shunt_insertion', origin='natural'):
        labels={'annotators':{'a':{'name':'A'}},'tasks':{task:{
            'groups':[{'episode_range':[39,39], 'failure_origin':origin}],
            'failure_onsets':{39:{'a':{'views':{'fix':{'frame_index':3}, 'wrist':{'frame_index':2}}}}}}}}
        return frozen_latencies(result, {'task':task,'episode_index':39,'variant':'main','method':'rnd',
            'requested_detector_views':['both']}, payload, labels, {'protocol_id':'test'})[0]

    def test_early_only_is_miss_and_complete_trace_is_retained(self):
        result, payload = self.replay([2.,0.,0.,0.])
        self.assertEqual(len(result['events']),4)
        self.assertEqual(result['stop_reason'],'source_exhausted')
        self.assertEqual(result['alarms']['both']['raw_frame'],0)
        self.assertTrue(all('both' in e['thresholds'] for e in result['events']))
        row=self.label(result,payload)
        self.assertFalse(row['post_alarm']); self.assertTrue(row['early_alarm'])
        self.assertIsNone(row['delay_s'])

    def test_shared_earlier_onset_and_persistence_negative_delay(self):
        result, payload=self.replay([0.,2.,2.,0.])
        row=self.label(result,payload)
        self.assertTrue(row['post_alarm']); self.assertEqual(row['onset_raw_frame'],2)
        self.assertAlmostEqual(row['delay_s'],-.99)
        row=self.label(result,payload,'needle_pickup','induced')
        self.assertAlmostEqual(row['delay_s'],.01)

    def test_ensemble_first_score_requires_previous_frame(self):
        result, _=self.replay([100.,0.,2.], 'act_ensemble_pos')
        self.assertEqual([e['endpoint'] for e in result['events']],[1,2])
        self.assertEqual(result['alarms']['both']['raw_frame'],2)

    def test_frozen_vlm_early_only_and_persistent_alarms(self):
        from vlm_online_benchmark import summarize_trial
        from benchmark_frozen_protocol import load_protocol
        spec=load_protocol();aid=spec['detection']['primary_annotator']
        task_labels={'groups':[{'episode_range':[39,39],'failure_origin':'natural'}],
                     'failure_onsets':{39:{aid:{'views':{'fix':{'frame_index':2},'wrist':{'frame_index':2}}}}}}
        labels={'tasks':{'shunt_insertion':task_labels}}
        plan={'config':{'evaluation_protocol':'lerobot/configs/benchmark/evaluation_protocol.yaml',
                        'evaluation':{'primary_annotator_id':aid}},'labels':labels}
        trial={'task':'shunt_insertion','view':'both','episode':39,'outcome':'failure'}
        # VLM runs are one view each; remap this scheduler fixture to fixed.
        trial['view']='fix'
        for scores,detected in [([2.,0.,0.,0.],False),([0.,2.,2.,0.],True)]:
            replay,payload=self.replay(scores);payload['preload_s']=0.
            replay['alarms']={'fix':replay['alarms']['both']}
            for event in replay['events']:
                event['scores']={'fix':event['scores']['both']}
                event['thresholds']={'fix':event['thresholds']['both']}
            result=summarize_trial(plan,trial,payload,replay,[])
            self.assertEqual(bool(result['alarm']),detected)
            self.assertTrue(result['any_time_alarm'])
            if detected:self.assertAlmostEqual(result['post_onset_evidence_delay_s'],-.99)


if __name__ == '__main__': unittest.main()
