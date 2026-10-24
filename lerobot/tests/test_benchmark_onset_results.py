"""CPU-only onset FDR invariants; no scoring or inference."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from benchmark_onset_results import assess, assess_stream, onset_frame, automatic_latencies, frozen_curves, refresh_offline_latency, refresh_online_latencies

class OnsetResultsTests(unittest.TestCase):
    def test_early_then_later_counts_and_strict_boundary(self):
        c = {'raw': np.array([0, 1, 2, 3]), 'score': np.array([2., -1., 0., 1.])}
        a = assess(c, 2)
        self.assertTrue(a['early_alarm']); self.assertTrue(a['post_alarm'])
        self.assertEqual(a['post_score_max'], 1)
        self.assertTrue(assess(c, 3)['post_alarm'])
        c['score'][-1] = 0
        self.assertFalse(assess(c, 2)['post_alarm'])

    def test_missing_and_after_end(self):
        c = {'raw': [0, 1], 'score': [2., 1.]}
        self.assertIsNone(assess(c, None)['post_alarm'])
        a = assess(c, 2)
        self.assertFalse(a['post_alarm']); self.assertEqual(a['post_score_max'], '-inf')

    def test_matching_views_and_earlier_combined(self):
        a = {'views': {'fix': {'frame_index': 20}, 'wrist': {'frame_index': 10}}}
        self.assertEqual(onset_frame(a, 'fix'), 20)
        self.assertEqual(onset_frame(a, 'wrist'), 10)
        self.assertEqual(onset_frame(a, 'both'), 10)
        del a['views']['fix']
        self.assertIsNone(onset_frame(a, 'both'))
        self.assertEqual(onset_frame({'frame_index': 3}, 'both'), 3)

    def test_combined_latency_selects_once_and_is_idempotent(self):
        r = {'view': 'both', 'annotator_id': 'a', 'onset_view': 'fix', 'onset_time_s': 2., 'status': 'detected', 'delay_s': 2.}
        rows = [r, {**r, 'onset_view': 'wrist', 'onset_time_s': 1., 'delay_s': 3.}]
        output = automatic_latencies(rows)
        self.assertEqual(len(output), 1); self.assertEqual(output[0]['delay_s'], 3.)
        self.assertEqual(automatic_latencies(output), output)
        rows[1]['onset_time_s'] = None
        self.assertEqual(automatic_latencies(rows)[0]['status'], 'unannotated')

    def test_streaming_early_stop_is_unknown_not_a_miss(self):
        e = {'raw_frame': 1, 'scores': {'fix': 1.}, 'thresholds': {'fix': .5}}
        r = {'status': 'complete', 'replay': {'events': [e], 'stop_reason': 'all_selected_views_alarmed'}}
        self.assertIsNone(assess_stream(r, 'fix', 3)['post_alarm'])
        self.assertTrue(assess_stream(r, 'fix', 1)['post_alarm'])
        r['replay']['stop_reason'] = 'source_exhausted'
        self.assertFalse(assess_stream(r, 'fix', 3)['post_alarm'])
        r['replay']['events'].append({**e, 'raw_frame': 4})
        self.assertTrue(assess_stream(r, 'fix', 3)['post_alarm'])

    def test_frozen_threshold_and_functional_score_orientation(self):
        source = {0: {'raw': np.array([0, 1, 2]), 'wall': np.array([0., 1., 2.]), 'score': np.array([9., 3., 7.])}}
        records = [{'bin_index': None, 'threshold': 5., 'scale': 2.}]
        c = frozen_curves(source, records, {0: {'wall': [0.]}})[0]
        np.testing.assert_array_equal(c['score'], [2., -1., 1.])
        a = assess(c, 1)
        self.assertTrue(a['early_alarm']); self.assertTrue(a['post_alarm'])
        self.assertEqual(a['post_score_max'], 1.)
        c['roc_score'] = np.array([20., 10., 12.])
        self.assertEqual(assess(c, 1)['post_score_max'], 12.)
        np.testing.assert_array_equal(source[0]['score'], [9., 3., 7.])

    def test_offline_reannotation_changes_status_without_changing_alarm(self):
        row = {'first_alarm_raw_frame': 2, 'first_alarm_wall_timestamp': 102.,
               'last_scored_wall_timestamp': 104., 'status': 'pre_onset_alarm'}
        wall = np.arange(100., 106.)
        revised = refresh_offline_latency(row, {'frame_index': 1}, wall)
        self.assertEqual(revised['status'], 'detected')
        self.assertEqual(revised['detection_delay_s'], 1.)
        self.assertEqual(revised['first_alarm_raw_frame'], row['first_alarm_raw_frame'])
        self.assertEqual(row['status'], 'pre_onset_alarm')
        self.assertEqual(refresh_offline_latency(row, {'frame_index': 3}, wall)['status'], 'pre_onset_alarm')
        self.assertEqual(refresh_offline_latency({**row, 'first_alarm_raw_frame': None}, {'frame_index': 5}, wall)['status'], 'onset_after_scored_end')
        self.assertEqual(refresh_offline_latency(row, None, wall)['status'], 'unannotated')

    def test_online_reannotation_uses_alarm_completion_and_earlier_joint_onset(self):
        trial = {'variant':'main','task':'t','method':'wm_nc','episode_index':0,
                 'calibration':'episode','requested_detector_views':['both']}
        alarm = {'raw_frame':2,'source_time_s':2.,'alarm_time_s':2.4}
        result = {'alarms': {'both': alarm}}
        labels = {'annotators': {'a': {'name':'A'}}, 'tasks': {'t': {'failure_onsets': {
            0: {'a': {'views': {'fix': {'frame_index':3}, 'wrist': {'frame_index':1}}}}}}}}
        records = refresh_online_latencies(trial,result,labels,np.arange(100.,106.))
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['status'],'detected')
        self.assertAlmostEqual(records[0]['delay_s'],1.4)
        self.assertEqual(records[0]['onset_view'],'wrist')
        labels['tasks']['t']['failure_onsets'][0]['a']['views']['wrist']['frame_index']=4
        records=refresh_online_latencies(trial,result,labels,np.arange(100.,106.))
        self.assertEqual(records[0]['status'],'pre_onset_evidence')
        self.assertIsNone(records[0]['delay_s'])
        self.assertEqual(alarm['alarm_time_s'],2.4)

    def test_vlm_reannotation_retains_late_status_and_evidence_rule(self):
        from benchmark_vlm_results import refresh_trial_onset
        result = {'task':'t','view':'wrist','episode':0,'outcome':'failure',
                  'alarm':{'source_time_s':2.,'alarm_time_s':3.5},'late_alarm':True}
        labels = {'tasks': {'t': {'failure_onsets': {0: {'a': {'views': {'wrist': {'frame_index':3}}}}}}}}
        old = refresh_trial_onset(result,labels,'a',np.arange(100.,106.))
        self.assertTrue(old['pre_onset_evidence']); self.assertFalse(old['pre_onset_alarm'])
        self.assertIsNone(old['post_onset_evidence_delay_s'])
        labels['tasks']['t']['failure_onsets'][0]['a']['views']['wrist']['frame_index']=1
        revised = refresh_trial_onset(result,labels,'a',np.arange(100.,106.))
        self.assertEqual(revised['post_onset_evidence_delay_s'],2.5)
        self.assertTrue(revised['late_alarm'])
        self.assertEqual(revised['alarm'], result['alarm'])

if __name__ == '__main__': unittest.main()
