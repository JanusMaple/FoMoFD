"""Mean wall-time reference and replay re-interpretation: CPU only."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from benchmark_consensus_onsets import mean_annotation, select_annotation, elapsed_onset
from benchmark_onset_results import automatic_latencies
from benchmark_frozen_protocol import alarm_run, apply_offline


class ConsensusTests(unittest.TestCase):
    def test_irregular_clock_mean_is_not_mean_frame(self):
        wall = np.array([100., 100.1, 100.2, 101.8, 102.])
        a = mean_annotation({'a': {'frame_index': 1}, 'b': {'frame_index': 2},
                             'c': {'frame_index': 4}}, wall)
        self.assertAlmostEqual(a['elapsed_seconds'], 2.3 / 3)
        self.assertEqual(a['frame_index'], 3)
        curve = {'raw': [1, 2, 3, 4], 'wall': [.1, .2, 1.8, 2.], 'score': [1, 1, 1, 1]}
        r = alarm_run(curve, a['frame_index'], elapsed_onset(a, wall),
                      allow_anticipation=True, first_post_onset=True)
        self.assertAlmostEqual(r['detection_delay_s'], 1.8 - 2.3 / 3)
        self.assertTrue(r['sustained_pre_onset_alarm'])
        self.assertFalse(r['early_alarm'])

    def test_copies_keep_timestamp_and_provenance(self):
        a = mean_annotation({'a': {'frame_index': 1},
                             'b': {'frame_index': 1, 'copied_from_annotator': 'a'},
                             'c': {'frame_index': 1, 'copied_from_annotator': 'a'}}, [0, .3, .7])
        self.assertEqual(a['frame_index'], 1)
        self.assertAlmostEqual(a['elapsed_seconds'], .3)
        self.assertEqual(a['copied_contributors'], ['b', 'c'])

    def test_missing_annotation_is_not_dropped_from_mean(self):
        for frame in [None, 5, -1, 1.5]:
            with self.assertRaises(ValueError):
                mean_annotation({'a': {'frame_index': 0}, 'b': {'frame_index': frame}}, [0., .1])

    def test_combined_uses_earlier_mean_even_in_same_frame_interval(self):
        entry = {'views': {'fix': {'frame_index': 3, 'wall_timestamp': .9},
                           'wrist': {'frame_index': 3, 'wall_timestamp': .6}}}
        self.assertEqual(select_annotation(entry, 'both')['wall_timestamp'], .6)
        rows = [dict(view='both', onset_view=v, onset_raw_frame=3, onset_time_s=t)
                for v, t in [('fix', .9), ('wrist', .6)]]
        self.assertEqual(automatic_latencies(rows)[0]['onset_view'], 'wrist')

    def test_censored_trace_is_unknown_after_mean_moves(self):
        r = alarm_run({'raw': [0, 1], 'wall': [.1, .2], 'score': [1., 1.]}, 2, .3,
                      allow_anticipation=True, first_post_onset=True, complete=False)
        self.assertIsNone(r['post_alarm'])
        self.assertIsNone(r['detection_delay_s'])

    def test_offline_interpreter_uses_mean_not_ceiling_timestamp(self):
        from unified_benchmark import report_detector
        view = {'frame_index': 2, 'wall_timestamp': 100.6, 'elapsed_seconds': .6}
        labels = {'annotators': {'consensus_mean': {'name': 'Mean'}}, 'tasks': {'task': {
            'groups': [{'episode_range': [39, 39], 'outcome': 'failure', 'split': 'test',
                        'failure_mode': 'Drop', 'failure_origin': 'natural'}],
            'failure_onsets': {39: {'consensus_mean': {'views': {'fix': view, 'wrist': view}}}}}}}
        curve = {'raw': np.array([0, 1, 2]), 'frame': np.array([0, 1, 2]),
                 'wall': np.array([100., 100.5, 101.]), 'score': np.array([-1., 1., 1.])}
        job = SimpleNamespace(task='task', dataset=Path('/unused'))
        rows = report_detector(job, 'fix', 'wm_nc', 'episode', {39: curve}, {39: curve}, {}, labels, ['consensus_mean'])
        spec = {'protocol_id': 'test', 'detection': {'primary_annotator': 'consensus_mean'},
                'latency': {'detection_start': 'first_positive_decision_with_source_endpoint_at_or_after_onset'}}
        apply_offline(rows, job, 'fix', {39: curve}, {39: curve}, {}, labels, spec)
        self.assertAlmostEqual(rows['latencies'][0]['detection_delay_s'], .4)
        self.assertTrue(rows['episodes'][0]['alarm'])
        self.assertFalse(rows['episodes'][0]['early_alarm'])


if __name__ == '__main__':
    unittest.main()
