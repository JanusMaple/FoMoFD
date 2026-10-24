"""v2 trace interpretation; no models or GPU."""
from pathlib import Path
import sys
import unittest
import tempfile
import csv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from benchmark_frozen_protocol import alarm_run, export_interpreted
from online_benchmark import frozen_latencies


class PostOnsetProtocolTests(unittest.TestCase):
    def run_trace(self, scores, *, times=None, natural=True, complete=True, frames=None):
        return alarm_run({'raw':list(range(len(scores))) if frames is None else frames,
                          'score':scores,'wall':np.arange(len(scores))+.4 if times is None else times},
                         3,3.,allow_anticipation=natural,complete=complete,first_post_onset=True)

    def test_sustained_run_uses_actual_post_completion_not_zero(self):
        r=self.run_trace([-1,1,1,1])
        self.assertAlmostEqual(r['detection_delay_s'],.4)
        self.assertEqual(r['detection_raw_frame'],3)
        self.assertTrue(r['any_pre_onset_alarm']);self.assertTrue(r['sustained_pre_onset_alarm'])
        self.assertFalse(r['early_alarm'])

    def test_disconnected_early_run_is_retained(self):
        r=self.run_trace([1,-1,1,1])
        self.assertTrue(r['sustained_pre_onset_alarm']);self.assertTrue(r['early_alarm'])

    def test_simulated_induced_has_no_early_exemption(self):
        r=self.run_trace([1,1,1,1],natural=False)
        self.assertFalse(r['sustained_pre_onset_alarm']);self.assertTrue(r['early_alarm'])
        self.assertAlmostEqual(r['detection_delay_s'],.4)

    def test_indeterminate_or_normal_breaks_run(self):
        for value in [0,-1,np.nan]:
            r=self.run_trace([1,1,value,1])
            self.assertFalse(r['sustained_pre_onset_alarm']);self.assertTrue(r['early_alarm'])

    def test_early_only_and_censored(self):
        r=self.run_trace([1,1,-1,-1]);self.assertFalse(r['post_alarm'])
        self.assertIsNone(r['detection_delay_s'])
        r=self.run_trace([1,1],complete=False);self.assertIsNone(r['post_alarm'])

    def test_source_frame_controls_eligibility(self):
        r=self.run_trace([-1,1,1,1],times=[2,3.2,4,5])
        self.assertEqual(r['detection_delay_s'],2)
        self.assertFalse(r['sustained_pre_onset_alarm']);self.assertTrue(r['early_alarm'])

    def test_skips_use_observed_decisions_without_interpolation(self):
        r=self.run_trace([1,1],frames=[1,7],times=[1.4,7.4])
        self.assertAlmostEqual(r['detection_delay_s'],4.4)
        self.assertEqual(r['detection_raw_frame'],7)

    def test_v1_unchanged(self):
        r=alarm_run({'raw':[0,1,2,3],'score':[-1,1,1,1],'wall':[0,1,2,3]},3,3.,allow_anticipation=True)
        self.assertEqual(r['detection_delay_s'],-2.)

    def test_online_writer_passes_selected_protocol_to_interpreter(self):
        labels = {'annotators': {'a': {'name': 'Annotator'}}, 'tasks': {'needle_pickup': {
            'groups': [{'episode_range': [39, 39], 'failure_origin': 'natural'}],
            'failure_onsets': {39: {'a': {'frame_index': 3}}}}}}
        trial = {'variant': 'main', 'task': 'needle_pickup', 'method': 'wm_nc',
                 'episode_index': 39, 'requested_detector_views': ['fix']}
        events = [{'raw_frame': i, 'scores': {'fix': value}, 'thresholds': {'fix': 0.},
                   'alarm_ready_s': i + .4} for i, value in enumerate([-1, 1, 1, 1, -1])]
        result = {'events': events, 'stop_reason': 'source_exhausted', 'alarms': {'fix': {}}}
        payload = {'raw_wall': np.arange(5, dtype=float)}
        v1 = {'protocol_id': 'v1', 'latency': {}}
        v2 = {'protocol_id': 'v2', 'latency': {
            'detection_start': 'first_positive_decision_with_source_endpoint_at_or_after_onset'}}
        old = frozen_latencies(result, trial, payload, labels, v1)[0]
        current = frozen_latencies(result, trial, payload, labels, v2)[0]
        self.assertAlmostEqual(old['delay_s'], -1.6)
        self.assertAlmostEqual(current['delay_s'], .4)
        self.assertEqual(current['detection_raw_frame'], 3)
        self.assertTrue(current['sustained_pre_onset_alarm'])
        self.assertFalse(current['early_alarm'])
        self.assertTrue(current['any_pre_onset_alarm'])
        self.assertEqual(current['alarm'], old['alarm'])

        # The actual post-onset completion, rather than an earlier sustained
        # alarm, must also determine whether the decision was late.
        events[3]['alarm_ready_s'] = 4.5
        events[4]['alarm_ready_s'] = 4.8
        current = frozen_latencies(result, trial, payload, labels, v2)[0]
        self.assertTrue(current['late_alarm'])
        self.assertAlmostEqual(current['delay_s'], 1.5)

    def test_invalid_clock_not_silently_clipped(self):
        with self.assertRaisesRegex(ValueError,'completed before onset'):
            self.run_trace([-1,-1,-1,1],times=[0,.1,.2,.3])

    def test_export_handles_success_rows_with_null_onset_assessment(self):
        data={'created':'2026-10-16','evaluation_protocol':{'protocol_id':'v2'},
              'latencies':{'columns':[],'rows':[]},'online':[],'online_runs':[],'vlm_runs':[],
              'onset_analysis':{'labels_sha256':'test'},
              'episodes':{'columns':['variant','task','method','view','calibration','episode_index','onset_assessments'],
                          'rows':[['main','task','wm_nc','fix','episode',19,None],
                                  ['main','task','wm_nc','fix','episode',39,{'a':{'post_alarm':True}}]]}}
        with tempfile.TemporaryDirectory() as folder:
            export_interpreted(data,Path(folder))
            with (Path(folder)/'analyses/onset_detection/episode_assessments.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['episode_index'],'39')


if __name__=='__main__':unittest.main()
