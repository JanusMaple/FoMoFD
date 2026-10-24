"""Synthetic onset-boundary and protocol-lock checks; no datasets/models/GPU."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from benchmark_frozen_protocol import alarm_run, allows_anticipation, load_protocol, validate_config, recorded_curve, resampling_post_max, REPO
from calibration_sensitivity import evaluate_splits
from benchmark_roc import operating_point


class FrozenProtocolTests(unittest.TestCase):
    def assess(self, values, *, simulated=False, complete=True, times=None):
        return alarm_run({'raw': np.arange(len(values)), 'score': values,
                          'wall': np.arange(len(values), dtype=float) if times is None else times},
                         3, 3., allow_anticipation=not simulated, complete=complete)

    def test_spanning_early_run_is_negative(self):
        r = self.assess([-1, 1, 1, 1, -1])
        self.assertTrue(r['post_alarm']); self.assertTrue(r['early_alarm'])
        self.assertTrue(r['detected_before_human_label'])
        self.assertEqual(r['detection_delay_s'], -2.)

    def test_simulated_induced_never_gets_negative_credit(self):
        r = self.assess([-1, 1, 1, 1], simulated=True)
        self.assertEqual(r['detection_delay_s'], 0.)
        self.assertFalse(r['detected_before_human_label'])
        self.assertFalse(allows_anticipation('needle_pickup', 'induced'))
        self.assertTrue(allows_anticipation('needle_pickup', 'natural'))
        self.assertTrue(allows_anticipation('shunt_insertion', 'induced'))

    def test_interrupted_or_indeterminate_run_has_no_negative_credit(self):
        for gap in [-1, 0, np.nan]:
            r = self.assess([1, 1, gap, 1])
            self.assertEqual(r['detection_delay_s'], 0.)
            self.assertFalse(r['persistent_early'])

    def test_later_run_cannot_validate_isolated_early_alarm(self):
        r = self.assess([1, 1, 1, -1, 1])
        self.assertEqual(r['detection_delay_s'], 1.)
        self.assertFalse(r['persistent_early'])

    def test_early_only_is_not_detection_and_stopped_trace_is_unknown(self):
        r = self.assess([1, 1, -1, -1])
        self.assertFalse(r['post_alarm']); self.assertTrue(r['early_alarm'])
        self.assertIsNone(r['detection_delay_s'])
        r = self.assess([1, 1], complete=False)
        self.assertIsNone(r['post_alarm']); self.assertEqual(r['status'], 'unobserved_after_stop')

    def test_online_uses_return_time_not_source_time(self):
        r = self.assess([-1, 1, 1, 1], times=[2., 3.5, 4., 5.])
        self.assertTrue(r['persistent_early'])
        self.assertEqual(r['detection_delay_s'], .5)
        self.assertFalse(r['detected_before_human_label'])

    def test_equality_is_not_alarm_and_onset_equality_is_post(self):
        r = self.assess([0, 0, 0, 1])
        self.assertFalse(r['early_alarm']); self.assertTrue(r['post_alarm'])
        self.assertEqual(r['detection_delay_s'], 0.)
        r = self.assess([-np.inf]*4)
        self.assertFalse(r['post_alarm']); self.assertEqual(r['status'],'missed')

    def test_protocol_rejects_changed_support(self):
        c = yaml.safe_load((REPO/'lerobot/configs/benchmark/unified_benchmark.yaml').read_text())
        c['evaluation_protocol'] = 'lerobot/configs/benchmark/evaluation_protocol_v3.yaml'
        validate_config(c)
        c['paper_protocol']['frame_limits'] = []
        with self.assertRaises(ValueError): validate_config(c)

    def test_input_hash_lock(self):
        spec = load_protocol('lerobot/configs/benchmark/evaluation_protocol_v3.yaml'); spec['frozen_inputs']['onsets']['sha256'] = 'bad'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'spec.yaml'; path.write_text(yaml.safe_dump(spec))
            with self.assertRaises(ValueError): load_protocol(path)

    def test_sensitivity_post_onset_fdr_keeps_auc_and_threshold(self):
        q = {i:float(i) for i in range(39)} | {39:100.}
        split = {'split_id':0, 'calibration_ids':list(range(19)), 'test_success_ids':list(range(19,39))}
        old = evaluate_splits(q,list(range(39)),[39],[split])[0]
        new = evaluate_splits(q,list(range(39)),[39],[split],failure_detection_scores={39:0.})[0]
        self.assertEqual(old['FDR'], 1.); self.assertEqual(new['FDR'], 0.)
        for key in ['threshold','FAR','episode_AUROC']: self.assertEqual(old[key],new[key])

    def test_matched_far_uses_post_onset_failure_maximum(self):
        rows = [{'outcome':'success','score_max':i} for i in range(20)]
        rows.append({'outcome':'failure','score_max':100.,'post_score_max':1.,'detection_definition':'at_or_after_onset'})
        self.assertEqual(operating_point(rows,.05)['FDR'],0.)

    def test_recorded_threshold_omission_never_invents_scores(self):
        events = [dict(raw_frame=0, scores={'fix':2.}, thresholds={'fix':1.}, alarm_ready_s=0.),
                  dict(raw_frame=1, scores={'fix':3.}, thresholds={}, alarm_ready_s=1.),
                  dict(raw_frame=2, scores={}, thresholds={}, alarm_ready_s=2.)]
        curve = recorded_curve(events,'fix','episode')
        self.assertEqual(curve['score'][:2],[1.,2.]); self.assertTrue(np.isnan(curve['score'][2]))
        self.assertTrue(np.isnan(recorded_curve(events,'fix','functional_band')['score'][1]))

    def test_resampling_post_max_is_not_normalized_margin(self):
        c = {'raw':np.array([0,1,2]), 'score':np.array([9.,-1.,2.]), 'raw_score':np.array([190.,90.,120.])}
        self.assertEqual(resampling_post_max(c,1),120.)
        c['roc_score'] = np.array([8.,1.,4.])
        self.assertEqual(resampling_post_max(c,1),4.)
        self.assertEqual(resampling_post_max(c,3),-np.inf)


if __name__ == '__main__': unittest.main()
