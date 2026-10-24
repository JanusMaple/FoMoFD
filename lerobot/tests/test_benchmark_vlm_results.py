"""CPU-only denominator, delay pooling and model-presentation safeguards."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from benchmark_vlm_results import aggregate, episode_row
from results_inspector import display_variant


class CuratedResultsTests(unittest.TestCase):
    def test_main_anytime_includes_late_alarms_without_inventing_scores(self):
        trial = {'task':'needle_pickup','view':'fix','episode':39}
        result = {'status':'complete','outcome':'failure','alarm':{'alarm_time_s':21.},
                  'late_alarm':True,'pre_onset_alarm':False}
        group = {'split':'test','outcome':'failure','failure_mode':'Needle Drop','failure_origin':'natural'}
        row = episode_row(trial,result,group,'vlm_qwen_v9')
        self.assertTrue(row['alarm'])
        self.assertTrue(row['late_alarm'])
        self.assertIsNone(row['score_max'])
        self.assertEqual(row['calibration'],'binary_anytime')
        self.assertEqual(row['failure_origin'],'natural')
        alternate = episode_row(trial,result,group,'vlm_qwen_v13_definitions',method='qwen_v13_definitions')
        self.assertEqual(alternate['method'], 'qwen_v13_definitions')
        self.assertNotEqual(alternate['method'], row['method'])
        self.assertEqual(alternate['alarm'], row['alarm'])
        with self.assertRaisesRegex(ValueError,'complete trials'):
            episode_row(trial,{**result,'status':'indeterminate'},group,'vlm_qwen_v9')

    def test_variant_alias_preserves_trial_identity_and_input(self):
        row = {'variant':'old_desktop', 'trial_id':'old_desktop_episode_39', 'delay_s':1.2}
        result = display_variant(row, {'old_desktop':'seed1'})
        self.assertEqual(result['variant'], 'seed1')
        self.assertEqual(result['trial_id'], row['trial_id'])
        self.assertEqual(row['variant'], 'old_desktop')
        self.assertEqual(display_variant(row, {}), row)

    def test_pooled_rates_use_counts_and_keep_unknown_denominators(self):
        rows = []
        for task,n,far,tp,unknown,delay,sd in [('a',10,1,2,0,2.,0.), ('b',30,6,12,1,4.,1.)]:
            rows.append({'model':'qwen', 'task':task, 'view':'fix',
                         'FAR_denominator':n, 'FAR_unknown':0, 'FAR_any_count':far, 'FAR_on_time_count':far,
                         'FDR_denominator':n, 'FDR_unknown':unknown, 'FDR_any_count':tp, 'FDR_on_time_count':tp,
                         'conditional_delay_n':2, 'conditional_delay_mean_s':delay, 'conditional_delay_sd_s':sd})
        pooled = aggregate(rows)[0]
        self.assertEqual(pooled['FAR_on_time'], 7/40)
        self.assertEqual(pooled['FDR_any_count'], 14)
        self.assertIsNone(pooled['FDR_on_time'])
        self.assertEqual(pooled['FDR_any_lower_bound'],14/40)
        self.assertEqual(pooled['FDR_any_upper_bound'],15/40)
        self.assertEqual(pooled['conditional_delay_mean_s'], 3.)
        self.assertAlmostEqual(pooled['conditional_delay_sd_s'], 1.5**.5)


if __name__ == '__main__':
    unittest.main()
