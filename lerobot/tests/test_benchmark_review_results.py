"""Reviewer summaries must preserve denominator and latency semantics."""
from pathlib import Path
import math
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'utils'))
from benchmark_review_results import wilson, describe, latency_counts, markdown_sections, verify_population_counts


class ReviewerEvidenceTests(unittest.TestCase):
    def test_wilson_zero_and_all_success_do_not_claim_certainty(self):
        lo,hi=wilson(0,20)
        self.assertAlmostEqual(lo,0.)
        self.assertAlmostEqual(hi,.16112515805281938)
        lo2,hi2=wilson(20,20)
        self.assertAlmostEqual(lo2,1-hi)
        self.assertAlmostEqual(hi2,1.)
        self.assertIsNone(wilson(0,0))
        with self.assertRaises(ValueError):wilson(21,20)

    def test_missing_latency_is_not_zero_and_sd_is_sample_sd(self):
        self.assertIsNone(describe([])['mean'])
        self.assertIsNone(describe([2])['sd'])
        self.assertAlmostEqual(describe([0,2])['sd'],math.sqrt(2))
        with self.assertRaises(ValueError):describe([float('nan')])

    def test_early_evidence_late_and_missing_keep_denominator(self):
        normal={'status':'detected','alarm':True,'delay_s':1.,'signed_alarm_delay_s':1.,
                'onset_raw_frame':10,'late_alarm':False}
        rows=[normal,{**normal,'delay_s':8.,'late_alarm':True},
              {**normal,'status':'pre_onset_evidence','delay_s':None,'signed_alarm_delay_s':.5},
              {**normal,'status':'pre_onset_evidence','delay_s':None,'signed_alarm_delay_s':-.5},
              {**normal,'status':'missed','alarm':False,'delay_s':None,'signed_alarm_delay_s':None},
              {**normal,'status':'unannotated','alarm':False,'delay_s':None,'onset_raw_frame':None,'signed_alarm_delay_s':None}]
        result=latency_counts(rows)
        self.assertEqual(result['selected'],6)
        self.assertEqual(result['alarms'],4)
        self.assertEqual(result['early_evidence'],2)
        self.assertEqual(result['early_completion'],1)
        self.assertEqual(result['late_alarms'],1)
        self.assertEqual(result['no_alarm'],2)
        self.assertEqual(result['missing_onset'],1)
        self.assertEqual(result['delay_s']['n'],1)
        self.assertEqual(result['within_deadline']['1'],1)
        self.assertEqual(result['within_deadline']['10'],1)

    def test_invalid_detected_latency_rejected(self):
        with self.assertRaises(ValueError):
            latency_counts([{'status':'detected','alarm':True,'delay_s':None}])

    def test_markdown_keeps_cell_boundaries(self):
        text='\n'.join(markdown_sections([{'title':'Pair','note':'Same episodes',
             'columns':['Name','Value'],'rows':[['main|wm_nc|wrist','1\n2']]}]))
        self.assertIn('main\\|wm_nc\\|wrist',text)
        self.assertIn('1 2',text)

    def test_population_audit_checks_ids_not_just_equal_counts(self):
        audit={'rows':[{'task':'t','view':'wrist','paper_detected':2,'completed_replay':1,
                        'paper_only_ids':[40],'replay_only_ids':[],'intersection_ids':[39]}]}
        common={'variant':'main','method':'wm_nc','calibration':'episode','task':'t'}
        episodes=[{**common,'view':'wrist','outcome':'failure','alarm':True,'episode_index':i} for i in [39,40]]
        replay=[{**common,'episode_index':39,'requested_detector_views':['wrist']}]
        verify_population_counts(audit,episodes,replay)
        replay[0]['episode_index']=40
        with self.assertRaisesRegex(ValueError,'stale'):
            verify_population_counts(audit,episodes,replay)

if __name__=='__main__':unittest.main()
