"""CPU-only coverage and missing-result checks for the compact reports."""
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
import results_inspector as inspector


class ResultSummaryTests(unittest.TestCase):
    def test_recovery_coverage_preserves_failed_attempt_and_deduplicates(self):
        runs = [{'source':'original','state':{'status':'failed','completed_trials':1}},
                {'source':'recovery','state':{'status':'complete','completed_trials':2}}]
        plans = {'original':{'a','b','c'},'recovery':{'b','c'}}
        result = inspector.online_coverage(runs,plans,{'a','b','c'})
        self.assertEqual(result,{'status':'complete','completed_trials':3,'total_trials':3,'missing_trials':0})
        self.assertEqual(runs[0]['covered_trials'],3)
        self.assertEqual(runs[0]['state'],{'status':'failed','completed_trials':1})
        # An equally sized set of other trials must never cover a missing one.
        incomplete = inspector.online_coverage(runs,plans,{'a','b'})
        self.assertEqual(incomplete['missing_trials'],1)
        self.assertEqual(runs[0]['coverage_status'],'incomplete')
        with self.assertRaisesRegex(ValueError,'no current trial plan'):
            inspector.online_coverage(runs,plans,{'a','b','different_model'})

    def test_combined_view_plans_keep_distinct_trial_ids(self):
        trial = {'variant':'main','task':'needle_pickup','method':'wm_nc_logpzo_native_or',
                 'episode_index':39,'input_views':['fix','wrist'],'fusion':{'output_view':'fix'}}
        fixed = inspector.trial_id(trial)
        self.assertTrue(fixed.endswith('_fix_wrist_out_fix_000039'))
        trial['fusion']['output_view'] = 'wrist'
        self.assertNotEqual(inspector.trial_id(trial), fixed)
        trial.pop('fusion')
        trial['score_aggregation']='joint'
        self.assertTrue(inspector.trial_id(trial).endswith('_out_joint_000039'))
        self.assertEqual(inspector.trial_id({'trial_id':'saved'}), 'saved')

    def test_reports_include_new_variants_and_failed_zero_trial_run(self):
        runs = []
        episodes = []
        for variant, family in [('main','main'),('fused','view_fusion'),('small','data_size'),('seed1','seed'),('seed2_repeat','seed_repeat')]:
            runs.append({'id':variant,'family':family,'title':variant,'path':'offline/'+variant,
                         'archive_sha256':variant,'methods':['wm_nc'],'alpha':.05,
                         'protocols':{'paper_protocol':{'window':1},'latency_protocol':{'window':1}}})
            for method, view in [('wm_nc','both' if family=='view_fusion' else 'wrist')] + ([('wm_nc_or','both')] if variant=='main' else []):
                for ep, split, outcome, alarm, score in [(0,'calibration','success',False,1.),
                        (19,'test','success',False,1.),(39,'test','failure',True,3.)]:
                    episodes.append({'variant':variant,'task':'needle_pickup','method':method,'view':view,
                        'calibration':'episode','episode_index':ep,'split':split,'outcome':outcome,
                        'failure_origin':'natural' if outcome=='failure' else 'not_applicable',
                        'failure_mode':'Drop' if outcome=='failure' else None,
                        'score_max':score,'alarm':alarm,'n_supported_scores':10})
        state = {'status':'failed','completed_trials':0,'total_trials':2,
                 'finished':'2026-10-10T23:04:25+00:00','error':'Parity failed'}
        data = {'created':'2026-10-11T12:00:00+00:00','runs':runs,'episodes':inspector.packed(episodes),
                'settings':[],'annotators':{'a':'A'},'online':[],'plans':[{},{}],
                'online_runs':[{'source':'experiment/online','title':'New models','path':'online/new','state':state}]}
        with tempfile.TemporaryDirectory() as folder:
            inspector.write_markdown_summary(data, Path(folder))
            text = (Path(folder)/'summary.md').read_text()
            summary = yaml.safe_load((Path(folder)/'summary.yaml').read_text())
        self.assertIn('failed, 0/2 trials recorded', text)
        self.assertIn('combined_monitors', summary['tables'])
        self.assertIn('data_size_and_seed', summary['tables'])
        self.assertIn('fused_view_fused', summary['tables'])
        self.assertEqual(summary['tables']['fused_view_fused']['rows'][0][0], 'WM-NC combined')
        self.assertEqual(summary['tables']['fused_view_fused']['rows'][0][3], '0.0/100.0/1.000')
        self.assertEqual(set(summary['source_reports']), {'main','fused','small','seed1','seed2_repeat'})
        self.assertTrue(any(row[0]=='seed2_repeat' for row in summary['tables']['data_size_and_seed']['rows']))
        self.assertIn('do not replace the active models', text)
        self.assertEqual(summary['online_runs'][0]['completed_trials'], 0)
        # Empty functional-band rows stay unavailable, never zero or copied
        # from constant thresholds. Original main metrics remain exact.
        self.assertEqual(summary['tables']['main_methods']['rows'][0][1], '0.0/100.0/1.000')
        self.assertEqual(summary['tables']['main_methods']['rows'][0][2], '—')


if __name__ == '__main__': unittest.main()
