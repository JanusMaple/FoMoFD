"""CPU-only checks for exact-count exports and offline/online scope separation."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
import export_benchmark_tables as export
from results_inspector import packed


def fixture():
    base = dict(variant='main', task='task', method='wm_nc', view='wrist', calibration='episode', split='test')
    rows = []
    for episode, outcome, alarm, score, post in ((19,'success',False,-1.,None), (20,'success',True,1.,None),
                                               (39,'failure',True,4.,2.), (40,'failure',False,3.,-1.)):
        rows.append({**base, 'episode_index':episode, 'outcome':outcome, 'alarm':alarm,
                     'failure_mode':'Drop' if outcome=='failure' else None,
                     'failure_origin':'natural' if outcome=='failure' else 'not_applicable',
                     'score_max':score, 'post_score_max':post, 'n_supported_scores':5,
                     'detection_definition':'at_or_after_onset' if outcome=='failure' else None,
                     'onset_assessments':{'a':{'post_alarm':alarm}} if outcome=='failure' else None})
    latencies = [{**r, 'annotator_id':'a', 'post_alarm':r['alarm'],
                  'detection_delay_s':.4 if r['alarm'] else None} for r in rows if r['outcome']=='failure']
    plan = {**rows[2], 'requested_detector_views':['wrist'], 'trial_id':'t'}
    data = {'evaluation_protocol':{'protocol_id':'v2','detection':{'primary_annotator':'a'}},
            'runs':[{'id':'main','family':'main','title':'Main'}], 'episodes':packed(rows),
            'latencies':packed(latencies), 'plans':[plan], 'online':[{**plan, 'path':'trace.json',
                         'latencies':[{**latencies[0], 'late_alarm':False}]}]}
    labels = {'tasks':{'task':{'groups':[
        {'split':'test','outcome':'success','episode_range':[19,20]},
        {'split':'test','outcome':'failure','failure_mode':'Drop','failure_origin':'natural','episode_range':[39,40]},
    ]}}}
    return data, labels


class TableExportTests(unittest.TestCase):
    def test_study_refresh_preserves_primary_tables_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root/'source', root/'published'
            folder = source/'analyses/study'
            folder.mkdir(parents=True)
            (output/'details').mkdir(parents=True)
            (source/'index.html').write_text('snapshot')
            (source/'summary.md').write_text('Updated study: median (IQR).\n')
            (folder/'episode_maximum_summary.csv').write_text('cohort,median,iqr\nexpert_validation,15,32.5\n')
            (folder/'episodes.csv').write_text('episode_index,max_score\n0,15\n')
            original = {'created_utc': 'frozen', 'evaluation_protocol': {'protocol_id': 'v2'},
                'tables': {'main': {'columns': ['FAR'], 'rows': [[.05]]},
                           'expert_policy_score_distributions': {'columns': ['median'], 'rows': [[10]]}}}
            (output/'summary.yaml').write_text(yaml.safe_dump(original | {'detailed_exports': {'guide': 'README.md'}}))
            (output/'details/manifest.json').write_text(json.dumps({'files': {}}))
            current = deepcopy(original)
            current['tables']['expert_policy_score_distributions']['rows'] = [[15]]
            (source/'summary.yaml').write_text(yaml.safe_dump(current))
            snapshot = {'studies': [{'id': 'expert_policy_score_distributions', 'kind': 'score_distributions',
                'status': 'complete', 'dataset_validity_applied': True, 'episode_quantile_method': 'linear',
                'path': 'analyses/study', 'source_sha256': {}}]}
            with patch.object(export, 'read_viewer_snapshot', return_value=snapshot):
                export.export_study_refresh(source, output)
                result = yaml.safe_load((output/'summary.yaml').read_text())
                self.assertEqual(result['tables']['main'], original['tables']['main'])
                self.assertEqual(result['tables']['expert_policy_score_distributions']['rows'], [[15]])
                self.assertTrue((output/'details/expert_policy_episode_maxima_summary.csv').is_file())
                before = (output/'summary.yaml').read_bytes()
                current['tables']['main']['rows'] = [[.10]]
                (source/'summary.yaml').write_text(yaml.safe_dump(current))
                with self.assertRaisesRegex(ValueError, 'Primary table changed'):
                    export.export_study_refresh(source, output)
                self.assertEqual((output/'summary.yaml').read_bytes(), before)

    def test_mode_counts_and_post_onset_matched_far_use_different_score_support(self):
        data, labels = fixture()
        rows, _ = export.offline_tables(data, labels)
        mode = next(r for r in rows if r['group_type']=='failure_mode')
        self.assertEqual((mode['alarm_n'],mode['population_n'],mode['FDR']), (1,2,.5))
        self.assertEqual(mode['FAR'], .5)
        self.assertEqual(mode['episode_AUROC'], 1.)  # Both whole-episode failure maxima exceed successes.
        self.assertEqual(mode['matched_threshold'], 1.)
        self.assertEqual(mode['matched_detected_n'], 1)  # Only one post-onset maximum exceeds it.
        self.assertEqual(mode['matched_FAR'], 0.)
        self.assertEqual(mode['latency_n'], 1)
        self.assertEqual(mode['episode_ids'], '39;40')

    def test_skipped_failures_and_successes_are_not_observed_online_misses(self):
        data, labels = fixture()
        _, groups = export.offline_tables(data, labels)
        rows, records = export.online_tables(data, labels, groups)
        mode = next(r for r in rows if r['group_type']=='failure_mode')
        self.assertEqual((mode['population_n'],mode['selected_n'],mode['recorded_n']), (2,1,1))
        self.assertEqual(mode['not_recorded_n'],1)
        self.assertEqual(mode['observed_negative_n'],0)
        self.assertEqual(mode['selected_detection_rate'],1.)
        self.assertIsNone(mode['FDR'])
        success = next(r for r in rows if r['group_type']=='success_test')
        self.assertIsNone(success['FAR'])
        self.assertIsNone(success['selected_detection_rate'])
        self.assertEqual(len(records),1)

    def test_late_detections_are_separate_from_on_time_delay(self):
        data, labels = fixture()
        data['online'][0]['latencies'][0]['late_alarm'] = True
        _, groups = export.offline_tables(data, labels)
        rows, _ = export.online_tables(data, labels, groups)
        row = next(r for r in rows if r['group_type']=='failure_mode')
        self.assertEqual((row['alarm_n'], row['on_time_alarm_n'],row['late_alarm_n'],row['latency_n']), (1,0,1,0))
        self.assertIsNone(row['latency_mean_s'])

    def test_unknown_and_duplicate_episodes_remain_explicit(self):
        rows = [{'episode_index':1,'alarm':True}, {'episode_index':2,'alarm':None}]
        result = export.observed_counts(rows, {1,2,3})
        self.assertIsNone(result['rate'])
        self.assertEqual((result['unknown_n'],result['not_recorded_n']), (1,1))
        self.assertEqual((result['rate_lower'],result['rate_upper']), (1/3,1.))
        with self.assertRaises(ValueError):
            export.observed_counts(rows + rows, {1,2,3})

    def test_vlm_full_test_allows_fdr_and_far_with_late_alarm_separation(self):
        data, labels = fixture()
        vlm = deepcopy(export.unpack(data['episodes']))
        for row in vlm:
            row.update(variant='vlm_qwen_v9',method='qwen_v9',calibration='binary_anytime',late_alarm=False)
            row['latencies'] = []
            if row['outcome']=='failure':
                row['latencies'] = [{**row,'annotator_id':'a','post_alarm':row['alarm'],
                                     'detection_delay_s':2. if row['alarm'] else None}]
        data['vlm_runs'] = [{'episodes':vlm}]
        _, groups = export.offline_tables(data, labels)
        rows, _ = export.online_tables(data, labels, groups)
        rows = [r for r in rows if r['method']=='qwen_v9']
        self.assertEqual(next(r['FDR'] for r in rows if r['group_type']=='failure_mode'),.5)
        self.assertEqual(next(r['FAR'] for r in rows if r['group_type']=='success_test'),.5)

    def test_disagreeing_source_decisions_are_rejected(self):
        data, labels = fixture()
        columns = data['latencies']['columns']
        data['latencies']['rows'][0][columns.index('post_alarm')] = False
        with self.assertRaisesRegex(ValueError,'populations disagree'):
            export.offline_tables(data, labels)


if __name__ == '__main__':
    unittest.main()
