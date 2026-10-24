"""CPU-only invariants for shared-episode conditional latency comparisons."""
import copy
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from benchmark_shared_latency import cohort, largest_groups, online_index, pairwise_comparisons


def index_for(sets):
    return {d: {'detected': {('task', e, 1.): {'task': 'task', 'episode_index': e, 'delay_s': e + .1,
                   'failure_origin': 'natural', 'failure_mode': 'drop'} for e in episodes},
                'records': {('task', e): {} for e in episodes}} for d, episodes in sets.items()}


class SharedLatencyTests(unittest.TestCase):
    def test_true_joint_intersection_not_pairwise_overlap(self):
        index = index_for({'a': {1, 2}, 'b': {2, 3}, 'c': {1, 3}})
        result = cohort(index, ['a', 'b', 'c'], 1)
        self.assertEqual(result['shared_n'], 0)
        self.assertFalse(result['eligible'])
        search = largest_groups(index, ['a', 'b', 'c'], 1)
        self.assertEqual(search['maximum_detectors'], 2)
        self.assertEqual(len(search['groups']), 3)

    def test_keep_all_shared_episodes_and_pair_differences(self):
        index = index_for({'a': {1, 2, 3, 4}, 'b': {1, 2, 3}})
        for r in index['b']['detected'].values(): r['delay_s'] += 2
        result = cohort(index, ['a', 'b'], 2)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['shared_n'], 3)
        self.assertEqual(result['summaries'][0]['excluded_from_own_detections'], 1)
        self.assertAlmostEqual(result['paired_differences'][0]['mean_s'], -2.)
        self.assertEqual(result['paired_differences'][0]['a_faster_n'], 3)
        self.assertFalse(cohort(index, ['a', 'b'], 4)['eligible'])

    def test_missing_detector_never_silently_dropped(self):
        index = index_for({'a': {1, 2}, 'b': {1, 2}})
        self.assertEqual(cohort(index, ['a', 'b', 'unmeasured'])['shared_n'], 0)
        with self.assertRaises(ValueError): cohort(index, ['a', 'a'])

    def test_matching_onset_and_task_required(self):
        index = index_for({'a': {1, 2}, 'b': {1, 2}})
        old = index['b']['detected'].pop(('task', 1, 1.))
        index['b']['detected']['task', 1, 2.] = old
        old = index['b']['detected'].pop(('task', 2, 1.))
        index['b']['detected']['another_task', 2, 1.] = old
        result = cohort(index, ['a', 'b'])
        self.assertEqual(result['shared_n'], 0)
        self.assertEqual(result['different_onset_excluded_n'], 1)

    def test_search_complete_matches_brute_force_and_ignores_delays(self):
        index = index_for({'a': {1, 2, 3, 4}, 'b': {2, 3, 4}, 'c': {1, 2, 4}, 'd': {1, 3}, 'e': {5}})
        for minimum in (1, 2, 3, 4):
            all_groups = []
            for n in range(2, 6):
                for group in itertools.combinations(index, n):
                    common = set.intersection(*(set(index[d]['detected']) for d in group))
                    if len(common) >= minimum: all_groups.append((group, len(common)))
            maximum = max((len(g) for g, _ in all_groups), default=0)
            expected = {(g, count) for g, count in all_groups if len(g) == maximum}
            result = largest_groups(index, list(reversed(index)), minimum)
            self.assertTrue(result['complete'])
            self.assertEqual(result['maximum_detectors'], maximum)
            self.assertEqual({(tuple(g['detectors']), g['shared_n']) for g in result['groups']}, expected)
            changed = copy.deepcopy(index)
            for bucket in changed.values():
                for r in bucket['detected'].values(): r['delay_s'] *= 10000
            self.assertEqual(largest_groups(changed, list(index), minimum), result)

    def test_search_bound_is_explicit(self):
        index = index_for({str(i): {1, 2} for i in range(21)})
        self.assertFalse(largest_groups(index, list(index), 1)['complete'])
        self.assertEqual(cohort(index, list(index))['shared_n'], 2)

    def test_filters_no_annotator_pooling_and_no_preonset_inclusion(self):
        def row(view, onset_view, annotator='a', status='detected', delay=1., onset=2.):
            return {'variant': 'main', 'method': 'wm_nc', 'view': view, 'onset_view': onset_view,
                    'annotator_id': annotator, 'status': status, 'delay_s': delay, 'onset_time_s': onset}
        trial = {'variant': 'main', 'task': 't', 'method': 'wm_nc', 'episode_index': 1, 'calibration': 'episode',
                 'failure_origin': 'natural', 'failure_mode': 'drop', 'path': 'trace.json',
                 'latencies': [row('fix', 'fix'), row('wrist', 'wrist', status='pre_onset_evidence', delay=None),
                               row('both', 'fix'), row('both', 'wrist', onset=3.), row('fix', 'fix', 'b', delay=100.)]}
        data = {'online': [trial]}
        index = online_index(data, 'a')
        self.assertEqual(len(index['main|wm_nc|fix']['detected']), 1)
        self.assertEqual(len(index['main|wm_nc|wrist']['detected']), 0)
        self.assertEqual(cohort(index, ['main|wm_nc|fix', 'main|wm_nc|both'])['shared_n'], 1)
        self.assertEqual(cohort(online_index(data, 'a', combined_onset='wrist'), ['main|wm_nc|fix', 'main|wm_nc|both'])['shared_n'], 0)
        self.assertEqual(online_index(data, 'a', origin='induced'), {})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            online_index({'online': [trial, trial]}, 'a')

    def test_pairwise_reference_is_recomputed_for_each_intersection(self):
        index = index_for({'main': {1,2,3,4}, 'baseline': {1,2}, 'K': {3,4}})
        detectors = {k:{'view':'wrist','family':'baseline','label':k} for k in ['main','baseline','K','action']}
        rows = {r['comparator']:r for r in pairwise_comparisons(index,detectors,'main')}
        self.assertEqual(rows['baseline']['shared_n'],2)
        self.assertAlmostEqual(rows['baseline']['summaries'][0]['mean_s'],1.6)
        self.assertAlmostEqual(rows['K']['summaries'][0]['mean_s'],3.6)
        self.assertEqual(rows['action']['status'],'not_measured_online')
        self.assertIsNone(rows['action']['summaries'][1]['mean_s'])
        self.assertEqual(pairwise_comparisons(index,detectors,'main',3)[0]['status'],'below_minimum')

    def test_wrist_reference_pairs_fixed_detections_using_each_own_onset(self):
        index = index_for({'wrist': {1,2}, 'fixed': {1,2}})
        for bucket in index.values():
            for row in bucket['detected'].values():
                row.update(onset_time_s=1., onset_raw_frame=10)
        row = index['fixed']['detected'].pop(('task',1,1.))
        row.update(onset_time_s=2., onset_raw_frame=20, delay_s=.5)
        index['fixed']['detected']['task',1,2.] = row
        detectors = {k:{'view':v,'family':'baseline','label':k}
                     for k,v in [('wrist','wrist'),('fixed','fix')]}
        result = pairwise_comparisons(index,detectors,'wrist')[0]
        self.assertEqual(result['comparator'],'fixed')
        self.assertEqual(result['shared_n'],2)
        self.assertEqual(result['different_onset_included_n'],1)
        self.assertEqual(result['different_onset_excluded_n'],0)
        self.assertAlmostEqual(result['summaries'][0]['mean_s'],1.6)
        self.assertAlmostEqual(result['summaries'][1]['mean_s'],1.3)
        self.assertAlmostEqual(result['paired_differences'][0]['mean_s'],.3)
        # The optional multi-method common-onset analysis retains its stricter rule.
        self.assertEqual(cohort(index,['wrist','fixed'])['shared_n'],1)
        del index['fixed']['detected']['task',2,1.]
        self.assertEqual(pairwise_comparisons(index,detectors,'wrist')[0]['shared_n'],1)

    def test_vlm_late_rule_applies_to_both_methods_and_remains_optional(self):
        def row(method,late=False):
            return {'variant':'main' if method=='wm_nc' else 'vlm_qwen', 'method':method,'view':'wrist',
                    'onset_view':'wrist','annotator_id':'a','status':'detected','delay_s':1.,'onset_time_s':2.,
                    'late_alarm':late}
        common={'task':'t','episode_index':1,'failure_origin':'natural','failure_mode':'drop'}
        trial={**common,'calibration':'episode','path':'wm.json','latencies':[row('wm_nc',True)]}
        vlm={**common,'outcome':'failure','calibration':'binary_anytime','trace_path':'qwen.json','latencies':[row('qwen_v9')]}
        data={'online':[trial],'vlm_runs':[{'episodes':[vlm]}]}
        self.assertNotIn('vlm_qwen|qwen_v9|wrist',online_index(data,'a'))
        both=online_index(data,'a',include_vlm=True)
        self.assertEqual(cohort(both,['main|wm_nc|wrist','vlm_qwen|qwen_v9|wrist'])['shared_n'],1)
        on_time=online_index(data,'a',include_vlm=True,on_time_only=True)
        self.assertEqual(len(on_time['main|wm_nc|wrist']['detected']),0)
        self.assertEqual(len(on_time['main|wm_nc|wrist']['records']),1)
        trial['latencies'][0]['late_alarm']=False;vlm['latencies'][0]['late_alarm']=True
        on_time=online_index(data,'a',include_vlm=True,on_time_only=True)
        self.assertEqual(len(on_time['vlm_qwen|qwen_v9|wrist']['detected']),0)


if __name__ == '__main__':
    unittest.main()
