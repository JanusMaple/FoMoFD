"""CPU-only calibration sensitivity: partitioning, direction, pairing and reuse."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
import calibration_sensitivity as sensitivity
import functional_conformal as functional
import calibration_sensitivity_metrics as extra


class SplitTests(unittest.TestCase):
    def test_500_unique_splits_reproducible_and_complete(self):
        ids = list(range(39))
        for task in sensitivity.TASKS:
            rows = sensitivity.make_splits(ids, task=task, seed=20261013, n_splits=500, unique=True)
            self.assertEqual(len(rows), 500)
            self.assertEqual(len({tuple(r['calibration_ids']) for r in rows}), 500)
            self.assertEqual(rows, sensitivity.make_splits(ids[::-1], task=task, seed=20261013,
                                                         n_splits=500, unique=True))
            for row in rows:
                sensitivity.validate_partition(row, ids)

    def test_all_200_disjoint_complete_and_reproducible(self):
        ids = list(range(39))
        rows = sensitivity.make_splits(ids, task='needle_pickup', seed=42)
        self.assertEqual(len(rows), 200)
        self.assertEqual([r['split_id'] for r in rows], list(range(200)))
        for row in rows:
            sensitivity.validate_partition(row, ids)
            self.assertEqual(len(set(row['calibration_ids']) & set(row['test_success_ids'])), 0)
        self.assertEqual(rows, sensitivity.make_splits(ids[::-1], task='needle_pickup', seed=42))
        self.assertNotEqual(rows, sensitivity.make_splits(ids, task='needle_pickup', seed=43))
        self.assertNotEqual(rows, sensitivity.make_splits(ids, task='shunt_insertion', seed=42))

    def test_task_generation_order_does_not_change_splits(self):
        tasks = ['needle_pickup', 'shunt_insertion']
        forward = {t: sensitivity.make_splits(list(range(39)), task=t, seed=4) for t in tasks}
        backward = {t: sensitivity.make_splits(list(range(39)), task=t, seed=4) for t in tasks[::-1]}
        self.assertEqual(forward, backward)

    def test_invalid_partitions_are_rejected(self):
        ids = list(range(39))
        good = {'calibration_ids': ids[:19], 'test_success_ids': ids[19:]}
        for bad in [dict(good, calibration_ids=ids[:18] + [19]),
                    dict(good, test_success_ids=ids[19:-1]),
                    dict(good, calibration_ids=[0]*19)]:
            with self.assertRaises(ValueError):
                sensitivity.validate_partition(bad, ids)
        with self.assertRaises(ValueError):
            sensitivity.make_splits([0]*39, task='x', seed=0)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.ids = list(range(39))
        self.original = {'split_id': 'original', 'calibration_ids': self.ids[:19], 'test_success_ids': self.ids[19:]}
        self.q = {i: float(i) for i in self.ids}
        self.q.update({100: 18., 101: 18.01})

    def test_maximum_rank_19_strict_ties_and_denominators(self):
        self.q[19] = 18.
        r = sensitivity.evaluate_splits(self.q, self.ids, [100, 101], [self.original])[0]
        self.assertEqual(r['threshold'], 18.)
        self.assertEqual(functional.order_statistic(range(19), .05), (18., 19))
        self.assertEqual(r['false_alarms'], 19)
        self.assertEqual(r['FAR'], 19/20)
        self.assertEqual(r['detected_failures'], 1)
        self.assertEqual(r['FDR'], .5)

    def test_test_and_failure_values_never_change_threshold(self):
        before = sensitivity.evaluate_splits(self.q, self.ids, [100, 101], [self.original])[0]
        for e in self.ids[19:] + [100, 101]:
            self.q[e] = 1e12
        after = sensitivity.evaluate_splits(self.q, self.ids, [100, 101], [self.original])[0]
        self.assertEqual(before['threshold'], after['threshold'])
        self.assertEqual(after['FAR'], 1.)
        self.assertEqual(after['FDR'], 1.)

    def test_auroc_ties_and_threshold_independence(self):
        self.q.update({100: 30., 101: 38.})
        before = sensitivity.evaluate_splits(self.q, self.ids, [100, 101], [self.original])[0]
        expected = sensitivity.benchmark.episode_auroc(
            [False]*20+[True]*2, [self.q[e] for e in self.ids[19:]+[100,101]])
        self.assertEqual(before['episode_AUROC'], expected)
        for e in self.original['calibration_ids']:
            self.q[e] = 1e10
        after = sensitivity.evaluate_splits(self.q, self.ids, [100, 101], [self.original])[0]
        self.assertEqual(before['episode_AUROC'], after['episode_AUROC'])
        self.assertNotEqual(before['FDR'], after['FDR'])

    def test_auroc_changes_with_held_out_success_ids(self):
        self.q.update({100: 30., 101: 38.})
        swapped = {'split_id': 1, 'calibration_ids': self.ids[20:], 'test_success_ids': self.ids[:20]}
        a,b = sensitivity.evaluate_splits(self.q, self.ids, [100,101], [self.original,swapped])
        self.assertGreater(b['episode_AUROC'], a['episode_AUROC'])

    def test_signed_functional_residual_and_wall_origin(self):
        source = {1: {'score': np.array([-100., -3., 6.]), 'wall': np.array([102., 103., 108.])}}
        timeline = {1: {'wall': np.array([100.])}}
        template = {'grid_s': [0., 2., 4.], 'mu': [100., 3., 2.], 'g': [1., 1., 2.],
                    'score_orientation': 'higher_is_more_abnormal', 'extrapolation': 'hold_boundary_values'}
        saved = deepcopy(template)
        self.assertEqual(sensitivity.episode_q(source, timeline), {1: 6.})
        self.assertEqual(sensitivity.episode_q(source, timeline, template), {1: 2.})
        self.assertEqual(template, saved)
        with self.assertRaises(ValueError):
            sensitivity.episode_q(source, timeline, orientation='lower_is_more_abnormal')

    def test_cached_logpzo_scores_are_not_negated(self):
        # A nonconformity maximum remains an upper-tail statistic even if all
        # scores happen to be negative; do not confuse scores with log_prob.
        source = {0: {'score': np.array([-10., -2., -7.]), 'wall': np.arange(3)}}
        self.assertEqual(sensitivity.episode_q(source, {0: {'wall': [0.]}}), {0: -2.})

    def test_no_splits_are_discarded_even_if_repeated_or_unfavorable(self):
        rows = [dict(self.original, split_id=i) for i in range(200)]
        results = sensitivity.evaluate_splits(self.q, self.ids, [100, 101], rows)
        self.assertEqual(len(results), 200)
        self.assertTrue(all(r['FAR'] == 1 for r in results))
        self.assertEqual(sensitivity.summarize(results, results[0])['FAR_sd'], 0)

    def test_invalid_scores_and_failure_leakage_fail(self):
        for q, failures in [(dict(self.q, bad=1), [100, 101]),
                            (dict(self.q, **{}), [0, 100, 101]),
                            ({**self.q, 100: np.nan}, [100, 101])]:
            with self.assertRaises(ValueError):
                sensitivity.evaluate_splits(q, self.ids, failures, [self.original])

    def test_vectorized_results_match_direct_per_split_evaluation(self):
        rng = np.random.default_rng(7)
        q = {e: float(rng.normal()) for e in self.ids + [100, 101]}
        splits = sensitivity.make_splits(self.ids, task='needle_pickup', seed=8)
        results = sensitivity.evaluate_splits(q, self.ids, [100, 101], splits)
        for split, result in zip(splits, results):
            h = max(q[e] for e in split['calibration_ids'])
            self.assertEqual(result['threshold'], h)
            self.assertEqual(result['false_alarms'], sum(q[e] > h for e in split['test_success_ids']))
            self.assertEqual(result['detected_failures'], sum(q[e] > h for e in [100, 101]))


class SummaryTests(unittest.TestCase):
    def test_offline_latency_preserves_early_missed_unannotated_and_strict_ties(self):
        source={39:{'raw':np.array([1,2,3]),'wall':np.array([10.,11.,12.]),'score':np.array([5.,1.,6.])}}
        annotation=dict(task='x',method='wm_nc',view='fix',calibration='episode',annotator_id='a',
                        annotator='A',onset_view='fix',copied_from_annotator='',episode_index='39',
                        onset_raw_frame='2',onset_wall_timestamp='11.')
        early=extra.latency_rows(source,[annotation],4.)[0]
        self.assertEqual(early['status'],'pre_onset_alarm')
        self.assertIsNone(early['detection_delay_s'])
        late=extra.latency_rows(source,[annotation],5.)[0]
        self.assertEqual(late['status'],'detected')
        self.assertEqual(late['detection_delay_s'],1.)
        self.assertEqual(extra.latency_rows(source,[annotation],6.)[0]['status'],'missed')
        missing=dict(annotation,onset_raw_frame='',onset_wall_timestamp='')
        self.assertEqual(extra.latency_rows(source,[missing],4.)[0]['status'],'unannotated')

    def test_undefined_delay_is_preserved_with_its_denominator(self):
        self.assertEqual(extra.distribution([None,None])['defined_splits'],0)
        self.assertIsNone(extra.distribution([None,None])['mean'])
        summary=extra.distribution([None,1.,3.])
        self.assertEqual(summary['defined_splits'],2)
        self.assertEqual(summary['mean'],2.)

    def test_overall_uses_pooled_counts_and_macro_auroc(self):
        rows=[dict(variant='main',method='wm_nc',view='fix',calibration='episode',split_id=0,
                   task=str(i),false_alarms=i,test_successes=20,detected_failures=i,
                   test_failures=10 if i<2 else 20,episode_AUROC=i/3) for i in range(4)]
        row=extra.overall_results(rows)[0]
        self.assertEqual(row['FAR'],6/80)
        self.assertEqual(row['FDR'],6/60)
        self.assertEqual(row['episode_AUROC'],.5)
        with self.assertRaises(ValueError):extra.overall_results(rows[:-1])

    def test_paired_differences_align_ids_not_row_positions(self):
        left = [{'split_id': 0, 'FAR': .1, 'FDR': .8}, {'split_id': 1, 'FAR': 0., 'FDR': .5}]
        right = [{'split_id': 1, 'FAR': .1, 'FDR': .5}, {'split_id': 0, 'FAR': .1, 'FDR': .6}]
        rows = sensitivity.paired_rows(left, right)
        summary = sensitivity.paired_summary(rows)
        self.assertAlmostEqual(summary['delta_FDR_mean'], .1)
        self.assertEqual(summary['fraction_FDR_win'], .5)
        self.assertEqual(summary['fraction_FDR_tie'], .5)
        self.assertEqual(summary['fraction_FDR_win_without_higher_FAR'], .5)
        with self.assertRaises(ValueError):
            sensitivity.paired_rows(left, right[:1])

    def test_sample_sd_percentiles_and_original_tail_ties(self):
        values = [0., .1, .2]
        s = sensitivity.stats(values)
        self.assertAlmostEqual(s['sd'], .1)
        self.assertAlmostEqual(s['p05'], .01)
        p = sensitivity.original_position([0., 0., .1, .2], .1)
        self.assertEqual(p['fraction_below_original'], .5)
        self.assertEqual(p['fraction_at_or_above_original'], .5)
        self.assertFalse(p['original_above_resampling_p95'])

    def test_frozen_input_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'state.json').write_text('{"status":"complete"}')
            (root/'data.json').write_text('{"value":1}')
            digest = sensitivity.sha(root/'data.json')
            (root/'artifact_manifest.json').write_text(json.dumps({'files': {'data.json': digest}}))
            inputs = sensitivity.Inputs(root)
            self.assertEqual(inputs.read('data.json'), {'value': 1})
            (root/'data.json').write_text('{"value":2}')
            with self.assertRaises(ValueError):
                inputs.assert_unchanged()


if __name__ == '__main__':
    unittest.main()
