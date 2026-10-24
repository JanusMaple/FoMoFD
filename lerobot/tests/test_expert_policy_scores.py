"""CPU-only checks of episode weighting, maxima, and paired score provenance."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
import expert_policy_scores as scores


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.curves = {0: {'score': np.array([0.])}, 1: {'score': np.array([10., 10., 10.])}}

    def test_long_episodes_do_not_dominate_balanced_windows(self):
        a = scores.distribution(self.curves, [0, 1], 'window_pooled')
        b = scores.distribution(self.curves, [0, 1], 'window_episode_balanced')
        self.assertAlmostEqual(scores.describe_distribution(*a)['mean'], 7.5)
        self.assertAlmostEqual(scores.describe_distribution(*b)['mean'], 5.)
        np.testing.assert_allclose(b[1], [.5, 1/6, 1/6, 1/6])

    def test_episode_maxima_are_per_episode_not_pooled(self):
        self.curves[1]['score'] = np.array([1., 5., 10.])
        values, weights = scores.distribution(self.curves, [0, 1], 'episode_maximum')
        np.testing.assert_array_equal(values, [0., 10.])
        np.testing.assert_array_equal(weights, [.5, .5])

    def test_episode_median_and_iqr_use_linear_quantiles_for_even_cohort(self):
        values = np.array([0., 10., 20., 100.])
        result = scores.describe_distribution(values, np.full(4, .25), quantile_method='linear')
        self.assertEqual(result['median'], 15.)
        self.assertEqual(result['q25'], 7.5)
        self.assertEqual(result['q75'], 40.)
        self.assertEqual(result['iqr'], 32.5)
        with self.assertRaisesRegex(ValueError, 'equal episode weights'):
            scores.describe_distribution(values, np.array([.1, .2, .3, .4]), quantile_method='linear')

    def test_validity_mask_truncates_policy_only_and_matching_camera(self):
        protocol = {'smoothing': 'trailing', 'window': 1, 'wm_frame_policy': 'all_endpoints'}
        config = {'window_protocol': protocol}
        corrections = [{'task': 'needle_pickup', 'episode_index': 5, 'raw_frame_stop': 122, 'views': ['wrist']}]
        arrays = {'episode_index': np.full(4, 5), 'frame_index': np.arange(4),
                  'raw_frame_index': np.arange(120, 124), 'wall_timestamp': np.arange(4.),
                  'scores': np.array([[1.], [2.], [100.], [200.]])}
        job = SimpleNamespace(task='needle_pickup', mode=False)
        expert = scores.benchmark.curves(arrays, job, 0, protocol)
        selected = scores.policy_protocol(config, 'needle_pickup', 'wrist', corrections)
        policy = scores.benchmark.curves(arrays, job, 0, selected)
        self.assertEqual(float(expert[5]['score'].max()), 200.)
        self.assertEqual(float(policy[5]['score'].max()), 2.)
        self.assertNotIn('frame_limits', protocol)
        self.assertNotIn('frame_limits', scores.policy_protocol(config, 'needle_pickup', 'fix', corrections))
        self.assertNotIn('frame_limits', scores.policy_protocol(config, 'ring_over_post', 'wrist', corrections))

    def test_comparison_orientation_and_ties(self):
        a = (np.array([0., 0.]), np.array([.5, .5]))
        b = (np.array([2., 2., 2.]), np.full(3, 1/3))
        result = scores.compare_distributions(a, b)
        self.assertEqual(result['KS_distance_descriptive'], 1.)
        self.assertEqual(result['wasserstein_raw_units'], 2.)
        self.assertEqual(result['prob_policy_higher_half_ties'], 1.)
        self.assertEqual(scores.compare_distributions(b, a)['prob_policy_higher_half_ties'], 0.)
        self.assertEqual(scores.compare_distributions(a, a)['prob_policy_higher_half_ties'], .5)

    def test_nonfinite_empty_and_duplicate_selections_rejected(self):
        for ids in ([], [0, 0]):
            with self.assertRaises(ValueError):
                scores.distribution(self.curves, ids, 'episode_maximum')
        self.curves[1]['score'][0] = np.nan
        with self.assertRaises(ValueError):
            scores.distribution(self.curves, [0, 1], 'window_pooled')

    def test_cached_scoring_protocol_and_validation_ids_must_match(self):
        base = {'task': 't', 'method': 'wm_nc', 'views': ['fix', 'wrist'], 'seed': 0,
                'batch_size': 8, 'device': 'cuda', 'checkpoint': '/ckpt',
                'sha256': {'/ckpt': 'hash', '/model.py': 'source_hash'},
                'merged_config': {'data': {'dataset_root': 'expert', 'history_len': 4}},
                'split': 'val', 'episode_ids': [2, 7]}
        policy = copy.deepcopy(base)
        policy['merged_config']['data']['dataset_root'] = 'policy'
        policy['split'] = 'all'
        policy.pop('episode_ids')
        scores.validate_pair(base, policy, [2, 7])
        for field, value in [('batch_size', 1), ('seed', 1)]:
            changed = copy.deepcopy(policy)
            changed[field] = value
            with self.assertRaises(ValueError):
                scores.validate_pair(base, changed, [2, 7])
        with self.assertRaises(ValueError):
            scores.validate_pair(base, policy, [2, 8])
        changed = copy.deepcopy(policy)
        changed['sha256']['/model.py'] = 'different'
        with self.assertRaises(ValueError):
            scores.validate_pair(base, changed, [2, 7])


if __name__ == '__main__':
    unittest.main()
