"""Partial template coverage must not hide an independently completed detector."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from benchmark_endpoint_protocol import evaluate_available_bands
import unified_benchmark as benchmark
from test_functional_conformal import constant_template
from test_unified_benchmark import toy_suite


class FunctionalCoverageTests(unittest.TestCase):
    def test_completed_joint_detector_survives_missing_companion_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            scored, labels, config = toy_suite(Path(tmp))
            original = next(item for item in scored if item[0].method == 'ace')
            pos, rot = deepcopy(original), deepcopy(original)
            pos[0].method = 'act_ensemble_pos'
            rot[0].method = 'act_ensemble_rot'
            templates = {(pos[0].task, 'act_ensemble_pos', 'both', name): constant_template()
                         for name in ('paper', 'latency')}
            reports, ready = evaluate_available_bands([pos, rot], labels, config,
                                                      list(labels['annotators']), templates)
            self.assertEqual(ready, 1)
            self.assertTrue(reports['episodes'])
            self.assertEqual({r['method'] for r in reports['episodes']}, {'act_ensemble_pos'})
            self.assertEqual({r['calibration'] for r in reports['episodes']}, {'functional_band'})
            self.assertTrue(all(r['n_cal'] == 19 for r in reports['calibration']))
            methods = list(dict.fromkeys(r['method'] for r in reports['episodes']))
            header, table = benchmark.paper_table(reports['episodes'], config, labels,
                                                  list(labels['tasks']), methods)
            self.assertEqual(len(header), 3)
            self.assertTrue(table)

    def test_missing_one_view_does_not_create_partial_camera_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            scored, labels, config = toy_suite(Path(tmp))
            item = next(item for item in scored if item[0].method == 'wm_nc')
            templates = {(item[0].task, 'wm_nc', 'fix', name): constant_template()
                         for name in ('paper', 'latency')}
            reports, ready = evaluate_available_bands([item], labels, config,
                                                      list(labels['annotators']), templates)
            self.assertEqual(ready, 0)
            self.assertTrue(all(not rows for rows in reports.values()))


if __name__ == '__main__':
    unittest.main()
