"""Ablations must pool measured counts on matching, complete episode cohorts."""
from copy import deepcopy
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import plot_sensitivity as plot


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name) / "measured.csv"
        self.rows = []
        variants = sorted({v for _, _, points in plot.PANELS for _, _, v in points})
        for variant in variants:
            for view in plot.VIEWS:
                for task, n, d in zip(plot.TASKS, (1, 1, 1, 9), (1, 1, 1, 0)):
                    self.rows.append(dict(variant=variant, view=view, task=task, method="wm_nc",
                        calibration="episode", group_type="all_failures", measurement="dense_offline",
                        annotator_id="consensus_mean", evaluation_protocol="test_mean_protocol",
                        population_n=n, recorded_n=n, not_recorded_n=0, unknown_n=0,
                        episode_ids=";".join(map(str, range(n))), matched_detected_n=d,
                        matched_FDR=d/n, matched_FAR_target=.05, matched_FAR=.05,
                        matched_false_alarm_n=1, success_test_n=20))

    def write_rows(self, rows):
        with self.source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_count_pooling_differs_from_task_percentage_average(self):
        self.write_rows(self.rows)
        values, selected = plot.load_values(self.source)
        self.assertEqual(len(values), 28)
        self.assertEqual(len(selected), 96)
        for point in values:
            self.assertEqual(point["detected_n"], 3)
            self.assertEqual(point["total_n"], 12)
            self.assertEqual(point["FDR_percent"], 25)  # Task-percentage mean would be 75%.
            self.assertEqual(point["test_false_alarm_n"], 4)
            self.assertEqual(point["success_test_n"], 80)

    def test_failure_mode_and_other_method_rows_do_not_double_count(self):
        ignored = [{**self.rows[0], "group_type": "failure_mode"},
                   {**self.rows[0], "method": "wm_pe"},
                   {**self.rows[0], "calibration": "functional_band"}]
        self.write_rows(self.rows + ignored)
        values, _ = plot.load_values(self.source)
        self.assertTrue(all(p["total_n"] == 12 and p["detected_n"] == 3 for p in values))

    def test_incomplete_duplicate_mixed_or_different_cohorts_rejected(self):
        cases = [(self.rows[1:], "coverage"), (self.rows + [self.rows[0]], "Duplicate")]
        for change, error in [({"evaluation_protocol": "old_protocol"}, "mix evaluation"),
                              ({"annotator_id": "individual"}, "mean-onset"),
                              ({"unknown_n": 1}, "complete offline"),
                              ({"matched_false_alarm_n": 2, "matched_FAR": .1}, "test-FAR"),
                              ({"matched_FDR": 0}, "test-FAR"),
                              ({"episode_ids": "999"}, "populations differ")]:
            rows = deepcopy(self.rows)
            # Mutate an ablation row; the main cohort remains the reference.
            index = next(i for i, r in enumerate(rows) if r["variant"] != "main" and r["population_n"] == 1)
            rows[index].update(change)
            cases.append((rows, error))
        for rows, error in cases:
            with self.subTest(error=error):
                self.write_rows(rows)
                with self.assertRaisesRegex(ValueError, error):
                    plot.load_values(self.source)

    def test_main_reference_and_exact_training_fractions(self):
        self.write_rows(self.rows)
        values, _ = plot.load_values(self.source)
        for view in plot.VIEWS:
            main = [p for p in values if p["view"] == view and p["is_main"]]
            self.assertEqual([p["setting"] for p in main], [4, 4, 1])
            self.assertEqual({p["variant"] for p in main}, {"main"})
        fractions = [p["setting"] for p in values if p["panel"] == "training_data" and p["view"] == "wrist"]
        self.assertEqual(fractions, [1/3, 2/3, 1])


if __name__ == "__main__":
    unittest.main()
