"""Scientific figure selection, exact raw endpoint alignment, and SVG export."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import plot_failure_examples as figure
from replay_inspector import Episode


class FailureFigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.raw = {}
        self.episodes = {}
        for ep in (0, 39):
            times = np.arange(15, dtype=float) * .17
            # A nonuniform recorded clock exposes frame/FPS substitutions.
            times[9:] += .45
            image = np.zeros((15, 8, 8, 3), dtype=np.uint8)
            image[:, :, :, 0] = np.arange(15)[:, None, None]
            self.raw["needle_pickup", ep] = root / f"raw_{ep}.npz"
            np.savez(self.raw["needle_pickup", ep], timestamps=times+1000*ep)
            self.episodes[ep] = Episode("needle_pickup", ep, f"raw_{ep}.npz", image, image, times, None)
        self.labels = {"annotators": {"author": {"name": "Rater A"}, "second": {"name": "Rater B"},
                                      "third": {"name": "Rater C"}},
            "tasks": {"needle_pickup": {"failure_onsets": {39: {
                "author": {"mode": "separate", "views": {"wrist": {"frame_index": 6}, "fix": {"frame_index": 9}}},
                "second": {"mode": "separate", "views": {"wrist": {"frame_index": 10}}},
                "third": {"mode": "separate", "views": {"wrist": {"frame_index": 11},
                                                         "fix": {"frame_index": 12}}}}}, "groups": [
            {"split": "calibration", "outcome": "success", "episode_range": [0, 0]},
            {"split": "test", "outcome": "failure", "episode_range": [39, 39]}]}}}
        labels_file = root / "labels.yaml"
        labels_file.write_text(yaml.safe_dump(self.labels))
        self.onset_protocol = root / "protocol.yaml"
        self.onset_protocol.write_text(yaml.safe_dump({"protocol_id": "test_mean_protocol", "detection": {
            "onset_reference": {"kind": "mean_wall_time", "annotator_ids": ["author", "second", "third"],
                                "required_annotations_per_view": 3, "missing_policy": "error"}}}))
        recordings = SimpleNamespace(paths=self.raw, labels=SimpleNamespace(read=lambda: self.labels),
            catalog={"tasks": [{"id": "needle_pickup", "groups": [
                {"failure_mode": "Blood Occlusion", "episode_ids": [39], "outcome": "failure"}]}]},
            episode=lambda task, ep: self.episodes[ep])
        cache = root / "scores.npz"
        raw = np.tile(np.arange(8, 14), 2)
        wall = np.concatenate([self.episodes[ep].times[8:14] + 1000*ep for ep in (0, 39)])
        np.savez(cache, episode_index=np.repeat([0, 39], 6), frame_index=np.tile(np.arange(7, 13), 2),
                 raw_frame_index=raw, wall_timestamp=wall,
                 scores=np.array([[100, 1], [100, 2], [100, 1], [100, 90], [100, 90], [100, 90],
                                  [100, 1], [100, 7], [100, 3], [100, 90], [100, 90], [100, 90]]))
        self.report = {"protocols": {"paper_protocol": {"window": 1, "smoothing": "centered", "wm_frame_policy": "replay_legacy"}},
            "score_caches": [{"task": "needle_pickup", "method": "wm_nc", "views": ["fix", "wrist"],
                "path": str(cache), "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "scoring_settings": {"K": 4, "L": 4, "checkpoint": "frozen/checkpoints/best.pt"}}],
            "calibration": [{"task": "needle_pickup", "method": "wm_nc", "view": "wrist", "protocol": "paper",
                "calibration": "episode", "threshold": 5., "n_calibration": 1, "alpha": .05}]}
        self.report["calibration"].append({**self.report["calibration"][0], "view": "fix", "threshold": 150.})
        (root / "report.json").write_text(json.dumps(self.report))
        with patch.object(figure, "Recordings", return_value=recordings), patch.object(figure, "LABELS", labels_file):
            self.data = figure.FigureData(root, onset_protocol=self.onset_protocol)
        self.config = {"schema_version": 1, "view": "wrist", "panels": [
            {"task": "needle_pickup", "failure_mode": "Blood Occlusion"} for _ in range(4)]}
        self.output = root / "figure.svg"

    def test_frozen_threshold_view_and_raw_endpoint_alignment(self):
        payload = self.data.episode("needle_pickup", 39, "wrist")
        self.assertEqual(payload["raw_frames"], [8, 9, 10])
        self.assertEqual(payload["scores"], [1, 7, 3])
        self.assertEqual(payload["threshold"], 5)  # Calibration maximum is 2.
        np.testing.assert_allclose(payload["times"], [1.36, 1.98, 2.15])
        np.testing.assert_allclose(payload["calibration"][0]["times"], [1.36, 1.98, 2.15])

    def test_wrong_episode_and_unscored_or_reversed_frames_rejected(self):
        for change in ({"episode": 0}, {"frames": [7, 9]}, {"frames": [10, 8]}, {"frames": [8, 8]}):
            selection = deepcopy(self.config)
            selection["panels"][0].update(change)
            with self.assertRaises(ValueError):
                self.data.normalize(selection)
        selection = self.data.normalize(self.config)
        self.assertEqual(selection["panels"][0]["frames"], [8, 9])
        self.assertEqual(selection["panels"][0]["category"], "IV")

    def test_timestamp_offset_cannot_silently_pair_another_episode(self):
        np.savez(self.raw["needle_pickup", 39], timestamps=self.episodes[39].times + 39001)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            self.data.episode("needle_pickup", 39, "wrist")

    def test_tampered_score_cache_rejected(self):
        cache = Path(self.report["score_caches"][0]["path"])
        cache.write_bytes(cache.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.data.score_data("needle_pickup", "wrist")

    def test_onset_is_exact_raw_timestamp_for_selected_camera_and_annotator(self):
        wrist = self.data.episode("needle_pickup", 39, "wrist")
        self.assertEqual(wrist["onsets"]["author"]["raw_frame"], 6)  # Not a scored endpoint.
        self.assertAlmostEqual(wrist["onsets"]["author"]["time"], 1.02)
        self.assertEqual(wrist["onsets"]["second"]["raw_frame"], 10)
        fixed = self.data.episode("needle_pickup", 39, "fix")
        self.assertEqual(fixed["onsets"]["author"]["raw_frame"], 9)
        self.assertNotIn("second", fixed["onsets"])
        self.labels["tasks"]["needle_pickup"]["failure_onsets"][39]["author"]["views"]["wrist"]["frame_index"] = 0
        zero = self.data.episode("needle_pickup", 39, "wrist")["onsets"]["author"]
        self.assertEqual(zero["raw_frame"], 0)
        self.assertEqual(zero["time"], 0)

    def test_missing_view_onset_and_hidden_onset_are_not_drawn(self):
        selection = {**self.config, "view": "fix", "onset_annotator": "second"}
        self.assertNotIn(b'failure-onset-', figure.render_figure(self.data, selection))
        selection = {**self.config, "show_onset": False}
        self.assertNotIn(b'failure-onset-', figure.render_figure(self.data, selection))

    def test_mean_onset_defaults_to_exact_wall_time_not_average_frame_or_evidence_time(self):
        mean = self.data.episode("needle_pickup", 39, "wrist")["onsets"][figure.REFERENCE]
        self.assertEqual(self.data.catalog()["annotators"][0]["id"], figure.REFERENCE)
        self.assertEqual(self.data.normalize(self.config)["onset_annotator"], figure.REFERENCE)
        # Frames 6, 10 and 11 average to frame 9, but their irregular wall times
        # average to 1.83 s, BEFORE frame 9's recorded time of 1.98 s.
        self.assertEqual(mean["raw_frame"], 9)
        self.assertAlmostEqual(mean["time"], 1.83)
        self.assertAlmostEqual(mean["annotation"]["wall_timestamp"], 39001.83)
        self.assertAlmostEqual(mean["contributors"]["author"]["time"], 1.02)
        self.assertEqual(mean["annotation"]["contributing_frames"], {"author": 6, "second": 10, "third": 11})
        # Check the actual plotted coordinate on both axes, not just the payload.
        original = figure.Figure.savefig
        for axis, expected in (("seconds", 1.83), ("frames", 9)):
            captured = []
            def capture(fig, *args, **kwargs):
                captured.extend(line.get_xdata()[0] for ax in fig.axes for line in ax.lines
                                if (line.get_gid() or "").startswith("failure-onset-"))
                return original(fig, *args, **kwargs)
            with patch.object(figure.Figure, "savefig", capture):
                figure.render_figure(self.data, {**self.config, "time_axis": axis})
            np.testing.assert_allclose(captured, [expected] * 4)

    def test_incomplete_mean_is_not_replaced_with_individual_or_other_view(self):
        payload = self.data.episode("needle_pickup", 39, "fix")
        self.assertNotIn(figure.REFERENCE, payload["onsets"])
        self.assertIn("Rater B", payload["onset_errors"][figure.REFERENCE])
        selection = {**self.config, "view": "fix"}
        with self.assertRaisesRegex(ValueError, "missing onset from Rater B"):
            self.data.normalize(selection)
        # Existing individual selections and hiding the line still work.
        self.data.normalize({**selection, "onset_annotator": "author"})
        self.data.normalize({**selection, "show_onset": False})
        entries = self.labels["tasks"]["needle_pickup"]["failure_onsets"][39]
        entries["second"]["views"]["fix"] = {"frame_index": 4}
        fixed = self.data.episode("needle_pickup", 39, "fix")["onsets"][figure.REFERENCE]
        self.assertAlmostEqual(fixed["time"], (1.98 + .68 + 2.49) / 3)
        self.assertNotAlmostEqual(fixed["time"], 1.83)

    def test_copied_contributors_and_zero_onset_keep_provenance(self):
        entries = self.labels["tasks"]["needle_pickup"]["failure_onsets"][39]
        for aid in ("author", "second", "third"):
            entries[aid]["views"]["wrist"] = {"frame_index": 0}
            if aid != "author":
                entries[aid]["views"]["wrist"]["copied_from_annotator"] = "author"
        mean = self.data.episode("needle_pickup", 39, "wrist")["onsets"][figure.REFERENCE]
        self.assertEqual(mean["time"], 0)
        self.assertEqual(mean["raw_frame"], 0)
        self.assertEqual(mean["annotation"]["copied_contributors"], ["second", "third"])
        self.assertEqual(mean["contributors"]["third"]["annotation"]["copied_from_annotator"], "author")

    def test_defaults_restore_exact_choices_without_export_metadata(self):
        selection = self.data.normalize(self.config)
        selection["panels"][0]["frames"] = [9, 10]
        path = self.output.with_name("failure_example.yaml")
        figure.save_defaults(self.data, selection, path)
        loaded = figure.read_config(path)
        self.assertEqual(self.data.normalize(loaded), selection)
        self.assertNotIn("provenance", loaded)
        self.assertNotIn("category", loaded["panels"][0])
        original = path.read_bytes()
        selection["panels"][0]["episode"] = 0
        with self.assertRaises(ValueError):
            figure.save_defaults(self.data, selection, path)
        self.assertEqual(path.read_bytes(), original)

    def test_svg_embeds_eight_original_frames_and_keeps_vector_text(self):
        result = figure.export_figure(self.data, self.config, self.output)
        svg = ET.fromstring(self.output.read_bytes())
        ns = {"s": "http://www.w3.org/2000/svg"}
        images = svg.findall(".//s:image", ns)
        self.assertEqual(len(images), 8)
        for img in images:
            self.assertTrue(img.attrib["{http://www.w3.org/1999/xlink}href"].startswith("data:image/png;base64,"))
        self.assertGreater(len(svg.findall(".//s:path", ns)), 30)
        self.assertIn("Needle Pickup: Blood Occlusion", "".join(svg.itertext()))
        self.assertNotIn("Category", "".join(svg.itertext()))
        self.assertEqual(len([g for g in svg.findall('.//s:g', ns)
                              if g.get('id', '').startswith('failure-onset-')]), 4)
        review = ET.parse(result["review_svg"])
        self.assertIn("Needle Pickup", "".join(review.getroot().itertext()))
        self.assertTrue(svg.findall(".//s:text", ns))
        config = json.loads(Path(result["selection"]).read_text())
        self.assertEqual(config["provenance"]["panels"][0]["threshold"], 5)
        self.assertEqual(config["onset_annotator"], figure.REFERENCE)
        onset = config["provenance"]["panels"][0]["onset"]
        self.assertEqual(onset["raw_frame"], 9)
        self.assertAlmostEqual(onset["time"], 1.83)
        self.assertEqual(onset["contributors"]["third"]["raw_frame"], 11)
        self.assertEqual(config["provenance"]["onset_reference"]["protocol_id"], "test_mean_protocol")
        self.assertEqual(config["provenance"]["onset_reference"]["protocol_sha256"],
                         hashlib.sha256(self.onset_protocol.read_bytes()).hexdigest())
        self.assertEqual(self.data.normalize(config)["panels"], self.data.normalize(self.config)["panels"])
        config["provenance"]["report_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "different report"):
            self.data.normalize(config)


if __name__ == "__main__":
    unittest.main()
