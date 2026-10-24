"""Verify fixed episode splits, ablation isolation and shared model identity."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lerobot"))

from models.flow_matching_wm.dataset import EndpointWindowDataset, select_training_episode_ids, split_episode_ids
from models.flow_matching_wm.train import load_config
from models.flow_matching_wm.score import infer_train_config_path_from_checkpoint


class ExperimentTests(unittest.TestCase):
    def test_nested_training_subsets_keep_validation_disjoint(self):
        train, val = split_episode_ids(list(range(250)), 0.1, 0)
        previous = set()
        for fraction, count in [(0.25, 56), (0.5, 112), (0.75, 168), (1.0, 225)]:
            subset = select_training_episode_ids(train, fraction, 0)
            self.assertEqual(len(subset), count)
            self.assertTrue(previous <= set(subset))
            self.assertFalse(set(subset) & set(val))
            self.assertEqual(subset, select_training_episode_ids(train, fraction, 0))
            previous = set(subset)
        self.assertEqual(select_training_episode_ids(train), train)
        for invalid in (0, -1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                select_training_episode_ids(train, invalid)

    def test_windows_stay_in_selected_episodes(self):
        cfg = load_config(REPO / "lerobot/configs/ring_over_post/data_size_ablation_k4_l4/data25.yaml")
        cfg.data.normalize_vfmf_vae_latents = False
        cfg.data.normalize_actions = False
        indices = {ep: list(range(ep * 20, ep * 20 + 20)) for ep in range(10)}
        train, val = split_episode_ids(list(indices), 0.1, 0)
        chosen = select_training_episode_ids(train, cfg.data.train_episode_fraction, 0)
        ds = EndpointWindowDataset(None, indices, chosen, cfg)
        self.assertEqual({s.episode_id for s in ds.samples}, set(chosen))
        self.assertFalse({s.episode_id for s in ds.samples} & set(val))
        for sample in ds.samples:
            self.assertGreaterEqual(sample.anchor_pos_in_episode - cfg.data.history_len + 1, 0)
            self.assertLess(sample.anchor_pos_in_episode + cfg.data.pred_horizon, len(indices[sample.episode_id]))

    def test_all_config_groups_share_full_model_and_isolate_changes(self):
        def differences(left, right, prefix=""):
            result = set()
            for key in left.keys() | right.keys():
                name = prefix + key
                a, b = left.get(key), right.get(key)
                if isinstance(a, dict) and isinstance(b, dict):
                    result |= differences(a, b, name + ".")
                elif a != b:
                    result.add(name)
            return result

        allowed = {
            "k_ablation_l4": {"experiment_name", "data.pred_horizon", "data.action_chunk_len"},
            "l_ablation_k4": {"experiment_name", "data.history_len"},
            "action_ablation_k4_l4": {"experiment_name", "action_conditioner.use_cross_attention", "action_conditioner.use_action_adaln"},
            "data_size_ablation_k4_l4": {"experiment_name", "data.train_episode_fraction"},
        }
        for task in ("needle_pickup", "ring_over_post", "tissue_retraction", "shunt_insertion"):
            root = REPO / "lerobot/configs" / task
            base = yaml.safe_load((root / "flow_matching_wm.yaml").read_text())["fm_wm"]
            full_paths = ["k_ablation_l4/k4.yaml", "l_ablation_k4/l4.yaml",
                          "action_ablation_k4_l4/cross_attention_and_adaln.yaml", "data_size_ablation_k4_l4/data100.yaml"]
            for path in full_paths:
                self.assertEqual(yaml.safe_load((root / path).read_text())["fm_wm"], base)
            identities = {}
            for folder, keys in allowed.items():
                paths = list((root / folder).glob("*.yaml"))
                self.assertEqual(len(paths), 4)
                for path in paths:
                    cfg = load_config(path)
                    raw = yaml.safe_load(path.read_text())["fm_wm"]
                    self.assertTrue(differences(base, raw) <= keys, path)
                    self.assertEqual(cfg.data.output_dir, base["data"]["output_dir"])
                    if cfg.experiment_name in identities:
                        self.assertEqual(raw, identities[cfg.experiment_name])
                    identities[cfg.experiment_name] = raw

    def test_scorer_prefers_effective_config_and_supports_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "checkpoints").mkdir()
            checkpoint = run / "checkpoints/best.pt"
            legacy = run / "flow_matching_wm.yaml"
            legacy.write_text("fm_wm: {}\n")
            self.assertEqual(infer_train_config_path_from_checkpoint(checkpoint), legacy)
            effective = run / "config.json"
            effective.write_text(json.dumps({"seed": 1}))
            self.assertEqual(infer_train_config_path_from_checkpoint(checkpoint), effective)


if __name__ == "__main__":
    unittest.main()
