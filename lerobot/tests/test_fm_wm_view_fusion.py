"""CPU tests of joint FM-WM training, transport, saved configs and latent input."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'lerobot'))
from models.flow_matching_wm.model import FlowMatchingWorldModel, SingleViewFMBranch
from models.flow_matching_wm.train import load_config, build_config
from models.flow_matching_wm.dataset import EndpointWindowDataset
from models.flow_matching_wm.score import (
    load_merged_config, load_model_from_checkpoint, _compute_reactive_scores_from_eps_hat,
)


def tiny_config(fusion='concat_channels'):
    cfg = load_config(REPO / 'lerobot/configs/needle_pickup/view_fusion_ablation_k4_l4/concat_channels.yaml')
    cfg.backbone.view_fusion = fusion
    cfg.backbone.hidden_dim = 12
    cfg.backbone.depth = 1
    cfg.backbone.heads = 2
    cfg.backbone.mlp_dim = 24
    cfg.backbone.dropout = 0
    for conditioner in (cfg.action_conditioner, cfg.state_conditioner):
        conditioner.token_dim = 12
        conditioner.hidden_dim = 24
        conditioner.num_heads = 2
        conditioner.num_layers = 1
        conditioner.dropout = 0
    cfg.action_conditioner.global_cond_dim = 12
    cfg.action_conditioner.global_hidden_dim = 24
    cfg.data.latent_grid_height = cfg.data.latent_grid_width = 2
    cfg.data.latent_channels = 2
    cfg.data.normalize_vfmf_vae_latents = cfg.data.normalize_actions = False
    cfg.runtime.device = 'cpu'
    cfg.runtime.use_amp = False
    cfg.validate()
    return cfg


def inputs():
    return {'fix_hist': torch.randn(2, 4, 2, 2, 2), 'wrist_hist': torch.randn(2, 4, 2, 2, 2),
            'state_hist': None, 'action_chunk': torch.randn(2, 4, 10),
            'future_fix': torch.randn(2, 2, 2, 2), 'future_wrist': torch.randn(2, 2, 2, 2),
            'next_fix': torch.randn(2, 2, 2, 2), 'next_wrist': torch.randn(2, 2, 2, 2)}


class ViewFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_all_four_configs_change_only_fusion_and_run_name(self):
        for task in ('needle_pickup', 'ring_over_post', 'tissue_retraction', 'shunt_insertion'):
            root = REPO / 'lerobot/configs' / task
            base = yaml.safe_load((root / 'k_ablation_l4/k4.yaml').read_text())['fm_wm']
            new = yaml.safe_load((root / 'view_fusion_ablation_k4_l4/concat_channels.yaml').read_text())['fm_wm']
            self.assertNotEqual(base['experiment_name'], new['experiment_name'])
            self.assertEqual(new['backbone'].pop('view_fusion'), 'concat_channels')
            new['experiment_name'] = base['experiment_name']
            self.assertEqual(base, new)
            cfg = load_config(root / 'view_fusion_ablation_k4_l4/concat_channels.yaml')
            self.assertEqual((cfg.data.history_len, cfg.data.pred_horizon, cfg.data.latent_channels), (4, 4, 8))

    def test_one_branch_channels_and_public_shape(self):
        cfg = tiny_config(); model = FlowMatchingWorldModel(cfg)
        self.assertIsNone(model.fix_branch); self.assertIsNone(model.wrist_branch)
        self.assertEqual(sum(isinstance(m, SingleViewFMBranch) for m in model.modules()), 1)
        self.assertEqual(model.joint_branch.fc_in.in_features, 4)
        self.assertEqual(model.joint_branch.fc_out.linear.out_features, 4)
        self.assertEqual(cfg.data.latent_channels, 2)  # no mutation of the dataset layout
        batch = inputs()
        future = torch.stack((batch['future_fix'], batch['future_wrist']), dim=1)
        context = {k: batch[k] for k in ('fix_hist', 'wrist_hist', 'state_hist', 'action_chunk')}
        with patch.object(model.joint_branch, 'predict_velocity', wraps=model.joint_branch.predict_velocity) as call:
            output = model.predict_velocity(**context, x_t=future, tau=torch.ones(2))
            self.assertEqual(call.call_count, 1)
        self.assertEqual(output.shape, future.shape)
        self.assertEqual(model.predict_next_map(**context).shape, future.shape)
        packed = model._concat_views(batch['future_fix'], batch['future_wrist'])
        torch.testing.assert_close(packed[..., :2], batch['future_fix'])
        torch.testing.assert_close(model._unpack_joint_map(packed), future)

    def test_flow_and_auxiliary_keep_sum_of_view_mse_and_backward(self):
        model = FlowMatchingWorldModel(tiny_config())
        batch = inputs()
        batch['future_fix'].fill_(1); batch['future_wrist'].fill_(3)
        batch['next_fix'].fill_(2); batch['next_wrist'].fill_(4)
        with torch.no_grad():
            model.joint_branch.aux_head[-1].weight.zero_()
            model.joint_branch.aux_head[-1].bias.zero_()
        with patch.object(model, '_sample_base_noise', side_effect=torch.zeros_like):
            losses, stats = model.compute_loss(**batch)
        self.assertEqual(losses.flow.item(), 10.)  # 1**2 + 3**2, not their mean
        self.assertEqual(losses.aux_one_step.item(), 20.)
        self.assertEqual(losses.total.item(), 20.)  # 10 + .5 * 20
        self.assertEqual(stats['flow_fix_loss'].item(), 1.)
        self.assertEqual(stats['flow_wrist_loss'].item(), 9.)
        losses.total.backward()
        self.assertGreater(model.joint_branch.fc_out.linear.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.joint_branch.aux_head[-1].weight.grad.abs().sum().item(), 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        model(**batch)['loss'].backward()
        self.assertGreater(model.joint_branch.fc_in.weight.grad.abs().sum().item(), 0)

    def test_fixed_output_really_depends_on_wrist_inputs(self):
        torch.manual_seed(4)
        model = FlowMatchingWorldModel(tiny_config()).eval()
        # DiT's zero output initialization intentionally hides input sensitivity
        # before the first update. Use nonzero output weights to test coupling.
        with torch.no_grad():
            model.joint_branch.fc_out.linear.weight.normal_(std=.1)
        batch = inputs(); batch['wrist_hist'].requires_grad_()
        future = torch.stack((batch['future_fix'], batch['future_wrist']), dim=1).requires_grad_()
        context = {k: batch[k] for k in ('fix_hist', 'wrist_hist', 'state_hist', 'action_chunk')}
        output = model.predict_velocity(**context, x_t=future, tau=torch.ones(2))
        output[:, 0].square().sum().backward()
        self.assertGreater(batch['wrist_hist'].grad.abs().sum().item(), 0)
        self.assertGreater(future.grad[:, 1].abs().sum().item(), 0)

    def test_ode_reverse_transport_sampling_and_joint_score(self):
        model = FlowMatchingWorldModel(tiny_config()).eval()
        batch = inputs()
        context = {k: batch[k] for k in ('fix_hist', 'wrist_hist', 'state_hist', 'action_chunk')}
        future = torch.stack((batch['future_fix'], batch['future_wrist']), dim=1)
        with torch.no_grad():
            model.joint_branch.fc_out.linear.bias.fill_(.25)
            transported = model.transport_observed_endpoint_for_scoring(future, **context)
            torch.testing.assert_close(transported, future - .25)
            parts = _compute_reactive_scores_from_eps_hat(transported, 'squared_norm', 1.)
            torch.testing.assert_close(parts['score_union'], parts['score_view'].sum(-1))
            _, score = model.score_from_base(transported)
            torch.testing.assert_close(score, parts['score_union'])
            sampled = model.sample_endpoint_latent(**context, num_samples=2)
            self.assertEqual(sampled.shape, (2, 2, 2, 2, 2, 2))
            self.assertTrue(torch.isfinite(sampled).all())

    def test_legacy_default_stays_separate_and_checkpoint_types_do_not_mix(self):
        cfg = tiny_config('separate')
        raw = asdict(cfg); raw['backbone'].pop('view_fusion')
        legacy = FlowMatchingWorldModel(build_config(raw))
        self.assertIsNone(legacy.joint_branch)
        self.assertFalse(any('joint_branch' in key for key in legacy.state_dict()))
        explicit = FlowMatchingWorldModel(cfg)
        explicit.load_state_dict(legacy.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            FlowMatchingWorldModel(tiny_config()).load_state_dict(legacy.state_dict(), strict=True)

    def test_saved_config_and_score_override_retain_joint_architecture(self):
        cfg = tiny_config()
        model = FlowMatchingWorldModel(cfg).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'checkpoints').mkdir()
            import json
            (root/'config.json').write_text(json.dumps(asdict(cfg)))
            checkpoint = root/'checkpoints/best.pt'
            torch.save({'model': model.state_dict(), 'config': asdict(cfg)}, checkpoint)
            score_config = root/'score.yaml'
            score_config.write_text('fm_wm:\n  data:\n    dataset_root: fake_prl\n')
            merged, *_ = load_merged_config(score_config, checkpoint)
            self.assertEqual(merged.backbone.view_fusion, 'concat_channels')
            self.assertEqual(merged.data.latent_channels, 2)
            restored = load_model_from_checkpoint(merged, checkpoint, torch.device('cpu'))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])

    def test_dataset_normalization_and_windows_unchanged(self):
        cfg = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'latent_stats.npz'
            np.savez(path, fix_z_per_channel_mean=[1., 1.], fix_z_per_channel_std=[2., 2.],
                     wrist_z_per_channel_mean=[10., 10.], wrist_z_per_channel_std=[5., 5.])
            cfg.data.normalize_vfmf_vae_latents = True
            cfg.data.vfmf_vae_latent_stats_path = str(path)
            raw = [{cfg.data.fix_key: torch.full((2, 2, 2), 3.),
                    cfg.data.wrist_key: torch.full((2, 2, 2), 20.),
                    cfg.data.action_key: torch.ones(10), 'frame_index': i} for i in range(12)]
            fused = EndpointWindowDataset(raw, {0: list(range(12))}, [0], cfg)
            separate_cfg = deepcopy(cfg); separate_cfg.backbone.view_fusion = 'separate'
            separate = EndpointWindowDataset(raw, {0: list(range(12))}, [0], separate_cfg)
            self.assertEqual(len(fused), len(separate))
            for key, value in fused[0].items():
                torch.testing.assert_close(value, separate[0][key])
            torch.testing.assert_close(fused[0]['fix_hist'], torch.ones(4, 2, 2, 2))
            torch.testing.assert_close(fused[0]['wrist_hist'], torch.full((4, 2, 2, 2), 2.))

    def test_missing_view_or_malformed_channels_fail(self):
        cfg = tiny_config(); cfg.data.use_wrist_camera = False; cfg.data.num_cameras = 1
        with self.assertRaisesRegex(ValueError, 'both fixed and wrist'):
            cfg.validate()
        model = FlowMatchingWorldModel(tiny_config()); batch = inputs()
        for wrist in (None, torch.randn(2, 2, 2, 4)):
            with self.assertRaises(ValueError):
                model._concat_views(batch['future_fix'], wrist)
        batch['next_wrist'] = None
        with self.assertRaises(ValueError):
            model.compute_loss(**batch)


if __name__ == '__main__':
    unittest.main()
