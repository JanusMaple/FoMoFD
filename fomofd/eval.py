"""
fomofd/eval.py

Rollout a trained LeRobot policy in Isaac Lab via ZMQ (REQ/REP).

- Runs the same ManagerBasedEnv task (e.g., needle_pickup).
- Extracts fix + wrist RGB from Isaac obs (torch.Tensor HWC uint8).
- Encodes to JPEG and sends to LeRobot eval server over ZMQ.
- Receives action (float32, dim=8) and steps the env.

Example:
  # In isaacsim env (py3.11): run rollout client
  python fomofd/eval.py --task needle_pickup --enable_cameras --report_gripper

  python fomofd/eval.py --task needle_pickup \
    --enable_cameras \
    --report_gripper \
    --record \
    --action_execution_mismatch \
    --mismatch_start_step 50 \
    --mismatch_gain 0.85 \
    --mismatch_reversal_gain 0.2 \
    --mismatch_reversal_steps 4 \
    --mismatch_deadband_m 2e-4 \
    --mismatch_coupling_xy 1.0 \
    --mismatch_coupling_yx -0.5

  # In lerobot env (py3.10): run server
  python lerobot/eval_server.py --task needle_pickup --method act --including_gripper
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.quat_util import *
from fomofd.utils.ZmqClient import ZmqPolicyClient, ZmqControlClient

# ----------------------------
# Task specs
# ----------------------------

def _import_needle_pickup_cfg():
    from fomofd.tasks.needle_pickup.environment.env_cfg import NeedlePickupEnvCfg
    return NeedlePickupEnvCfg

def _import_ring_over_post_cfg():
    from fomofd.tasks.ring_over_post.environment.env_cfg import RingOverPostEnvCfg
    return RingOverPostEnvCfg

THIS_DIR = Path(__file__).resolve().parent
TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "needle_pickup": {
        "cfg_loader": _import_needle_pickup_cfg,
        "obs_groups": {"fix": "fix_camera", "wrist": "wrist_camera"},
        "rgb_keys": {"fix": "rgb", "wrist": "rgb"},
        "action_dim": 10,
        "default_out_dir": str(THIS_DIR / "tasks" / "needle_pickup" / "prl_data" / "raw"),
    },
    "ring_over_post": {
        "cfg_loader": _import_ring_over_post_cfg,
        "obs_groups": {"fix": "fix_camera", "wrist": "wrist_camera"},
        "rgb_keys": {"fix": "rgb", "wrist": "rgb"},
        "action_dim": 10,
        "default_out_dir": str(THIS_DIR / "tasks" / "ring_over_post" / "prl_data" / "raw"),
    },
}

# ----------------------------
# Utilities
# ----------------------------

def set_viewports_to_cameras():
    import asyncio
    import omni.ui as ui    # type: ignore
    import omni.kit.app     # type: ignore
    from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name # type: ignore

    async def setup_dual_viewports():
        create_viewport_window(name="Viewport 2")
        
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        
        vp_fix = get_viewport_from_window_name("Viewport")
        vp_wrist = get_viewport_from_window_name("Viewport 2")
        win_fix = ui.Workspace.get_window("Viewport")
        win_wrist = ui.Workspace.get_window("Viewport 2")

        if win_fix and win_wrist:
            win_wrist.dock_in(win_fix, ui.DockPosition.RIGHT, ratio=0.5)
            win_fix.focus()

        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        configs = [
            (vp_fix, "/World/envs/env_0/FixCamera"),
            (vp_wrist, "/World/envs/env_0/Robot/dvrk_psm_si/tool_roll_link/WristCamera")
        ]

        for vp, path in configs:
            if vp:
                # Essential: Set path first, then resolution
                vp.camera_path = path
                vp.fill_frame = False        # Disables auto-resize
                vp.resolution_scale = 1.0    # Locks to 100%
                vp.resolution = (224, 224)   # Forces 224x224
                
                # Kick the renderer to wake it up
                vp.updates_enabled = True

    asyncio.ensure_future(setup_dual_viewports())

def extract_rgb_u8(obs: dict, obs_group: str, rgb_key: str) -> torch.Tensor:
    """Return RGB torch.uint8 tensor [H,W,3]."""
    rgb = obs[obs_group][rgb_key]
    if not isinstance(rgb, torch.Tensor):
        raise TypeError("Expected torch.Tensor")

    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    if rgb.dtype.is_floating_point:
        rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)
    else:
        rgb = rgb.to(torch.uint8)

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB [H,W,3], got shape {rgb.shape}")

    return rgb

def warmup_env(env, action: torch.Tensor, steps: int = 5) -> dict:
    obs = None
    for _ in range(int(steps)):
        out = env.step(action)
        # IsaacLab env.step API varies; handle common cases
        if isinstance(out, tuple):
            obs = out[0]
        else:
            obs = out
    return obs

def encode_jpg_from_rgb_u8(rgb_hwc_u8: torch.Tensor, quality: int = 85) -> bytes:
    """Encode torch.uint8 [H,W,3] to JPEG bytes."""
    img = rgb_hwc_u8.contiguous().cpu().numpy()  # HWC uint8
    buf = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(
        buf, format="JPEG", quality=int(quality), subsampling=0
    )
    return buf.getvalue()

def _gaussian_kernel_2d(sigma: float, device, dtype) -> torch.Tensor:
    sigma = float(max(0.1, sigma))
    radius = max(1, int(round(3.0 * sigma)))
    size = 2 * radius + 1

    x = torch.arange(size, device=device, dtype=dtype) - radius
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1e-12)


def _blur_rgb_u8(rgb_hwc_u8: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return rgb_hwc_u8

    x = rgb_hwc_u8.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    kernel = _gaussian_kernel_2d(sigma, device=rgb_hwc_u8.device, dtype=x.dtype)
    k = int(kernel.shape[0])
    weight = kernel.view(1, 1, k, k).repeat(3, 1, 1, 1)

    y = F.conv2d(x, weight, padding=k // 2, groups=3)
    return (y.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0).to(torch.uint8)


def make_blood_corruption_state(h: int, w: int, device) -> dict:
    rng = np.random.default_rng()
    n = int(rng.integers(3, 6))  # 3 to 5 droplets

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    color = np.zeros((h, w, 3), dtype=np.float32)

    droplets_meta = []

    for _ in range(n):
        cx = float(rng.uniform(0.18 * w, 0.82 * w))
        cy = float(rng.uniform(0.18 * h, 0.78 * h))
        rx = float(rng.uniform(0.045 * w, 0.105 * w))
        ry = float(rng.uniform(0.045 * h, 0.125 * h))
        theta = float(rng.uniform(0.0, np.pi))

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx, dy = xx - cx, yy - cy
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy

        ang = np.arctan2(yr / max(ry, 1e-6), xr / max(rx, 1e-6))
        boundary_noise = (
            1.0
            + 0.14 * np.sin(3.0 * ang + float(rng.uniform(0, 2 * np.pi)))
            + 0.09 * np.sin(7.0 * ang + float(rng.uniform(0, 2 * np.pi)))
        )

        r = np.sqrt((xr / max(rx, 1e-6)) ** 2 + (yr / max(ry, 1e-6)) ** 2)
        blob = 1.0 / (1.0 + np.exp((r - boundary_noise) * 14.0))

        # Downward smear / streak.
        smear_cx = cx + float(rng.uniform(-0.25, 0.25) * rx)
        smear_cy = cy + float(rng.uniform(0.75, 1.55) * ry)
        smear_rx = rx * float(rng.uniform(0.30, 0.65))
        smear_ry = ry * float(rng.uniform(1.7, 3.2))

        sdx, sdy = xx - smear_cx, yy - smear_cy
        sxr = cos_t * sdx + sin_t * sdy
        syr = -sin_t * sdx + cos_t * sdy
        sr = np.sqrt((sxr / max(smear_rx, 1e-6)) ** 2 + (syr / max(smear_ry, 1e-6)) ** 2)
        smear = 1.0 / (1.0 + np.exp((sr - 1.0) * 10.0))

        a = np.maximum(
            blob * float(rng.uniform(0.55, 0.78)),
            smear * float(rng.uniform(0.16, 0.34)),
        ).astype(np.float32)

        blood_rgb = np.array([
            rng.uniform(80, 145),
            rng.uniform(0, 35),
            rng.uniform(0, 28),
        ], dtype=np.float32)

        replace = a > alpha
        color[replace] = blood_rgb
        alpha = np.maximum(alpha, a)

        # Small wet specular highlight.
        hx = cx - 0.28 * rx
        hy = cy - 0.32 * ry
        hrx = max(2.0, 0.18 * rx)
        hry = max(1.5, 0.11 * ry)

        hl = np.exp(-(((xx - hx) / hrx) ** 2 + ((yy - hy) / hry) ** 2)).astype(np.float32)
        hl_alpha = hl * float(rng.uniform(0.12, 0.26))
        highlight_rgb = np.array([230.0, 205.0, 205.0], dtype=np.float32)

        color = color * (1.0 - hl_alpha[..., None]) + highlight_rgb * hl_alpha[..., None]
        alpha = np.clip(alpha + 0.35 * hl_alpha, 0.0, 0.92)

        droplets_meta.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry, "theta": theta})

    return {
        "alpha": torch.from_numpy(alpha[..., None]).to(device=device, dtype=torch.float32),
        "color": torch.from_numpy(color).to(device=device, dtype=torch.float32),
        "blur_sigma": float(rng.uniform(2.0, 4.5)),
        "num_droplets": n,
        "droplets": droplets_meta,
    }


def apply_blood_corruption(rgb_hwc_u8: torch.Tensor, blood_state: dict | None) -> torch.Tensor:
    if blood_state is None:
        return rgb_hwc_u8

    alpha = blood_state["alpha"].to(device=rgb_hwc_u8.device, dtype=torch.float32)
    color = blood_state["color"].to(device=rgb_hwc_u8.device, dtype=torch.float32)
    blur_sigma = float(blood_state.get("blur_sigma", 3.0))

    base = rgb_hwc_u8.float()
    blurred = _blur_rgb_u8(rgb_hwc_u8, sigma=blur_sigma).float()

    mixed_under_lens = base * (1.0 - alpha) + blurred * alpha
    out = mixed_under_lens * (1.0 - 0.82 * alpha) + color * (0.82 * alpha)

    return out.clamp(0, 255).to(torch.uint8)


class ActionExecutionMismatch:
    """Cable/tendon-like mismatch between commanded and executed tool motion.

    Only translation is corrupted. Rotation and gripper are left unchanged.
    This preserves realistic frame-level observations but breaks action-conditioned
    dynamics: the policy command says "move this way", while the robot
    under-responds and/or moves with cross-axis tendon/transmission coupling,
    especially after direction reversals.
    """

    def __init__(
        self,
        gain: float = 0.55,
        reversal_gain: float = 0.10,
        reversal_steps: int = 6,
        deadband_m: float = 2e-4,
        coupling_xy: float = 0.0,
        coupling_yx: float = 0.0,
        coupling_xz: float = 0.0,
        coupling_zx: float = 0.0,
        coupling_yz: float = 0.0,
        coupling_zy: float = 0.0,
    ):
        self.gain = float(gain)
        self.reversal_gain = float(reversal_gain)
        self.reversal_steps = int(reversal_steps)
        self.deadband_m = float(deadband_m)

        # Cross-axis transmission coupling. Naming convention:
        # coupling_xy means commanded x translation leaks into executed y translation.
        self.coupling_xy = float(coupling_xy)
        self.coupling_yx = float(coupling_yx)
        self.coupling_xz = float(coupling_xz)
        self.coupling_zx = float(coupling_zx)
        self.coupling_yz = float(coupling_yz)
        self.coupling_zy = float(coupling_zy)

        self.prev_cmd_dp: torch.Tensor | None = None
        self.reversal_steps_left = 0

    def reset(self) -> None:
        self.prev_cmd_dp = None
        self.reversal_steps_left = 0

    def apply(self, commanded_action: torch.Tensor) -> torch.Tensor:
        """Return executed action tensor with translation lag/backlash.

        Args:
            commanded_action: torch.Tensor [N, 8], usually [dp(3), quat(4), g(1)].
        """
        executed = commanded_action.clone()
        cmd_dp = commanded_action[0, 0:3].detach().clone()

        if self.prev_cmd_dp is not None:
            # Direction reversal is where cable slack/backlash is most visible.
            if torch.dot(cmd_dp, self.prev_cmd_dp).item() < 0.0:
                self.reversal_steps_left = self.reversal_steps

        if torch.linalg.norm(cmd_dp).item() < self.deadband_m:
            exec_dp = torch.zeros_like(cmd_dp)
        elif self.reversal_steps_left > 0:
            exec_dp = self.reversal_gain * cmd_dp
            self.reversal_steps_left -= 1
        else:
            exec_dp = self.gain * cmd_dp

        # Add cross-axis coupling on top of the gain/backlash response.
        # This simulates tendon/cable transmission or calibration error:
        # the tool moves plausibly, but not along the commanded direction.
        if any(
            abs(v) > 0.0
            for v in (
                self.coupling_xy, self.coupling_yx,
                self.coupling_xz, self.coupling_zx,
                self.coupling_yz, self.coupling_zy,
            )
        ):
            coupled = exec_dp.clone()
            coupled[0] = coupled[0] + self.coupling_yx * cmd_dp[1] + self.coupling_zx * cmd_dp[2]
            coupled[1] = coupled[1] + self.coupling_xy * cmd_dp[0] + self.coupling_zy * cmd_dp[2]
            coupled[2] = coupled[2] + self.coupling_xz * cmd_dp[0] + self.coupling_yz * cmd_dp[1]
            exec_dp = coupled

        executed[0, 0:3] = exec_dp
        self.prev_cmd_dp = cmd_dp
        return executed
    

class SceneObjectMotionAttack:
    """Inject horizontal velocity into a configured scene object.

    Used for:
      - ring_over_post: move ring
      - needle_pickup: move needle

    This does not alter robot actions. It changes the scene dynamics directly.
    """

    def __init__(self, env, cfg):
        self.env = env

        self.asset_name = str(getattr(cfg, "scene_motion_attack_asset_name"))
        self.attack_type = str(getattr(cfg, "scene_motion_attack_type", "scene_object_motion_velocity"))
        self.asset = env.scene[self.asset_name]

        self.speed_mps = float(getattr(cfg, "scene_motion_attack_speed_mps", 0.003))
        self.duration_steps = int(getattr(cfg, "scene_motion_attack_duration_steps", 90))
        self.zero_xy_on_stop = bool(getattr(cfg, "scene_motion_attack_zero_xy_velocity_on_stop", True))

        direction_xy = getattr(cfg, "scene_motion_attack_direction_xy", (1.0, 0.0))
        d = torch.tensor(direction_xy, device=env.device, dtype=torch.float32).reshape(2)
        self.direction_xy = d / torch.linalg.norm(d).clamp_min(1e-8)

        self.active = False
        self.steps_left = 0

    def reset(self) -> None:
        if self.active and self.zero_xy_on_stop:
            self._zero_xy_velocity()
        self.active = False
        self.steps_left = 0

    def start(self, step_i: int) -> dict:
        self.active = True
        self.steps_left = self.duration_steps

        return {
            "step": int(step_i),
            "type": self.attack_type,
            "asset_name": self.asset_name,
            "speed_mps": float(self.speed_mps),
            "direction_xy": [
                float(self.direction_xy[0].detach().cpu().item()),
                float(self.direction_xy[1].detach().cpu().item()),
            ],
            "duration_steps": int(self.duration_steps),
            "horizontal_plane": "world_xy_velocity",
        }

    def _get_root_velocity(self) -> torch.Tensor:
        if hasattr(self.asset.data, "root_vel_w"):
            return self.asset.data.root_vel_w.clone()
        return self.asset.data.root_state_w[:, 7:13].clone()

    def _write_root_velocity(self, root_vel: torch.Tensor) -> None:
        if hasattr(self.asset, "write_root_velocity_to_sim"):
            self.asset.write_root_velocity_to_sim(root_vel)
        else:
            root_state = self.asset.data.root_state_w.clone()
            root_state[:, 7:13] = root_vel
            self.asset.write_root_state_to_sim(root_state)

    def _zero_xy_velocity(self) -> None:
        root_vel = self._get_root_velocity()
        root_vel[:, 0:2] = 0.0
        self._write_root_velocity(root_vel)

    def apply_one_step(self) -> None:
        if not self.active:
            return

        if self.duration_steps >= 0 and self.steps_left <= 0:
            if self.zero_xy_on_stop:
                self._zero_xy_velocity()
            self.active = False
            return

        root_vel = self._get_root_velocity()

        # Only overwrite world XY velocity.
        # Preserve z velocity and angular velocity so contacts stay physical.
        v_xy = self.direction_xy * self.speed_mps
        root_vel[:, 0] = v_xy[0]
        root_vel[:, 1] = v_xy[1]

        self._write_root_velocity(root_vel)

        if self.duration_steps >= 0:
            self.steps_left -= 1


class EvalEpisodeRecorder:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episode_idx = self._infer_next_episode_idx()
        self.reset()

    def reset(self):
        self.images_fix = []
        self.images_wrist = []
        self.actions = []                       # main recorded action, e.g. act8
        self.executed_actions = []              # env-stepped action, e.g. act8
        self.policy_action = []                 # first raw policy action, act10
        self.policy_action_chunks = []          # raw predicted chunk, [K, 10]
        self.policy_action_chunk_batches = []   # raw sampled chunk batch, [B, K, 10]
        self.state = []
        self.timestamps = []

    def has_data(self) -> bool:
        return len(self.timestamps) > 0

    def push(self, rgb_fix, rgb_wrist, action, state, t_wall,
            policy_action=None, policy_action_chunk=None, policy_action_chunk_batch=None,
            executed_action=None):
        self.images_fix.append(rgb_fix.detach().cpu().numpy())
        self.images_wrist.append(rgb_wrist.detach().cpu().numpy())

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        self.actions.append(np.asarray(action, dtype=np.float32))

        if executed_action is None:
            executed_action = action
        if isinstance(executed_action, torch.Tensor):
            executed_action = executed_action.detach().cpu().numpy()
        self.executed_actions.append(np.asarray(executed_action, dtype=np.float32))

        if policy_action is None:
            self.policy_action.append(np.full((10,), np.nan, dtype=np.float32))
        else:
            if isinstance(policy_action, torch.Tensor):
                policy_action = policy_action.detach().cpu().numpy()
            self.policy_action.append(np.asarray(policy_action, dtype=np.float32))

        if policy_action_chunk is None:
            self.policy_action_chunks.append(np.full((1, 10), np.nan, dtype=np.float32))
        else:
            if isinstance(policy_action_chunk, torch.Tensor):
                policy_action_chunk = policy_action_chunk.detach().cpu().numpy()
            self.policy_action_chunks.append(np.asarray(policy_action_chunk, dtype=np.float32))

        if policy_action_chunk_batch is None:
            self.policy_action_chunk_batches.append(np.full((1, 1, 10), np.nan, dtype=np.float32))
        else:
            if isinstance(policy_action_chunk_batch, torch.Tensor):
                policy_action_chunk_batch = policy_action_chunk_batch.detach().cpu().numpy()
            self.policy_action_chunk_batches.append(np.asarray(policy_action_chunk_batch, dtype=np.float32))

        self.state.append(np.asarray(state, dtype=np.float32))
        self.timestamps.append(float(t_wall))

    def save(self, meta: dict) -> Path:
        save_path = self.out_dir / f"episode_{self.episode_idx:05d}.npz"

        chunk_shapes = {tuple(x.shape) for x in self.policy_action_chunks}
        if len(chunk_shapes) != 1:
            raise RuntimeError(
                f"Inconsistent policy_action_chunk shapes in episode: {sorted(chunk_shapes)}"
            )

        batch_shapes = {tuple(x.shape) for x in self.policy_action_chunk_batches}
        if len(batch_shapes) != 1:
            raise RuntimeError(
                f"Inconsistent policy_action_chunk_batch shapes in episode: {sorted(batch_shapes)}"
            )

        np.savez_compressed(
            save_path,
            images_fix=np.stack(self.images_fix, axis=0),
            images_wrist=np.stack(self.images_wrist, axis=0),
            actions=np.stack(self.actions, axis=0),                            # [T, 8]
            executed_actions=np.stack(self.executed_actions, axis=0),          # [T, 8]
            policy_action=np.stack(self.policy_action, axis=0),                # [T, 10]
            policy_action_chunks=np.stack(self.policy_action_chunks, axis=0),  # [T, K, 10]
            policy_action_chunk_batches=np.stack(self.policy_action_chunk_batches, axis=0),  # [T, B, K, 10]
            states=np.stack(self.state, axis=0),
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
            meta=np.asarray(json.dumps(meta)),
        )
        self.episode_idx += 1
        self.reset()
        return save_path

    def _infer_next_episode_idx(self) -> int:
        max_idx = -1
        pat = re.compile(r"^episode_(\d+)\.npz$")

        for p in self.out_dir.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if m is None:
                continue
            max_idx = max(max_idx, int(m.group(1)))

        return max_idx + 1

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LeRobot policy in Isaac Lab via ZMQ.")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASK_SPECS.keys()))
    parser.add_argument("--policy_addr", type=str, default="ipc:///tmp/fomofd_policy.sock", help="Control socket exposed by eval_server.py")
    parser.add_argument("--control_addr", type=str, default="ipc:///tmp/fomofd_control.sock", help="Control socket exposed by eval_server.py")
    parser.add_argument("--zmq_timeout_ms", type=int, default=300)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=2000, help="Max env steps before exit.")
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--report_gripper", action="store_true",
                        help="Report gripper jaw joint position as part of state")
    parser.add_argument("--record", action="store_true", help="Record eval episodes to NPZ.")
    parser.add_argument("--record_out_dir", type=str, default=None, help="Output dir for recorded eval episodes.")
    parser.add_argument("--policy_chunk_batch_size", type=int, default=32,
                        help="Number of sampled policy action chunks to record per step when --record is on.")

    parser.add_argument("--control_poll_every", type=int, default=10)
    parser.add_argument("--attack_translate_m", type=float, default=0.004,
                        help="Total translation attack distance over one chunk, in meters.")
    parser.add_argument("--attack_translation_mode", type=str, default="total",
                        choices=["total", "per_step"],
                        help="'total' spreads attack_translate_m over the chunk; 'per_step' applies it every step.")
    parser.add_argument("--attack_chunk_steps", type=int, default=4,
                        help="Fallback chunk length if policy chunk shape is unavailable.")
    parser.add_argument("--attack_gripper_value", type=float, default=0.4,
                        help="Injected gripper command for f attack. 0.5=open, 0/0.09=closed.")

    parser.add_argument("--action_execution_mismatch", action="store_true",
                        help="Enable cable/tendon-like command-execution mismatch. The policy command is normal, but the env receives an attenuated/lagged action.")
    parser.add_argument("--mismatch_start_step", type=int, default=0,
                        help="Step at which action-execution mismatch starts after each reset.")
    parser.add_argument("--mismatch_duration_steps", type=int, default=-1,
                        help="Number of steps to keep mismatch active. Use -1 for persistent until reset/end.")
    parser.add_argument("--mismatch_gain", type=float, default=0.55,
                        help="Translation gain during normal mismatch; 1.0 disables attenuation.")
    parser.add_argument("--mismatch_reversal_gain", type=float, default=0.10,
                        help="Translation gain immediately after commanded direction reversals.")
    parser.add_argument("--mismatch_reversal_steps", type=int, default=6,
                        help="Number of post-reversal steps using mismatch_reversal_gain.")
    parser.add_argument("--mismatch_deadband_m", type=float, default=2e-4,
                        help="Commanded translation norm below this value is executed as zero motion.")
    parser.add_argument("--mismatch_coupling_xy", type=float, default=0.0,
                        help="Cross-axis coupling: commanded x translation leaks into executed y translation.")
    parser.add_argument("--mismatch_coupling_yx", type=float, default=0.0,
                        help="Cross-axis coupling: commanded y translation leaks into executed x translation.")
    parser.add_argument("--mismatch_coupling_xz", type=float, default=0.0,
                        help="Cross-axis coupling: commanded x translation leaks into executed z translation.")
    parser.add_argument("--mismatch_coupling_zx", type=float, default=0.0,
                        help="Cross-axis coupling: commanded z translation leaks into executed x translation.")
    parser.add_argument("--mismatch_coupling_yz", type=float, default=0.0,
                        help="Cross-axis coupling: commanded y translation leaks into executed z translation.")
    parser.add_argument("--mismatch_coupling_zy", type=float, default=0.0,
                        help="Cross-axis coupling: commanded z translation leaks into executed y translation.")

    # Isaac Lab args
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Optional: quiet logging channels
    extra = " --/log/channels/omni.usd=disable --/log/channels/carb=disable"
    args.kit_args = (getattr(args, "kit_args", "") or "") + extra

    spec = TASK_SPECS[args.task]
    obs_groups = spec["obs_groups"]
    rgb_keys = spec["rgb_keys"]
    record_out_dir = Path(args.record_out_dir or spec["default_out_dir"])

    # Launch app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import env after app launches
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import subtract_frame_transforms

    EnvCfgCls = spec["cfg_loader"]()
    cfg = EnvCfgCls()

    # Respect CLI device if present
    if hasattr(args, "device") and args.device is not None:
        cfg.sim.device = args.device

    env = ManagerBasedEnv(cfg=cfg)

    scene_motion_attack = None
    if args.task in {"ring_over_post", "needle_pickup"} and hasattr(cfg, "scene_motion_attack_asset_name"):
        scene_motion_attack = SceneObjectMotionAttack(env, cfg)

    # Recorder
    robot = env.scene["psm_si"]
    fix_cam = env.scene["fix_camera"]
    from isaaclab.managers import SceneEntityCfg
    robot_entity_cfg = SceneEntityCfg("psm_si", body_names=["tool_yaw_link"])
    robot_entity_cfg.resolve(env.scene)
    ee_body_id = robot_entity_cfg.body_ids[0]
    jaw_entity_cfg = SceneEntityCfg("psm_si", joint_names=["jaw_joint_1"], body_names=[])
    jaw_entity_cfg.resolve(env.scene)
    jaw_joint_id = jaw_entity_cfg.joint_ids[0]
    def get_record_state() -> np.ndarray:
        ee_pose_w = robot.data.body_state_w[0, ee_body_id, 0:7]
        p_fix_w = fix_cam.data.pos_w[0]
        q_fix_w = fix_cam.data.quat_w_world[0]
        p_ee_fix, q_ee_fix = subtract_frame_transforms(
            p_fix_w.unsqueeze(0), q_fix_w.unsqueeze(0),
            ee_pose_w[0:3].unsqueeze(0), ee_pose_w[3:7].unsqueeze(0),
        )
        p = p_ee_fix[0].detach().cpu().numpy().astype(np.float32, copy=False)
        q = q_ee_fix[0].detach().cpu().numpy().astype(np.float32, copy=False)
        jaw = np.array([float(robot.data.joint_pos[0, jaw_joint_id].detach().cpu().item())], dtype=np.float32)
        return np.concatenate([p, q, jaw], axis=0).astype(np.float32, copy=False)  # (8,)

    # Default action (keep consistent with your training action space = 8 dims)
    def get_default_action() -> torch.Tensor:
        dp = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
        dq = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
        dq[:, 0] = 1.0  # w=1
        gr_cfg = env.cfg.actions.psm_gripper
        gr = torch.full((env.num_envs, 1), float(gr_cfg.default_gripper), device=env.device, dtype=torch.float32)
        return torch.cat([dp, dq, gr], dim=-1)

    last_action = get_default_action()

    # Fetch state: EE pose in fix camera frame from the scene + (maybe) gripper jaw joint position
    def get_cur_state(env, report_gripper) -> np.ndarray:
        """
        Returns float32 (10,) = [p_fix(3), rot6d(6), jaw(1)]
        - pose is EE in fix_camera frame
        - jaw is robot joint_pos for jaw_joint_1 (radians, float32)
        """
        robot = env.scene["psm_si"]
        fix_cam = env.scene["fix_camera"]

        # Resolve EE body id once (tool_yaw_link, same as record.py)
        if not hasattr(get_cur_state, "_ee_body_id"):
            robot_entity_cfg = SceneEntityCfg("psm_si", body_names=["tool_yaw_link"])
            robot_entity_cfg.resolve(env.scene)
            get_cur_state._ee_body_id = robot_entity_cfg.body_ids[0]

        ee_body_id = get_cur_state._ee_body_id

        # EE pose in world
        ee_pose_w = robot.data.body_state_w[0, ee_body_id, 0:7]  # pos(3), quat(wxyz)(4)

        # Fix camera pose in world
        p_fix_w = fix_cam.data.pos_w[0]          # (3,)
        q_fix_w = fix_cam.data.quat_w_world[0]   # (4,) wxyz

        # EE pose in fix cam frame: fix^{-1} * ee
        p_ee_fix, q_ee_fix = subtract_frame_transforms(
            p_fix_w.unsqueeze(0), q_fix_w.unsqueeze(0),
            ee_pose_w[0:3].unsqueeze(0), ee_pose_w[3:7].unsqueeze(0),
        )
        p = p_ee_fix[0].detach().cpu().numpy().astype(np.float32, copy=False)  # (3,)
        q = q_ee_fix[0].detach().cpu().numpy().astype(np.float32, copy=False)  # (4,) wxyz

        r6 = quat_wxyz_to_rot6d(q).astype(np.float32, copy=False)              # (6,)

        if report_gripper:
            # Resolve jaw joint id once (known name)
            if not hasattr(get_cur_state, "_jaw_joint_id"):
                jaw_entity_cfg = SceneEntityCfg("psm_si", joint_names=["jaw_joint_1"], body_names=[])
                jaw_entity_cfg.resolve(env.scene)
                get_cur_state._jaw_joint_id = jaw_entity_cfg.joint_ids[0]

            jaw_joint_id = get_cur_state._jaw_joint_id

            # Jaw joint position
            jaw = robot.data.joint_pos[0, jaw_joint_id]
            jaw = float(jaw.detach().cpu().item())
            jaw = np.array([jaw], dtype=np.float32)                                     # (1,)

        if report_gripper:
            return np.concatenate([p, r6, jaw], axis=0).astype(np.float32, copy=False)  # (10,)
        else:
            return np.concatenate([p, r6], axis=0).astype(np.float32, copy=False)       # (10,)

    # ZMQ client
    policy_client = ZmqPolicyClient(args.policy_addr, timeout_ms=args.zmq_timeout_ms)
    control_client = ZmqControlClient(args.control_addr, timeout_ms=args.zmq_timeout_ms)

    # Recorder
    recorder = EvalEpisodeRecorder(record_out_dir) if args.record else None
    recording_armed = False          # stays False until first RESET

    # Attacks
    ATTACK_DIRS_FIX = {
        "ATTACK_W": np.array([+1.0,  0.0, 0.0], dtype=np.float32),  # forward
        "ATTACK_S": np.array([-1.0,  0.0, 0.0], dtype=np.float32),  # backward
        "ATTACK_A": np.array([ 0.0, +1.0, 0.0], dtype=np.float32),  # left
        "ATTACK_D": np.array([ 0.0, -1.0, 0.0], dtype=np.float32),  # right
    }
    pending_motion_attack = None
    pending_gripper_attack = False
    pending_blood_attack = False
    active_motion_delta = None
    active_motion_steps_left = 0
    active_gripper_steps_left = 0
    active_blood_corruption_fix = None
    active_blood_corruption_wrist = None
    action_mismatch = ActionExecutionMismatch(
        gain=args.mismatch_gain,
        reversal_gain=args.mismatch_reversal_gain,
        reversal_steps=args.mismatch_reversal_steps,
        deadband_m=args.mismatch_deadband_m,
        coupling_xy=args.mismatch_coupling_xy,
        coupling_yx=args.mismatch_coupling_yx,
        coupling_xz=args.mismatch_coupling_xz,
        coupling_zx=args.mismatch_coupling_zx,
        coupling_yz=args.mismatch_coupling_yz,
        coupling_zy=args.mismatch_coupling_zy,
    ) if args.action_execution_mismatch else None
    mismatch_started = False
    attack_events = []
    def infer_chunk_len(chunk) -> int:
        try:
            if chunk is not None and np.ndim(chunk) == 2:
                return max(1, int(chunk.shape[0]))
        except Exception:
            pass
        return max(1, int(args.attack_chunk_steps))

    def mismatch_active_at(step: int) -> bool:
        if not args.action_execution_mismatch:
            return False
        start = max(0, int(args.mismatch_start_step))
        if step < start:
            return False
        duration = int(args.mismatch_duration_steps)
        return duration < 0 or step < start + duration

    try:
        # Reset
        obs, _ = env.reset()

        set_viewports_to_cameras()

        # Warmup camera buffers
        obs = warmup_env(env, last_action, steps=args.warmup_steps)

        print(f"[INFO] Rollout task={args.task}")
        print(f"[INFO] Policy addr={args.policy_addr} timeout={args.zmq_timeout_ms}ms jpeg_q={args.jpeg_quality}")
        if args.action_execution_mismatch:
            print(
                "[FAULT] action_execution_mismatch enabled: "
                f"start={args.mismatch_start_step}, duration={args.mismatch_duration_steps}, "
                f"gain={args.mismatch_gain}, reversal_gain={args.mismatch_reversal_gain}, "
                f"reversal_steps={args.mismatch_reversal_steps}, deadband_m={args.mismatch_deadband_m}, "
                f"coupling_xy={args.mismatch_coupling_xy}, coupling_yx={args.mismatch_coupling_yx}, "
                f"coupling_xz={args.mismatch_coupling_xz}, coupling_zx={args.mismatch_coupling_zx}, "
                f"coupling_yz={args.mismatch_coupling_yz}, coupling_zy={args.mismatch_coupling_zy}"
            )

        step_i = 0
        t0 = time.time()

        while simulation_app.is_running():
            try:
                if step_i % max(1, int(args.control_poll_every)) == 0:
                    ctrl_cmd = control_client.poll_command()
                else:
                    ctrl_cmd = None
            except Exception as e:
                ctrl_cmd = None
                if step_i % max(1, int(args.print_every)) == 0:
                    print(f"[WARN] control poll failed: {e}")
            if ctrl_cmd in {"RESET", "RESET_NOSAVE"}:
                save_episode = ctrl_cmd == "RESET"
                print(f"[INFO] {ctrl_cmd} requested from eval_server. Resetting environment...")

                if args.record:
                    if save_episode and recording_armed and recorder.has_data():
                        save_path = recorder.save(meta={
                            "task": args.task,
                            "end_reason": ctrl_cmd.lower(),
                            "attacks": attack_events,
                        })
                        print(f"[REC] Saved {save_path}")
                    else:
                        recorder.reset()
                        if ctrl_cmd == "RESET_NOSAVE":
                            print("[REC] Discarded current episode.")

                    recording_armed = True

                pending_motion_attack = None
                pending_gripper_attack = False
                pending_blood_attack = False
                active_motion_delta = None
                active_motion_steps_left = 0
                active_gripper_steps_left = 0
                active_blood_corruption_fix = None
                active_blood_corruption_wrist = None
                if scene_motion_attack is not None:
                    scene_motion_attack.reset()
                if action_mismatch is not None:
                    action_mismatch.reset()
                mismatch_started = False
                attack_events.clear()

                obs, _ = env.reset()
                last_action = get_default_action()
                obs = warmup_env(env, last_action, steps=args.warmup_steps)
                step_i = 0
                t0 = time.time()
                continue

            elif ctrl_cmd in ATTACK_DIRS_FIX:
                pending_motion_attack = ctrl_cmd
                print(f"[ATTACK] queued translation attack {ctrl_cmd}")

            elif ctrl_cmd == "ATTACK_F":
                pending_gripper_attack = True
                print(f"[ATTACK] queued gripper attack: g={args.attack_gripper_value}")

            elif ctrl_cmd == "ATTACK_B":
                pending_blood_attack = True
                print("[ATTACK] queued blood corruption attack on both cameras")

            elif ctrl_cmd == "ATTACK_M":
                if scene_motion_attack is None:
                    print(f"[ATTACK] ignored scene motion attack: not configured for task={args.task}")
                else:
                    attack_events.append(scene_motion_attack.start(step_i))

            if step_i >= int(args.max_steps):
                print(f"[INFO] Reached max_steps={args.max_steps}. Exiting.")
                break

            # Build observation images for policy
            rgb_fix = extract_rgb_u8(obs, obs_group=obs_groups["fix"], rgb_key=rgb_keys["fix"])
            rgb_wrist = extract_rgb_u8(obs, obs_group=obs_groups["wrist"], rgb_key=rgb_keys["wrist"])

            # Apply attacks to observations
            if pending_blood_attack:
                active_blood_corruption_fix = make_blood_corruption_state(
                    h=int(rgb_fix.shape[0]),
                    w=int(rgb_fix.shape[1]),
                    device=rgb_fix.device,
                )
                active_blood_corruption_wrist = make_blood_corruption_state(
                    h=int(rgb_wrist.shape[0]),
                    w=int(rgb_wrist.shape[1]),
                    device=rgb_wrist.device,
                )

                attack_events.append({
                    "step": int(step_i),
                    "type": "blood_corruption",
                    "camera": "both",
                    "persistent_until_reset": True,
                    "fix": {
                        "num_droplets": int(active_blood_corruption_fix["num_droplets"]),
                        "blur_sigma": float(active_blood_corruption_fix["blur_sigma"]),
                    },
                    "wrist": {
                        "num_droplets": int(active_blood_corruption_wrist["num_droplets"]),
                        "blur_sigma": float(active_blood_corruption_wrist["blur_sigma"]),
                    },
                    "note": "Independent random blood corruption states for fix and wrist views.",
                })

                pending_blood_attack = False

            if active_blood_corruption_fix is not None:
                rgb_fix = apply_blood_corruption(rgb_fix, active_blood_corruption_fix)

            if active_blood_corruption_wrist is not None:
                rgb_wrist = apply_blood_corruption(rgb_wrist, active_blood_corruption_wrist)

            # Send to policy server, receive action
            act10 = None
            act_chunk10 = None
            act_chunk_batch10 = None
            commanded_action8_for_record = last_action[0].detach().clone()
            executed_action8_for_record = last_action[0].detach().clone()

            try:
                jpg_fix = encode_jpg_from_rgb_u8(rgb_fix, quality=args.jpeg_quality)
                jpg_wrist = encode_jpg_from_rgb_u8(rgb_wrist, quality=args.jpeg_quality)
                state = get_cur_state(env, args.report_gripper)
                if args.record and int(args.policy_chunk_batch_size) > 0:
                    act10, act_chunk10, act_chunk_batch10 = policy_client.predict_action_chunk_batch(
                        jpg_fix,
                        jpg_wrist,
                        state,
                        num_samples=int(args.policy_chunk_batch_size),
                    )
                else:
                    act10, act_chunk10 = policy_client.predict_action_chunk(jpg_fix, jpg_wrist, state)
                    act_chunk_batch10 = None
                if act10.shape[0] != 10:
                    raise RuntimeError(f"Expected action_dim=10, got {act10.shape}")

                dp = act10[0:3]
                drot6d = act10[3:9]
                g = act10[9:10]

                dq_wxyz = rot6d_to_quat_wxyz(drot6d)  # (4,)

                act8 = np.concatenate([dp, dq_wxyz, g], axis=0).astype(np.float32, copy=False)

                last_action[0, :] = torch.from_numpy(act8).to(env.device, dtype=torch.float32)
                commanded_action8_for_record = last_action[0].detach().clone()

                chunk_len = infer_chunk_len(act_chunk10)

                if pending_motion_attack is not None:
                    direction = ATTACK_DIRS_FIX[pending_motion_attack]
                    delta = float(args.attack_translate_m) * direction

                    if args.attack_translation_mode == "total":
                        delta = delta / float(chunk_len)

                    active_motion_delta = torch.from_numpy(delta).to(env.device, dtype=torch.float32)
                    active_motion_steps_left = chunk_len

                    attack_events.append({
                        "step": int(step_i),
                        "type": "translation",
                        "cmd": pending_motion_attack,
                        "distance_m": float(args.attack_translate_m),
                        "mode": args.attack_translation_mode,
                        "chunk_steps": int(chunk_len),
                    })

                    pending_motion_attack = None

                if pending_gripper_attack:
                    active_gripper_steps_left = chunk_len

                    attack_events.append({
                        "step": int(step_i),
                        "type": "gripper",
                        "value": float(args.attack_gripper_value),
                        "chunk_steps": int(chunk_len),
                    })

                    pending_gripper_attack = False

                if active_motion_steps_left > 0 and active_motion_delta is not None:
                    last_action[0, 0:3] += active_motion_delta
                    active_motion_steps_left -= 1

                if active_gripper_steps_left > 0:
                    last_action[0, 7] = float(args.attack_gripper_value)
                    active_gripper_steps_left -= 1

                if action_mismatch is not None and mismatch_active_at(step_i):
                    if not mismatch_started:
                        attack_events.append({
                            "step": int(step_i),
                            "type": "action_execution_mismatch",
                            "model": "translation_gain_reversal_backlash_axis_coupling",
                            "gain": float(args.mismatch_gain),
                            "reversal_gain": float(args.mismatch_reversal_gain),
                            "reversal_steps": int(args.mismatch_reversal_steps),
                            "deadband_m": float(args.mismatch_deadband_m),
                            "coupling_xy": float(args.mismatch_coupling_xy),
                            "coupling_yx": float(args.mismatch_coupling_yx),
                            "coupling_xz": float(args.mismatch_coupling_xz),
                            "coupling_zx": float(args.mismatch_coupling_zx),
                            "coupling_yz": float(args.mismatch_coupling_yz),
                            "coupling_zy": float(args.mismatch_coupling_zy),
                            "duration_steps": int(args.mismatch_duration_steps),
                        })
                        mismatch_started = True
                    last_action = action_mismatch.apply(last_action)

                executed_action8_for_record = last_action[0].detach().clone()

            except Exception as e:
                # Fallback: hold last action (or set default). Holding is usually smoother.
                if step_i % max(1, int(args.print_every)) == 0:
                    print(f"[WARN] policy failed @step {step_i}: {e} (holding last action)")

            # Apply scene-level post motion before stepping physics.
            if scene_motion_attack is not None:
                scene_motion_attack.apply_one_step()

            # Step env
            out = env.step(last_action)
            if isinstance(out, tuple):
                obs = out[0]
            else:
                obs = out

            if not args.headless:
                simulation_app.update()

            # Record
            if args.record and recording_armed:
                rec_rgb_fix = extract_rgb_u8(obs, obs_group=obs_groups["fix"], rgb_key=rgb_keys["fix"])
                rec_rgb_wrist = extract_rgb_u8(obs, obs_group=obs_groups["wrist"], rgb_key=rgb_keys["wrist"])
                if active_blood_corruption_fix is not None:
                    rec_rgb_fix = apply_blood_corruption(rec_rgb_fix, active_blood_corruption_fix)
                if active_blood_corruption_wrist is not None:
                    rec_rgb_wrist = apply_blood_corruption(rec_rgb_wrist, active_blood_corruption_wrist)
                rec_state = get_record_state()
                # For action-execution mismatch, save the commanded action as the main
                # `actions` key so action-conditioned monitors see the mismatch. The
                # actual env-stepped action is also saved as `executed_actions`.
                record_action8 = commanded_action8_for_record if args.action_execution_mismatch else last_action[0]
                recorder.push(
                    rgb_fix=rec_rgb_fix,
                    rgb_wrist=rec_rgb_wrist,
                    action=record_action8,
                    executed_action=executed_action8_for_record,
                    policy_action=act10,                            # raw policy act10 before rot6d->quat conversion
                    policy_action_chunk=act_chunk10,                # full raw chunk [K,10]
                    policy_action_chunk_batch=act_chunk_batch10,    # full raw chunk batch [B,K,10]
                    state=rec_state,
                    t_wall=time.time(),
                )

            step_i += 1
            if step_i % max(1, int(args.print_every)) == 0:
                dt = time.time() - t0
                fps = step_i / max(1e-9, dt)
                print(f"[INFO] step={step_i} approx_fps={fps:.2f}")

    finally:
        print("\n[INFO] Shutting down eval client...")
        try:
            policy_client.close()
            control_client.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass
        print("[INFO] Done.")

if __name__ == "__main__":
    main()