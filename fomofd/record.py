"""
fomofd/record.py

Teleoperation runner using ManagerBasedEnv + NeedlePickupEnvCfg.

- Receives ROS UDP commands at ~100 Hz (non-blocking).
- Steps Isaac Lab env at 30 Hz (env.decimation=8, sim dt=1/240).
- Applies action: [x,y,z,qw,qx,qy,qz, gripper] (Isaac quat is wxyz).
- Records 30 Hz transitions (RGB + action) into per-episode NPZ files.

Run:
  python fomofd/record.py --task needle_pickup --enable_camera --device cuda:0
"""


from __future__ import annotations

import argparse

import time
from pathlib import Path
from typing import Any, Dict
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from isaaclab.app import AppLauncher
from fomofd.utils.quat_util import *
from fomofd.utils.EpisodeRecorder import EpisodeRecorder
from fomofd.utils.UdpTeleop import UdpTeleop
from fomofd.utils.HybridRelativeActionConverter import HybridRelativeActionConverter

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
        "obs_groups": {
            "fix": "fix_camera",
            "wrist": "wrist_camera",
        },
        "rgb_keys": {
            "fix": "rgb",
            "wrist": "rgb",
        },
        "action_dim": 8,
        "default_out_dir": str(THIS_DIR / "tasks" / "needle_pickup" / "data" / "raw"),
    },
    "ring_over_post": {
        "cfg_loader": _import_ring_over_post_cfg,
        "obs_groups": {
            "fix": "fix_camera",
            "wrist": "wrist_camera",
        },
        "rgb_keys": {
            "fix": "rgb",
            "wrist": "rgb",
        },
        "action_dim": 8,
        "default_out_dir": str(THIS_DIR / "tasks" / "ring_over_post" / "data" / "raw"),
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

def sync_psm_ee_action_to_current_pose(env, robot, ee_body_id):
    from isaaclab.utils.math import subtract_frame_transforms

    ee_term = env.action_manager._terms["psm_ee"]   # adapt if needed for your IsaacLab version

    ee_pose_w = robot.data.body_state_w[:, ee_body_id, 0:7]
    root_pose_w = robot.data.root_state_w[:, 0:7]

    p_ee_w = ee_pose_w[:, 0:3]
    q_ee_w = quat_canon(quat_norm(ee_pose_w[:, 3:7]))

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        p_ee_w, q_ee_w,
    )
    ee_quat_b = quat_canon(quat_norm(ee_quat_b))

    goal_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    ee_term._base_ee_goal_b = goal_b[0].clone()
    ee_term._current_ee_goal_b[:] = goal_b

    ee_term._p_cmd_w[:] = p_ee_w
    ee_term._q_cmd_w[:] = q_ee_w

    ee_term._raw.zero_()
    ee_term._dp_fix.zero_()
    ee_term._dq_ee.zero_()
    ee_term._dq_ee[:, 0] = 1.0

    ee_term._cmd_initialized = True

def warmup_env(env, action: torch.Tensor, steps: int = 5) -> dict:
    obs = None
    for _ in range(steps):
        obs, *_ = env.step(action)
    return obs

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Record teleoperation episodes for Isaac Lab tasks.")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASK_SPECS.keys()))
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--udp_listen_ip", type=str, default="0.0.0.0")
    parser.add_argument("--udp_listen_port", type=int, default=15151)
    parser.add_argument("--udp_report_ip", type=str, default="127.0.0.1")
    parser.add_argument("--udp_report_port", type=int, default=15150)
    parser.add_argument("--max_seconds", type=float, default=10.0)
    parser.add_argument("--warmup_steps", type=int, default=5)

    # Isaac Lab args
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    extra = " --/log/channels/omni.usd=disable --/log/channels/carb=disable"
    args.kit_args = (getattr(args, "kit_args", "") or "") + extra

    spec = TASK_SPECS[args.task]
    out_dir = Path(args.out_dir or spec["default_out_dir"])

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

    teleop = UdpTeleop(
        args.udp_listen_ip, args.udp_listen_port,
        state_dst_ip=args.udp_report_ip,
        state_dst_port=args.udp_report_port,
        default_EE_pose=cfg.get_default_teleop_state()
    )

    obs_groups = spec["obs_groups"]
    rgb_keys = spec["rgb_keys"]

    # Initialize hybrid relative action converter
    converter = HybridRelativeActionConverter(device=env.device)
    converter.reset()

    # Reset env
    obs, _ = env.reset()
    robot = env.scene["psm_si"]  # Articulation

    # Adjust Viewports for Fix and Wrist cameras
    set_viewports_to_cameras()

    # Fetch robot end-effector body ID
    robot_entity_cfg = SceneEntityCfg(
        "psm_si",
        body_names=["tool_yaw_link"],
    )
    robot_entity_cfg.resolve(env.scene)
    ee_body_id = robot_entity_cfg.body_ids[0]

    # Fetch gripper jaw joint ID
    jaw_entity_cfg = SceneEntityCfg(
        "psm_si",
        joint_names=[cfg.actions.psm_gripper.gripper_joint_name],  # "jaw_joint_1"
    )
    jaw_entity_cfg.resolve(env.scene)
    jaw_joint_id = jaw_entity_cfg.joint_ids[0]

    # Initialize recorder
    recorder = EpisodeRecorder(out_dir=out_dir)
    recorder.reset()

    def get_default_action():
        # Relative EE translation: zero (no motion)
        delta_p = torch.zeros((env.num_envs, 3),
                            device=env.device,
                            dtype=torch.float32)

        # Relative EE rotation: identity quaternion (wxyz)
        delta_q = torch.zeros((env.num_envs, 4),
                            device=env.device,
                            dtype=torch.float32)
        delta_q[:, 0] = 1.0  # w = 1

        # Gripper default
        gr_cfg = env.cfg.actions.psm_gripper
        default_gripper = torch.full(
            (env.num_envs, 1),
            gr_cfg.default_gripper,
            device=env.device,
            dtype=torch.float32,
        )

        # Final action
        return torch.cat([delta_p, delta_q, default_gripper], dim=-1)

    last_action = get_default_action()

    # Warmup for camera buffers
    warmup_env(env, last_action, steps=args.warmup_steps)

    print(f"[INFO] Recording task={args.task} -> {out_dir}")
    print("[INFO] Waiting for UDP commands...")

    while simulation_app.is_running():
        # Update latest teleop cmd
        if teleop.poll_latest() and not teleop.msg_received:
            teleop.msg_received = True
            if converter.wait_mode:
                try:
                    converter.update_previous_pose(env, ee_pos_b, ee_quat_b)
                except:
                    print("[ERROR] Failed to initialize hybrid action converter history.")
                    return

        # Handle recording command
        record_cmd = teleop.consume_record_cmd()
        if record_cmd == 1:     # Start new recording
            recorder.start()
            print("[REC] Started new recording...")
        elif record_cmd == 2:   # Stop and save recording
            if len(recorder.images_fix) > 0:
                recorder.save(meta={"task": args.task, "end_reason": "stop_command"})
                print(f"[REC] Stopped and saved current recording: index {recorder.episode_idx - 1}")
            else:
                print("[REC] No data recorded, nothing to save.")
        elif record_cmd == 3:   # Discard current recording
            recorder.reset()
            print("[REC] Discarded current recording.")
        elif record_cmd == 4:   # Pause recording
            recorder.pause()
            print("[REC] Paused current recording.")
        elif record_cmd == 5:   # Resume recording
            if not recorder.is_recording:
                recorder.start()
                print("[REC] Resumed recording.")
        else:                   # No recording command
            pass

        # Handle external reset -> finalize episode
        if teleop.consume_reset_flag():
            teleop.reset()
            converter.reset()
            obs, _ = env.reset()
            # sync_psm_ee_action_to_current_pose(env, robot, ee_body_id)

            last_action = get_default_action()
            warmup_env(env, last_action, steps=args.warmup_steps)

            ee_pose_w = robot.data.body_state_w[0, ee_body_id, 0:7]
            root_pose_w = robot.data.root_state_w[0, 0:7]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[0:3].unsqueeze(0), root_pose_w[3:7].unsqueeze(0),
                ee_pose_w[0:3].unsqueeze(0),  ee_pose_w[3:7].unsqueeze(0),
            )
            teleop.maybe_send_state_from_isaac(ee_pos_b, ee_quat_b, force=True)
            continue

        # Update last_action if we have a cmd
        absolute_pose = teleop.make_action_wxyz(env.device)
        if absolute_pose is not None:
            p = absolute_pose[0, 0:3]
            q = absolute_pose[0, 3:7]
            gr = absolute_pose[0, 7:8]

            dp_fix, dq_ee = converter.convert(env, p, q)
            last_action[:] = torch.cat(
                [dp_fix.view(1,3), dq_ee.view(1,4), gr.view(1,1)],
                dim=-1
            )

        # Step env (rate determined by cfg.decimation and cfg.sim.dt)
        obs, _ = env.step(last_action)
        # Read current EE pose in world
        ee_pose_w = robot.data.body_state_w[0, ee_body_id, 0:7]     # [pos(3), quat(4)] in world
        root_pose_w = robot.data.root_state_w[0, 0:7]               # robot base in world
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[0:3].unsqueeze(0), root_pose_w[3:7].unsqueeze(0),
            ee_pose_w[0:3].unsqueeze(0),  ee_pose_w[3:7].unsqueeze(0),
        )
        teleop.maybe_send_state_from_isaac(ee_pos_b, ee_quat_b)

        if not args.headless:
            simulation_app.update()

        # Record at env step frequency
        if recorder.is_recording:
            rgb_fix = extract_rgb_u8(obs, obs_group=obs_groups["fix"], rgb_key=rgb_keys["fix"])
            rgb_wrist = extract_rgb_u8(obs, obs_group=obs_groups["wrist"], rgb_key=rgb_keys["wrist"])

            p_ee_w = ee_pose_w[0:3]
            q_ee_w = quat_canon(quat_norm(ee_pose_w[3:7]))                      # wxyz
            fix_cam = env.scene["fix_camera"]
            p_fix_w = fix_cam.data.pos_w[0]
            q_fix_w = quat_canon(quat_norm(fix_cam.data.quat_w_world[0]))       # wxyz
            p_ee_fix = quat_rotate(quat_inv(q_fix_w), p_ee_w - p_fix_w)
            q_ee_fix = quat_mul(quat_inv(q_fix_w), q_ee_w)
            q_ee_fix = quat_canon(quat_norm(q_ee_fix))
            jaw = robot.data.joint_pos[0, jaw_joint_id].to(dtype=torch.float32) # scalar
            state_fix = torch.cat([p_ee_fix, q_ee_fix, jaw.view(1)], dim=0)     # (8,)

            recorder.push(rgb_fix=rgb_fix, rgb_wrist=rgb_wrist, action=last_action[0], state=state_fix, t_wall=time.time())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()