from __future__ import annotations
from typing import Sequence

import torch
import random

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    TerminationTermCfg,
    EventTermCfg,
    SceneEntityCfg,
)

import isaaclab.envs.mdp as mdp

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

from fomofd.utils.quat_util import *
from fomofd.tasks.needle_pickup.environment.scene_cfg import (
    dvrkBlockNeedleSceneCfg as NeedlePickupSceneCfg,
)

# -----------------------------------------------------------------------------
# Observation term function: RGB camera observation
# ObservationTermCfg.func is a callable taking env first, then params.
# -----------------------------------------------------------------------------

def obs_camera(env, sensor_name: str = "fix_camera", data_key: str = "rgb"):
    cam = env.scene[sensor_name]
    img = cam.data.output[data_key][0]  # often [H,W,C] for a single env
    # rgb handling
    if data_key == "rgb":
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype.is_floating_point:
            img = (img.clamp(0, 1) * 255).to(torch.uint8)
        else:
            img = img.to(torch.uint8)
        return img
    # depth handling
    elif data_key == "depth":
        # keep as float32 (meters); squeeze channel if present
        img = img.to(torch.float32)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        return img
    else:
        raise ValueError(f"Unsupported data_key: {data_key}")

class PSMHybridRelativeEEAction(ActionTerm):
    """
    7D action:
      dp_fix (3): translation increment expressed in fix_camera frame
      dq_ee  (4): local EE-tip-frame rotation increment (wxyz)

    Convention:
      p_cmd_next = p_cmd + R_fix @ dp_fix
      q_cmd_next = q_cmd ⊗ dq_ee
    """

    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self._robot = env.scene[cfg.asset_name]

        self._fix_cam = env.scene[cfg.fix_camera_name]

        self._robot_entity_cfg = SceneEntityCfg(
            cfg.asset_name,
            joint_names=list(cfg.arm_joint_names),
            body_names=[cfg.ee_body_name],
        )
        self._robot_entity_cfg.resolve(env.scene)

        # Jacobian index shift
        if self._robot.is_fixed_base:
            self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0] - 1
        else:
            self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0]

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=cfg.ik_method,
        )
        self._ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

        self._raw = torch.zeros(env.num_envs, 7, device=env.device)
        self._dp_fix = torch.zeros(env.num_envs, 3, device=env.device)
        self._dq_ee = torch.zeros(env.num_envs, 4, device=env.device)

        self._dp_scale = float(getattr(cfg, "delta_pos_scale", 1.0))
        
        self._cmd_initialized = False

        self._p_cmd_w = torch.zeros(env.num_envs, 3, device=env.device)
        self._q_cmd_w = torch.zeros(env.num_envs, 4, device=env.device); self._q_cmd_w[:, 0] = 1.0

        self._p_fix0_w = torch.zeros(env.num_envs, 3, device=env.device)
        self._q_fix0_w = torch.zeros(env.num_envs, 4, device=env.device); self._q_fix0_w[:, 0] = 1.0

        self._base_ee_goal_b = torch.tensor(cfg.default_ee_goal, device=env.device, dtype=torch.float32)
        self._current_ee_goal_b = self._base_ee_goal_b.clone().repeat(env.num_envs, 1)
        self._ee_pos_noise_range = getattr(cfg, "ee_pos_noise_range", (0.0, 0.0))

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw  # if you don’t do extra processing
    
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._cmd_initialized = False
        self._raw.zero_()
        self._dp_fix.zero_()
        self._dq_ee.zero_()
        self._dq_ee[:, 0] = 1.0

        self._p_cmd_w.zero_()
        self._q_cmd_w.zero_()
        self._q_cmd_w[:, 0] = 1.0

        if env_ids is None:
            env_ids = slice(None)
            num_reset = self.num_envs
        else:
            num_reset = len(env_ids)

        noise = torch.zeros((num_reset, 3), device=self.device)
        noise.uniform_(self._ee_pos_noise_range[0], self._ee_pos_noise_range[1])
        self._current_ee_goal_b[env_ids, :3] = self._base_ee_goal_b[:3] + noise

    def process_actions(self, actions: torch.Tensor) -> None:
        # --- Store raw actions ---
        self._raw[:] = actions
        self._dp_fix[:] = actions[:, 0:3] * self._dp_scale
        self._dq_ee[:] = quat_canon(quat_norm(actions[:, 3:7]))

        # --- Latch fixed/endoscope camera frame at action arrival ---
        p_fix_w = self._fix_cam.data.pos_w
        q_fix_w = quat_canon(quat_norm(self._fix_cam.data.quat_w_world))

        self._p_fix0_w[:] = p_fix_w
        self._q_fix0_w[:] = q_fix_w

        # --- Apply dp in fixed/endoscope camera frame ---
        # p_cmd_next = p_cmd + R_fix @ dp_fix
        p_cmd_fix = quat_rotate(
            quat_inv(self._q_fix0_w),
            self._p_cmd_w - self._p_fix0_w,
        )
        p_cmd_fix = p_cmd_fix + self._dp_fix
        self._p_cmd_w[:] = self._p_fix0_w + quat_rotate(self._q_fix0_w, p_cmd_fix)

        # --- Apply dq in local EE-tip frame ---
        # q_cmd_next = q_cmd ⊗ dq_ee
        self._q_cmd_w[:] = quat_mul(self._q_cmd_w, self._dq_ee)
        self._q_cmd_w[:] = quat_canon(quat_norm(self._q_cmd_w))

    def apply_actions(self) -> None:
        # --- Read current EE pose in world ---
        ee_pose_w = self._robot.data.body_state_w[:, self._robot_entity_cfg.body_ids[0], 0:7]
        p_ee_w = ee_pose_w[:, 0:3]
        q_ee_w = quat_canon(quat_norm(ee_pose_w[:, 3:7]))

        # --- Read current robot base pose in world ---
        root_pose_w = self._robot.data.root_state_w[:, 0:7]
        p_root_w = root_pose_w[:, 0:3]
        q_root_w = quat_canon(quat_norm(root_pose_w[:, 3:7]))

        if not self._cmd_initialized:
            p_goal_b = self._current_ee_goal_b[:, :3]
            q_goal_b = self._current_ee_goal_b[:, 3:7]

            self._p_cmd_w[:] = p_root_w + quat_rotate(q_root_w, p_goal_b)
            self._q_cmd_w[:] = quat_canon(quat_norm(quat_mul(q_root_w, q_goal_b)))
            
            self._cmd_initialized = True

        # frozen target expressed in root frame
        p_cmd_b = quat_rotate(quat_inv(q_root_w), (self._p_cmd_w - p_root_w))
        q_cmd_b = quat_mul(quat_inv(q_root_w), self._q_cmd_w)
        q_cmd_b = quat_canon(quat_norm(q_cmd_b))

        # --- IK step ---
        jacobians = self._robot.root_physx_view.get_jacobians()
        jacobian = jacobians[:, self._ee_jacobi_idx, :, [0, 2, 4, 5, 6, 7]]  # 6 cols

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            p_ee_w, q_ee_w
        )

        joint_pos = self._robot.data.joint_pos[:, self._robot_entity_cfg.joint_ids]
        self._ik.set_command(torch.cat([p_cmd_b, q_cmd_b], dim=-1))
        joint_des = self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        self._robot.set_joint_position_target(joint_des, joint_ids=self._robot_entity_cfg.joint_ids)

class PSMGripperAction(ActionTerm):
    """1D action: [gripper] -> jaw joint target."""

    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self._asset_name: str = cfg.asset_name
        self._robot = env.scene[self._asset_name]

        self._gripper_entity_cfg = SceneEntityCfg(
            self._asset_name,
            joint_names=[cfg.gripper_joint_name],
            body_names=[],
        )
        self._gripper_entity_cfg.resolve(env.scene)

        self._raw = torch.zeros(env.num_envs, 1, device=env.device)
        self._processed = torch.zeros_like(self._raw)

        self._default_gripper = float(cfg.default_gripper)
        self._gpr_target = torch.full((env.num_envs, 1), self._default_gripper, device=env.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._raw.zero_()
            self._processed.zero_()
            self._gpr_target[:] = self._default_gripper
        else:
            env_ids = list(env_ids)
            self._raw[env_ids].zero_()
            self._processed[env_ids].zero_()
            self._gpr_target[env_ids] = self._default_gripper

    def process_actions(self, actions: torch.Tensor) -> None:
        # actions: [num_envs, 1]
        self._raw[:] = actions
        self._processed[:] = actions
        self._gpr_target[:] = self._processed

    def apply_actions(self) -> None:
        self._robot.set_joint_position_target(self._gpr_target, joint_ids=self._gripper_entity_cfg.joint_ids)

# -----------------------------------------------------------------------------
# Config containers
# -----------------------------------------------------------------------------

@configclass
class PSMHybridRelativeEEActionCfg(ActionTermCfg):
    arm_joint_names: list[str] = ()
    ee_body_name: str = ""
    ik_method: str = "dls"
    default_ee_goal: list[float] = (0.89253896, 0.04946792, -0.07037026, -0.4092562, 0.61447084, 0.6591015, -0.1432485)  # 7D
    ee_pos_noise_range: tuple[float, float] = (-0.002, 0.002)
    fix_camera_name: str = "fix_camera"
    delta_pos_scale: float = 1.0

@configclass
class PSMGripperActionCfg(ActionTermCfg):
    gripper_joint_name: str = ""
    default_gripper: float = 0.5

@configclass
class NeedlePickupActionsCfg:
    """Action terms (configured; ActionManager will instantiate them)."""

    psm_ee: PSMHybridRelativeEEActionCfg = PSMHybridRelativeEEActionCfg(
        class_type=PSMHybridRelativeEEAction,
        asset_name="psm_si",
        arm_joint_names=[
            "j1",
            "j2",
            "outer_insertion",
            "tool_roll_joint",
            "tool_pitch_joint",
            "tool_yaw_joint",
        ],
        ee_body_name="tool_yaw_link",
        ik_method="dls",
        fix_camera_name="fix_camera",
    )

    psm_gripper: PSMGripperActionCfg = PSMGripperActionCfg(
        class_type=PSMGripperAction,
        asset_name="psm_si",
        gripper_joint_name="jaw_joint_1",
        default_gripper=0.5,
    )

@configclass
class NeedlePickupObservationsCfg:
    """Observation groups."""

    @configclass
    class FixCameraCfg(ObservationGroupCfg):
        """Observations from the fix camera."""

        rgb: ObservationTermCfg = ObservationTermCfg(
            func=obs_camera,
            params={"sensor_name": "fix_camera", "data_key": "rgb"},
        )
        depth: ObservationTermCfg = ObservationTermCfg(
            func=obs_camera,
            params={"sensor_name": "fix_camera", "data_key": "depth"},
        )

        def __post_init__(self):
            # Keep dict output
            self.concatenate_terms = False
            self.enable_corruption = False

    @configclass
    class WristCameraCfg(ObservationGroupCfg):
        """Observations from the wrist camera."""

        rgb: ObservationTermCfg = ObservationTermCfg(
            func=obs_camera,
            params={"sensor_name": "wrist_camera", "data_key": "rgb"},
        )
        depth: ObservationTermCfg = ObservationTermCfg(
            func=obs_camera,
            params={"sensor_name": "wrist_camera", "data_key": "depth"},
        )

        def __post_init__(self):
            # Keep dict output
            self.concatenate_terms = False
            self.enable_corruption = False

    fix_camera: FixCameraCfg = FixCameraCfg()
    wrist_camera: WristCameraCfg = WristCameraCfg()

@configclass
class NeedlePickupTerminationsCfg:
    """Termination terms.
    """

    # Uses built-in "time_out" term from mdp in typical IsaacLab tasks, but to avoid version mismatch,
    # implement a simple time limit here based on env.episode_length_buf if needed. 
    time_out: TerminationTermCfg = TerminationTermCfg(
        func=lambda env, max_steps: (env.episode_length_buf >= max_steps),
        params={"max_steps": 10 * 30},  # 10 seconds at 30 Hz env-step
    )

def disable_robot_organ_collisions(env, env_ids, robot_prim_name: str, organ_prim_name: str):
    """Natively disables physics collisions by targeting exact collision meshes."""
    import omni.usd                     # type: ignore
    from pxr import UsdPhysics, Usd     # type: ignore

    stage = omni.usd.get_context().get_stage()
    if not stage:
        return
        
    if env_ids is None:
        env_id_list = range(env.num_envs)
    else:
        env_id_list = env_ids.tolist()
        
    # Helper function to recursively find all actual collision meshes
    def get_collision_prims(root_prim):
        prims = []
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prims.append(prim)
        return prims

    for env_id in env_id_list:
        env_path = f"{env.scene.env_ns}/env_{env_id}"
        
        robot_prim = stage.GetPrimAtPath(f"{env_path}/{robot_prim_name}")
        organ_prim = stage.GetPrimAtPath(f"{env_path}/{organ_prim_name}")
        
        if not robot_prim.IsValid() or not organ_prim.IsValid():
            continue
            
        # Extract the exact physical meshes from inside the parent folders
        robot_col_prims = get_collision_prims(robot_prim)
        organ_col_prims = get_collision_prims(organ_prim)
        
        # Cross-link every organ mesh to ignore every robot mesh
        for organ_mesh in organ_col_prims:
            if not organ_mesh.HasAPI(UsdPhysics.FilteredPairsAPI):
                filter_api = UsdPhysics.FilteredPairsAPI.Apply(organ_mesh)
            else:
                filter_api = UsdPhysics.FilteredPairsAPI(organ_mesh)
                
            for robot_mesh in robot_col_prims:
                filter_api.GetFilteredPairsRel().AddTarget(robot_mesh.GetPath())

def randomize_light_intensity(env, env_ids, prim_name: str, intensity_range: tuple):
    """Randomizes light intensity natively using OpenUSD API."""
    import omni.usd  # type: ignore
    
    stage = omni.usd.get_context().get_stage()
    if not stage:
        return
        
    for env_id in env_ids.tolist():
        env_path = f"{env.scene.env_ns}/env_{env_id}"
        light_path = f"{env_path}/{prim_name}"
        
        prim = stage.GetPrimAtPath(light_path)
        if prim.IsValid():
            intensity_attr = prim.GetAttribute("inputs:intensity")
            if intensity_attr.IsValid():
                new_intensity = random.uniform(intensity_range[0], intensity_range[1])
                intensity_attr.Set(new_intensity)

@configclass
class NeedlePickupEventCfg:
    """Configuration for reset events (Domain Randomization)."""

    reset_needle_pose = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",                       # Triggers during env.reset()
        params={
            "pose_range": {
                "x": (-0.005, 0.005),       # +/- 5mm shift along the X axis
                "y": (-0.01, 0.0),          # +/- 5mm shift along the Y axis
                "z": (-0.01, -0.01),        # Keep Z unchanged so it stays on the surface
                "roll": (0.0, 0.0),         # Keep it flat
                "pitch": (0.0, 0.0),        # Keep it flat
                "yaw": (-0.15, 0.3736)      # +/- 15 degree random rotation
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            },
            "asset_cfg": SceneEntityCfg("needle"), 
        },
    )

    reset_endoscope_light = EventTermCfg(
        func=randomize_light_intensity,
        mode="reset",
        params={
            "prim_name": "EndoscopeLight",             
            # Randomize the base 250000 intensity by +/- 20%
            "intensity_range": (200000.0, 300000.0),   
        },
    )

    filter_collisions = EventTermCfg(
        func=disable_robot_organ_collisions,
        mode="startup", 
        params={
            "robot_prim_name": "Robot",
            "organ_prim_name": "SurgicalBackground",
        }
    )

@configclass
class NeedlePickupEnvCfg(ManagerBasedEnvCfg):
    """Manager-based environment configuration."""

    # Simulation parameters
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 240,
        render_interval=8,  # 30 Hz render (240/8)
        device="cuda:0",
    )

    # Scene
    scene: NeedlePickupSceneCfg = NeedlePickupSceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )

    # Managers (config containers)
    actions: NeedlePickupActionsCfg = NeedlePickupActionsCfg()
    observations: NeedlePickupObservationsCfg = NeedlePickupObservationsCfg()
    terminations: NeedlePickupTerminationsCfg = NeedlePickupTerminationsCfg()
    events: NeedlePickupEventCfg = NeedlePickupEventCfg()

    # Make env.step() run at 30 Hz:
    # 240 Hz physics / 8 = 30 Hz environment step
    decimation: int = 240 // 30

    # Episode length (seconds). Mirrored into termination max_steps in __post_init__.
    episode_length_s: float = 300.0  # 5 minutes

    # Scene-level needle motion attack, used by fomofd/eval.py only.
    # Triggered by pressing "m" in eval_server.py.
    scene_motion_attack_asset_name: str = "needle"
    scene_motion_attack_type: str = "needle_motion_velocity"
    # Horizontal velocity
    scene_motion_attack_speed_mps: float = 0.08
    # World XY direction. Z velocity is preserved by physics.
    scene_motion_attack_direction_xy: tuple[float, float] = (1.0, -0.5)
    # Number of env steps after pressing "m".
    scene_motion_attack_duration_steps: int = 8
    # Stop commanded XY velocity after attack duration.
    scene_motion_attack_zero_xy_velocity_on_stop: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Derive episode length steps for termination (env steps, not sim steps)
        self.scene.replicate_physics = False
        max_steps = int(self.episode_length_s * (240 // self.decimation))
        self.terminations.time_out.params["max_steps"] = max_steps

    def get_default_teleop_state(self) -> list[list[float]]:
        """
        Fetches the default end-effector pose and gripper state from the action configuration.
        Returns format: [[x, y, z, qw, qx, qy, qz, gripper]]
        """
        ee_pose = list(self.actions.psm_ee.default_ee_goal)
        gripper_val = float(self.actions.psm_gripper.default_gripper)
        return [ee_pose + [gripper_val]]
