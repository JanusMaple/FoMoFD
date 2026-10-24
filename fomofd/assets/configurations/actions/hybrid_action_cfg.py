from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors.camera import TiledCamera

import torch

class HybridActionTerm(ActionTerm):
    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._camera: TiledCamera = self._env.scene[self.cfg.camera_name]

        self._ee_body_id = 8
        self._wrist_body_id = 6
        self._jacobian_joint_ids = [0, 2, 4, 5, 6, 7, 8]
        self._command_joint_ids = [0, 1, 4, 5, 6, 7, 8]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        self._ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
        self._ik_controller = DifferentialIKController(self._ik_cfg, num_envs=self.num_envs, device=self.device)
        self._ik_commands = torch.zeros(self.num_envs, self._ik_controller.action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._ee_body_id - 1, :, self._jacobian_joint_ids]
    
    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian
    
    def process_actions(self, actions: torch.Tensor):
        """
        Process raw actions per environment step. 

        Args:
            actions: shape (num_envs, 6): [Relative_Displacement Relative_Rotation]
        """
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        # Compute relative position command
        camera_quat_w = self._camera.data.quat_w_world
        camera_rotmat_w = math_utils.matrix_from_quat(camera_quat_w)
        root_quat_w = self._asset.data.root_state_w[:, 3:7]
        root_rotmat_w = math_utils.matrix_from_quat(root_quat_w)
        rel_pos_w = torch.bmm(camera_rotmat_w, self._processed_actions[:, 0:3].unsqueeze(-1))       # (num_envs, 3, 1)
        self._ik_commands[:, 0:3] = torch.bmm(root_rotmat_w.transpose(1, 2), rel_pos_w).squeeze(-1) # (num_envs, 3)
        # Compute relative rotation command
        wrist_quat_w = self._asset.data.body_state_w[:, self._wrist_body_id, 3:7]                   # (num_envs, 4)
        wrist_rotmat_w = math_utils.matrix_from_quat(wrist_quat_w)
        root2wrist_rotmat = torch.bmm(root_quat_w.transpose(1, 2), wrist_rotmat_w)
        dq_wrist = math_utils.quat_from_euler_xyz(self._processed_actions[:, 3:6])
        dr_wrist = math_utils.matrix_from_quat(dq_wrist)
        dr_root = torch.bmm(root2wrist_rotmat, dr_wrist)
        dq_root = math_utils.quat_from_matrix(dr_root)
        self._ik_commands[:, 3:6] = math_utils.euler_xyz_from_quat(dq_root)
        # Set IK command for the controller
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._ik_commands, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        """
        Apply actions to the robot per simulation step. 
        """
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_ett_pos = self._asset.data.joint_pos[:, self._command_joint_ids]
        joint_des_pos = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, self.jacobian_b, joint_ett_pos)
        self._asset.set_joint_position_target(joint_des_pos, self._command_joint_ids)

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame. 

        Returns:
            A tuple of the EE's position and orientation in the root frame. 
        """
        # Retrieve data from current scene
        ee_pose_w = self._asset.data.body_state_w[:, self._ee_body_id, 0:7]
        root_pose_w = self._asset.data.root_state_w[:, 0:7]
        # Compute current current EE pose in base frame
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
        return ee_pos_b, ee_quat_b

@configclass
class HybridActionTermCfg(ActionTermCfg):
    class_type = HybridActionTerm
    asset_name: str = "psm_si"
    camera_name: str = "fix_camera"
    action_dim: int = 6
    sacle: float = 1.0
    debug_vis = True
