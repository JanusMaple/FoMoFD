# ----------------------------
# Hybrid Relative Action Converter
# ----------------------------

import torch
from fomofd.utils.quat_util import *


class HybridRelativeActionConverter:
    """
    SRT-style hybrid relative action converter.

    Outputs:
      dp_fix:  (3,) translation delta expressed in the fixed/endoscope camera frame at t
      dq_ee:   (4,) local EE-tip-frame rotation delta, wxyz

    Convention:
      p_next = p_prev + R_fix_t @ dp_fix
      q_next = q_prev ⊗ dq_ee
    """

    def __init__(self, device):
        self.device = device
        self.wait_mode = True

        self.prev_p_ee_w = None
        self.prev_q_ee_w = None
        self.prev_p_fix_w = None
        self.prev_q_fix_w = None

    def reset(self):
        self.wait_mode = True
        self.prev_p_ee_w = None
        self.prev_q_ee_w = None
        self.prev_p_fix_w = None
        self.prev_q_fix_w = None

    def _ee_pose_base_to_world(self, env, p_ee_b, q_ee_b):
        robot = env.scene["psm_si"]

        root_pose_w = robot.data.root_state_w[0, 0:7]
        p_root_w = root_pose_w[0:3]
        q_root_w = quat_canon(quat_norm(root_pose_w[3:7]))  # wxyz

        p_ee_b = p_ee_b.view(3)
        q_ee_b = quat_canon(quat_norm(q_ee_b.view(4)))

        p_ee_w = p_root_w + quat_rotate(q_root_w, p_ee_b)
        q_ee_w = quat_mul(q_root_w, q_ee_b)
        q_ee_w = quat_canon(quat_norm(q_ee_w))

        return p_ee_w, q_ee_w

    def update_previous_pose(self, env, p_ee_b, q_ee_b):
        fix_cam = env.scene["fix_camera"]

        self.prev_p_fix_w = fix_cam.data.pos_w[0].clone()
        self.prev_q_fix_w = quat_canon(
            quat_norm(fix_cam.data.quat_w_world[0])
        ).clone()

        p_ee_w, q_ee_w = self._ee_pose_base_to_world(env, p_ee_b, q_ee_b)

        self.prev_p_ee_w = p_ee_w.clone()
        self.prev_q_ee_w = q_ee_w.clone()

        self.wait_mode = False

    def convert(self, env, p_ee_b, q_ee_b):
        """
        Inputs:
          p_ee_b: (3,) EE position in robot base/root frame
          q_ee_b: (4,) EE orientation in robot base/root frame, wxyz

        Outputs:
          dp_fix: (3,) delta position in fixed/endoscope camera frame at t
          dq_ee:  (4,) local EE-tip-frame delta rotation, wxyz
        """
        fix_cam = env.scene["fix_camera"]

        p_fix_w = fix_cam.data.pos_w[0]
        q_fix_w = quat_canon(quat_norm(fix_cam.data.quat_w_world[0]))

        p_ee_w, q_ee_w = self._ee_pose_base_to_world(env, p_ee_b, q_ee_b)

        # First step: zero action, initialize history
        if self.prev_p_ee_w is None:
            self.prev_p_ee_w = p_ee_w.clone()
            self.prev_q_ee_w = q_ee_w.clone()
            self.prev_p_fix_w = p_fix_w.clone()
            self.prev_q_fix_w = q_fix_w.clone()

            dp_fix = torch.zeros(3, device=self.device)
            dq_ee = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            return dp_fix, dq_ee

        # --- Δp in fixed/endoscope camera frame at timestep t ---
        # Use previous fix camera frame as the action frame.
        dp_w = p_ee_w - self.prev_p_ee_w
        dp_fix = quat_rotate(quat_inv(self.prev_q_fix_w), dp_w)

        # --- Δq in previous EE-tip frame ---
        # q_curr = q_prev ⊗ dq_ee
        dq_ee = quat_mul(quat_inv(self.prev_q_ee_w), q_ee_w)
        dq_ee = quat_canon(quat_norm(dq_ee))

        if not self.wait_mode:
            self.prev_p_ee_w = p_ee_w.clone()
            self.prev_q_ee_w = q_ee_w.clone()
            self.prev_p_fix_w = p_fix_w.clone()
            self.prev_q_fix_w = q_fix_w.clone()

        return dp_fix, dq_ee