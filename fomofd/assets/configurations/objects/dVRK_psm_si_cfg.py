"""
    This script is a configuration file for a PSM Si robot arm
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

DVRK_PSM_SI_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir, os.pardir)}/models/dvrk_psm_si.usd",
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=200.0,
            max_angular_velocity=200.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link = True
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.0001,
            rest_offset=0.0,
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2), rot=(1, 0, 0, 0),
        joint_pos={"j1": 0.0, "j2": 0.0, "j3": 0.0, "j4": 0.0, "outer_insertion": 0.1, "tool_roll_joint": 0.0,
                   "tool_pitch_joint": 0.0, "tool_yaw_joint": 0.0, "jaw_joint_1": 0.0, "jaw_joint_2": 0.0}
    ),
    debug_vis=True,
    actuators={
        "j1": ImplicitActuatorCfg(
            joint_names_expr = "j1",
            stiffness = 2000.0,
            damping = 100.0,
            effort_limit = 25.0,
            velocity_limit=1.0
        ),
        "j2": ImplicitActuatorCfg(
            joint_names_expr = "j2",
            stiffness = 2000.0,
            damping = 100.0,
            effort_limit = 25.0,
            velocity_limit=1.0
        ),
        # Mimic Joint
        "j3": ImplicitActuatorCfg(
            joint_names_expr = "j3",
            stiffness = 0,
            damping = 0,
            effort_limit = 0
        ),
        # Mimic Joint
        "j4": ImplicitActuatorCfg(
            joint_names_expr = "j4",
            stiffness = 0,
            damping = 0,
            effort_limit = 0
        ),
        "outer_insertion": ImplicitActuatorCfg(
            joint_names_expr = "outer_insertion",
            stiffness = 60000.0,
            damping = 800.0,
            effort_limit = 120.0,
            velocity_limit=1.0
        ),
        "tool_roll_joint": ImplicitActuatorCfg(
            joint_names_expr = "tool_roll_joint",
            stiffness = 800.0,
            damping = 40.0,
            effort_limit = 12.0,
            velocity_limit=1.0
        ),
        "tool_pitch_joint": ImplicitActuatorCfg(
            joint_names_expr = "tool_pitch_joint",
            stiffness = 800.0,
            damping = 40.0,
            effort_limit = 12.0,
            velocity_limit=1.0
        ),
        "tool_yaw_joint": ImplicitActuatorCfg(
            joint_names_expr = "tool_yaw_joint",
            stiffness = 800.0,
            damping = 40.0,
            effort_limit = 12.0,
            velocity_limit=1.0
        ),
        "jaw_joint_1": ImplicitActuatorCfg(
            joint_names_expr = "jaw_joint_1",
            effort_limit=0.1,
            velocity_limit=0.2,
            stiffness=200,
            damping=10.0,
        ),
        # Mimic Joint
        "jaw_joint_2": ImplicitActuatorCfg(
            joint_names_expr = "jaw_joint_2",
            stiffness = 0,
            damping = 0,
            effort_limit=0.1,
            velocity_limit=0.2,
        ),
    }
)
