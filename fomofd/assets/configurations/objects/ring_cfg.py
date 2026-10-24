import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

RING_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir, os.pardir)}/models//ring.usdc",
        scale=(0.04, 0.04, 0.04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
            max_angular_velocity=60,
            max_linear_velocity=3,
            max_depenetration_velocity=0.5,
            kinematic_enabled=False,
            disable_gravity=False,
            linear_damping=5.0,
            angular_damping=5.0,
            sleep_threshold=0.1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.0002,
            rest_offset=0.0,
        ),
    ),
)