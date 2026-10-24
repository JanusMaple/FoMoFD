"""
    This script is a configuration file for different scenes of surgical tasks
"""

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.sensors.camera.camera_cfg import PinholeCameraCfg
from isaaclab.scene import InteractiveSceneCfg

from fomofd.assets.configurations.objects.dVRK_psm_si_cfg import DVRK_PSM_SI_CFG
from fomofd.assets.configurations.objects.needle_cfg import NEEDLE_CFG
from fomofd.assets.configurations.objects.surgical_background_cfg import SURGICAL_BACKGROUND_CFG

WALL_H = 1.0
WALL_T = 0.02

XMIN, XMAX = 0.45, 1.15
YMIN, YMAX = -0.45, 0.45
ZC = WALL_H / 2.0

@configclass
class dvrkBlockNeedleSceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.1), 
            
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.3, 0.4), # Surgical Drape Blue
                roughness=1.0,                 # Pure Matte
                metallic=0.0
            ),
            
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.05),
        )
    )

    # front_wall = AssetBaseCfg(
    #     prim_path="/World/FrontWall",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(WALL_T, YMAX - YMIN, WALL_H),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.0, 0.3, 0.4),
    #             roughness=1.0,
    #             metallic=0.0,
    #         ),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(XMAX + WALL_T / 2.0, (YMIN + YMAX) / 2.0, ZC),
    #     ),
    # )

    # back_wall = AssetBaseCfg(
    #     prim_path="/World/BackWall",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(WALL_T, YMAX - YMIN, WALL_H),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.0, 0.3, 0.4),
    #             roughness=1.0,
    #             metallic=0.0,
    #         ),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(XMIN - WALL_T / 2.0, (YMIN + YMAX) / 2.0, ZC),
    #     ),
    # )

    # left_wall = AssetBaseCfg(
    #     prim_path="/World/LeftWall",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(XMAX - XMIN, WALL_T, WALL_H),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.0, 0.3, 0.4),
    #             roughness=1.0,
    #             metallic=0.0,
    #         ),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=((XMIN + XMAX) / 2.0, YMAX + WALL_T / 2.0, ZC),
    #     ),
    # )

    # right_wall = AssetBaseCfg(
    #     prim_path="/World/RightWall",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(XMAX - XMIN, WALL_T, WALL_H),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.0, 0.3, 0.4),
    #             roughness=1.0,
    #             metallic=0.0,
    #         ),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=((XMIN + XMAX) / 2.0, YMIN - WALL_T / 2.0, ZC),
    #     ),
    # )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=100.0, color=(0.75, 0.75, 0.75))
    )
    
    endoscope_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/EndoscopeLight",
        spawn=sim_utils.SphereLightCfg(
            radius=0.01,                    # Tiny radius mimics a small LED source
            intensity=250000,               # High intensity because it attenuates quickly
            color=(1.0, 0.95, 0.95)         # Slightly warm/sterile white
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.89282, 0.03116, 0.2439),
            rot=(0.6740914, -0.17675164, 0.18190286, -0.69373698)
        )
    )

    wrist_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/dvrk_psm_si/tool_roll_link/WristLight",
        spawn=sim_utils.SphereLightCfg(
            radius=0.005,
            intensity=100000,               # Lower than dome, but enough for local illumination
            color=(1.0, 0.95, 0.95)         # Warm surgical LED
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.02, 0.0, -0.02),
            rot=(0.18301, 0.68301, 0.68301, 0.18301)
        )
    )

    # robot
    psm_si: ArticulationCfg = DVRK_PSM_SI_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            # pos=(0.325, -0.195, 0.82), rot=(0.90984, 0.16043, 0.37688, 0.06646),
            pos=(0.06963, -0.44893, 0.25373), rot=(0.96363, 0.0, 0.0, 0.26724),
            joint_pos={"j1": 0.0, "j2": 0.0, "j3": 0.0, "j4": 0.0, "outer_insertion": 0.15, "tool_roll_joint": 0.0,
                    "tool_pitch_joint": 0.0, "tool_yaw_joint": 0.0, "jaw_joint_1": 0.5, "jaw_joint_2": -0.5}
                    ))

    # needle
    needle: RigidObjectCfg = NEEDLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Needle",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.82, 0.05, 0.16),
            rot=(0.5, 0, 0, 0.86603)))

    # sensors
    fix_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/FixCamera",
        width=224, height=224,
        data_types=["rgb", "depth"],
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 5.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # pos=(0.87, 0.04, 0.2445),
            # rot=(0.68898, -0.15906, 0.15906, -0.68898),
            pos=(0.83562, 0.06059, 0.23128),
            rot=(0.63729, -0.09999, 0.0839, -0.75949),
            convention="opengl"
        ),
        update_period=1/30,                                                                     # 30.0 fps
    )
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/dvrk_psm_si/tool_roll_link/WristCamera",
        width=224, height=224,
        data_types=["rgb", "depth"],
        spawn=PinholeCameraCfg(
            focal_length=36.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.02, 0.0, -0.01),
            rot=(-0.2706, -0.65328, -0.65328, -0.2706),
            convention="opengl"
        ),
        update_period=1/30,                                                                     # 30.0 fps
    )

    # surgical background
    surgical_background: AssetBaseCfg = SURGICAL_BACKGROUND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/SurgicalBackground",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.67, -0.24, 0.0),
            rot=(0.70711, 0.70711, 0.0, 0.0))
    )
