# fomofd/assets/configurations/objects/surgical_background_cfg.py

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

SURGICAL_BACKGROUND_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/SurgicalBackground",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir, os.pardir)}/models/OR_scene_s1371/surgical_background.usd",
        scale=(0.01, 0.01, 0.01),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.70711, 0.70711, 0.0, 0.0)
    )
)