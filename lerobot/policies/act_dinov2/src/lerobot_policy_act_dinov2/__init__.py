# __init__.py
"""
LeRobot Policy: VQ-BeT Multi-View.
"""

try:
    import lerobot
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_act_dinov2 import ACTDINOv2Config
from .modeling_act_dinov2 import ACTDINOv2Policy
from .processor_act_dinov2 import make_act_dinov2_pre_post_processors

__all__ = [
    "ACTDINOv2Config",
    "ACTDINOv2Policy",
    "make_act_dinov2_pre_post_processors",
]