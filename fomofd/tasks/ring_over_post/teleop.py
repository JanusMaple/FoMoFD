"""
Teleoperation Test for dVRK PSM in Isaac Lab on Task: ring over post
Run by: python ./fomofd/tasks/ring_over_post/teleop.py --enable_camera
"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation Test for dVRK PSM in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--display_marker", action=argparse.BooleanOptionalAction, help="Whether to display coordinate markers.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac Lab Related"""
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils.math import subtract_frame_transforms

"""User Related"""
import socket
import json
import time
import torch
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from fomofd.tasks.ring_over_post.environment.scene_cfg import dvrkRingPostSceneCfg as dvrkSceneCfg

"""Utility"""
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

def disable_robot_organ_collisions():
    import omni.usd                     # type: ignore
    from pxr import UsdPhysics, Usd     # type: ignore
    stage = omni.usd.get_context().get_stage()
    if not stage:
        return
    def get_collision_prims(root_prim):
        prims = []
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prims.append(prim)
        return prims
    robot_prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot")
    organ_prim = stage.GetPrimAtPath(f"/World/envs/env_0/SurgicalBackground")
    if not robot_prim.IsValid() or not organ_prim.IsValid():
        return
    # Extract the exact physical meshes from inside the parent folders
    robot_col_prims = get_collision_prims(robot_prim)
    organ_col_prims = get_collision_prims(organ_prim)
    # Disable collisions between all pairs of robot and organ collision primitives
    for organ_mesh in organ_col_prims:
        if not organ_mesh.HasAPI(UsdPhysics.FilteredPairsAPI):
            filter_api = UsdPhysics.FilteredPairsAPI.Apply(organ_mesh)
        else:
            filter_api = UsdPhysics.FilteredPairsAPI(organ_mesh)
            
        for robot_mesh in robot_col_prims:
            filter_api.GetFilteredPairsRel().AddTarget(robot_mesh.GetPath())

"""dVRK Teleoperation"""
class Teleoperator:
    def __init__(self):
        """Initialization Scene in Simulator"""
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1/240, render_interval=4)
        self.sim = SimulationContext(sim_cfg)
        # Design scene
        scene_cfg = dvrkSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)
        # Play the simulator
        self.sim.reset()
        # Set up viewports
        set_viewports_to_cameras()
        disable_robot_organ_collisions()
        """Initialize simulator properties"""
        # Extract scene entities
        self.robot = self.scene["psm_si"]           # Articulation Class
        # self.camera = self.scene["tiled_camera"]    # TiledCamera Class
        # Set main camera
        self.sim.set_camera_view([2.5, 1.0, 3.0], [0.0, 0.0, 0.5])
        # Create IK controller
        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(
            self.diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)
        self.default_ee_goal = torch.tensor([0.89253896, 0.04946792, -0.07037026,
                                             -0.4092562, 0.61447084, 0.6591015, -0.1432485],
                                             device=self.sim.device)
        self.ik_commands = torch.zeros(
            self.scene.num_envs, self.diff_ik_controller.action_dim, device=self.robot.device)
        # Gripper Target Position
        self.gpr_target = torch.zeros(self.scene.num_envs, 1, device=self.robot.device)
        # Specify robot-specific parameters
        self.robot_entity_cfg = SceneEntityCfg("psm_si", 
                                          joint_names=["j1", "j2", "outer_insertion", "tool_roll_joint", 
                                                       "tool_pitch_joint", "tool_yaw_joint"],
                                          body_names=["tool_yaw_link"])
        self.gripper_entity_cfg = SceneEntityCfg("psm_si", joint_names=["jaw_joint_1"], body_names=[])
        # Resolving all important entities
        self.robot_entity_cfg.resolve(self.scene)
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]
        self.gripper_entity_cfg.resolve(self.scene)
        # Create EE Marker
        if args_cli.display_marker:
            self.frame_marker_cfg = FRAME_MARKER_CFG.copy()
            self.frame_marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
            self.ee_marker = VisualizationMarkers(self.frame_marker_cfg.replace(prim_path="/Visuals/ee"))
            self.frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self.base_marker = VisualizationMarkers(self.frame_marker_cfg.replace(prim_path="/Visuals/base"))
        # Define simulation stepping
        self.sim_dt = self.sim.get_physics_dt()
        """ Initialize UDP Sockets """
        self.udp_isaac_listen_ip = "0.0.0.0"
        self.udp_isaac_listen_port = 15151

        self.udp_ros_ip = "127.0.0.1"
        self.udp_ros_port = 15150

        self.sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_cmd.bind((self.udp_isaac_listen_ip, self.udp_isaac_listen_port))
        self.sock_cmd.setblocking(False)

        self.sock_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.pending_reset = False
        """Initialization Completed"""
        print("[INFO]: Initialization complete...")

    def _udp_send_state_to_ros(self, ee_pos_b, ee_quat_b):
        # Isaac quat is [w,x,y,z] -> ROS expects [x,y,z,w]
        pos = [float(ee_pos_b[0][0]), float(ee_pos_b[0][1]), float(ee_pos_b[0][2])]
        quat_xyzw = [float(ee_quat_b[0][1]), float(ee_quat_b[0][2]), float(ee_quat_b[0][3]), float(ee_quat_b[0][0])]

        msg = {"type": "psm_state", "t": time.time(), "pos": pos, "quat_xyzw": quat_xyzw}
        self.sock_state.sendto(json.dumps(msg).encode("utf-8"), (self.udp_ros_ip, self.udp_ros_port))

    def _udp_poll_cmd_from_ros(self):
        latest = None
        while True:
            try:
                data, _ = self.sock_cmd.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception:
                break
            try:
                pkt = json.loads(data.decode("utf-8"))
                if pkt.get("type") == "psm_cmd":
                    latest = pkt
            except Exception:
                continue

        if latest is None:
            return

        if bool(latest.get("reset", False)):
            self.pending_reset = True

        if int(latest.get("record_cmd")) > 0:
            return    # Ignore toggle_record command in Isaac side

        p = latest.get("pos", None)
        q = latest.get("quat_xyzw", None)
        if p is not None and q is not None:
            # Convert ROS [x,y,z,w] -> Isaac IK command expects [x,y,z,w,x,y,z]
            ee_goal = torch.tensor([p[0], p[1], p[2], q[3], q[0], q[1], q[2]], device=self.sim.device)
            self.ik_commands[:] = ee_goal
            self.diff_ik_controller.set_command(self.ik_commands)

        if "gripper" in latest:
            self.gpr_target[:] = float(latest["gripper"])

    """Initialize the simulation loop"""
    def reset_simulator(self):
        joint_pos, joint_vel = self.robot.data.default_joint_pos.clone(), self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.diff_ik_controller.reset()
        self.ik_commands[:] = self.default_ee_goal
        self.diff_ik_controller.set_command(self.ik_commands)
        for obj in self.scene.rigid_objects.values():
            root = obj.data.default_root_state.clone()
            root[:, :3] += self.scene.env_origins
            obj.write_root_pose_to_sim(root[:, :7])
            obj.write_root_velocity_to_sim(root[:, 7:])
            obj.reset()
        for obj in self.scene.deformable_objects.values():
            root = obj.data.default_nodal_state_w.clone()
            obj.write_nodal_state_to_sim(root)
            obj.reset()
        self.scene.reset()

    """Runs the simulation loop."""
    def run_simulator(self):
        self.reset_simulator()
        # Simulation loop
        while simulation_app.is_running():
            # Receive UDP Command
            self._udp_poll_cmd_from_ros()
            if self.pending_reset:
                self.reset_simulator()
                self.pending_reset = False
            # Get Jacobians and Poses
            jacobians = self.robot.root_physx_view.get_jacobians()
            jacobian = jacobians[:, self.ee_jacobi_idx, :, [0, 2, 4, 5, 6, 7]]
            ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = self.robot.data.root_state_w[:, 0:7]
            if args_cli.display_marker:
                self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                self.base_marker.visualize(root_pose_w[:, 0:3], root_pose_w[:, 3:7])
            joint_ett_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # Get robot movement goal
            joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_ett_pos)
            # Set robot movement goal
            self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
            # Set gripper movement goal
            self.robot.set_joint_position_target(self.gpr_target, joint_ids=self.gripper_entity_cfg.joint_ids)
            
            # Send UDP State
            self._udp_send_state_to_ros(ee_pos_b, ee_quat_b)
            
            # Write data to sim
            self.scene.write_data_to_sim()
            # Perform step
            self.sim.step()
            # Update buffers
            self.scene.update(self.sim_dt)

if __name__ == "__main__":
    # Run the test
    teleoperator = Teleoperator()
    teleoperator.run_simulator()
    # Close sim app
    simulation_app.close()