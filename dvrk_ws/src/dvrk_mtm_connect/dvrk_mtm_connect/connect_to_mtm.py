import rclpy                # type: ignore
from rclpy.node import Node # type: ignore

from rcl_interfaces.msg import ParameterType, ParameterDescriptor   # type: ignore
from sensor_msgs.msg import Joy, JointState                         # type: ignore
from std_msgs.msg import Empty                                      # type: ignore
from std_msgs.msg import Bool                                       # type: ignore
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, WrenchStamped   # type: ignore

from scipy.spatial.transform import Rotation as R
import numpy as np

import socket
import json
import time

"""
MTM_Connecter communicates with a standard dvRK MTM console and: 
    1. Records MTM cartesian pose (cp) change and translate it into teleoperation commands; 
    2. Deals with foot pedal buttons and try to reproduce a real teleoperation experience;
Run this node with the following steps: 
    1. In a separate terminal (or even PC): ros2 run dvrk_console_json -j console-MTMR-FP.json
    2. Power on and bring up MTM console:   Click "Home" on the GUI
       or:                                  ros2 topic pub /MTMR/body/servo_cf geometry_msgs/msg/WrenchStamped "{}" -1
    3. Bring up this node:                  ros2 run dvrk_mtm_connect mtm_connect --ros-args -p scale:=0.8
       or:                                  ros2 run dvrk_mtm_connect mtm_connect --ros-args -p scale:=0.8 -p io_mode:=ros
    4. Enable Simulated PSM:                python Isaac_test/test_teleop.py --num_envs 1 --enable_camera
       or:                                  python Isaac_test/test_teleop_udp.py --num_envs 1 --enable_camera
    5. Step on clutch and release it to initiazte MTM alignment process, wait until terminal prompts "MTM aligned!"
    6. Step on MONO to enable teleoperation; step on clutch while stepping on MONO to enable position adjustment
Note:
    1. orbit flag is only effective when io_mode is 'ros', it specifies whether using an Orbit dVRK setup for test only purpose;
    2. toggle_record flag is only effective when io_mode is 'udp', it will trigger Isaac side to start/stop recording data. 
"""
class MTM_Connecter(Node):
    def __init__(self):
        super().__init__('mtm_connecter')
        """ROS2 Node Parameters"""
        scale_param_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, 
            description='Scale of Teleoperation')
        self.declare_parameter('scale', 0.2, scale_param_descriptor)
        orbit_param_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL,
            description='Whether Using Orbit dVRK Setup')
        self.declare_parameter('is_orbit', False, orbit_param_descriptor)
        io_mode_param_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description="PSM communication mode: 'ros' or 'udp'"
        )
        self.declare_parameter('io_mode', 'udp', io_mode_param_descriptor)

        """Robot Constant"""
        self.scale = self.get_parameter('scale').get_parameter_value().double_value
        self.is_orbit = self.get_parameter('is_orbit').get_parameter_value().bool_value
        self.io_mode = self.get_parameter('io_mode').get_parameter_value().string_value.lower()
        self.MTM_home_position = Point(x=0.0, y=-0.3645, z=-0.1288)
        self.__maximum_gripper_angle = 75.0 / 180.0 * 3.1415926
        if self.is_orbit:
            self.__minimum_gripper_angle = 10.0 / 180.0 * 3.1415926
        else:
            self.__minimum_gripper_angle = 5.0 / 180.0 * 3.1415926
        # Actual => command:    [min, min + 20% * (max - min)) => {min} 
        # and:                  [min + 20% * (max - min), max] => [min, max]
        self.__gripper_angle_minimum_percent = 0.2
        self.__gripper_threshold_angle = self.__minimum_gripper_angle + self.__gripper_angle_minimum_percent * (
            self.__maximum_gripper_angle - self.__minimum_gripper_angle)
        self.align_dis_threshold = 0.005
        self.align_ori_threshould = 5.0 / 180.0 * 3.1415926

        """Orientation Offset Data"""
        if self.is_orbit:
            self.r_pi = R.from_quat([-0.5, 0.5, -0.5, 0.5])
            self.r_to = R.from_quat([-0.5, -0.5, -0.5,  0.5])
        else:
            self.r_pi = R.from_quat([-0.5, -0.5, 0.5, 0.5])
            self.r_to = R.from_quat([-0.5, 0.5, 0.5, 0.5])
        self.r_mi = R.from_quat([0.4934, 0.7696, 0.0589, -0.4010])        
        self.r_pi_inv = self.r_pi.inv()
        self.r_mi_inv = self.r_mi.inv()
        self.r_to_inv = self.r_to.inv()

        """Robot Data"""
        self.mtm_cp = Pose()
        self.psm_cp = Pose()
        self.mtm_position_init = Point()
        self.psm_position_init = Point()
        self.align_tar_pose = Pose()
        self.gripper_js = 0.0
        
        """Button Data"""
        self.coag_state = 0
        self.clutch_state = 0
        # mtm_state = -3: Newborn; -2: PSM Heard; -1: Unaligned; -0.5: Aligning; 0: Hold; 1: Teleoperation; 2: Adjusting
        self.mtm_state = -3
        self.recording_state = 0    # 0: not recording; 1: recording; 2: pausing;

        """Socket Connection"""
        if self.io_mode == "udp":
            self.udp_ros_listen_ip = "0.0.0.0"      # Not listening to UDP
            self.udp_ros_listen_port = 15150

            self.udp_isaac_ip = "127.0.0.1"         # Send to localhost
            self.udp_isaac_port = 15151

            self.sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock_rx.bind((self.udp_ros_listen_ip, self.udp_ros_listen_port))
            self.sock_rx.setblocking(False)

            self.sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        """Subscriptions"""
        self.coag_listener = self.create_subscription(
            Joy,
            '/footpedals/coag',
            self.coag_changed,
            10
        )
        self.clutch_listener = self.create_subscription(
            Joy, 
            '/footpedals/clutch',
            self.clutch_changed,
            10
        )
        self.camera_listener = self.create_subscription(
            Joy,
            '/footpedals/camera',
            self.camera_changed,
            10
        )
        self.mtm_cp_listener = self.create_subscription(
            PoseStamped,
            '/MTMR/measured_cp',
            self.update_mtm_cp,
            10
        )
        self.psm_cp_listener = self.create_subscription(
            PoseStamped,
            'PSMI/local/measure_cp',
            self.update_psm_cp,
            10
        )
        self.mtm_gripper_listener = self.create_subscription(
            JointState,
            'MTMR/gripper/measured_js',
            self.update_gripper_js,
            10
        )

        """Publishers"""
        self.mtm_gc_controller = self.create_publisher(
            Bool,
            '/MTMR/use_gravity_compensation',
            10
        )
        self.mtm_cp_controller = self.create_publisher(
            PoseStamped,
            '/MTMR/move_cp',
            10
        )
        self.mtm_cf_controller = self.create_publisher(
            WrenchStamped,
            '/MTMR/body/servo_cf',
            10
        )
        self.mtm_orientation_locker = self.create_publisher(
            Quaternion,
            '/MTMR/lock_orientation',
            10
        )
        self.mtm_orientation_unlocker = self.create_publisher(
            Empty,
            '/MTMR/unlock_orientation',
            10
        )
        self.psm_cp_controller = self.create_publisher(
            PoseStamped,
            'PSMI/move_cp',
            1
        )
        self.psm_gpr_controller = self.create_publisher(
            JointState,
            'PSMI/gripper/move_js',
            1
        )
        self.psm_reset_controller = self.create_publisher(
            Empty,
            'PSMI/reset',
            1
        )

        """Enable Gravity Compensation at the beginning"""
        gc_msg = Bool()
        gc_msg.data = True
        self.mtm_gc_controller.publish(gc_msg)

        """Timers"""
        self.timer = self.create_timer(0.01, self.timer_callback)

        self.get_logger().info('Node created, waiting for PSM side message...')

    def _udp_send_cmd_to_isaac(self, pose: Pose, gripper: float, reset: bool, record_cmd: int = None):
        if self.io_mode != "udp":
            return
        if record_cmd is None:
            if self.recording_state == 1:
                record_cmd = 5    # Continue Recording
            else:
                record_cmd = 0    # No Action
        msg = {
            "type": "psm_cmd",
            "t": time.time(),
            "pos": [pose.position.x, pose.position.y, pose.position.z],
            "quat_xyzw": [pose.orientation.x, pose.orientation.y,
                        pose.orientation.z, pose.orientation.w],
            "gripper": float(gripper),
            "reset": bool(reset),
            "record_cmd": record_cmd            # 0: No Action; 1: Start Recording; 2: Save Recording; 3: Discard Recording; 4: Pause Recording; 5: Resume Recording
        }
        self.sock_tx.sendto(
            json.dumps(msg).encode("utf-8"),
            (self.udp_isaac_ip, self.udp_isaac_port),
        )

    def _udp_poll_state_from_isaac(self):
        if self.io_mode != "udp":
            return

        latest = None
        while True:             # Read all queued packets; keep the most recent
            try:
                data, _ = self.sock_rx.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception as e:
                self.get_logger().warning(f"UDP recv error: {e}")
                break

            try:
                pkt = json.loads(data.decode("utf-8"))
                if pkt.get("type") == "psm_state":
                    latest = pkt
            except Exception:
                continue

        if latest is None:
            return

        # Update self.psm_cp
        p = latest["pos"]
        q = latest["quat_xyzw"]

        pose = Pose()
        pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        self.psm_cp = pose

        if self.mtm_state == -3:
            self.mtm_state = self.mtm_state + 1
            self.get_logger().info("PSM side UDP message received, waiting for MTM side message...")

    def coag_changed(self, msg: Joy):                                           # Mono foot pedal stepped or released
        buttons = msg.buttons.tolist()
        self.coag_state = buttons[0]
        if self.coag_state == 1 and self.mtm_state == 0:                        # MTM Mode: Hold -> Teleoperation
            self.mtm_position_init = self.mtm_cp.position
            self.psm_position_init = self.psm_cp.position
            gc_msg = Bool()
            gc_msg.data = True
            self.mtm_gc_controller.publish(gc_msg)
            loose_msg = WrenchStamped()
            self.mtm_cf_controller.publish(loose_msg)
            self.mtm_state = 1
            if self.recording_state == 2:
                self.recording_state = 1
                self.get_logger().info('Resume Recording.')
        elif self.coag_state == 0 and self.mtm_state == 1:                      # MTM Mode: Teleoperation -> Hold
            hold_msg = PoseStamped()
            hold_msg.pose = self.mtm_cp
            self.mtm_cp_controller.publish(hold_msg)
            self.mtm_state = 0
            if self.recording_state == 1:
                self.recording_state = 2
                self._udp_send_cmd_to_isaac(
                    pose=Pose(),
                    gripper=0.0,
                    reset=False,
                    record_cmd=4
                )
                self.get_logger().info('Pause Recording.')
        elif self.coag_state == 0 and self.mtm_state == 2:                      # MTM Mode: Adjusting -> Hold
            unlock_msg = Empty()
            self.mtm_orientation_unlocker.publish(unlock_msg)
            self.mtm_state = 1
            hold_msg = PoseStamped()
            hold_msg.pose = self.mtm_cp
            self.mtm_cp_controller.publish(hold_msg)
            self.mtm_state = 0

    def clutch_changed(self, msg: Joy):                                         # Clutch foot pedal stepped or released
        buttons = msg.buttons.tolist()
        self.clutch_state = buttons[0]
        if self.clutch_state == 0 and self.mtm_state == -1:                     # MTM Mode: Unaligned -> Aligning
            self.get_logger().info('MTM aligning...')
            align_msg = PoseStamped()
            align_pose = Pose()
            align_pose.position = self.MTM_home_position
            align_pose.orientation = self.__rotation_convert_PSM2MTM(self.psm_cp.orientation)
            align_msg.pose = align_pose
            self.align_tar_pose = align_pose
            self.mtm_cp_controller.publish(align_msg)
            self.mtm_state = -0.5
        elif self.clutch_state == 1 and self.mtm_state == 0:                    # MTM Mode: Hold -> Unaligned
            self.get_logger().info('PSMI Reset')
            reset_msg = Empty()
            if self.io_mode == "ros":
                self.psm_reset_controller.publish(reset_msg)
            elif self.io_mode == "udp":
                if self.recording_state > 0:
                    self.recording_state = 0
                    self.get_logger().info('Discard Recording.')
                    record_cmd = 3                                              # Discard Recording
                else:
                    record_cmd = 0
                self._udp_send_cmd_to_isaac(
                    pose=Pose(),
                    gripper=0.0,
                    reset=True,
                    record_cmd=record_cmd
                )
            self.mtm_state = -1
        elif self.clutch_state == 1 and self.mtm_state == 1:                    # MTM Mode: Teleoperation -> Adjusting
            # Lock Orientation
            lock_msg = self.mtm_cp.orientation
            self.mtm_orientation_locker.publish(lock_msg)
            self.mtm_state = 2
            if self.recording_state == 1:
                self.recording_state = 2
                self._udp_send_cmd_to_isaac(
                    pose=Pose(),
                    gripper=0.0,
                    reset=False,
                    record_cmd=4
                )
                self.get_logger().info('Pause Recording.')
        elif self.clutch_state == 0 and self.mtm_state == 2:                    # MTM Mode: Adjusting -> Teleoperation
            self.mtm_position_init = self.mtm_cp.position
            self.psm_position_init = self.psm_cp.position
            # Unlock Orientation
            unlock_msg = Empty()
            self.mtm_orientation_unlocker.publish(unlock_msg)
            self.mtm_state = 1
            if self.recording_state == 2:
                self.recording_state = 1
                self.get_logger().info('Resume Recording.')

    def camera_changed(self, msg: Joy):
        if not self.mtm_state == 0:                                             # Only respond when in Hold mode
            return
        buttons = msg.buttons.tolist()
        camera_state = buttons[0]
        if camera_state == 0:                                                   # Camera foot pedal released
            if self.recording_state == 0:
                self.recording_state = 1
                self._udp_send_cmd_to_isaac(
                    pose=Pose(),
                    gripper=0.0,
                    reset=False,
                    record_cmd=1                                                # Start Recording
                    )
                self.get_logger().info('Start Recording.')
            else:
                self.recording_state = 0
                self._udp_send_cmd_to_isaac(
                    pose=Pose(),
                    gripper=0.0,
                    reset=False,
                    record_cmd=2                                                # Save Recording
                    )
                self.get_logger().info('Save Recording.')

    def update_mtm_cp(self, msg: PoseStamped):
        if self.mtm_state == -2:
            self.mtm_state = self.mtm_state + 1
            self.get_logger().info('MTM side message received, waiting for MTM alignment...')
        self.mtm_cp = msg.pose

    def update_psm_cp(self, msg: PoseStamped):
        if self.io_mode != "ros":
            return
        
        if self.mtm_state == -3:
            self.mtm_state = self.mtm_state + 1
            self.get_logger().info('PSM side ROS message received, waiting for MTM side message...')
        self.psm_cp = msg.pose

    def update_gripper_js(self, msg: JointState):
        self.gripper_js = ((min(max(msg.position[0], -1.0), 1.0) + 1.0) / 2.0 * (
                self.__maximum_gripper_angle - self.__minimum_gripper_angle) + self.__minimum_gripper_angle) / 2.0
        if self.gripper_js < self.__gripper_threshold_angle:
            self.gripper_js = self.__minimum_gripper_angle
        else:
            self.gripper_js = (self.gripper_js - self.__gripper_threshold_angle) / (
                self.__maximum_gripper_angle - self.__gripper_threshold_angle) * (
                    self.__maximum_gripper_angle - self.__minimum_gripper_angle) + self.__minimum_gripper_angle
    def timer_callback(self):
        if self.io_mode == "udp":
            self._udp_poll_state_from_isaac()

        if self.mtm_state == 1:                                                     # Control PSMI via psm_cp_controller
            ctrl_msg = PoseStamped()
            target_pose = Pose()
            target_pose.position = self.__displacement_convert_MTM2PSM(self.mtm_cp.position)
            target_pose.orientation = self.__rotation_convert_MTM2PSM(self.mtm_cp.orientation)
            ctrl_msg.pose = target_pose

            if self.io_mode == "ros":
                self.psm_cp_controller.publish(ctrl_msg)
                gpr_msg = JointState()
                gpr_msg.position = [self.gripper_js, 0.0]
                self.psm_gpr_controller.publish(gpr_msg)
            elif self.io_mode == "udp":
                self._udp_send_cmd_to_isaac(
                    pose=target_pose,
                    gripper=self.gripper_js,
                    reset=False,
                )
        elif self.mtm_state == -0.5:                                                # Aligning state check
            if self.__get_pose_distance(self.mtm_cp, self.align_tar_pose) < self.align_dis_threshold:
                if self.__get_pose_defelction(self.mtm_cp, self.align_tar_pose) < self.align_ori_threshould:
                    self.mtm_state = 0
                    self.get_logger().info('MTM aligned!')

    def __displacement_convert_MTM2PSM(self, p_m: Point):
        delta_p = self.r_mi_inv.as_matrix() @ np.array(
            [(p_m.x - self.mtm_position_init.x) * self.scale, 
             (p_m.y - self.mtm_position_init.y) * self.scale, 
             (p_m.z - self.mtm_position_init.z) * self.scale])
        p_p = self.r_pi.as_matrix() @ (self.r_to.as_matrix() @ delta_p)
        return Point(
            x=p_p[0] + self.psm_position_init.x, 
            y=p_p[1] + self.psm_position_init.y, 
            z=p_p[2] + self.psm_position_init.z)

    def __rotation_convert_PSM2MTM(self, orientation: Quaternion):
        q_p = [orientation.x, orientation.y, orientation.z, orientation.w]
        r_p = R.from_quat(q_p)
        r_m = self.r_mi * self.r_to_inv * self.r_pi_inv * r_p * self.r_to
        q_m = r_m.as_quat()
        return Quaternion(x=q_m[0], y=q_m[1], z=q_m[2], w=q_m[3])
    
    def __rotation_convert_MTM2PSM(self, orientation: Quaternion):
        q_m = [orientation.x, orientation.y, orientation.z, orientation.w]
        r_m = R.from_quat(q_m)
        r_p = self.r_pi * self.r_to * self.r_mi_inv * r_m * self.r_to_inv
        q_p = r_p.as_quat()
        return Quaternion(x=q_p[0], y=q_p[1], z=q_p[2], w=q_p[3])
    
    @staticmethod
    def __get_pose_distance(pose1: Pose, pose2: Pose) -> float:
        pos1 = np.array([pose1.position.x, pose1.position.y, pose1.position.z])
        pos2 = np.array([pose2.position.x, pose2.position.y, pose2.position.z])
        distance = np.linalg.norm(pos1 - pos2)
        return distance
    
    @staticmethod
    def __get_pose_defelction(pose1: Pose, pose2: Pose) -> float:
        r1 = R.from_quat([pose1.orientation.x, pose1.orientation.y, 
                      pose1.orientation.z, pose1.orientation.w])
        r2 = R.from_quat([pose2.orientation.x, pose2.orientation.y, 
                        pose2.orientation.z, pose2.orientation.w])
        relative_rotation = r1.inv() * r2
        return relative_rotation.magnitude()

def main(args=None):
    rclpy.init(args=args)

    mtm_connecter = MTM_Connecter()

    try:
        rclpy.spin(mtm_connecter)
    except KeyboardInterrupt:
        print('Shutting Down...')
    finally:
        mtm_connecter.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()