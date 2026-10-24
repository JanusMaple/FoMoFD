from __future__ import annotations

import argparse
import json
import random
import socket
import time
from typing import Optional

import numpy as np

"""
Example Usage:
  ros2 run real_dvrk_bridge real_psm_bridge \
    --mode record \
    --task tissue_retraction \
    --arm PSM2 \
    --enable_reset_on_clutch
or:
  ros2 run real_dvrk_bridge real_psm_bridge \
    --mode rollout \
    --task shunt_insertion \
    --arm PSM2 \
    --enable_reset_on_clutch \
    --reset_on_clutch_event press \
    --reset_duration_s 3.0
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy, JointState
from geometry_msgs.msg import PoseStamped


def quat_norm(q, eps=1e-8):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / max(float(np.linalg.norm(q)), eps)


def quat_canon(q):
    q = quat_norm(q)
    return -q if q[0] < 0 else q


def quat_mul(q1, q2):
    w1, x1, y1, z1 = quat_norm(q1)
    w2, x2, y2, z2 = quat_norm(q2)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_lerp(q0, q1, a):
    q0 = quat_norm(q0)
    q1 = quat_norm(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    return quat_canon((1.0 - a) * q0 + a * q1)


def pose_to_np_wxyz(pose_msg: PoseStamped):
    p = pose_msg.pose.position
    q = pose_msg.pose.orientation
    pos = np.array([p.x, p.y, p.z], dtype=np.float64)
    quat_wxyz = quat_canon(np.array([q.w, q.x, q.y, q.z], dtype=np.float64))
    return pos, quat_wxyz

def normalize_task_name(task: str) -> str:
    return str(task).strip().lower().replace("-", "_")


TASK_INIT_POSES = {
    "tissue_retraction": {
        "pos": [
            -0.1335065569810792,
            0.06209585838009683,
            -0.1253270310980986
        ],
        "quat_xyzw": [
            -0.20147677212752074,
            0.8957673261389112,
            0.18353709609247623,
            -0.3511725246555624,
        ],
    },
    "shunt_insertion": {
        "pos": [
            -0.13159899883998177,
            0.05796048030665117,
            -0.12522943992512509,
        ],
        "quat_xyzw": [
            -0.3344033413453463,
            0.8675844210102588,
            0.2750790149453987,
            -0.24454695511093796,
        ],
    },
}


class RealPSMBridge(Node):
    JAW_OPEN = 1.43

    def __init__(
        self,
        mode: str,
        task: str,
        arm: str,
        udp_ip: str,
        udp_port: int,
        udp_listen_port: int,
        state_hz: float,
        init_pos_jitter_m: float,
        enable_reset_on_clutch: bool,
        reset_on_clutch_event: str,
        reset_duration_s: float,
        action_interp_s: float,
    ):
        super().__init__("real_psm_bridge")

        if mode not in {"record", "rollout"}:
            raise ValueError(f"Invalid mode: {mode}")

        self.mode = mode
        self.task = normalize_task_name(task)
        if self.task not in TASK_INIT_POSES:
            raise ValueError(
                f"Unknown task={task}. Available tasks: {sorted(TASK_INIT_POSES.keys())}"
            )

        init_pose = TASK_INIT_POSES[self.task]
        self.init_pos = np.asarray(init_pose["pos"], dtype=np.float64)
        self.init_quat_xyzw = [float(x) for x in init_pose["quat_xyzw"]]
        self.init_q_wxyz = quat_canon(np.array([
            self.init_quat_xyzw[3],
            self.init_quat_xyzw[0],
            self.init_quat_xyzw[1],
            self.init_quat_xyzw[2],
        ], dtype=np.float64))
        self.arm = arm
        self.init_pos_jitter_m = float(init_pos_jitter_m)
        self.enable_reset_on_clutch = bool(enable_reset_on_clutch)
        self.reset_on_clutch_event = str(reset_on_clutch_event)
        self.reset_duration_s = float(reset_duration_s)

        self.latest_measured_cp: Optional[PoseStamped] = None
        self.latest_setpoint_cp: Optional[PoseStamped] = None
        self.latest_jaw_measured: Optional[JointState] = None
        self.latest_jaw_setpoint: Optional[JointState] = None

        self.cmd_p = None
        self.cmd_q = None
        self.cmd_gripper = float(self.JAW_OPEN)
        self.control_active = False

        self.reset_active = False
        self.reset_start_time = 0.0
        self.reset_start_p = None
        self.reset_start_q = None
        self.reset_start_jaw = float(self.JAW_OPEN)
        self.reset_target_p = np.asarray(self.init_pos, dtype=np.float64)
        q_xyzw = self.init_quat_xyzw
        self.reset_target_q = quat_canon(np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64))
        self.reset_target_jaw = float(self.JAW_OPEN)

        self.action_interp_s = float(action_interp_s)

        self.interp_active = False
        self.interp_start_time = 0.0
        self.interp_duration_s = self.action_interp_s

        self.interp_start_p = None
        self.interp_start_q = None
        self.interp_start_jaw = float(self.JAW_OPEN)

        self.interp_target_p = None
        self.interp_target_q = None
        self.interp_target_jaw = float(self.JAW_OPEN)

        self.last_pub_p = None
        self.last_pub_q = None
        self.last_pub_jaw = float(self.JAW_OPEN)

        self.pedal_state = {
            "clutch": None,
            "camera": None,
            "coag": None,
        }

        self.udp_ip = udp_ip
        self.udp_port = int(udp_port)
        self.sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx.bind(("0.0.0.0", int(udp_listen_port)))
        self.sock_rx.setblocking(False)

        self.create_subscription(PoseStamped, f"/{arm}/measured_cp", self.on_measured_cp, 10)
        self.create_subscription(PoseStamped, f"/{arm}/setpoint_cp", self.on_setpoint_cp, 10)
        self.create_subscription(JointState, f"/{arm}/jaw/measured_js", self.on_jaw_measured, 10)
        self.create_subscription(JointState, f"/{arm}/jaw/setpoint_js", self.on_jaw_setpoint, 10)

        self.create_subscription(Joy, "/footpedals/clutch", lambda m: self.on_pedal("clutch", m), 10)
        self.create_subscription(Joy, "/footpedals/camera", lambda m: self.on_pedal("camera", m), 10)
        self.create_subscription(Joy, "/footpedals/coag", lambda m: self.on_pedal("coag", m), 10)

        self.pub_move_cp = self.create_publisher(PoseStamped, f"/{arm}/move_cp", 10)
        self.pub_servo_cp = self.create_publisher(PoseStamped, f"/{arm}/servo_cp", 10)
        self.pub_jaw_move_jp = self.create_publisher(JointState, f"/{arm}/jaw/move_jp", 10)
        self.pub_jaw_servo_jp = self.create_publisher(JointState, f"/{arm}/jaw/servo_jp", 10)

        period = 1.0 / max(float(state_hz), 1.0)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f"real_psm_bridge started: mode={mode}, task={self.task}, arm={arm}, "
            f"UDP TX={udp_ip}:{udp_port}, UDP RX=0.0.0.0:{udp_listen_port}, "
            f"state_hz={state_hz}, reset_duration_s={self.reset_duration_s}"
        )

    @staticmethod
    def pose_to_dict(msg: PoseStamped) -> dict:
        p = msg.pose.position
        q = msg.pose.orientation
        return {
            "pos": [float(p.x), float(p.y), float(p.z)],
            "quat_xyzw": [float(q.x), float(q.y), float(q.z), float(q.w)],
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "frame_id": str(msg.header.frame_id),
        }

    @staticmethod
    def joint_to_dict(msg: JointState) -> dict:
        return {
            "name": list(msg.name),
            "position": [float(x) for x in msg.position],
            "velocity": [float(x) for x in msg.velocity],
            "effort": [float(x) for x in msg.effort],
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
        }

    def udp_send(self, pkt: dict) -> None:
        pkt["t_bridge"] = time.time()
        self.sock_tx.sendto(json.dumps(pkt).encode("utf-8"), (self.udp_ip, self.udp_port))

    def on_measured_cp(self, msg: PoseStamped) -> None:
        self.latest_measured_cp = msg

    def on_setpoint_cp(self, msg: PoseStamped) -> None:
        self.latest_setpoint_cp = msg

    def on_jaw_measured(self, msg: JointState) -> None:
        self.latest_jaw_measured = msg

    def on_jaw_setpoint(self, msg: JointState) -> None:
        self.latest_jaw_setpoint = msg

    def on_pedal(self, pedal: str, msg: Joy) -> None:
        pressed = bool(msg.buttons[0]) if msg.buttons else False
        prev = self.pedal_state[pedal]
        self.pedal_state[pedal] = pressed

        if prev is None or prev == pressed:
            return

        event = "press" if pressed else "release"
        reset_triggered = False

        if (
            self.enable_reset_on_clutch
            and pedal == "clutch"
            and event == self.reset_on_clutch_event
            and not bool(self.pedal_state.get("coag", False))
        ):
            self.begin_reset(reason=f"clutch_{event}")
            reset_triggered = True

        self.udp_send({
            "type": "pedal",
            "pedal": pedal,
            "pressed": pressed,
            "event": event,
            "reset_triggered": reset_triggered,
        })

        if reset_triggered:
            self.udp_send({
                "type": "reset_event",
                "reason": f"clutch_{event}",
                "action": "reset_robot_and_policy",
            })

    def make_init_pose_msg(self) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""

        pos = list(self.init_pos)
        if self.init_pos_jitter_m > 0:
            pos = [
                pos[0] + random.uniform(-self.init_pos_jitter_m, self.init_pos_jitter_m),
                pos[1] + random.uniform(-self.init_pos_jitter_m, self.init_pos_jitter_m),
                pos[2] + random.uniform(-self.init_pos_jitter_m, self.init_pos_jitter_m),
            ]

        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])

        q = self.init_quat_xyzw
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        return msg

    def get_current_jaw(self) -> float:
        if self.latest_jaw_measured is not None and len(self.latest_jaw_measured.position) > 0:
            return float(self.latest_jaw_measured.position[0])
        if self.latest_jaw_setpoint is not None and len(self.latest_jaw_setpoint.position) > 0:
            return float(self.latest_jaw_setpoint.position[0])
        return float(self.cmd_gripper)

    def begin_reset(self, reason: str) -> None:
        src = self.latest_measured_cp or self.latest_setpoint_cp
        if src is None:
            self.get_logger().warning("Reset requested but no measured/setpoint CP available.")
            return

        self.control_active = False

        self.interp_active = False

        self.reset_start_p, self.reset_start_q = pose_to_np_wxyz(src)
        self.reset_start_jaw = self.get_current_jaw()

        target_p = np.asarray(self.init_pos, dtype=np.float64).copy()
        if self.init_pos_jitter_m > 0:
            target_p += np.random.uniform(
                -self.init_pos_jitter_m,
                self.init_pos_jitter_m,
                size=3,
            )

        self.reset_target_p = target_p
        q_xyzw = self.init_quat_xyzw
        self.reset_target_q = quat_canon(np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64))
        self.reset_target_jaw = float(self.JAW_OPEN)

        self.reset_start_time = time.time()
        self.reset_active = True

        self.get_logger().warning(
            f"RESET begin: reason={reason}, duration={self.reset_duration_s:.2f}s, "
            f"jaw {self.reset_start_jaw:.3f}->{self.reset_target_jaw:.3f}"
        )

    def update_reset(self) -> None:
        if not self.reset_active:
            return

        now = time.time()
        raw_a = (now - self.reset_start_time) / max(self.reset_duration_s, 1e-6)
        a = min(max(raw_a, 0.0), 1.0)

        p = (1.0 - a) * self.reset_start_p + a * self.reset_target_p
        q = quat_lerp(self.reset_start_q, self.reset_target_q, a)
        jaw = (1.0 - a) * self.reset_start_jaw + a * self.reset_target_jaw

        self.publish_servo_pose(p, q)
        self.publish_jaw_servo(jaw)

        if a >= 1.0:
            self.reset_active = False

            self.cmd_p = self.reset_target_p.copy()
            self.cmd_q = self.reset_target_q.copy()
            self.cmd_gripper = float(self.reset_target_jaw)
            self.control_active = False

            self.publish_servo_pose(self.cmd_p, self.cmd_q)
            self.publish_jaw_servo(self.cmd_gripper)

            self.udp_send({
                "type": "reset_complete",
                "reason": "linear_pose_and_jaw_reset_complete",
            })

            self.get_logger().warning("RESET complete.")

    def poll_udp_commands(self) -> None:
        while True:
            try:
                data, _ = self.sock_rx.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception as e:
                self.get_logger().warning(f"UDP recv error: {e}")
                break

            try:
                pkt = json.loads(data.decode("utf-8"))
            except Exception:
                continue

            typ = pkt.get("type")
            if typ == "absolute_reset":
                self.begin_reset(reason="udp_absolute_reset")
                self.udp_send({"type": "reset_ack", "reason": "udp_absolute_reset"})
            elif typ == "relative_action":
                self.handle_relative_action(pkt)

    def start_action_interpolation(self, target_p, target_q, target_jaw) -> None:
        if self.last_pub_p is not None and self.last_pub_q is not None:
            start_p = self.last_pub_p.copy()
            start_q = self.last_pub_q.copy()
            start_jaw = float(self.last_pub_jaw)
        elif self.latest_setpoint_cp is not None:
            start_p, start_q = pose_to_np_wxyz(self.latest_setpoint_cp)
            start_jaw = self.get_current_jaw()
        elif self.latest_measured_cp is not None:
            start_p, start_q = pose_to_np_wxyz(self.latest_measured_cp)
            start_jaw = self.get_current_jaw()
        else:
            start_p = np.asarray(target_p, dtype=np.float64).copy()
            start_q = quat_canon(target_q).copy()
            start_jaw = float(target_jaw)

        self.interp_start_p = start_p
        self.interp_start_q = start_q
        self.interp_start_jaw = float(start_jaw)

        self.interp_target_p = np.asarray(target_p, dtype=np.float64).copy()
        self.interp_target_q = quat_canon(target_q).copy()
        self.interp_target_jaw = float(target_jaw)

        self.interp_start_time = time.time()
        self.interp_duration_s = max(float(self.action_interp_s), 1e-4)
        self.interp_active = True

    def handle_relative_action(self, pkt: dict) -> None:
        if self.mode != "rollout":
            return
        if self.reset_active:
            return

        a = np.asarray(pkt.get("action", []), dtype=np.float64).reshape(-1)
        if a.size != 8:
            self.get_logger().warning(f"Bad relative_action size: {a.size}")
            return

        if self.cmd_p is None or self.cmd_q is None:
            self.reset_rollout_command_target_from_robot()

        dp = a[0:3]
        dq = quat_canon(a[3:7])
        g = float(a[7])

        # Integrate policy action ONCE into the next absolute target.
        target_p = self.cmd_p + dp
        target_q = quat_canon(quat_mul(self.cmd_q, dq))
        target_g = g

        self.cmd_p = target_p.copy()
        self.cmd_q = target_q.copy()
        self.cmd_gripper = float(target_g)

        self.control_active = True

        # Smoothly publish current target over the next control cycle
        self.start_action_interpolation(target_p, target_q, target_g)

    def reset_rollout_command_target_from_robot(self) -> None:
        src = self.latest_setpoint_cp or self.latest_measured_cp
        if src is None:
            self.cmd_p = np.asarray(self.init_pos, dtype=np.float64).copy()
            q_xyzw = self.init_quat_xyzw
            self.cmd_q = quat_canon(np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64))
            self.cmd_gripper = float(self.JAW_OPEN)
            self.control_active = False
            return

        self.cmd_p, self.cmd_q = pose_to_np_wxyz(src)
        self.cmd_gripper = self.get_current_jaw()
        self.control_active = False

    def publish_rollout_hold_command(self) -> None:
        if self.mode != "rollout":
            return
        if self.reset_active:
            return
        if not self.control_active:
            return
        if self.cmd_p is None or self.cmd_q is None:
            return

        if self.interp_active:
            now = time.time()
            a = (now - self.interp_start_time) / max(self.interp_duration_s, 1e-6)
            a = min(max(a, 0.0), 1.0)

            p = (1.0 - a) * self.interp_start_p + a * self.interp_target_p
            q = quat_lerp(self.interp_start_q, self.interp_target_q, a)
            jaw = (1.0 - a) * self.interp_start_jaw + a * self.interp_target_jaw

            self.publish_servo_pose(p, q)
            self.publish_jaw_servo(jaw)

            if a >= 1.0:
                self.interp_active = False
            return

        # Hold latest absolute target.
        self.publish_servo_pose(self.cmd_p, self.cmd_q)
        self.publish_jaw_servo(self.cmd_gripper)

    def publish_servo_pose(self, p, q_wxyz) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""

        msg.pose.position.x = float(p[0])
        msg.pose.position.y = float(p[1])
        msg.pose.position.z = float(p[2])

        q = quat_canon(q_wxyz)
        msg.pose.orientation.w = float(q[0])
        msg.pose.orientation.x = float(q[1])
        msg.pose.orientation.y = float(q[2])
        msg.pose.orientation.z = float(q[3])

        self.pub_servo_cp.publish(msg)

        self.last_pub_p = np.asarray(p, dtype=np.float64).copy()
        self.last_pub_q = quat_canon(q).copy()

    def publish_jaw_servo(self, jaw: float) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [float(jaw)]
        self.pub_jaw_servo_jp.publish(msg)

        self.last_pub_jaw = float(jaw)

    def send_robot_state(self) -> None:
        if self.latest_measured_cp is None:
            return

        pkt = {
            "type": "robot_state",
            "mode": self.mode,
            "arm": self.arm,
            "measured_cp": self.pose_to_dict(self.latest_measured_cp),
            "setpoint_cp": self.pose_to_dict(self.latest_setpoint_cp)
            if self.latest_setpoint_cp is not None else None,
            "jaw_measured": self.joint_to_dict(self.latest_jaw_measured)
            if self.latest_jaw_measured is not None else None,
            "jaw_setpoint": self.joint_to_dict(self.latest_jaw_setpoint)
            if self.latest_jaw_setpoint is not None else None,
            "pedals": {
                k: bool(v) if v is not None else None
                for k, v in self.pedal_state.items()
            },
        }
        self.udp_send(pkt)

    def timer_callback(self) -> None:
        self.poll_udp_commands()
        self.update_reset()
        self.publish_rollout_hold_command()
        self.send_robot_state()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["record", "rollout"], default="record")
    parser.add_argument("--task", choices=["tissue_retraction", "shunt_insertion"], default="tissue_retraction")
    parser.add_argument("--arm", type=str, default="PSM2")
    parser.add_argument("--udp_ip", type=str, default="127.0.0.1")
    parser.add_argument("--udp_port", type=int, default=15151)
    parser.add_argument("--udp_listen_port", type=int, default=15150)
    parser.add_argument("--state_hz", type=float, default=100.0)
    parser.add_argument("--init_pos_jitter_m", type=float, default=0.0)
    parser.add_argument("--enable_reset_on_clutch", action="store_true")
    parser.add_argument("--reset_on_clutch_event", choices=["press", "release"], default="press")
    parser.add_argument("--reset_duration_s", type=float, default=3.0)
    parser.add_argument("--action_interp_s", type=float, default=0.10)

    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = RealPSMBridge(
        mode=args.mode,
        task=args.task,
        arm=args.arm,
        udp_ip=args.udp_ip,
        udp_port=args.udp_port,
        udp_listen_port=args.udp_listen_port,
        state_hz=args.state_hz,
        init_pos_jitter_m=args.init_pos_jitter_m,
        enable_reset_on_clutch=args.enable_reset_on_clutch,
        reset_on_clutch_event=args.reset_on_clutch_event,
        reset_duration_s=args.reset_duration_s,
        action_interp_s=args.action_interp_s,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()