from __future__ import annotations

import argparse
import io
import json
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from utils.ZmqClient import ZmqPolicyClient, ZmqControlClient

"""
python eval_real.py \
  --task tissue_retraction \
  --policy_addr tcp://anonymous225.169:5555 \
  --control_addr tcp://anonymous225.169:5556 \
  --fix_dev /dev/video2 \
  --wrist_dev /dev/video0 \
  --fix_width 640 \
  --fix_height 480 \
  --wrist_width 640 \
  --wrist_height 480 \
  --policy_hz 10 \
  --record \
  --display

python eval_real.py \
  --task shunt_insertion \
  --policy_addr tcp://anonymous225.169:5555 \
  --control_addr tcp://anonymous225.169:5556 \
  --fix_dev /dev/video2 \
  --wrist_dev /dev/video0 \
  --fix_width 640 \
  --fix_height 480 \
  --wrist_width 640 \
  --wrist_height 480 \
  --policy_hz 10 \
  --record \
  --display \
  --attack_gripper_peak_bias_rad 1.1 \
  --attack_gripper_steps 7
"""

WRIST_V4L2_BASELINE_CONTROLS = {
    "auto_exposure": 1,
    "exposure_dynamic_framerate": 0,
    "gain": 4,
    "white_balance_automatic": 0,
    "white_balance_temperature": 4000,
    "power_line_frequency": 2,
    "sharpness": 0,
    "contrast": 8,
    "gamma": 100,
    "brightness": 120,
    "backlight_compensation": 0,
    "exposure_time_absolute": 12,
}

WRIST_V4L2_TASK_OVERRIDES = {
    "tissue_retraction": {},
    "shunt_insertion": {
        "exposure_time_absolute": 4,
    },
}


def quat_norm(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    return q / max(float(np.linalg.norm(q)), eps)


def quat_canon(q: np.ndarray) -> np.ndarray:
    q = quat_norm(q)
    return -q if q[0] < 0 else q


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = quat_norm(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul_raw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Raw Hamilton product. Do NOT normalize operands here.
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float32).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float32).reshape(4)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # q rotates vector v while preserving vector magnitude.
    q = quat_norm(q)
    v = np.asarray(v, dtype=np.float32).reshape(3)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul_raw(quat_mul_raw(q, vq), quat_conj(q))[1:].astype(np.float32)


def xyzw_to_wxyz(q_xyzw: list[float]) -> np.ndarray:
    return quat_canon(np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32))


def _rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
    r6 = np.asarray(r6, dtype=np.float32).reshape(6,)
    a1 = r6[0:3]
    a2 = r6[3:6]

    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - b1 * float(np.dot(b1, a2))

    if np.linalg.norm(a2) < 1e-6:
        tmp = np.array([1, 0, 0], dtype=np.float32) if abs(b1[0]) < 0.9 else np.array([0, 1, 0], dtype=np.float32)
        a2 = tmp - b1 * float(np.dot(b1, tmp))

    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    b3 = b3 / (np.linalg.norm(b3) + 1e-8)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return quat_canon(np.array([w, x, y, z], dtype=np.float32))


def rot6d_to_quat_wxyz(r6: np.ndarray) -> np.ndarray:
    return _rotmat_to_quat_wxyz(_rot6d_to_rotmat(r6))


def quat_wxyz_to_rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    q = quat_norm(q_wxyz)
    w, x, y, z = q.tolist()

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ], dtype=np.float32)

    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)



def normalize_task_name(task: str) -> str:
    return str(task).strip().lower().replace("-", "_")


def get_wrist_v4l2_controls(task: str) -> dict:
    task_key = normalize_task_name(task)
    controls = WRIST_V4L2_BASELINE_CONTROLS.copy()
    controls.update(WRIST_V4L2_TASK_OVERRIDES.get(task_key, {}))
    return controls


def build_v4l2_cmd(device: str, controls: dict) -> list[str]:
    cmd = ["v4l2-ctl", "-d", device]
    for key, value in controls.items():
        cmd.extend(["-c", f"{key}={value}"])
    return cmd


def apply_wrist_baseline(device: str, task: str) -> dict:
    controls = get_wrist_v4l2_controls(task)
    cmd = build_v4l2_cmd(device, controls)

    print(f"[CAM] Applying wrist V4L2 config for task={task}:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)

    return controls


def center_square_resize_bgr(frame_bgr: np.ndarray, out_size: int = 224) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w > h:
        x0 = (w - h) // 2
        frame_bgr = frame_bgr[:, x0:x0 + h]
    elif h > w:
        y0 = (h - w) // 2
        frame_bgr = frame_bgr[y0:y0 + w, :]
    return cv2.resize(frame_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)


def encode_jpg_from_rgb_u8(rgb_hwc_u8: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb_hwc_u8, mode="RGB").save(buf, format="JPEG", quality=int(quality), subsampling=0)
    return buf.getvalue()


class CameraReader:
    def __init__(self, name: str, device: str, width: int, height: int, fps: int, fourcc: str):
        self.name = name
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.fourcc = fourcc

        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_time = 0.0
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {self.name}: {self.device}")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(5):
            cap.read()

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[CAM] {self.name}: {self.device}, requested {self.width}x{self.height}@{self.fps}, got {aw}x{ah}@{afps:.2f}")
        return cap

    def start(self):
        self.cap = self.open()
        self.thread.start()

    def _run(self):
        fail_count = 0
        while not self.stop_flag:
            if self.cap is None:
                try:
                    self.cap = self.open()
                except Exception as e:
                    print(f"[CAM][WARN] reopen {self.name} failed: {e}")
                    time.sleep(0.5)
                    continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                fail_count += 1
                print(f"[CAM][WARN] failed read {self.name}: {fail_count}")
                if fail_count >= 3:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    fail_count = 0
                    time.sleep(0.2)
                continue

            fail_count = 0
            with self.lock:
                self.latest_frame = frame
                self.latest_time = time.time()

    def get_latest(self) -> tuple[Optional[np.ndarray], float]:
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), float(self.latest_time)

    def stop(self):
        self.stop_flag = True
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        if self.cap is not None:
            self.cap.release()


class UDPBridgeClient:
    def __init__(self, listen_ip: str, listen_port: int, bridge_ip: str, bridge_port: int):
        self.latest_robot_state = None
        self.events = []

        self.sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx.bind((listen_ip, int(listen_port)))
        self.sock_rx.setblocking(False)

        self.sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bridge_addr = (bridge_ip, int(bridge_port))

    def send(self, pkt: dict) -> None:
        pkt["t_eval"] = time.time()
        self.sock_tx.sendto(json.dumps(pkt).encode("utf-8"), self.bridge_addr)

    def send_relative_action(self, action8: np.ndarray) -> None:
        self.send({
            "type": "relative_action",
            "action": np.asarray(action8, dtype=np.float32).reshape(8).tolist(),
        })

    def send_absolute_reset(self) -> None:
        self.send({"type": "absolute_reset"})

    def poll(self) -> None:
        while True:
            try:
                data, _ = self.sock_rx.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception as e:
                print(f"[UDP][WARN] recv failed: {e}")
                break

            try:
                pkt = json.loads(data.decode("utf-8"))
            except Exception:
                continue

            typ = pkt.get("type")
            if typ == "robot_state":
                self.latest_robot_state = pkt
            elif typ in {"pedal", "reset_event", "reset_complete", "reset_ack"}:
                self.events.append(pkt)

    def pop_events(self) -> list[dict]:
        ev = self.events
        self.events = []
        return ev


class RealEvalEpisodeRecorder:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episode_idx = self._infer_next_episode_idx()
        self.reset()

    def reset(self):
        self.images_fix = []
        self.images_wrist = []
        self.actions = []
        self.executed_actions = []
        self.policy_action = []
        self.policy_action_chunks = []
        self.policy_action_chunk_batches = []
        self.state = []
        self.timestamps = []

    def has_data(self) -> bool:
        return len(self.timestamps) > 0

    def push(
        self,
        rgb_fix,
        rgb_wrist,
        action,
        state,
        t_wall,
        policy_action=None,
        policy_action_chunk=None,
        policy_action_chunk_batch=None,
        executed_action=None,
    ):
        self.images_fix.append(np.asarray(rgb_fix, dtype=np.uint8))
        self.images_wrist.append(np.asarray(rgb_wrist, dtype=np.uint8))

        action = np.asarray(action, dtype=np.float32)
        self.actions.append(action)

        if executed_action is None:
            executed_action = action
        self.executed_actions.append(np.asarray(executed_action, dtype=np.float32))

        if policy_action is None:
            self.policy_action.append(np.full((10,), np.nan, dtype=np.float32))
        else:
            self.policy_action.append(np.asarray(policy_action, dtype=np.float32))

        if policy_action_chunk is None:
            self.policy_action_chunks.append(np.full((1, 10), np.nan, dtype=np.float32))
        else:
            self.policy_action_chunks.append(np.asarray(policy_action_chunk, dtype=np.float32))

        if policy_action_chunk_batch is None:
            self.policy_action_chunk_batches.append(np.full((1, 1, 10), np.nan, dtype=np.float32))
        else:
            self.policy_action_chunk_batches.append(np.asarray(policy_action_chunk_batch, dtype=np.float32))

        self.state.append(np.asarray(state, dtype=np.float32))
        self.timestamps.append(float(t_wall))

    def save(self, meta: dict) -> Path:
        chunk_shapes = {tuple(x.shape) for x in self.policy_action_chunks}
        if len(chunk_shapes) != 1:
            raise RuntimeError(f"Inconsistent policy_action_chunk shapes: {sorted(chunk_shapes)}")

        batch_shapes = {tuple(x.shape) for x in self.policy_action_chunk_batches}
        if len(batch_shapes) != 1:
            raise RuntimeError(f"Inconsistent policy_action_chunk_batch shapes: {sorted(batch_shapes)}")

        save_path = self.out_dir / f"episode_{self.episode_idx:05d}.npz"
        np.savez_compressed(
            save_path,
            images_fix=np.stack(self.images_fix, axis=0),
            images_wrist=np.stack(self.images_wrist, axis=0),
            actions=np.stack(self.actions, axis=0),
            executed_actions=np.stack(self.executed_actions, axis=0),
            policy_action=np.stack(self.policy_action, axis=0),
            policy_action_chunks=np.stack(self.policy_action_chunks, axis=0),
            policy_action_chunk_batches=np.stack(self.policy_action_chunk_batches, axis=0),
            states=np.stack(self.state, axis=0),
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
            meta=np.asarray(json.dumps(meta)),
        )
        self.episode_idx += 1
        self.reset()
        return save_path

    def _infer_next_episode_idx(self) -> int:
        max_idx = -1
        pat = re.compile(r"^episode_(\d+)\.npz$")
        for p in self.out_dir.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if m is not None:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1


def pose_dict_to_pos_quat_wxyz(pose: dict) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pose["pos"], dtype=np.float32).reshape(3)
    q = xyzw_to_wxyz(pose["quat_xyzw"])
    return p, q


def robot_state_to_state8_and_state10(robot_state: dict) -> tuple[np.ndarray, np.ndarray]:
    measured = robot_state.get("measured_cp")
    if measured is None:
        raise RuntimeError("robot_state missing measured_cp")

    p, q = pose_dict_to_pos_quat_wxyz(measured)

    jaw = 0.0
    jm = robot_state.get("jaw_measured")
    if jm is not None and len(jm.get("position", [])) > 0:
        jaw = float(jm["position"][0])

    state8 = np.concatenate([p, q, np.array([jaw], dtype=np.float32)], axis=0).astype(np.float32)
    state10 = np.concatenate([p, quat_wxyz_to_rot6d(q), np.array([jaw], dtype=np.float32)], axis=0).astype(np.float32)
    return state8, state10


def action10_to_action8(act10: np.ndarray) -> np.ndarray:
    act10 = np.asarray(act10, dtype=np.float32).reshape(10)
    dp = act10[0:3]
    dq = rot6d_to_quat_wxyz(act10[3:9])
    g = act10[9:10]
    return np.concatenate([dp, dq, g], axis=0).astype(np.float32)


def prepare_camera_frame(frame_bgr: np.ndarray, out_size: int) -> np.ndarray:
    bgr = center_square_resize_bgr(frame_bgr, out_size)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def put_status(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="tissue_retraction")
    ap.add_argument("--policy_addr", type=str, required=True)
    ap.add_argument("--control_addr", type=str, required=True)
    ap.add_argument("--zmq_timeout_ms", type=int, default=300)
    ap.add_argument("--jpeg_quality", type=int, default=85)

    ap.add_argument("--fix_dev", type=str, default="/dev/video2")
    ap.add_argument("--wrist_dev", type=str, default="/dev/video0")
    ap.add_argument("--fix_width", type=int, default=640)
    ap.add_argument("--fix_height", type=int, default=480)
    ap.add_argument("--wrist_width", type=int, default=640)
    ap.add_argument("--wrist_height", type=int, default=480)
    ap.add_argument("--camera_fps", type=int, default=30)
    ap.add_argument("--policy_hz", type=float, default=10.0)
    ap.add_argument("--fourcc", type=str, default="YUYV")
    ap.add_argument("--out_size", type=int, default=224)

    ap.add_argument("--udp_listen_ip", type=str, default="0.0.0.0")
    ap.add_argument("--udp_listen_port", type=int, default=15151)
    ap.add_argument("--bridge_ip", type=str, default="127.0.0.1")
    ap.add_argument("--bridge_port", type=int, default=15150)

    ap.add_argument("--record", action="store_true")
    ap.add_argument("--record_out_dir", type=str, default=None)
    ap.add_argument("--policy_chunk_batch_size", type=int, default=32)
    ap.add_argument("--reset_cooldown_s", type=float, default=2.0)
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--no_apply_wrist_baseline", action="store_true")
    ap.add_argument("--print_every", type=int, default=20)

    ap.add_argument("--control_poll_every", type=int, default=1)
    # Position attack: add 0.3 mm relative dp bias for 30 policy steps by default.
    # At 10 Hz, this lasts 3 seconds.
    ap.add_argument("--attack_position_step_m", type=float, default=0.0003)
    ap.add_argument("--attack_position_steps", type=int, default=30)
    # Gripper attack: absolute gripper command bias profile.
    # Default profile = +0.1,+0.2,+0.3,+0.4,+0.5,+0.4,+0.3,+0.2,+0.1 rad.
    ap.add_argument("--attack_gripper_peak_bias_rad", type=float, default=0.5)
    ap.add_argument("--attack_gripper_steps", type=int, default=9)
    ap.add_argument("--gripper_open_angle", type=float, default=1.43)
    # Camera lag attack: pressing v freezes both camera views and reuses
    # the trigger-time images for the next k policy steps.
    ap.add_argument("--attack_camera_lag_steps", type=int, default=10)

    args = ap.parse_args()

    if args.record_out_dir is None:
        record_out_dir = Path("tasks") / args.task / "prl_data" / "raw"
    else:
        record_out_dir = Path(args.record_out_dir)

    wrist_v4l2_controls = get_wrist_v4l2_controls(args.task)
    if not args.no_apply_wrist_baseline:
        wrist_v4l2_controls = apply_wrist_baseline(args.wrist_dev, args.task)
        time.sleep(0.2)

    fix_cam = CameraReader("fix", args.fix_dev, args.fix_width, args.fix_height, args.camera_fps, args.fourcc)
    wrist_cam = CameraReader("wrist", args.wrist_dev, args.wrist_width, args.wrist_height, args.camera_fps, args.fourcc)
    fix_cam.start()
    wrist_cam.start()

    bridge = UDPBridgeClient(args.udp_listen_ip, args.udp_listen_port, args.bridge_ip, args.bridge_port)

    policy_client = ZmqPolicyClient(args.policy_addr, timeout_ms=args.zmq_timeout_ms)
    control_client = ZmqControlClient(args.control_addr, timeout_ms=max(100, args.zmq_timeout_ms))

    recorder = RealEvalEpisodeRecorder(record_out_dir) if args.record else None

    class EvalState:
        WAITING = "WAITING"      # robot reset/safe, waiting for camera/clutch/coag command
        RUNNING = "RUNNING"      # policy streaming + recording
        RESETTING = "RESETTING"  # bridge is resetting robot; no policy streaming

    eval_state = EvalState.WAITING
    reset_block_until = 0.0
    step_i = 0
    t0 = time.time()
    next_policy_t = time.time()

    ATTACK_DIRS_TIP = {
        "ATTACK_W": np.array([0.0, +1.0, 0.0], dtype=np.float32),
        "ATTACK_S": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "ATTACK_A": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "ATTACK_D": np.array([+1.0, 0.0, 0.0], dtype=np.float32),
    }
    active_position_delta_base = np.zeros(3, dtype=np.float32)
    active_position_steps_left = 0
    active_gripper_biases: list[float] = []
    active_camera_lag_fix_rgb: Optional[np.ndarray] = None
    active_camera_lag_wrist_rgb: Optional[np.ndarray] = None
    active_camera_lag_steps_left = 0
    attack_events = []

    meta_base = {
        "task": args.task,
        "real_world": True,
        "eval_rollout": True,
        "state_frame": "PSM base / dVRK Cartesian frame",
        "action_translation_frame": "PSM base / dVRK Cartesian frame",
        "action_rotation_frame": "local EE-tip frame",
        "fix_dev": args.fix_dev,
        "wrist_dev": args.wrist_dev,
        "fix_width": int(args.fix_width),
        "fix_height": int(args.fix_height),
        "wrist_width": int(args.wrist_width),
        "wrist_height": int(args.wrist_height),
        "camera_fps": int(args.camera_fps),
        "wrist_v4l2_controls": wrist_v4l2_controls,
        "policy_hz": float(args.policy_hz),
        "image_size": int(args.out_size),
        "jpeg_quality": int(args.jpeg_quality),
        "policy_addr": args.policy_addr,
        "control_addr": args.control_addr,
        "control": "foot_pedals",
        "episode_start": "coag_press",
        "episode_end": "clutch_reset",
        "save_command": "camera_release_in_WAITING",
        "discard_command": "clutch_or_coag_in_WAITING",
    }

    def has_pending_episode() -> bool:
        return bool(args.record and recorder is not None and recorder.has_data())

    def save_pending_episode(reason: str) -> None:
        nonlocal recorder
        if not args.record or recorder is None:
            print("[REC] Save requested, but --record is off.")
            return
        if not recorder.has_data():
            print("[REC] Save requested, but no pending episode.")
            return
        path = recorder.save(meta={
            **meta_base,
            "end_reason": reason,
            "attacks": attack_events,
        })
        attack_events.clear()
        print(f"[REC] Saved pending episode: {path}")

    def clear_camera_lag() -> None:
        nonlocal active_camera_lag_fix_rgb, active_camera_lag_wrist_rgb, active_camera_lag_steps_left
        active_camera_lag_fix_rgb = None
        active_camera_lag_wrist_rgb = None
        active_camera_lag_steps_left = 0

    def discard_pending_episode(reason: str) -> None:
        nonlocal active_position_steps_left, active_position_delta_base, active_gripper_biases
        if args.record and recorder is not None and recorder.has_data():
            print(f"[REC] Discarded pending episode. reason={reason}")
        if recorder is not None:
            recorder.reset()

        active_position_steps_left = 0
        active_position_delta_base = np.zeros(3, dtype=np.float32)
        active_gripper_biases = []
        clear_camera_lag()
        attack_events.clear()

    def reset_policy(reason: str) -> None:
        try:
            ok = control_client.reset_policy()
            print(f"[POLICY] reset_policy reason={reason}: {ok}")
        except Exception as e:
            print(f"[POLICY][WARN] reset_policy failed reason={reason}: {e}")

    def enter_waiting(reason: str) -> None:
        nonlocal eval_state, reset_block_until, step_i, t0
        eval_state = EvalState.WAITING
        reset_block_until = time.time() + float(args.reset_cooldown_s)
        step_i = 0
        t0 = time.time()
        print(f"[STATE] -> WAITING reason={reason}. Pending episode={has_pending_episode()}")

    def enter_resetting(reason: str) -> None:
        nonlocal eval_state, reset_block_until
        nonlocal active_position_steps_left, active_position_delta_base, active_gripper_biases

        active_position_steps_left = 0
        active_position_delta_base = np.zeros(3, dtype=np.float32)
        active_gripper_biases = []
        clear_camera_lag()

        eval_state = EvalState.RESETTING
        reset_block_until = time.time() + float(args.reset_cooldown_s)
        print(f"[STATE] -> RESETTING reason={reason}. Pending episode={has_pending_episode()}")

    def enter_running(reason: str) -> None:
        nonlocal eval_state, step_i, t0, next_policy_t
        discard_pending_episode(reason=f"start_new_rollout_{reason}")
        reset_policy(reason=f"start_running_{reason}")
        eval_state = EvalState.RUNNING
        step_i = 0
        t0 = time.time()
        next_policy_t = time.time()
        print("[STATE] -> RUNNING. Policy streaming and episode recording started.")

    def read_prepared_rgb_pair() -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        fix_frame, _ = fix_cam.get_latest()
        wrist_frame, _ = wrist_cam.get_latest()
        if fix_frame is None or wrist_frame is None:
            return None, None
        fix_rgb = prepare_camera_frame(fix_frame, args.out_size)
        wrist_rgb = prepare_camera_frame(wrist_frame, args.out_size)
        return fix_rgb, wrist_rgb

    def get_effective_rgb_pair() -> tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        if (
            active_camera_lag_steps_left > 0
            and active_camera_lag_fix_rgb is not None
            and active_camera_lag_wrist_rgb is not None
        ):
            return active_camera_lag_fix_rgb.copy(), active_camera_lag_wrist_rgb.copy(), True

        if active_camera_lag_steps_left > 0:
            clear_camera_lag()

        fix_rgb, wrist_rgb = read_prepared_rgb_pair()
        return fix_rgb, wrist_rgb, False

    def consume_camera_lag_policy_step() -> None:
        nonlocal active_camera_lag_steps_left
        if active_camera_lag_steps_left <= 0:
            return
        active_camera_lag_steps_left -= 1
        if active_camera_lag_steps_left <= 0:
            clear_camera_lag()

    def start_camera_lag_attack() -> None:
        nonlocal active_camera_lag_fix_rgb, active_camera_lag_wrist_rgb, active_camera_lag_steps_left
        k = max(1, int(args.attack_camera_lag_steps))
        fix_rgb, wrist_rgb = read_prepared_rgb_pair()
        if fix_rgb is None or wrist_rgb is None:
            print("[ATTACK][WARN] V: cannot start camera lag; missing camera frame.")
            return

        active_camera_lag_fix_rgb = fix_rgb.copy()
        active_camera_lag_wrist_rgb = wrist_rgb.copy()
        active_camera_lag_steps_left = k

        attack_events.append({
            "step": int(step_i),
            "type": "camera_lag_both_views",
            "cmd": "ATTACK_V",
            "num_steps": int(k),
            "duration_s": float(k) / float(args.policy_hz),
            "views": ["fix_camera", "wrist_camera"],
            "note": "Trigger-time images are reused for policy input, recording, and display for the next k policy steps.",
        })

        print(
            f"[ATTACK] V: freezing both camera views for {k} policy steps "
            f"({float(k) / float(args.policy_hz):.2f}s at {float(args.policy_hz):.2f} Hz)"
        )

    print("[INFO] real_eval started.")
    print("[INFO] Pedal control:")
    print("       WAITING + camera release: save pending episode")
    print("       WAITING + clutch press/release: discard pending episode and reset robot")
    print("       WAITING + coag press: discard pending episode and start new rollout")
    print("       RUNNING + clutch press/release: stop rollout, keep episode pending, reset robot")
    print("       RESETTING: wait for bridge reset_complete")
    print(f"[STATE] -> {eval_state}")

    try:
        while True:
            bridge.poll()

            for ev in bridge.pop_events():
                typ = ev.get("type")

                if typ == "pedal":
                    pedal = ev.get("pedal")
                    event = ev.get("event")
                    reset_triggered = bool(ev.get("reset_triggered", False))

                    # Use release for camera to avoid double-trigger feel.
                    if eval_state == EvalState.WAITING:
                        if pedal == "camera" and event == "release":
                            save_pending_episode(reason="camera_save")
                            continue

                        if pedal == "coag" and event == "press":
                            enter_running(reason="coag_press")
                            continue

                        if pedal == "clutch" and event in {"press", "release"}:
                            # Bridge should trigger physical reset; we discard pending.
                            discard_pending_episode(reason=f"waiting_clutch_{event}")
                            if reset_triggered:
                                enter_resetting(reason=f"clutch_{event}")
                            else:
                                # Fallback: explicitly request bridge reset if this pedal event did not trigger it.
                                bridge.send_absolute_reset()
                                enter_resetting(reason=f"manual_absolute_reset_from_clutch_{event}")
                            continue

                    elif eval_state == EvalState.RUNNING:
                        if pedal == "clutch" and event in {"press", "release"}:
                            # End current rollout immediately. Keep recorder buffer as pending.
                            print(f"[RUNNING] clutch_{event}: stopping rollout; current episode is pending.")
                            if reset_triggered:
                                enter_resetting(reason=f"running_clutch_{event}")
                            else:
                                bridge.send_absolute_reset()
                                enter_resetting(reason=f"running_manual_absolute_reset_from_clutch_{event}")
                            continue

                        # Ignore camera/coag during active rollout.
                        continue

                    elif eval_state == EvalState.RESETTING:
                        # Ignore pedal commands while bridge is resetting.
                        continue

                elif typ == "reset_event":
                    # Bridge says a reset sequence has begun.
                    if eval_state == EvalState.RUNNING:
                        print("[RESET_EVENT] Reset began during RUNNING; stopping rollout and keeping pending episode.")
                    elif eval_state == EvalState.WAITING:
                        print("[RESET_EVENT] Reset began from WAITING.")
                    enter_resetting(reason=str(ev.get("reason", "bridge_reset_event")))
                    continue

                elif typ == "reset_complete":
                    # Bridge says robot is back at initialization pose.
                    enter_waiting(reason=str(ev.get("reason", "bridge_reset_complete")))
                    continue

                elif typ == "reset_ack":
                    # Optional ack from bridge for UDP absolute_reset; no state change required.
                    print(f"[RESET_ACK] {ev.get('reason', '')}")
                    continue

            if eval_state == EvalState.RUNNING and step_i % max(1, int(args.control_poll_every)) == 0:
                try:
                    ctrl_cmd = control_client.poll_command()
                except Exception as e:
                    ctrl_cmd = None
                    if step_i % max(1, args.print_every) == 0:
                        print(f"[WARN] control poll failed: {e}")

                if ctrl_cmd in ATTACK_DIRS_TIP:
                    robot_state = bridge.latest_robot_state
                    if robot_state is not None:
                        state8, _ = robot_state_to_state8_and_state10(robot_state)
                        q_tip_base = state8[3:7]  # wxyz

                        delta_tip = ATTACK_DIRS_TIP[ctrl_cmd] * float(args.attack_position_step_m)
                        active_position_delta_base = quat_rotate_np(q_tip_base, delta_tip)
                        active_position_steps_left = int(args.attack_position_steps)

                        attack_events.append({
                            "step": int(step_i),
                            "type": "tip_frame_position_bias",
                            "cmd": ctrl_cmd,
                            "per_step_m": float(args.attack_position_step_m),
                            "num_steps": int(args.attack_position_steps),
                            "duration_s": float(args.attack_position_steps) / float(args.policy_hz),
                            "delta_tip_per_step": delta_tip.tolist(),
                            "delta_base_per_step": active_position_delta_base.tolist(),
                            "note": "This is a relative dp bias added each policy step.",
                        })

                        print(
                            f"[ATTACK] {ctrl_cmd}: +{1000.0 * args.attack_position_step_m:.3f} mm/step "
                            f"for {args.attack_position_steps} steps"
                        )

                elif ctrl_cmd == "ATTACK_V":
                    start_camera_lag_attack()

                elif ctrl_cmd == "ATTACK_F":
                    # +0.1,+0.2,+0.3,+0.2,+0.1 for default peak 0.3 and 5 steps.
                    if int(args.attack_gripper_steps) == 5:
                        ratios = np.array([1/3, 2/3, 1.0, 2/3, 1/3], dtype=np.float32)
                    else:
                        n = max(1, int(args.attack_gripper_steps))
                        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
                        ratios = 1.0 - np.abs(2.0 * x - 1.0)
                        ratios = ratios / max(float(ratios.max()), 1e-8)

                    active_gripper_biases = [
                        float(r) * float(args.attack_gripper_peak_bias_rad)
                        for r in ratios
                    ]

                    attack_events.append({
                        "step": int(step_i),
                        "type": "gripper_open_bias",
                        "peak_bias_rad": float(args.attack_gripper_peak_bias_rad),
                        "num_steps": int(len(active_gripper_biases)),
                        "duration_s": float(len(active_gripper_biases)) / float(args.policy_hz),
                        "bias_profile": active_gripper_biases,
                        "gripper_open_angle": float(args.gripper_open_angle),
                        "note": "Bias is added to absolute policy gripper command and clamped to open angle.",
                    })

                    print(f"[ATTACK] F: gripper bias profile={active_gripper_biases}")

                elif ctrl_cmd in {"RESET", "RESET_NOSAVE"}:
                    print(f"[INFO] Ignoring eval_server reset command in real eval: {ctrl_cmd}")

            now = time.time()

            if args.display:
                fix_rgb, wrist_rgb, using_camera_lag = get_effective_rgb_pair()
                if fix_rgb is not None and wrist_rgb is not None:
                    lag_tag = f" | CAM_LAG {active_camera_lag_steps_left}" if using_camera_lag else ""
                    panel = np.concatenate([
                        cv2.cvtColor(put_status(fix_rgb, f"fix | {eval_state}{lag_tag}"), cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(put_status(wrist_rgb, f"wrist{lag_tag}"), cv2.COLOR_RGB2BGR),
                    ], axis=1)
                    cv2.imshow("real eval: fix | wrist", panel)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

            # No policy streaming unless RUNNING.
            if eval_state != EvalState.RUNNING:
                time.sleep(0.001)
                continue

            if now < reset_block_until:
                time.sleep(0.001)
                continue

            if now < next_policy_t:
                time.sleep(0.001)
                continue
            next_policy_t = now + 1.0 / max(float(args.policy_hz), 1e-6)

            robot_state = bridge.latest_robot_state
            if robot_state is None:
                print("[WARN] No robot_state yet.")
                continue

            fix_rgb, wrist_rgb, using_camera_lag = get_effective_rgb_pair()
            if fix_rgb is None or wrist_rgb is None:
                print("[WARN] Missing camera frame.")
                continue

            try:
                state8, state10 = robot_state_to_state8_and_state10(robot_state)
            except Exception as e:
                print(f"[WARN] Bad robot state: {e}")
                continue

            act10 = None
            act_chunk10 = None
            act_chunk_batch10 = None

            try:
                jpg_fix = encode_jpg_from_rgb_u8(fix_rgb, quality=args.jpeg_quality)
                jpg_wrist = encode_jpg_from_rgb_u8(wrist_rgb, quality=args.jpeg_quality)

                if args.record and int(args.policy_chunk_batch_size) > 0:
                    act10, act_chunk10, act_chunk_batch10 = policy_client.predict_action_chunk_batch(
                        jpg_fix,
                        jpg_wrist,
                        state10,
                        num_samples=int(args.policy_chunk_batch_size),
                    )
                else:
                    act10, act_chunk10 = policy_client.predict_action_chunk(
                        jpg_fix,
                        jpg_wrist,
                        state10,
                    )
                    act_chunk_batch10 = None

                if np.asarray(act10).reshape(-1).shape[0] != 10:
                    raise RuntimeError(f"Expected act10 dim 10, got {np.asarray(act10).shape}")

                nominal_action8 = action10_to_action8(act10)
                executed_action8 = nominal_action8.copy()

                # Position attack: action is relative, so add only a tiny per-step dp bias.
                if active_position_steps_left > 0:
                    executed_action8[0:3] += active_position_delta_base.astype(np.float32)
                    active_position_steps_left -= 1

                # Gripper attack: gripper command is absolute, so add bias directly.
                if active_gripper_biases:
                    g_bias = float(active_gripper_biases.pop(0))
                    executed_action8[7] = min(
                        float(args.gripper_open_angle),
                        float(executed_action8[7]) + g_bias,
                    )

                bridge.send_relative_action(executed_action8)

                if args.record and recorder is not None:
                    recorder.push(
                        rgb_fix=fix_rgb,
                        rgb_wrist=wrist_rgb,
                        action=nominal_action8,
                        executed_action=executed_action8,
                        policy_action=act10,
                        policy_action_chunk=act_chunk10,
                        policy_action_chunk_batch=act_chunk_batch10,
                        state=state8,
                        t_wall=now,
                    )

                if using_camera_lag:
                    consume_camera_lag_policy_step()

            except Exception as e:
                print(f"[WARN] policy/action failed @step {step_i}: {e}")

            step_i += 1
            if step_i % max(1, int(args.print_every)) == 0:
                fps = step_i / max(time.time() - t0, 1e-9)
                print(
                    f"[INFO] state={eval_state}, step={step_i}, "
                    f"approx_policy_fps={fps:.2f}, pending={has_pending_episode()}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt")
    finally:
        if args.record and recorder is not None and recorder.has_data():
            print("[INFO] Pending unsaved eval episode exists; not auto-saving on Ctrl+C.")
        policy_client.close()
        control_client.close()
        fix_cam.stop()
        wrist_cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] real_eval stopped.")


if __name__ == "__main__":
    main()