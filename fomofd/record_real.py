"""
Example Usage:
  python record_real.py \
    --task tissue_retraction \
    --fix_dev /dev/video2 \
    --wrist_dev /dev/video0 \
    --fix_width 640 \
    --fix_height 480 \
    --wrist_width 640 \
    --wrist_height 480 \
    --record_fps 10 \
    --display
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


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


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = quat_norm(q1)
    w2, x2, y2, z2 = quat_norm(q2)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quat_inv(q: np.ndarray) -> np.ndarray:
    return quat_conj(quat_norm(q))


def xyzw_to_wxyz(q_xyzw: list[float]) -> np.ndarray:
    return quat_canon(np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32))


def pose_dict_to_pos_quat_wxyz(pose: dict) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pose["pos"], dtype=np.float32).reshape(3)
    q = xyzw_to_wxyz(pose["quat_xyzw"])
    return p, q


def center_square_resize_bgr(frame_bgr: np.ndarray, out_size: int = 224) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w > h:
        x0 = (w - h) // 2
        frame_bgr = frame_bgr[:, x0:x0 + h]
    elif h > w:
        y0 = (h - w) // 2
        frame_bgr = frame_bgr[y0:y0 + w, :]
    return cv2.resize(frame_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)


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


class CameraReader:
    def __init__(
        self,
        name: str,
        device: str,
        width: int,
        height: int,
        fps: int,
        fourcc: str = "YUYV",
    ):
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
            raise RuntimeError(f"Failed to open {self.name} camera: {self.device}")

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


class RealEpisodeRecorder:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episode_idx = self._infer_next_episode_idx()
        self.reset()

    def reset(self):
        self.images_fix = []
        self.images_wrist = []
        self.actions = []
        self.states = []
        self.timestamps = []

    def has_data(self) -> bool:
        return len(self.timestamps) > 0

    def push(self, rgb_fix: np.ndarray, rgb_wrist: np.ndarray, action: np.ndarray, state: np.ndarray, t_wall: float):
        self.images_fix.append(np.asarray(rgb_fix, dtype=np.uint8))
        self.images_wrist.append(np.asarray(rgb_wrist, dtype=np.uint8))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.states.append(np.asarray(state, dtype=np.float32))
        self.timestamps.append(float(t_wall))

    def save(self, meta: dict) -> Path:
        if not self.has_data():
            raise RuntimeError("No data to save")

        save_path = self.out_dir / f"episode_{self.episode_idx:05d}.npz"
        np.savez_compressed(
            save_path,
            images_fix=np.stack(self.images_fix, axis=0),
            images_wrist=np.stack(self.images_wrist, axis=0),
            actions=np.stack(self.actions, axis=0),
            states=np.stack(self.states, axis=0),
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
        pkt["t_record"] = time.time()
        self.sock_tx.sendto(json.dumps(pkt).encode("utf-8"), self.bridge_addr)

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
            elif typ in {"pedal", "reset_event"}:
                self.events.append(pkt)

    def pop_events(self) -> list[dict]:
        ev = self.events
        self.events = []
        return ev


class RecordStateMachine:
    IDLE = "IDLE"
    ARMED = "ARMED"
    RECORDING = "RECORDING"
    HELD = "HELD"

    def __init__(self):
        self.state = self.IDLE

    def on_camera_release(self, recorder: RealEpisodeRecorder, meta_base: dict):
        if self.state == self.IDLE:
            recorder.reset()
            self.state = self.ARMED
            print("[REC] Armed. Press coag to start actual recording.")
            return None

        if self.state == self.HELD:
            if recorder.has_data():
                path = recorder.save(meta={**meta_base, "end_reason": "camera_save"})
                self.state = self.IDLE
                print(f"[REC] Saved {path}")
                return path
            recorder.reset()
            self.state = self.IDLE
            print("[REC] No data; returned to IDLE.")
            return None

        if self.state == self.ARMED:
            recorder.reset()
            self.state = self.IDLE
            print("[REC] Disarmed; no data saved.")
            return None

        print(f"[REC] Ignored camera release in state={self.state}")
        return None

    def on_coag_press(self):
        if self.state in {self.ARMED, self.HELD}:
            self.state = self.RECORDING
            print("[REC] Recording...")
        else:
            print(f"[REC] Ignored coag press in state={self.state}")

    def on_coag_release(self):
        if self.state == self.RECORDING:
            self.state = self.HELD
            print("[REC] Held. Press camera to save, or clutch-reset to discard.")
        else:
            print(f"[REC] Ignored coag release in state={self.state}")

    def discard(self, recorder: RealEpisodeRecorder, reason: str):
        if recorder.has_data() or self.state != self.IDLE:
            print(f"[REC] Discarded unsaved episode. reason={reason}")
        recorder.reset()
        self.state = self.IDLE


def build_state_action(robot_state: dict, prev_setpoint: Optional[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    measured = robot_state.get("measured_cp")
    setpoint = robot_state.get("setpoint_cp") or measured

    if measured is None or setpoint is None:
        raise RuntimeError("Missing measured/setpoint pose")

    p_meas, q_meas = pose_dict_to_pos_quat_wxyz(measured)
    p_set, q_set = pose_dict_to_pos_quat_wxyz(setpoint)

    jaw_measured = 0.0
    jm = robot_state.get("jaw_measured")
    if jm is not None and len(jm.get("position", [])) > 0:
        jaw_measured = float(jm["position"][0])

    jaw_action = jaw_measured
    js = robot_state.get("jaw_setpoint")
    if js is not None and len(js.get("position", [])) > 0:
        jaw_action = float(js["position"][0])

    state8 = np.concatenate([
        p_meas.astype(np.float32),
        q_meas.astype(np.float32),
        np.array([jaw_measured], dtype=np.float32),
    ], axis=0)

    if prev_setpoint is None:
        dp = np.zeros(3, dtype=np.float32)
        dq = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        p_prev, q_prev = prev_setpoint
        dp = (p_set - p_prev).astype(np.float32)
        dq = quat_canon(quat_mul(quat_inv(q_prev), q_set))

    action8 = np.concatenate([
        dp.astype(np.float32),
        dq.astype(np.float32),
        np.array([jaw_action], dtype=np.float32),
    ], axis=0)

    return state8, action8, (p_set.copy(), q_set.copy())


def put_status(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="needle_pickup")
    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument("--fix_dev", type=str, default="/dev/video2")
    parser.add_argument("--wrist_dev", type=str, default="/dev/video0")
    parser.add_argument("--fix_width", type=int, default=640)
    parser.add_argument("--fix_height", type=int, default=480)
    parser.add_argument("--wrist_width", type=int, default=640)
    parser.add_argument("--wrist_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=int, default=30)
    parser.add_argument("--record_fps", type=float, default=10.0)
    parser.add_argument("--fourcc", type=str, default="YUYV")
    parser.add_argument("--out_size", type=int, default=224)

    parser.add_argument("--udp_listen_ip", type=str, default="0.0.0.0")
    parser.add_argument("--udp_listen_port", type=int, default=15151)
    parser.add_argument("--bridge_ip", type=str, default="127.0.0.1")
    parser.add_argument("--bridge_port", type=int, default=15150)

    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no_apply_wrist_baseline", action="store_true")

    args = parser.parse_args()

    if args.out_dir is None:
        out_dir = Path("tasks") / args.task / "data" / "raw"
    else:
        out_dir = Path(args.out_dir)

    wrist_v4l2_controls = get_wrist_v4l2_controls(args.task)

    if not args.no_apply_wrist_baseline:
        wrist_v4l2_controls = apply_wrist_baseline(args.wrist_dev, args.task)
        time.sleep(0.2)

    fix_cam = CameraReader("fix", args.fix_dev, args.fix_width, args.fix_height, args.camera_fps, args.fourcc)
    wrist_cam = CameraReader("wrist", args.wrist_dev, args.wrist_width, args.wrist_height, args.camera_fps, args.fourcc)

    fix_cam.start()
    wrist_cam.start()

    bridge = UDPBridgeClient(
        listen_ip=args.udp_listen_ip,
        listen_port=args.udp_listen_port,
        bridge_ip=args.bridge_ip,
        bridge_port=args.bridge_port,
    )

    recorder = RealEpisodeRecorder(out_dir)
    sm = RecordStateMachine()

    prev_setpoint_for_action = None
    next_sample_t = time.time()

    meta_base = {
        "task": args.task,
        "real_world": True,
        "state_frame": "PSM base / dVRK Cartesian frame",
        "action_translation_frame": "PSM base / dVRK Cartesian frame",
        "action_rotation_frame": "local EE-tip frame",
        "fix_dev": args.fix_dev,
        "wrist_dev": args.wrist_dev,
        "record_fps": float(args.record_fps),
        "camera_fps": int(args.camera_fps),
        "image_size": int(args.out_size),
        "wrist_v4l2_controls": wrist_v4l2_controls,
    }

    print(f"[INFO] record_real started. out_dir={out_dir}")
    print("[INFO] Workflow: camera release=arm/save, coag press=start, coag release=hold, clutch release=discard/reset.")

    try:
        while True:
            bridge.poll()

            for ev in bridge.pop_events():
                typ = ev.get("type")

                if typ == "pedal":
                    pedal = ev.get("pedal")
                    event = ev.get("event")
                    reset_triggered = bool(ev.get("reset_triggered", False))

                    if pedal == "camera" and event == "release":
                        sm.on_camera_release(recorder, meta_base)
                        if sm.state == sm.IDLE:
                            prev_setpoint_for_action = None

                    elif pedal == "coag" and event == "press":
                        sm.on_coag_press()
                        if sm.state == sm.RECORDING and prev_setpoint_for_action is None:
                            prev_setpoint_for_action = None

                    elif pedal == "coag" and event == "release":
                        sm.on_coag_release()

                    elif pedal == "clutch" and event == "release":
                        sm.discard(recorder, reason="clutch_release")
                        prev_setpoint_for_action = None
                        if reset_triggered:
                            print("[REC] Bridge triggered PSM initialization.")

                elif typ == "reset_event":
                    sm.discard(recorder, reason=str(ev.get("reason", "reset_event")))
                    prev_setpoint_for_action = None

            now = time.time()

            if args.display:
                fix_frame, _ = fix_cam.get_latest()
                wrist_frame, _ = wrist_cam.get_latest()
                if fix_frame is not None and wrist_frame is not None:
                    fix_224_bgr = center_square_resize_bgr(fix_frame, args.out_size)
                    wrist_224_bgr = center_square_resize_bgr(wrist_frame, args.out_size)
                    panel = np.concatenate([
                        put_status(fix_224_bgr, f"fix | {sm.state}"),
                        put_status(wrist_224_bgr, "wrist"),
                    ], axis=1)
                    cv2.imshow("fix | wrist", panel)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

            if sm.state == sm.RECORDING and now >= next_sample_t:
                next_sample_t = now + 1.0 / max(float(args.record_fps), 1e-6)

                robot_state = bridge.latest_robot_state
                if robot_state is None:
                    print("[REC][WARN] No robot_state yet; skipping sample.")
                    continue

                fix_frame, _ = fix_cam.get_latest()
                wrist_frame, _ = wrist_cam.get_latest()
                if fix_frame is None or wrist_frame is None:
                    print("[REC][WARN] Missing camera frame; skipping sample.")
                    continue

                fix_bgr_224 = center_square_resize_bgr(fix_frame, args.out_size)
                wrist_bgr_224 = center_square_resize_bgr(wrist_frame, args.out_size)

                # Save RGB HWC uint8, matching simulation camera convention.
                fix_rgb_224 = cv2.cvtColor(fix_bgr_224, cv2.COLOR_BGR2RGB)
                wrist_rgb_224 = cv2.cvtColor(wrist_bgr_224, cv2.COLOR_BGR2RGB)

                try:
                    state8, action8, curr_setpoint = build_state_action(robot_state, prev_setpoint_for_action)
                except Exception as e:
                    print(f"[REC][WARN] Could not build state/action: {e}")
                    continue

                recorder.push(
                    rgb_fix=fix_rgb_224,
                    rgb_wrist=wrist_rgb_224,
                    action=action8,
                    state=state8,
                    t_wall=now,
                )
                prev_setpoint_for_action = curr_setpoint

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt")
    finally:
        if recorder.has_data():
            print("[INFO] Unsaved data exists; not auto-saving. Press camera to save during normal workflow.")
        fix_cam.stop()
        wrist_cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] record_real stopped.")


if __name__ == "__main__":
    main()