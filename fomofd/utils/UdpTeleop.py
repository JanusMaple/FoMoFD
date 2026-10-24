# ----------------------------
# UDP Teleop
# ----------------------------

import json
import time
import torch
import socket
from typing import Optional

class UdpTeleop:
    """
    RX expects incoming JSON packets like:
      {"type":"psm_cmd","pos":[x,y,z],"quat_xyzw":[x,y,z,w],"gripper":0.0,"reset":false}

    TX sends:
      {"type":"psm_state","t":..., "pos":[x,y,z], "quat_xyzw":[x,y,z,w]}
    """

    def __init__(
        self,
        listen_ip: str,
        listen_port: int,
        state_dst_ip: str = "127.0.0.1",
        state_dst_port: int = 15150,
        state_hz: float = 60.0,
        default_EE_pose = [[0.83, 0.0, -0.05, -0.5, 0.5, 0.5, -0.5, 0.5]]
    ):
        # --- RX (commands) ---
        self.sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_cmd.bind((listen_ip, int(listen_port)))
        self.sock_cmd.setblocking(False)

        # --- TX (state) ---
        self.state_dst = (state_dst_ip, int(state_dst_port))
        self.sock_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._state_period = 0.0
        if state_hz is not None and float(state_hz) > 0:
            self._state_period = 1.0 / float(state_hz)

        # --- Whether has received udp message ---
        self.msg_received = False

        # --- Reset Teleop Interface ---
        self.reset()

        # --- Default EE pose to use when no teleop command received ---
        self.default_EE_pose = default_EE_pose

    def reset(self) -> None:
        self._t_last_state = 0.0

        self.latest_pos = None
        self.latest_quat_xyzw = None
        self.latest_gripper = 0.5
        self.latest_reset = False
        self.latest_record_cmd = 0

        self.msg_received = False

        while True:     # Drain UDP buffer
            try:
                self.sock_cmd.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception:
                break

    def poll_latest(self) -> bool:
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
            return False
        
        if bool(latest.get("reset", False)):
            self.latest_reset = True
        
        self.latest_record_cmd = int(latest.get("record_cmd"))
        if self.latest_record_cmd == 0 or self.latest_record_cmd == 5:  # No record command or resume recording
            p = latest.get("pos", None)
            q = latest.get("quat_xyzw", None)
            if p is not None and q is not None:
                self.latest_pos = p
                self.latest_quat_xyzw = q

            if "gripper" in latest:
                self.latest_gripper = float(latest["gripper"])

        return True

    def consume_reset_flag(self) -> bool:
        r = self.latest_reset
        self.latest_reset = False
        return r
    
    def consume_record_cmd(self) -> int:
        cmd = self.latest_record_cmd
        self.latest_record_cmd = 0
        return cmd

    def make_action_wxyz(self, device: torch.device) -> Optional[torch.Tensor]:
        """
        Return [1,8] tensor: [x,y,z,qw,qx,qy,qz,gripper] in Isaac convention.
        """
        if self.latest_pos is None or self.latest_quat_xyzw is None:
            return None

        x, y, z = map(float, self.latest_pos)
        qx, qy, qz, qw = map(float, self.latest_quat_xyzw)  # ROS xyzw
        gr = float(self.latest_gripper)
        return torch.tensor([[x, y, z, qw, qx, qy, qz, gr]], dtype=torch.float32, device=device)

    def maybe_send_state_from_isaac(self,
                                    ee_pos_b: torch.Tensor,
                                    ee_quat_b_wxyz: torch.Tensor,
                                    force: bool = False) -> None:
        """
        ee_pos_b: (3,) or (1,3) torch
        ee_quat_b_wxyz: (4,) or (1,4) torch in Isaac wxyz
        Sends ROS-style quat_xyzw.
        """
        now = time.time()
        if (not force) and self._state_period > 0 and (now - self._t_last_state) < self._state_period:
            return
        self._t_last_state = now

        if ee_pos_b.ndim == 2:
            ee_pos_b = ee_pos_b[0]
        if ee_quat_b_wxyz.ndim == 2:
            ee_quat_b_wxyz = ee_quat_b_wxyz[0]

        pos = [float(ee_pos_b[0]), float(ee_pos_b[1]), float(ee_pos_b[2])]
        quat_xyzw = [
            float(ee_quat_b_wxyz[1]),
            float(ee_quat_b_wxyz[2]),
            float(ee_quat_b_wxyz[3]),
            float(ee_quat_b_wxyz[0]),
        ]
        msg = {"type": "psm_state", "t": now, "pos": pos, "quat_xyzw": quat_xyzw}
        self.sock_state.sendto(json.dumps(msg).encode("utf-8"), self.state_dst)