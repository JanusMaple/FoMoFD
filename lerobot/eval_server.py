"""
fomofd/lerobot/eval_server.py

The UDP evaluation server for running a pretrained policy.

Example:
    python lerobot/eval_server.py --task needle_pickup --method act_dinov2 --including_gripper
or:
    python lerobot/eval_server.py \
        --task tissue_retraction \
        --method act_dinov2 \
        --including_gripper \
        --action_bind tcp://0.0.0.0:5555 \
        --control_bind tcp://0.0.0.0:5556
or:
    python eval_server.py \
        --task needle_pickup \
        --method act_dinov2_pretrained_tokens \
        --including_gripper \
        --tokenizer_ckpt ../fomofd/tasks/needle_pickup/model/dino_act_wm_tokenizer_only/best_tokenizer.pt
"""

from __future__ import annotations

import argparse
import io
import json
import inspect
import importlib
import importlib.metadata as md
from pathlib import Path

import numpy as np
import torch
import zmq
from PIL import Image
from collections import deque

import atexit
import signal
import sys
import termios
import threading
import tty

from lerobot.policies.pretrained import PreTrainedPolicy    # type: ignore

import torch.nn.functional as F

"""
utils: path solver
"""
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _find_pretrained_dir(task: str, method: str, step: str = "last") -> Path:
    rr = _repo_root()
    base = rr / "fomofd" / "tasks" / task / "model" / method / "checkpoints"
    if step == "last":
        cand = base / "last" / "pretrained_model"
        if cand.exists():
            return cand
        # fallback: numeric max
        if base.exists():
            nums = sorted([p for p in base.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
            if nums:
                cand2 = nums[-1] / "pretrained_model"
                if cand2.exists():
                    return cand2
    else:
        cand = base / step / "pretrained_model"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Cannot find pretrained_model under {base} (step={step})")

"""
utils: jpg image handler
"""
def _decode_jpg_to_chw_u8(b: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)      # HWC
    return np.transpose(arr, (2, 0, 1))        # CHW

@torch.inference_mode()
def _extract_pretrained_tokens_online(
    img_fix_01: torch.Tensor,     # (1,3,H,W), float in [0,1]
    img_wrist_01: torch.Tensor,   # (1,3,H,W), float in [0,1]
    dinov2_extractor: torch.nn.Module,
    pretrained_tokenizer: torch.nn.Module,
    dino_norm_mean: torch.Tensor,
    dino_norm_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fix = (img_fix_01 - dino_norm_mean) / dino_norm_std
    wrist = (img_wrist_01 - dino_norm_mean) / dino_norm_std

    feat_fix = dinov2_extractor.forward_features(fix)["x_norm_patchtokens"]      # (1,256,768)
    feat_wrist = dinov2_extractor.forward_features(wrist)["x_norm_patchtokens"]  # (1,256,768)

    feat_fix = F.normalize(feat_fix, p=2, dim=2)
    feat_wrist = F.normalize(feat_wrist, p=2, dim=2)

    tok_fix = pretrained_tokenizer(
        patch_tokens=feat_fix,
        camera_id=0,
        sampled_patch_indices=None,
        return_attn=False,
    ).tokens  # (1,Q,D)

    tok_wrist = pretrained_tokenizer(
        patch_tokens=feat_wrist,
        camera_id=1,
        sampled_patch_indices=None,
        return_attn=False,
    ).tokens  # (1,Q,D)

    return tok_fix, tok_wrist

"""
utils: model and data loader
"""
def _resolve_policy_module_name(ptype: str) -> str:
    # 1) built-ins
    cand = f"lerobot.policies.{ptype}"
    try:
        importlib.import_module(cand)
        return cand
    except ModuleNotFoundError:
        pass

    # 2) plugins via entry points (Bring Your Own Policies)
    try:
        eps = md.entry_points()
        group = eps.select(group="lerobot.policies") if hasattr(eps, "select") else eps.get("lerobot.policies", [])
        for ep in group:
            if ep.name == ptype:
                # ep.value can be "pkg.module" or "pkg.module:obj"
                return ep.value.split(":")[0]
    except Exception:
        pass

    # 3) common plugin naming conventions
    for cand in (f"lerobot_policy_{ptype}", ptype):
        try:
            importlib.import_module(cand)
            return cand
        except ModuleNotFoundError:
            continue

    raise ModuleNotFoundError(
        f"Could not resolve policy module for type='{ptype}'. "
        f"Tried lerobot.policies.{ptype}, entry_points('lerobot.policies'), "
        f"lerobot_policy_{ptype}, and {ptype}."
    )

def _load_pretrained_tokenizer(tokenizer_ckpt: Path, device: str):
    raise NotImplementedError("Deprecated Feature....")
    return tokenizer

def _load_policy(pretrained_dir: Path, device: str):
    """
    Load a concrete policy (ACT, Diffusion, ...) from a local `pretrained_model/` folder.

    Steps:
      1) parse config.json to get policy type
      2) import lerobot.policies.<type>
      3) find a non-abstract subclass of PreTrainedPolicy
      4) call <PolicyClass>.from_pretrained(pretrained_dir)
    """
    cfg_path = pretrained_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {pretrained_dir}")

    cfg = json.loads(cfg_path.read_text())
    ptype = cfg.get("type", None)
    if not ptype:
        raise ValueError(f"config.json missing 'type': {cfg_path}")

    # Map policy type -> module path (extend as needed)
    module_name = _resolve_policy_module_name(ptype)

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise RuntimeError(f"Failed to import {module_name}: {e}") from e

    # Find a concrete Policy class in that module
    candidates = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if not issubclass(obj, PreTrainedPolicy):
            continue
        if obj is PreTrainedPolicy:
            continue
        # must be instantiable and have from_pretrained
        if inspect.isabstract(obj):
            continue
        if not hasattr(obj, "from_pretrained"):
            continue
        candidates.append(obj)

    if not candidates:
        # Sometimes policy class is in a submodule (common).
        # Try a few known submodules for this policy type.
        for sub in ("policy", "policies", f"policy_{ptype}", f"modeling_{ptype}"):
            try:
                mod2 = importlib.import_module(f"{module_name}.{sub}")
            except Exception:
                continue
            for name, obj in inspect.getmembers(mod2, inspect.isclass):
                if not issubclass(obj, PreTrainedPolicy):
                    continue
                if obj is PreTrainedPolicy:
                    continue
                if inspect.isabstract(obj):
                    continue
                if not hasattr(obj, "from_pretrained"):
                    continue
                candidates.append(obj)

    if not candidates:
        raise RuntimeError(
            f"Could not find a concrete PreTrainedPolicy subclass for type='{ptype}'. "
            f"Tried module={module_name} and common submodules."
        )

    # Prefer a class with a name containing the type (e.g., ACTPolicy)
    candidates = sorted(
        candidates,
        key=lambda c: (0 if ptype.lower() in c.__name__.lower() else 1, c.__name__),
    )
    PolicyCls = candidates[0]

    policy = PolicyCls.from_pretrained(pretrained_dir)
    policy.to(device).eval()
    return policy, f"{PolicyCls.__module__}.{PolicyCls.__name__}.from_pretrained"

def _action_dataset_root(task: str) -> Path:
    """Resolve expert action metadata in the original or unified dataset layout."""
    data_root = _repo_root() / "fomofd" / "tasks" / task / "data"
    candidates = [data_root / "processed_le_v21",
                  data_root / "processed_le_v21_vfmf_vae_fix_wrist"]
    for root in candidates:
        if (root / "meta" / "stats.json").is_file() or (root / "stats.json").is_file():
            return root
    raise FileNotFoundError(
        f"Expert action stats.json not found for task={task}. Checked:\n  "
        + "\n  ".join(map(str, candidates))
    )


def _load_action_mean_std(task: str) -> tuple[np.ndarray, np.ndarray, Path]:
    ds_root = _action_dataset_root(task)

    # LeRobot v2.1 stats are usually here
    stats_path = ds_root / "meta" / "stats.json"
    if not stats_path.exists():
        # fallback if layout differs
        stats_path = ds_root / "stats.json"

    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found for task={task} under:\n  {ds_root}")

    stats = json.loads(stats_path.read_text())
    a = stats.get("action", None)
    if not a or "mean" not in a or "std" not in a:
        raise KeyError(f"stats.json missing action mean/std at:\n  {stats_path}")

    a_mean = np.asarray(a["mean"], dtype=np.float32).reshape(-1)
    a_std  = np.asarray(a["std"], dtype=np.float32).reshape(-1)
    return a_mean, a_std, stats_path

def _load_gripper_transform(task: str) -> dict | None:
    ds_root = _action_dataset_root(task)
    meta_path = ds_root / "meta" / "gripper_transform.json"
    if not meta_path.exists():
        print(f"[INFO] No gripper_transform.json at {meta_path} (will not invert gripper).")
        return None
    cfg = json.loads(meta_path.read_text())
    # validate minimal keys
    for k in ("gripper_transform", "gripper_min", "gripper_max", "gripper_eps", "gripper_p_u_mean"):
        if k not in cfg:
            raise KeyError(f"gripper_transform.json missing '{k}': {meta_path}")
    if not cfg.get("gripper_transform", "none") == "centered_logit":
        return None
    return cfg

"""
utils: action chunk helper
"""
def _denorm_action_array(x: np.ndarray, a_mean: np.ndarray, a_std: np.ndarray, gripper_tfm: dict | None) -> np.ndarray:
    """
    x: (..., A) normalized action array
    returns: same shape, denormalized
    """
    y = x.astype(np.float32, copy=True)
    y = y * a_std + a_mean

    if gripper_tfm is not None and y.shape[-1] >= 1:
        flat = y.reshape(-1, y.shape[-1])
        for i in range(flat.shape[0]):
            g_z = float(flat[i, -1])
            flat[i, -1] = np.float32(_invert_gripper_z_to_angle(g_z, gripper_tfm))
        y = flat.reshape(y.shape)

    return y


def _policy_select_action_and_chunk(
    policy,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      actual_action: [B, A]
      raw_chunk:     [B, K, A]

    This helper is only valid for policies using temporal ensembling.
    It mirrors select_action() in that mode:
        raw_chunk = predict_action_chunk(batch)
        actual_action = temporal_ensembler.update(raw_chunk)
    """
    assert getattr(policy.config, "temporal_ensemble_coeff", None) is not None, (
        "_policy_select_action_and_chunk requires policy.config.temporal_ensemble_coeff to be enabled.")

    policy.eval()
    raw_chunk = policy.predict_action_chunk(batch)                  # [B, K, A]
    actual_action = policy.temporal_ensembler.update(raw_chunk)     # [B, A]
    return actual_action, raw_chunk


def _policy_select_action_chunk_and_batch(
    policy,
    batch: dict[str, torch.Tensor],
    num_chunk_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      actual_action: [1, A]
      raw_chunk:     [1, K, A]      # chunk actually associated with executed action
      raw_chunk_batch: [B, K, A]    # batch of sampled chunks for proactive analysis

    Assumes temporal ensembling is enabled.
    """
    if num_chunk_samples < 1:
        raise ValueError(f"num_chunk_samples must be >= 1, got {num_chunk_samples}")

    act_t, chunk_t = _policy_select_action_and_chunk(policy, batch)  # [1,A], [1,K,A]

    if num_chunk_samples == 1:
        chunk_batch_t = chunk_t
    else:
        chunk_batch_t = policy.predict_action_chunk_stochastic_batch(batch, num_samples=num_chunk_samples)  # [num_chunk_samples, K, A]

    return act_t, chunk_t, chunk_batch_t


"""
utils: gripper action converter
"""
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _logit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.log(x) - np.log1p(-x)

def _invert_gripper_z_to_angle(g_z: float, tfm: dict) -> float:
    """
    g_z is the *transformed gripper scalar* after MEAN/STD denorm.
    Invert:
      z = logit(u) - logit(p)
      => logit(u) = z + logit(p)
      => u = sigmoid(z + logit(p))
      => angle = u*(u_max-u_min) + u_min
    """
    eps = float(tfm["gripper_eps"])
    u_min = float(tfm["gripper_min"])
    u_max = float(tfm["gripper_max"])
    p = float(tfm["gripper_p_u_mean"])

    p = min(max(p, eps), 1.0 - eps)
    lp = float(_logit(np.array(p, dtype=np.float32)))

    u = float(_sigmoid(np.array(g_z + lp, dtype=np.float32)))
    u = min(max(u, eps), 1.0 - eps)

    return u * (u_max - u_min) + u_min

_TTY_FD = None
_TTY_OLD_SETTINGS = None


def setup_terminal_cbreak():
    global _TTY_FD, _TTY_OLD_SETTINGS

    if not sys.stdin.isatty():
        print("[WARN] stdin is not a TTY; keyboard control disabled.")
        return False

    _TTY_FD = sys.stdin.fileno()
    _TTY_OLD_SETTINGS = termios.tcgetattr(_TTY_FD)
    tty.setcbreak(_TTY_FD)
    return True


def restore_terminal():
    global _TTY_FD, _TTY_OLD_SETTINGS
    if _TTY_FD is not None and _TTY_OLD_SETTINGS is not None:
        try:
            termios.tcsetattr(_TTY_FD, termios.TCSADRAIN, _TTY_OLD_SETTINGS)
        except Exception:
            pass
        _TTY_FD = None
        _TTY_OLD_SETTINGS = None


def _signal_restore_and_raise(signum, frame):
    restore_terminal()
    raise KeyboardInterrupt

_KEY_TO_CONTROL_CMD = {
    "r": "RESET",          # reset + save current recorded episode
    "t": "RESET_NOSAVE",   # reset + discard current recorded episode
    "w": "ATTACK_W",
    "a": "ATTACK_A",
    "s": "ATTACK_S",
    "d": "ATTACK_D",
    "f": "ATTACK_F",
    "b": "ATTACK_B",
    "m": "ATTACK_M",
    "v": "ATTACK_V",
}

def keyboard_listener(command_queue, command_lock) -> None:
    print("[INFO] Keyboard:")
    print("       r = reset and save")
    print("       t = reset and discard current episode")
    print("       w/a/s/d = inject translation attack for next action chunk")
    print("       f = inject gripper-open attack for next action chunk")
    print("       b = inject blood corruption on wrist camera")
    print("       m = inject scene-object motion attack")
    print("       v = inject camera-lag attack on both camera views")

    while True:
        try:
            ch = sys.stdin.read(1).lower()
        except (EOFError, KeyboardInterrupt):
            break

        cmd = _KEY_TO_CONTROL_CMD.get(ch)
        if cmd is None:
            continue

        with command_lock:
            command_queue.append(cmd)

        print(f"\n[CTRL] queued {cmd}")

@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument(
        "--tokenizer_ckpt",
        type=str,
        default=None,
        help="Path to best_tokenizer.pt, required for token-input ACT policies unless default path exists.",
    )
    ap.add_argument("--step", default="last")
    ap.add_argument("--action_bind", default="ipc:///tmp/fomofd_policy.sock")
    ap.add_argument("--control_bind", default="ipc:///tmp/fomofd_control.sock")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--including_gripper", action="store_true",
                   help="Include gripper jaw joint position as part of state")
    args = ap.parse_args()

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    pretrained_dir = _find_pretrained_dir(args.task, args.method, args.step)
    policy, how = _load_policy(pretrained_dir, device)
    print(f"[INFO] Loaded {args.method} for {args.task} from:\n  {pretrained_dir}\n  via {how}\n  device={device}")

    uses_token_inputs = bool(getattr(policy.config, "token_features", {}))
    uses_image_inputs = bool(getattr(policy.config, "image_features", {}))

    dinov2_extractor = None
    pretrained_tokenizer = None
    dino_norm_mean = None
    dino_norm_std = None

    if uses_token_inputs:
        tokenizer_ckpt = Path(args.tokenizer_ckpt).expanduser().resolve() if args.tokenizer_ckpt else (
            _repo_root() / "fomofd" / "tasks" / args.task / "model" / "dino_act_wm_tokenizer_only" / "best_tokenizer.pt"
        )
        if not tokenizer_ckpt.exists():
            raise FileNotFoundError(
                f"Token-input policy detected, but tokenizer checkpoint not found:\n  {tokenizer_ckpt}\n"
                f"Pass --tokenizer_ckpt explicitly."
            )

        model_name = getattr(policy.config, "dinov2_model_name", "dinov2_vitb14_reg")
        dinov2_extractor = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        pretrained_tokenizer = _load_pretrained_tokenizer(tokenizer_ckpt, device)

        dino_norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        dino_norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        print(f"[INFO] Token-input ACT detected. Loaded DINOv2 + pretrained tokenizer:")
        print(f"       tokenizer_ckpt = {tokenizer_ckpt}")

    elif args.method == "act_dinov2" and uses_image_inputs:
        model_name = policy.config.dinov2_model_name if hasattr(policy, "config") else "dinov2_vitb14_reg"
        dinov2_extractor = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()

        dino_norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        dino_norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        print("[INFO] Initialized DINOv2 feature extractor for image-input ACT inference")

    ctx = zmq.Context.instance()
    action_sock = ctx.socket(zmq.REP)
    action_sock.setsockopt(zmq.LINGER, 0)
    action_sock.setsockopt(zmq.RCVTIMEO, 200)
    action_sock.bind(args.action_bind)

    control_sock = ctx.socket(zmq.REP)
    control_sock.setsockopt(zmq.LINGER, 0)
    control_sock.setsockopt(zmq.RCVTIMEO, 200)
    control_sock.bind(args.control_bind)

    a_mean, a_std, stats_path = _load_action_mean_std(args.task)
    print(f"[INFO] Loaded action mean/std from:\n  {stats_path}")

    gripper_tfm = _load_gripper_transform(args.task)
    if gripper_tfm is not None:
        print(f"[INFO] Gripper transform enabled: p_u_mean={gripper_tfm['gripper_p_u_mean']}, "
              f"u_min={gripper_tfm['gripper_min']}, u_max={gripper_tfm['gripper_max']}, eps={gripper_tfm['gripper_eps']}")

    tty_enabled = setup_terminal_cbreak()
    atexit.register(restore_terminal)
    signal.signal(signal.SIGINT, _signal_restore_and_raise)
    signal.signal(signal.SIGTERM, _signal_restore_and_raise)

    command_queue = deque()
    command_lock = threading.Lock()
    if tty_enabled:
        kb_thread = threading.Thread(
            target=keyboard_listener,
            args=(command_queue, command_lock),
            daemon=True,
        )
        kb_thread.start()

    try:
        while True:
            poller = zmq.Poller()
            poller.register(action_sock, zmq.POLLIN)
            poller.register(control_sock, zmq.POLLIN)

            try:
                events = dict(poller.poll(timeout=100))
            except KeyboardInterrupt:
                break

            # Handle control requests first
            if control_sock in events and events[control_sock] == zmq.POLLIN:
                try:
                    ctrl_parts = control_sock.recv_multipart()
                except zmq.Again:
                    ctrl_parts = None

                if ctrl_parts is not None:
                    if len(ctrl_parts) == 1 and ctrl_parts[0] == b"RESET_POLICY":
                        try:
                            if hasattr(policy, "reset"):
                                policy.reset()
                            control_sock.send_multipart([b"OK"])
                        except Exception as e:
                            control_sock.send_multipart([b"ERR", str(e).encode("utf-8")])
                        continue

                    if len(ctrl_parts) == 1 and ctrl_parts[0] == b"POLL":
                        with command_lock:
                            cmd = command_queue.popleft() if command_queue else None

                        if cmd is None:
                            control_sock.send_multipart([b"NONE"])
                            continue

                        try:
                            if cmd in {"RESET", "RESET_NOSAVE"} and hasattr(policy, "reset"):
                                policy.reset()
                            control_sock.send_multipart([cmd.encode("utf-8")])
                        except Exception as e:
                            control_sock.send_multipart([b"ERR", str(e).encode("utf-8")])
                        continue
                    else:
                        control_sock.send_multipart([b"ERR", b"bad_control_message"])
                        continue

            if action_sock not in events or events[action_sock] != zmq.POLLIN:
                continue

            try:
                parts = action_sock.recv_multipart()
            except zmq.Again:
                continue
            except KeyboardInterrupt:
                break

            req_type = b"ACT"

            if len(parts) >= 1 and parts[0] in {b"ACT", b"ACT_CHUNK", b"ACT_CHUNK_BATCH"}:
                req_type = parts[0]
                parts = parts[1:]

            num_chunk_samples = None
            if req_type == b"ACT_CHUNK_BATCH":
                if len(parts) == 5:
                    seq_b, n_samples_bytes, jpg_fix, jpg_wrist, state_bytes = parts
                    state_np = np.frombuffer(state_bytes, dtype=np.float32).reshape(-1)
                elif len(parts) == 4:
                    seq_b, n_samples_bytes, jpg_fix, jpg_wrist = parts
                    state_np = None
                else:
                    action_sock.send_multipart([b"ERR", b"bad_parts"])
                    continue

                n_arr = np.frombuffer(n_samples_bytes, dtype=np.int32)
                if n_arr.size != 1:
                    action_sock.send_multipart([b"ERR", b"bad_num_samples"])
                    continue
                num_chunk_samples = int(n_arr[0])
                if num_chunk_samples < 1:
                    action_sock.send_multipart([b"ERR", b"num_samples_must_be_positive"])
                    continue
            else:
                if len(parts) == 3:
                    seq_b, jpg_fix, jpg_wrist = parts
                    state_np = None
                elif len(parts) == 4:
                    seq_b, jpg_fix, jpg_wrist, state_bytes = parts
                    state_np = np.frombuffer(state_bytes, dtype=np.float32).reshape(-1)
                else:
                    action_sock.send_multipart([b"ERR", b"bad_parts"])
                    continue

            try:
                fix = _decode_jpg_to_chw_u8(jpg_fix)
                wrist = _decode_jpg_to_chw_u8(jpg_wrist)

                fix = np.array(fix, copy=True, order="C")
                wrist = np.array(wrist, copy=True, order="C")

                if args.including_gripper:
                    state_dim = 10
                else:
                    state_dim = 9
                if state_np is None:
                    state_t = torch.zeros((1, state_dim), dtype=torch.float32, device=device)
                else:
                    if state_np.size != state_dim:
                        raise ValueError(f"state dim mismatch: got {state_np.size} expected {state_dim}")
                    if not state_np.flags.writeable:
                        state_np = state_np.copy()
                    state_t = torch.from_numpy(state_np.astype(np.float32, copy=False)).unsqueeze(0).to(device)

                img_fix_t = torch.from_numpy(fix).unsqueeze(0).to(device).float().div_(255.0)
                img_wrist_t = torch.from_numpy(wrist).unsqueeze(0).to(device).float().div_(255.0)

                if uses_token_inputs:
                    tok_fix, tok_wrist = _extract_pretrained_tokens_online(
                        img_fix_01=img_fix_t,
                        img_wrist_01=img_wrist_t,
                        dinov2_extractor=dinov2_extractor,
                        pretrained_tokenizer=pretrained_tokenizer,
                        dino_norm_mean=dino_norm_mean,
                        dino_norm_std=dino_norm_std,
                    )

                    batch = {
                        "observation.tokens.fix_camera": tok_fix,
                        "observation.tokens.wrist_camera": tok_wrist,
                        "observation.state": state_t,
                    }

                else:
                    batch = {
                        "observation.images.fix_camera": img_fix_t,
                        "observation.images.wrist_camera": img_wrist_t,
                        "observation.state": state_t,
                    }

                    if dinov2_extractor is not None:
                        batch["pre_extracted_dinov2"] = {}
                        for cam_key in ["observation.images.fix_camera", "observation.images.wrist_camera"]:
                            img = batch[cam_key]
                            img = (img - dino_norm_mean) / dino_norm_std
                            features = dinov2_extractor.forward_features(img)
                            batch["pre_extracted_dinov2"][cam_key] = {
                                "x_norm_patchtokens": features["x_norm_patchtokens"],
                                "x_norm_clstoken": features["x_norm_clstoken"],
                            }

                with torch.inference_mode():
                    if req_type == b"ACT_CHUNK_BATCH":
                        act_t, chunk_t, chunk_batch_t = _policy_select_action_chunk_and_batch(
                            policy,
                            batch,
                            num_chunk_samples=num_chunk_samples,
                        )

                        act = act_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
                        chunk = chunk_t[0].detach().cpu().numpy().astype(np.float32)              # [K, A]
                        chunk_batch = chunk_batch_t.detach().cpu().numpy().astype(np.float32)     # [B, K, A]

                        act = _denorm_action_array(act, a_mean, a_std, gripper_tfm)
                        chunk = _denorm_action_array(chunk, a_mean, a_std, gripper_tfm)
                        chunk_batch = _denorm_action_array(chunk_batch, a_mean, a_std, gripper_tfm)

                        chunk_shape = np.asarray(chunk.shape, dtype=np.int32)
                        batch_shape = np.asarray(chunk_batch.shape, dtype=np.int32)

                        action_sock.send_multipart([
                            seq_b,
                            act.tobytes(),
                            chunk_shape.tobytes(),
                            chunk.tobytes(),
                            batch_shape.tobytes(),
                            chunk_batch.tobytes(),
                        ])
                    elif req_type == b"ACT_CHUNK":
                        act_t, chunk_t = _policy_select_action_and_chunk(policy, batch)

                        act = act_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
                        chunk = chunk_t[0].detach().cpu().numpy().astype(np.float32)   # [K, A]

                        act = _denorm_action_array(act, a_mean, a_std, gripper_tfm)
                        chunk = _denorm_action_array(chunk, a_mean, a_std, gripper_tfm)

                        chunk_shape = np.asarray(chunk.shape, dtype=np.int32)
                        action_sock.send_multipart([
                            seq_b,
                            act.tobytes(),
                            chunk_shape.tobytes(),
                            chunk.tobytes(),
                        ])
                    else:
                        out = policy.select_action(batch)

                        if isinstance(out, dict):
                            out = out.get("action", out.get("actions", out))
                        if not torch.is_tensor(out):
                            raise TypeError(f"Unexpected policy output type: {type(out)}")

                        act = out.detach().cpu().numpy().reshape(-1).astype(np.float32)
                        act = _denorm_action_array(act, a_mean, a_std, gripper_tfm)

                        action_sock.send_multipart([seq_b, act.tobytes()])

            except Exception as e:
                action_sock.send_multipart([b"ERR", str(e).encode("utf-8")])
    finally:
        print("\r[INFO] Shutting down ZMQ server...")
        action_sock.close(linger=0)
        control_sock.close(linger=0)
        ctx.term()
        restore_terminal()
        print("[INFO] Server exited cleanly.")

if __name__ == "__main__":
    main()
