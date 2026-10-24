"""
lerobot/utils/npz_to_hf.py

Convert fomofd raw NPZ episodes into a HuggingFace dataset.

Example Run:
    python utils/npz_to_hf.py --task needle_pickup --shift_actions_to_next_obs
"""

import argparse
from pathlib import Path
import numpy as np
import shutil

import hashlib
from pathlib import Path

def _raw_signature(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        st = p.stat()
        h.update(p.name.encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()[:16]

def _to_chw_u8(img_hwc: np.ndarray) -> np.ndarray:
    """Convert [H,W,3] uint8 -> [3,H,W] uint8."""
    if img_hwc.ndim != 3 or img_hwc.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {img_hwc.shape}")
    if img_hwc.dtype != np.uint8:
        img_hwc = img_hwc.astype(np.uint8, copy=False)
    return np.transpose(img_hwc, (2, 0, 1))

def _resolve_task_dirs(task: str, prl: bool = False) -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    base_name = "prl_data" if prl else "data"

    task_data_dir = repo_root / "fomofd" / "tasks" / task / base_name
    in_dir = task_data_dir / "raw"
    out_dir = task_data_dir / "processed_hf"
    return in_dir, out_dir

def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w,x,y,z) to 3x3 rotation matrix.
    Robust to non-unit quaternions (normalizes).
    """
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        # Fallback: identity rotation if quaternion is degenerate
        return np.eye(3, dtype=np.float32)
    w, x, y, z = (q / n).tolist()

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
    ], dtype=np.float32)
    return R

def _rotmat_to_rot6d_first_two_cols(R: np.ndarray) -> np.ndarray:
    """
    6D rotation representation: first two columns of R (3x3).
    Returns shape (6,) = [R00,R10,R20, R01,R11,R21].
    """
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    c0 = R[:, 0]
    c1 = R[:, 1]
    return np.concatenate([c0, c1], axis=0).astype(np.float32, copy=False)

def _action_quat_to_rot6d(act8: np.ndarray, quat_order: str = "wxyz") -> np.ndarray:
    """
    Convert [dp(3), quat(4), gripper(1)] -> [dp(3), rot6d(6), gripper(1)].
    Assumes quaternion is act8[3:7].
    """
    act8 = np.asarray(act8, dtype=np.float32).reshape(-1,)
    if act8.shape[0] != 8:
        raise ValueError(f"Expected 8D action [dp, quat, gripper], got shape {act8.shape}")

    dp = act8[0:3]
    quat = act8[3:7]
    g = act8[7:8]

    if quat_order.lower() == "xyzw":
        quat = quat[[3, 0, 1, 2]]  # -> wxyz
    elif quat_order.lower() != "wxyz":
        raise ValueError(f"Unsupported quat_order={quat_order} (expected 'wxyz' or 'xyzw').")

    R = _quat_wxyz_to_rotmat(quat)
    rot6d = _rotmat_to_rot6d_first_two_cols(R)

    act10 = np.concatenate([dp, rot6d, g], axis=0).astype(np.float32, copy=False)
    return act10

def _state_quat_to_rot6d(st8: np.ndarray, quat_order: str = "wxyz") -> np.ndarray:
    """
    Convert state [p(3), quat(4), gripper] -> [p(3), rot6d(6), gripper].
    """
    st8 = np.asarray(st8, dtype=np.float32).reshape(-1,)
    if st8.shape[0] != 8:
        raise ValueError(f"Expected 8D state [p, quat], got {st8.shape}")

    p = st8[0:3]
    quat = st8[3:7]
    gripper = st8[7:8]

    if quat_order.lower() == "xyzw":
        quat = quat[[3, 0, 1, 2]]  # -> wxyz
    elif quat_order.lower() != "wxyz":
        raise ValueError(f"Unsupported quat_order={quat_order}")

    R = _quat_wxyz_to_rotmat(quat)
    rot6d = _rotmat_to_rot6d_first_two_cols(R)
    return np.concatenate([p, rot6d, gripper], axis=0).astype(np.float32, copy=False)

def main():
    parser = argparse.ArgumentParser(
        description="Convert fomofd raw NPZ episodes into a LeRobot-compatible HF dataset."
    )
    parser.add_argument("--task", type=str, required=True, help="Task name, e.g. needle_pickup")
    parser.add_argument("--raw_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)

    # Optional: allow choosing representation without editing code later
    parser.add_argument(
        "--action_repr",
        type=str,
        default="rot6d",
        choices=("quat", "rot6d"),
        help="Store action orientation as quaternion (8D) or rot6d (10D). Default: rot6d",
    )
    parser.add_argument(
        "--state_repr",
        type=str,
        default="rot6d",
        choices=("quat", "rot6d"),
        help="Store state orientation as quaternion (7D) or rot6d (9D). Default: rot6d",
    )
    parser.add_argument(
        "--quat_order",
        type=str,
        default="wxyz",
        choices=("wxyz", "xyzw"),
        help="Quaternion order in recorded actions if action_repr=rot6d or state_repr=rot6d conversion is used. Default: wxyz",
    )
    parser.add_argument(
        "--shift_actions_to_next_obs",
        action="store_true",
        help="Store sample t with action from t+1, so action causes the next observation.",
    )
    parser.add_argument(
        "--prl",
        action="store_true",
        help="Use fomofd/tasks/<task>/prl_data/{raw,processed_hf} instead of data/{raw,processed_hf}.",
    )

    args = parser.parse_args()

    in_dir, out_dir = _resolve_task_dirs(args.task, args.prl)
    in_dir = args.raw_dir.resolve() if args.raw_dir else in_dir
    out_dir = args.output_dir.resolve() if args.output_dir else out_dir

    if not in_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found:\n  {in_dir}")

    episode_files = sorted(in_dir.glob("episode_*.npz"),
                           key=lambda path: int(path.stem.split("_")[-1]))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz found in {in_dir}")

    print(f"[INFO] Convert {len(episode_files)} episodes of task {args.task}")
    print(f"       from {in_dir}")
    print(f"       to {out_dir}")
    print(f"       action_repr={args.action_repr} (quat_order={args.quat_order})")
    print(f"       state_repr={args.state_repr} (quat_order={args.quat_order})")
    print(f"       shift_actions_to_next_obs={args.shift_actions_to_next_obs}")

    # LeRobot/HF datasets
    import datasets
    from datasets import Dataset, Features, Value, Sequence, Array3D        # type: ignore

    # 1. Peek at the first episode to determine dimensions for the schema
    first_data = np.load(episode_files[0], allow_pickle=True)
    _, H, W, _ = first_data["images_fix"].shape
    raw_action_dim = int(first_data["actions"].shape[1])
    raw_state_dim = int(first_data["states"].shape[1])

    action_dim = raw_action_dim if args.action_repr == "quat" else 10
    state_dim = raw_state_dim if args.state_repr == "quat" else 10

    # 2. Define schema upfront
    features = Features({
        "episode_index": Value("int32"),
        "step_index": Value("int32"),
        "timestamp": Value("float64"),
        "observation.images.fix_camera": Array3D(dtype="uint8", shape=(3, H, W)),
        "observation.images.wrist_camera": Array3D(dtype="uint8", shape=(3, H, W)),
        "observation.state": Sequence(Value("float32"), length=int(state_dim)),
        "action": Sequence(Value("float32"), length=int(action_dim)),
        "done": Value("bool"),
    })

    # 3. Create a generator to stream rows one at a time
    def episode_generator():
        for ep_idx, ep_file in enumerate(episode_files):
            data = np.load(ep_file, allow_pickle=True)

            images_fix = data["images_fix"]         # [T,H,W,3] uint8
            images_wrist = data["images_wrist"]     # [T,H,W,3] uint8
            actions = data["actions"]               # [T,Da] float32
            timestamps = data["timestamps"]         # [T]
            states = data["states"]                 # [T,Ds] float32

            T_raw = images_fix.shape[0]
            T_out = T_raw - 1 if args.shift_actions_to_next_obs else T_raw

            if T_out <= 0:
                print(f"[WARN] skipping short episode: {ep_file.name}")
                continue

            for t in range(T_out):
                obs_idx = t
                act_idx = t + 1 if args.shift_actions_to_next_obs else t

                fix_chw = _to_chw_u8(images_fix[obs_idx])
                wrist_chw = _to_chw_u8(images_wrist[obs_idx])

                act_raw = actions[act_idx].astype(np.float32, copy=False)
                state_raw = states[obs_idx].astype(np.float32, copy=False)

                if args.action_repr == "quat":
                    act_out = act_raw
                else:
                    act_out = _action_quat_to_rot6d(act_raw, quat_order=args.quat_order)

                if args.state_repr == "quat":
                    state_out = state_raw
                else:
                    state_out = _state_quat_to_rot6d(state_raw, quat_order=args.quat_order)

                yield {
                    "episode_index": ep_idx,
                    "step_index": t,
                    "timestamp": float(timestamps[obs_idx]),
                    "observation.images.fix_camera": fix_chw,
                    "observation.images.wrist_camera": wrist_chw,
                    "observation.state": state_out.tolist(),
                    "action": act_out.tolist(),
                    "done": (t == T_out - 1),
                }

    print("[INFO] Generating dataset (this will stream to disk to prevent memory overflow)...")
    
    # 4. Ingest via generator (handles chunking/batching automatically)
    episode_files = sorted(in_dir.glob("episode_*.npz"),
                           key=lambda path: int(path.stem.split("_")[-1]))
    raw_sig = _raw_signature(episode_files)

    cache_dir = out_dir.parent / ".hf_generator_cache" / raw_sig
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_generator(
        episode_generator,
        features=features,
        cache_dir=str(cache_dir),
    )

    # Save to disk
    if out_dir.exists():
        print(f"[WARN] Processed dataset already exists at:\n  {out_dir}")
        resp = input("Delete it and regenerate all episodes? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("[INFO] Aborted. Existing dataset kept.")
            return
        print("[INFO] Deleting existing processed dataset...")
        shutil.rmtree(out_dir)

    dataset.save_to_disk(out_dir)

    del dataset  # free memory before cleanup
    shutil.rmtree(cache_dir, ignore_errors=True)

    print(f"[DONE] Saved dataset to {out_dir}")

if __name__ == "__main__":
    main()
