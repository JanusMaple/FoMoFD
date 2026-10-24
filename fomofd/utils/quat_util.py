import torch
import numpy as np

def quat_conj(q):
    out = q.clone()
    out[..., 1:] *= -1
    return out

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def quat_norm(q, eps=1e-8):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)

def quat_inv(q):
    return quat_conj(quat_norm(q))  # inverse of unit quat

def quat_rotate(q, v):
    # q: (...,4) wxyz, v: (...,3)
    q = quat_norm(q)
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]

def quat_canon(q):
    # enforce w >= 0 for learning stability
    return torch.where(q[..., :1] < 0, -q, q)

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

    q = np.array([w, x, y, z], dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-8)
    return q

def rot6d_to_quat_wxyz(r6: np.ndarray) -> np.ndarray:
    return _rotmat_to_quat_wxyz(_rot6d_to_rotmat(r6))

def quat_wxyz_to_rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float32).reshape(4,)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        R = np.eye(3, dtype=np.float32)
    else:
        w, x, y, z = (q / n).tolist()
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        R = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
            [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
            [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
        ], dtype=np.float32)
    c0 = R[:, 0]
    c1 = R[:, 1]
    return np.concatenate([c0, c1], axis=0).astype(np.float32, copy=False)
