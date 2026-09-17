"""Physics token representation for the MotionCraft policy (docs/07 §1).

Per frame (30 fps), all computed from the tracked physics fields (body_pos [T,24,3], dof_state [T,69,2],
root_state [T,13], action [T,69]) after WINDOW canonicalisation (frame 0 root xy -> origin, frame 0
hip-across direction -> +x so the character faces +y), exactly as UniPhys `cano_seq_smpl_or_smplx` +
`get_repr(return_last=True)`:

  root  (15) : root_trans 3 | root_rot_6d 6 | root_trans_vel 3 | root_rot_vel 3
  body (420) : local_positions 72 | local_vel 72 | dof_pose_6d 138 | dof_vel 69 | action 69

`action[t]` is the (pre PD offset/scale) policy action recorded WITH frame t by the PHC recorder, i.e. the action
applied at step t whose result is the state stored in row t (state after the step). The closed-loop history buffer
uses the same pairing (state after the step, action that produced it); the first future row's action is the one to
execute from the current state.
Everything is vectorised numpy; `check_against_uniphys()` verifies equality with the UniPhys code.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT_DIM, BODY_DIM = 15, 420
ROOT_SLICES = dict(root_trans=(0, 3), root_rot_6d=(3, 9), root_trans_vel=(9, 12), root_rot_vel=(12, 15))
BODY_SLICES = dict(local_positions=(0, 72), local_vel=(72, 144), dof_pose_6d=(144, 282), dof_vel=(282, 351), action=(351, 420))
STATE_DIM = 351  # body without action
ACTION_DIM = 69
R_HIP, L_HIP = 5, 1  # MuJoCo body order (UniPhys face_joint_indx_robot = [5, 1])


# ----------------------------------------------------------------------------- quaternion helpers (w,x,y,z), numpy
def _qmul(q, r):
    w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], -1)


def _qinv(q):
    return q * np.array([1, -1, -1, -1], dtype=q.dtype)


def _qrot(q, v):
    """rotate v [...,3] by unit quaternion q [...,4] (w,x,y,z) — same as HumanML3D qrot."""
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def _qbetween(v0, v1):
    """quaternion rotating v0 to v1 (HumanML3D qbetween: w = |v0||v1| + v0·v1, xyz = v0×v1, normalised)."""
    v = np.cross(v0, v1)
    w = np.sqrt((v0 ** 2).sum(-1) * (v1 ** 2).sum(-1)) + (v0 * v1).sum(-1)
    q = np.concatenate([w[..., None], v], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------- canonicalisation (window frame 0)
def canonicalize(body_pos, root_state):
    """UniPhys cano_seq_smpl_or_smplx for one window. Returns (cano_body_pos [T,24,3], cano_root_state [T,13], transf [4,4])."""
    p = body_pos.astype(np.float64).copy()
    root_xy0 = p[0, 0] * np.array([1.0, 1.0, 0.0])
    p = p - root_xy0
    across = p[0, R_HIP] - p[0, L_HIP]
    x_axis = across.copy(); x_axis[-1] = 0.0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis); y_axis = y_axis / np.linalg.norm(y_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)  # [3,3], columns = new axes; p_new = p @ rot
    p = p @ rot
    T1 = np.eye(4); T1[0, 3] = -root_xy0[0]; T1[1, 3] = -root_xy0[1]
    T2 = np.eye(4); T2[:3, :3] = rot.T  # UniPhys: transf_matrix_2[:3,:3] = transf_rotmat.T in the single-seq version
    transf = T2 @ T1
    rs = root_state.astype(np.float64)
    T = len(rs)
    root_pos = rs[:, :3]; root_rotm = R.from_quat(rs[:, 3:7]).as_matrix()
    body_mat = np.zeros((T, 4, 4)); body_mat[:, :3, :3] = root_rotm; body_mat[:, :3, 3] = root_pos + (body_pos[0, 0] - root_pos[0]); body_mat[:, 3, 3] = 1
    new = transf[None] @ body_mat
    quat_new = R.from_matrix(new[:, :3, :3]).as_quat()
    pos_new = new[:, :3, 3]
    vel_new = rs[:, 7:10] @ transf[:3, :3].T
    angvel_new = rs[:, 10:13] @ transf[:3, :3].T
    return p, np.concatenate([pos_new, quat_new, vel_new, angvel_new], -1), transf


# ----------------------------------------------------------------------------- representation (UniPhys get_repr, return_last=True)
def heading_quat(body_pos):
    """per-frame quaternion (w,x,y,z) rotating the hip-derived forward direction onto +y.

    NOTE: UniPhys get_repr unpacks face_joint_indx_robot=[5,1] as (l_hip, r_hip), i.e. the opposite order
    of the canonicalisation step, so its per-frame local frame faces -y (a consistent 180-degree yaw for
    every frame, frame 0 forced to (0,0,0,1)). We reproduce that convention exactly so the two
    implementations agree numerically; it is self-consistent and harmless.
    """
    across = body_pos[:, L_HIP] - body_pos[:, R_HIP]
    across = across.copy(); across[:, -1] = 0
    across = across / np.sqrt((across ** 2).sum(-1))[:, None]
    forward = np.cross(np.array([[0, 0, 1.0]]), across)
    forward = forward / np.sqrt((forward ** 2).sum(-1))[..., None]
    q = _qbetween(forward, np.tile(np.array([[0, 1.0, 0]]), (len(forward), 1)))
    q[0] = np.array([0, 0, 0, 1.0])  # UniPhys sets frame 0 to (0,0,0,1) after canonicalisation (identity in their storage)
    bad = np.isnan(q).any(-1)
    for i in np.where(bad)[0]:
        q[i] = q[i - 1]
    return q


def get_repr(cano_body_pos, dof_state, cano_root_state):
    """Returns dict of per-frame arrays (T frames each), float32 — mirrors UniPhys get_repr(return_last=True)."""
    p = cano_body_pos.astype(np.float64); T, J, _ = p.shape
    q = heading_quat(p)
    local = p.copy(); local[..., 0] -= local[:, :1, 0]; local[..., 1] -= local[:, :1, 1]
    local = _qrot(np.repeat(q[:, None], J, 1), local)
    lvel = _qrot(np.repeat(q[:-1, None], J, 1), p[1:] - p[:-1])
    lvel = np.concatenate([lvel[:1], lvel], 0)
    rs = cano_root_state.astype(np.float64)
    root_rot_6d = R.from_quat(rs[:, 3:7]).as_matrix()[..., :-1].reshape(T, 6)
    dof_aa = dof_state[..., 0].reshape(T, -1, 3).astype(np.float64)
    dof_6d = R.from_rotvec(dof_aa.reshape(-1, 3)).as_matrix().reshape(T, -1, 3, 3)[..., :-1].reshape(T, -1)
    return dict(root_trans=rs[:, 0:3], root_rot_6d=root_rot_6d, root_trans_vel=rs[:, 7:10], root_rot_vel=rs[:, 10:13],
                local_positions=local.reshape(T, -1), local_vel=lvel.reshape(T, -1), dof_pose_6d=dof_6d,
                dof_vel=dof_state[..., 1].reshape(T, -1).astype(np.float64), heading_quat=q)


def window_tokens(body_pos, dof_state, root_state, action):
    """One window [T frames] of raw physics -> (root [T,15], body [T,420]) float32 in the window-canonical frame."""
    cp, crs, _ = canonicalize(body_pos, root_state)
    r = get_repr(cp, dof_state, crs)
    root = np.concatenate([r["root_trans"], r["root_rot_6d"], r["root_trans_vel"], r["root_rot_vel"]], -1)
    body = np.concatenate([r["local_positions"], r["local_vel"], r["dof_pose_6d"], r["dof_vel"], action.astype(np.float64)], -1)
    assert root.shape[1] == ROOT_DIM and body.shape[1] == BODY_DIM
    return root.astype(np.float32), body.astype(np.float32)


def hold_action(dof_pos, pd_offset, pd_scale):
    """policy action whose PD target equals the given joint angles (pd_tar = offset + scale * a)."""
    return (dof_pos - pd_offset) / pd_scale


def check_against_uniphys(body_pos, dof_state, root_state, atol=1e-4):
    """Compare with the UniPhys implementation on one window (requires the uniphys env on sys.path)."""
    import sys
    sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot/UniPhys")
    from uniphys.utils import motion_repr_utils as mru
    cj, cd, cr, _ = mru.cano_seq_smpl_or_smplx(body_pos.astype(np.float32), dof_state, root_state.astype(np.float32))
    ref = mru.get_repr(cj, cd, cr, return_last=True)
    cp, crs, _ = canonicalize(body_pos, root_state)
    ours = get_repr(cp, dof_state, crs)
    out = {}
    for k in ["root_trans", "root_rot_6d", "root_trans_vel", "root_rot_vel", "local_positions", "local_vel", "dof_pose_6d", "dof_vel"]:
        a = np.asarray(ref[k], dtype=np.float64).reshape(len(ours[k]), -1); b = ours[k].reshape(len(ours[k]), -1)
        out[k] = float(np.abs(a - b).max())
    ok = all(v < atol for v in out.values())
    return ok, out


# ----------------------------------------------------------------------------- batched version (closed loop, B windows)
def canonicalize_batch(body_pos, root_state):
    """body_pos [B,T,24,3], root_state [B,T,13] -> cano body_pos, cano root_state (same math as canonicalize)."""
    p = body_pos.astype(np.float64).copy(); B, T = p.shape[:2]
    root_xy0 = p[:, 0, 0] * np.array([1.0, 1.0, 0.0])
    p = p - root_xy0[:, None, None]
    across = p[:, 0, R_HIP] - p[:, 0, L_HIP]
    x_axis = across.copy(); x_axis[:, -1] = 0.0
    x_axis = x_axis / np.linalg.norm(x_axis, axis=-1, keepdims=True)
    z_axis = np.tile(np.array([0.0, 0.0, 1.0]), (B, 1))
    y_axis = np.cross(z_axis, x_axis); y_axis = y_axis / np.linalg.norm(y_axis, axis=-1, keepdims=True)
    rot = np.stack([x_axis, y_axis, z_axis], axis=2)  # [B,3,3] columns = axes; p_new = p @ rot
    p = np.einsum("btjk,bkl->btjl", p, rot)
    transf = np.zeros((B, 4, 4)); transf[:, :3, :3] = rot.transpose(0, 2, 1); transf[:, 3, 3] = 1
    transf[:, :3, 3] = np.einsum("bij,bj->bi", rot.transpose(0, 2, 1), -root_xy0)
    rs = root_state.astype(np.float64)
    root_rotm = R.from_quat(rs[:, :, 3:7].reshape(-1, 4)).as_matrix().reshape(B, T, 3, 3)
    body_mat = np.zeros((B, T, 4, 4)); body_mat[..., :3, :3] = root_rotm
    body_mat[..., :3, 3] = rs[:, :, :3] + (body_pos[:, 0, 0] - rs[:, 0, :3])[:, None]; body_mat[..., 3, 3] = 1
    new = np.einsum("bij,btjk->btik", transf, body_mat)
    quat_new = R.from_matrix(new[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(B, T, 4)
    pos_new = new[..., :3, 3]
    vel_new = np.einsum("btj,bij->bti", rs[:, :, 7:10], transf[:, :3, :3])
    angvel_new = np.einsum("btj,bij->bti", rs[:, :, 10:13], transf[:, :3, :3])
    return p, np.concatenate([pos_new, quat_new, vel_new, angvel_new], -1)


def window_tokens_batch(body_pos, dof_state, root_state, action):
    """[B,T,...] raw physics -> (root [B,T,15], body [B,T,420]) float32; same result as window_tokens per window."""
    B, T = body_pos.shape[:2]
    cp, crs = canonicalize_batch(body_pos, root_state)
    roots, bodies = [], []
    # per-frame heading + local positions are cheap; do per window to keep the reference implementation exact
    for b in range(B):
        r = get_repr(cp[b], dof_state[b], crs[b])
        roots.append(np.concatenate([r["root_trans"], r["root_rot_6d"], r["root_trans_vel"], r["root_rot_vel"]], -1))
        bodies.append(np.concatenate([r["local_positions"], r["local_vel"], r["dof_pose_6d"], r["dof_vel"], action[b].astype(np.float64)], -1))
    return np.stack(roots).astype(np.float32), np.stack(bodies).astype(np.float32)
