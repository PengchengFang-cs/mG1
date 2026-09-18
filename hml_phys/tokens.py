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
def canonicalize(body_pos, root_state, origin=0):
    """UniPhys cano_seq_smpl_or_smplx for one window, with a selectable origin frame.

    origin = index of the frame whose root xy goes to the world origin and whose hip-across direction is
    rotated onto +y (v1/v2 used the oldest window frame, 0; v3 uses the newest history frame, N_s - 1, so
    that the frames being predicted sit closest to the origin — see docs/07 §15 改动 3b).
    Returns (cano_body_pos [T,24,3], cano_root_state [T,13], transf [4,4]).
    """
    p = body_pos.astype(np.float64).copy()
    root_xy0 = p[origin, 0] * np.array([1.0, 1.0, 0.0])
    p = p - root_xy0
    across = p[origin, R_HIP] - p[origin, L_HIP]
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
    body_mat = np.zeros((T, 4, 4)); body_mat[:, :3, :3] = root_rotm
    body_mat[:, :3, 3] = root_pos + (body_pos[origin, 0] - root_pos[origin]); body_mat[:, 3, 3] = 1
    new = transf[None] @ body_mat
    quat_new = R.from_matrix(new[:, :3, :3]).as_quat()
    pos_new = new[:, :3, 3]
    vel_new = rs[:, 7:10] @ transf[:3, :3].T
    angvel_new = rs[:, 10:13] @ transf[:3, :3].T
    return p, np.concatenate([pos_new, quat_new, vel_new, angvel_new], -1), transf


# ----------------------------------------------------------------------------- representation (UniPhys get_repr, return_last=True)
def heading_quat(body_pos, origin=0):
    """per-frame quaternion (w,x,y,z) rotating the hip-derived forward direction onto +y.

    NOTE: UniPhys get_repr unpacks face_joint_indx_robot=[5,1] as (l_hip, r_hip), i.e. the opposite order
    of the canonicalisation step, so its per-frame local frame faces -y. We reproduce that convention
    exactly so the two implementations agree numerically; it is self-consistent and harmless.

    The CANONICALISATION frame is the degenerate case: there the hip-across direction is exactly +x, so
    forward is exactly -y and qbetween(-y, +y) has both a zero axis and a zero scalar part. UniPhys hard-
    codes it to (0,0,0,1) -- which is a 180-degree yaw about z, NOT the identity -- and that is the correct
    limit. Since v3 canonicalises on the newest history frame, the override must be applied at `origin`,
    not at row 0 (applying it at row 0 would impose the origin's heading on an unrelated frame).
    """
    across = body_pos[:, L_HIP] - body_pos[:, R_HIP]
    across = across.copy(); across[:, -1] = 0
    across = across / np.sqrt((across ** 2).sum(-1))[:, None]
    forward = np.cross(np.array([[0, 0, 1.0]]), across)
    forward = forward / np.sqrt((forward ** 2).sum(-1))[..., None]
    q = _qbetween(forward, np.tile(np.array([[0, 1.0, 0]]), (len(forward), 1)))
    q[origin] = np.array([0, 0, 0, 1.0])  # exact limit at the canonicalisation frame (180-degree yaw about z)
    bad = np.isnan(q).any(-1)
    for i in np.where(bad)[0]:
        q[i] = q[i - 1] if i > 0 else np.array([0, 0, 0, 1.0])
    return q


def get_repr(cano_body_pos, dof_state, cano_root_state, origin=0):
    """Returns dict of per-frame arrays (T frames each), float32 — mirrors UniPhys get_repr(return_last=True)."""
    p = cano_body_pos.astype(np.float64); T, J, _ = p.shape
    q = heading_quat(p, origin=origin)
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


def window_tokens(body_pos, dof_state, root_state, action, origin=0):
    """One window [T frames] of raw physics -> (root [T,15], body [T,420]) float32 in the window-canonical frame."""
    cp, crs, _ = canonicalize(body_pos, root_state, origin=origin)
    r = get_repr(cp, dof_state, crs, origin=origin)
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
def canonicalize_batch(body_pos, root_state, origin=0):
    """body_pos [B,T,24,3], root_state [B,T,13] -> cano body_pos, cano root_state (same math as canonicalize)."""
    p = body_pos.astype(np.float64).copy(); B, T = p.shape[:2]
    root_xy0 = p[:, origin, 0] * np.array([1.0, 1.0, 0.0])
    p = p - root_xy0[:, None, None]
    across = p[:, origin, R_HIP] - p[:, origin, L_HIP]
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
    body_mat[..., :3, 3] = rs[:, :, :3] + (body_pos[:, origin, 0] - rs[:, origin, :3])[:, None]; body_mat[..., 3, 3] = 1
    new = np.einsum("bij,btjk->btik", transf, body_mat)
    quat_new = R.from_matrix(new[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(B, T, 4)
    pos_new = new[..., :3, 3]
    vel_new = np.einsum("btj,bij->bti", rs[:, :, 7:10], transf[:, :3, :3])
    angvel_new = np.einsum("btj,bij->bti", rs[:, :, 10:13], transf[:, :3, :3])
    return p, np.concatenate([pos_new, quat_new, vel_new, angvel_new], -1)


def window_tokens_batch(body_pos, dof_state, root_state, action, origin=0):
    """[B,T,...] raw physics -> (root [B,T,15], body [B,T,420]) float32; same result as window_tokens per window."""
    B, T = body_pos.shape[:2]
    cp, crs = canonicalize_batch(body_pos, root_state, origin=origin)
    roots, bodies = [], []
    # per-frame heading + local positions are cheap; do per window to keep the reference implementation exact
    for b in range(B):
        r = get_repr(cp[b], dof_state[b], crs[b], origin=origin)
        roots.append(np.concatenate([r["root_trans"], r["root_rot_6d"], r["root_trans_vel"], r["root_rot_vel"]], -1))
        bodies.append(np.concatenate([r["local_positions"], r["local_vel"], r["dof_pose_6d"], r["dof_vel"], action[b].astype(np.float64)], -1))
    return np.stack(roots).astype(np.float32), np.stack(bodies).astype(np.float32)



# ============================================================================================
# Local (velocity) root — the KiMoDo / ARDY / MotionCraft root->body bridge (docs/07 §15 改动 1)
# 4 dims: [yaw rate, dx * fps, dy * fps, root height], computed from the 15-d root token by finite
# differences; the last valid frame copies its predecessor. Our frame is z-up, so the horizontal
# plane is xy and the height is z (KiMoDo is y-up and uses xz + y).
# ============================================================================================
LOCAL_ROOT_DIM = 4
FPS = 30.0


def _heading_from_rot6d(rot6d):
    """[..., 6] -> unit horizontal heading [..., 2] (the body x axis projected on the ground).

    The 6-d rotation is stored as `R.as_matrix()[..., :-1].reshape(6)` (tokens are built that way, copying
    UniPhys), i.e. the first two COLUMNS flattened row-major: [M00, M01, M10, M11, M20, M21]. The body x
    axis in world coordinates is column 0 = (M00, M10, M20), so its horizontal part is elements 0 and 2.
    (Taking elements 0 and 1 would be the world x axis expressed in the body frame, whose angle is the
    NEGATED yaw.)
    """
    h = rot6d[..., [0, 2]]
    n = np.linalg.norm(h, axis=-1, keepdims=True)
    return h / np.clip(n, 1e-8, None)


def root_to_local_root(root, fps=FPS, valid=None, frame_index=None):
    """root [T,15] (un-normalised) -> local root [T,4] = [yaw rate, dx/dt, dy/dt, z].

    frame_index: optional [T] true frame offsets. Rows of a window may be non-contiguous in time (the
    sparse long history), so the finite differences are divided by the actual gap dt in frames; with
    contiguous rows dt = 1 and this reduces to the KiMoDo formula (multiply by fps).
    valid: optional [T] bool; the last valid row has no successor and copies its predecessor, exactly as
    KimodoRootConditioner does (hy273_root_conditioning.py:94-99).
    """
    root = np.asarray(root, dtype=np.float64)
    T = root.shape[0]
    a, b = ROOT_SLICES["root_trans"]; pos = root[:, a:b]
    a, b = ROOT_SLICES["root_rot_6d"]; head = _heading_from_rot6d(root[:, a:b])
    out = np.zeros((T, LOCAL_ROOT_DIM), np.float64)
    if T >= 2:
        if frame_index is None:
            dt = np.ones(T - 1)
        else:
            dt = np.maximum(np.diff(np.asarray(frame_index, dtype=np.float64)), 1.0)
        cross = head[:-1, 0] * head[1:, 1] - head[:-1, 1] * head[1:, 0]
        dot = (head[:-1] * head[1:]).sum(-1)
        out[:-1, 0] = np.arctan2(cross, dot) * fps / dt
        out[:-1, 1:3] = (pos[1:, :2] - pos[:-1, :2]) * fps / dt[:, None]
    out[:, 3] = pos[:, 2]
    if T >= 2:  # the last valid row has no successor: copy its predecessor's velocity channels.
        # Padding may sit at either end, so locate the last valid row rather than assuming n_valid - 1.
        last = T - 1 if valid is None else int(np.max(np.where(np.asarray(valid).astype(bool))[0]))
        if last >= 1:
            out[last, :3] = out[last - 1, :3]
        else:
            out[0, :3] = 0.0
    elif T >= 1:
        out[0, :3] = 0.0
    return out


# ----------------------------------------------------------------------------- sparse long history (SCRIPT eq. 6)
def sample_sparse_history(l_distant, n_sparse, alpha, rng):
    """SCRIPT's non-linear history downsampling: indices into a distant span of length `l_distant`,
    biased towards the recent end (index l_distant-1 is the most recent distant frame).

        I_i = floor( L_distant * (1 + ln(1 - u_i (1 - e^-alpha)) / alpha) ),   u_i ~ U[0,1]

    alpha -> 0 degenerates to uniform sampling. Returns a sorted array of at most `n_sparse` distinct
    indices (duplicates are dropped and the shortfall is filled from the most recent unused frames).
    """
    if l_distant <= 0 or n_sparse <= 0:
        return np.zeros(0, dtype=np.int64)
    if l_distant <= n_sparse:
        return np.arange(l_distant, dtype=np.int64)
    def draw(k):
        u = rng.rand(k)
        if alpha <= 1e-6:
            v = np.floor(l_distant * (1.0 - u))
        else:
            v = np.floor(l_distant * (1.0 + np.log(1.0 - u * (1.0 - np.exp(-alpha))) / alpha))
        return np.clip(v, 0, l_distant - 1).astype(np.int64)

    idx = np.unique(draw(n_sparse))
    for _ in range(16):  # redraw from the SAME law instead of filling with the most recent frames,
        if len(idx) >= n_sparse:  # which would bias the distribution towards the recent end
            break
        idx = np.unique(np.concatenate([idx, draw(n_sparse - len(idx))]))
    if len(idx) > n_sparse:
        idx = idx[:n_sparse]
    if len(idx) < n_sparse:  # pathological alpha: fall back to the most recent unused frames
        taken = np.zeros(l_distant, bool); taken[idx] = True
        spare = np.where(~taken)[0][::-1][: n_sparse - len(idx)]
        idx = np.union1d(idx, spare)
    return np.sort(idx).astype(np.int64)
