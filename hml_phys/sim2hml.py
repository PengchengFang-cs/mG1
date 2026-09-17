"""Simulated SMPL-humanoid trajectories -> HumanML3D 22-joint positions -> 263-d features.

Conventions
-----------
* Isaac Gym / PHC humanoid: 24 rigid bodies in MuJoCo order
  ['Pelvis','L_Hip','L_Knee','L_Ankle','L_Toe','R_Hip','R_Knee','R_Ankle','R_Toe','Torso','Spine',
   'Chest','Neck','Head','L_Thorax','L_Shoulder','L_Elbow','L_Wrist','L_Hand','R_Thorax','R_Shoulder',
   'R_Elbow','R_Wrist','R_Hand'], z-up, 30 fps.
* HumanML3D: SMPL joint order, first 22 joints (hands dropped), y-up, 20 fps, then the canonical
  Guo et al. 2022 `process_file` (uniform skeleton, floor, origin, face Z+, 263-d).

The 263-d pipeline is the vendored KV-Control copy of the official HumanML3D code
(hml_phys/t2m/motion_process.py). Its globals are injected here once (see `_init_globals`).
"""
import os
import numpy as np
import torch

from hml_phys.t2m import motion_process as mp
from hml_phys.t2m.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from hml_phys.t2m.skeleton import Skeleton

HML_ROOT = "/iridisfs/scratch/pf2m24/data/HumanML3D/HumanML3D"
EXAMPLE_VECS = os.path.join(HML_ROOT, "new_joint_vecs", "000021.npy")  # reference skeleton clip (official t2m_tgt_skel_id)

# smpl_joint = mujoco_body[MUJOCO_2_SMPL]  (from PHC / CLoSD rep_util.py)
MUJOCO_2_SMPL = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 19, 13, 15, 20, 16, 21, 17, 22, 18, 23]
SMPL_2_MUJOCO = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]
N_HML_JOINTS = 22
HML_FPS = 20
SIM_FPS = 30
FEET_THRE = 0.002  # official HumanML3D foot-contact threshold

# proper rotation (det=+1): Isaac (x, y, z-up) -> HumanML3D (x, y-up, z): (x, y, z) -> (x, z, -y)
_Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [0.0, 1.0, 0.0]], dtype=np.float64)  # row-vector convention: p_new = p @ M

_INITIALISED = False


def _init_globals():
    """Inject the module-level constants that the official process_file() expects."""
    global _INITIALISED
    if _INITIALISED:
        return
    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.r_hip, mp.l_hip = 2, 1
    mp.joints_num = N_HML_JOINTS
    mp.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    mp.kinematic_chain = t2m_kinematic_chain
    # Reference offsets from the official 000021 clip. Positions are recovered from the official 263-d
    # features (the local new_joints/ directory is not consistent with new_joint_vecs/ for every clip).
    vecs = torch.from_numpy(np.load(EXAMPLE_VECS).astype(np.float32))
    example = mp.recover_from_ric(vecs, N_HML_JOINTS).numpy().astype(np.float64)
    skel = Skeleton(mp.n_raw_offsets, mp.kinematic_chain, "cpu")
    mp.tgt_offsets = skel.get_offsets_joints(torch.from_numpy(example[0]))
    _INITIALISED = True


def resample_time(x, src_fps, dst_fps):
    """Linear time-resampling of x [T, ...] from src_fps to dst_fps (frame 0 kept, t = i/fps)."""
    x = np.asarray(x)
    T = x.shape[0]
    if src_fps == dst_fps or T < 2:
        return x.copy()
    dur = (T - 1) / src_fps
    n_out = int(np.floor(dur * dst_fps)) + 1
    t_src = np.arange(T) / src_fps
    t_dst = np.arange(n_out) / dst_fps
    flat = x.reshape(T, -1)
    out = np.empty((n_out, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t_dst, t_src, flat[:, c])
    return out.reshape((n_out,) + x.shape[1:])


def isaac_body_pos_to_hml_joints(body_pos, src_fps=SIM_FPS, dst_fps=HML_FPS):
    """[T, 24, 3] Isaac/MuJoCo-order z-up positions -> [T', 22, 3] SMPL-order y-up joints at dst_fps."""
    body_pos = np.asarray(body_pos, dtype=np.float64)
    assert body_pos.ndim == 3 and body_pos.shape[1:] == (24, 3), body_pos.shape
    smpl = body_pos[:, MUJOCO_2_SMPL][:, :N_HML_JOINTS]
    yup = smpl @ _Z_UP_TO_Y_UP
    return resample_time(yup, src_fps, dst_fps)


def joints_to_hml263(joints22):
    """[T, 22, 3] y-up 20 fps joints -> ([T-1, 263] float32, processed joints [T, 22, 3]).

    Exactly the official HumanML3D `process_file` (uniform skeleton to the 000021 reference,
    floor at min height, root XZ at origin, first frame facing Z+, foot-contact threshold 0.002).
    """
    _init_globals()
    joints22 = np.asarray(joints22, dtype=np.float64)
    assert joints22.ndim == 3 and joints22.shape[1:] == (N_HML_JOINTS, 3), joints22.shape
    assert joints22.shape[0] >= 2, "need at least 2 frames"
    data, ground_positions, positions, l_velocity = mp.process_file(joints22.copy(), FEET_THRE)
    return data.astype(np.float32), ground_positions.astype(np.float32)


def hml263_to_joints(data):
    """[T, 263] (raw, un-normalised) -> [T, 22, 3] global y-up joints (recover_from_ric)."""
    t = torch.from_numpy(np.asarray(data, dtype=np.float32))
    return mp.recover_from_ric(t, N_HML_JOINTS).numpy()


def isaac_body_pos_to_hml263(body_pos, src_fps=SIM_FPS):
    j = isaac_body_pos_to_hml_joints(body_pos, src_fps=src_fps)
    return joints_to_hml263(j)
