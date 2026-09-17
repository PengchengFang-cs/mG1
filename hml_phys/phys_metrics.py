"""Physics-plausibility metrics on HumanML3D 22-joint trajectories (y-up, 20 fps, metres).

Floating / Penetration / Foot-sliding follow PhysDiff (Yuan et al. 2023) as implemented in CLoSD
(closd/diffusion_planner/data_loaders/humanml/utils/metrics.py); Skating ratio follows GMD.
Jerk is reported in two units because papers are inconsistent: m/s^3 (physical) and mm/frame^3.
All functions take a list of arrays [T_i, 22, 3] (one per clip, already cut to its valid length).
"""
import numpy as np
from scipy.ndimage import uniform_filter1d

FPS = 20.0
TOL = 0.005  # 5 mm tolerance (PhysDiff)
FEET = [10, 11]  # L_Foot, R_Foot in HumanML3D order


def floating_mm(joints_list):
    vals = []
    for j in joints_list:
        lowest = j[:, :, 1].min(axis=1)
        vals.append(np.clip(lowest - TOL, 0, None) * 1000.0)
    return float(np.mean(np.concatenate(vals)))


def penetration_mm(joints_list):
    vals = []
    for j in joints_list:
        lowest = j[:, :, 1].min(axis=1)
        vals.append(np.clip(lowest + TOL, None, 0) * -1000.0)
    return float(np.mean(np.concatenate(vals)))


def foot_sliding_mm(joints_list):
    """PhysDiff foot sliding: horizontal displacement (mm) of the lowest joint when it stays on the ground."""
    vals = []
    for j in joints_list:
        for t in range(j.shape[0] - 1):
            c = int(np.argmin(j[t, :, 1]))
            if j[t, c, 1] <= TOL and j[t + 1, c, 1] <= TOL:
                vals.append(np.linalg.norm(j[t + 1, c, [0, 2]] - j[t, c, [0, 2]]) * 1000.0)
            else:
                vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def skating_ratio(joints_list, thresh_height=0.05, thresh_vel=0.50, avg_window=5):
    """GMD skating ratio: fraction of frames where a foot below 5 cm moves faster than 0.5 m/s."""
    ratios = []
    for j in joints_list:
        feet = j[:, FEET]  # [T, 2, 3]
        vel = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1) * FPS  # [T-1, 2]
        vel_avg = uniform_filter1d(vel, axis=0, size=avg_window, mode="constant", origin=0)
        h = feet[:, :, 1]
        contact = (h[:-1] < thresh_height) & (h[1:] < thresh_height)
        skate = contact & (vel > thresh_vel) & (vel_avg > thresh_vel)
        skate = skate[:, 0] | skate[:, 1]
        ratios.append(skate.sum() / max(1, skate.shape[0]))
    return float(np.mean(ratios))


def jerk(joints_list):
    """Mean |third finite difference| over joints and frames. Returns (m/s^3, mm/frame^3)."""
    per_frame = []
    for j in joints_list:
        if j.shape[0] < 4:
            continue
        d3 = j[3:] - 3 * j[2:-1] + 3 * j[1:-2] - j[:-3]  # [T-3, 22, 3] in m/frame^3
        per_frame.append(np.linalg.norm(d3, axis=-1).mean(axis=1))
    if not per_frame:
        return float("nan"), float("nan")
    mmpf3 = float(np.mean(np.concatenate(per_frame))) * 1000.0
    return mmpf3 / 1000.0 * FPS ** 3, mmpf3


def all_metrics(joints_list):
    j_ms3, j_mmf3 = jerk(joints_list)
    return {
        "floating_mm": floating_mm(joints_list),
        "penetration_mm": penetration_mm(joints_list),
        "foot_sliding_mm": foot_sliding_mm(joints_list),
        "skating_ratio": skating_ratio(joints_list),
        "jerk_m_s3": j_ms3,
        "jerk_mm_frame3": j_mmf3,
    }
