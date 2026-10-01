"""SMPL -> G1 21-DoF retargeting, shared by the FRoM-W1 reproduction and the end-to-end G1 line.

The gradient fit is transcribed from FRoM-W1's own generic retargeter,
`external/FRoM-W1/H-ACT/retarget/body_retarget/grad_fit_robot.py:process_data`, with their hyperparameters
(`grad_fit_robot.py:51-55`) and their forward kinematics (`body_retarget/robot.py:Humanoid_Batch`). The
released G1 body shape, `assets/beta/shape_optimized_g1.pkl`, is used as-is: body proportions do not change
between the 21/23/29-DoF variants, which are one robot with joints locked.

Two things here differ from their deployment path, deliberately, and both are load-bearing (docs/09 §6.6):

  * The root keeps the motion-library convention, `gt_root_rot = R * Q^-1` with Q = quat(0.5,0.5,0.5,0.5)
    and NO pre-rotation of the SMPL root. Their `load_amass_data` pre-rotates by Q and then conjugates,
    which is right for a consumer that also applies their `[2,0,1]` axis permutation and wrong for a motion
    library. Mixing the two puts the whole clip in a Q-rotated world: the robot spawns on its side and
    every episode dies on step one, while the fit error stays at a healthy ~5 cm because the SMPL target
    was rotated by the same Q. `check_upright()` below is the only thing that catches it.
  * The root is grounded per clip (lowest body point to 0.08 m, as `grad_fit_h1.py:219` does) rather than
    pinned at a constant height by their `FIX_BASE_HEIGHT`, which would erase every crouch and jump.
"""
import os
import sys
import xml.etree.ElementTree as ETree
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from torch.autograd import Variable

REPO = Path(__file__).resolve().parents[1]
RETARGET = REPO / "external/FRoM-W1/H-ACT/retarget"
H2H = REPO / "external/FRoM-W1/H-ACT/human2humanoid"
XML21 = H2H / "legged_gym/resources/robots/g1/xml/g1_21dof.xml"
XML29 = H2H / "legged_gym/resources/robots/g1/xml/g1_29dof.xml"

# grad_fit_robot.py:51-55
SMOOTH_WEIGHT = 1e-3
MAX_ITER_PATIENCE = 100
MIN_LR = 1e-5
INIT_LR = 1e-1
WEIGHT_DECAY = 1e-5
MAX_ITER = 1000
TARGET_FR = 30
# cfg_g1/phc/phc_base.yaml: the 21 of 29 non-root joints the policy actuates
PICKED_JOINT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 22, 23, 24, 25]
N_EXTEND = 3


class Retargeter:
    """Holds the SMPL parser, the released G1 betas and the robot FK; `fit()` does one clip."""

    def __init__(self, device="cuda:0"):
        # The retarget package resolves "assets/...", "models/smpl" relative to its own root, and importing
        # body_retarget loads the SMPL parser and the G1 betas at module level, so chdir first.
        self._cwd = os.getcwd()
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(RETARGET))
        os.chdir(RETARGET)
        import joblib
        from body_retarget.robot import Humanoid_Batch
        from body_retarget.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES
        from hml_phys.g1_21dof_config import G121DOFConfig

        self.device = torch.device(device)
        self.cfg = G121DOFConfig(str(XML21), SMPL_BONE_ORDER_NAMES)
        self.robot = Humanoid_Batch(cfg=self.cfg, device=self.device)
        # joints_range from the MJCF has one row per JOINT (21); the dof variable has one row per non-root
        # BODY (23). Swap in the body-aligned table so clamp_ lines up and the two locked rows stay at 0.
        self.robot.joints_range = self.cfg.joints_range_expanded.to(self.device)

        shape, scale = joblib.load("assets/beta/shape_optimized_g1.pkl")
        self.shape, self.scale = shape.to(self.device), scale.to(self.device)
        self.smpl = SMPL_Parser(model_path="models/smpl", gender="neutral").to(self.device)
        os.chdir(self._cwd)

        self.names29 = body_names(XML29)
        self.nonroot29 = self.names29[1:]
        assert len(self.nonroot29) == 29, f"expected 29 non-root bodies, got {len(self.nonroot29)}"
        src_nonroot = self.cfg.body_names[1:]
        self.row_map = [self.nonroot29.index(n) for n in src_nonroot]
        self.unactuated = [i for i in range(29) if i not in PICKED_JOINT]
        assert len(self.unactuated) == 8

    def fit(self, poses, trans, mocap_fps, max_iter=MAX_ITER):
        """Raw AMASS `poses` (T,>=66) and `trans` (T,3) at `mocap_fps` -> one 29-layout library entry.

        Resampling follows `process_amass_db.py:175` -- the real framerate, not the hardcoded 30 that
        `grad_fit_robot.load_amass_data` leaves in place (which would play 120 Hz AMASS four times slow).
        """
        dev = self.device
        skip = max(1, int(round(float(mocap_fps) / TARGET_FR)))
        poses, trans_np = poses[::skip], trans[::skip]
        N = poses.shape[0]
        if N < 10:
            return None

        # SMPL, not SMPL-H: keep the 22 body joints, zero the hands (process_amass_db.py:206).
        pose_aa_np = np.concatenate([poses[:, :66], np.zeros((N, 6))], axis=-1).astype(np.float32)
        trans_t = torch.from_numpy(trans_np.astype(np.float32)).to(dev)
        pose_aa = torch.from_numpy(pose_aa_np).to(dev)

        with torch.no_grad():
            # zero-beta pass only to recover the root offset (grad_fit_robot.py:178)
            _, j0 = self.smpl.get_joints_verts(pose_aa, torch.zeros((1, 10)).to(dev), trans_t)
            root_trans = trans_t + (j0[:, 0] - trans_t)
            _, joints = self.smpl.get_joints_verts(pose_aa, self.shape, trans_t)
            root_pos = joints[:, 0:1]
            target = ((joints - root_pos) * self.scale + root_pos)[:, self.cfg.smpl_joint_pick_idx]

        gt_root_rot = torch.from_numpy(
            (sRot.from_rotvec(pose_aa_np[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()
            .astype(np.float32)
        ).to(dev)

        dof = Variable(torch.zeros((1, N, self.cfg.JOINT_NUM, 1), device=dev), requires_grad=True)
        opt = torch.optim.Adam([dof], lr=INIT_LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=10, factor=0.5, min_lr=MIN_LR)
        axis = self.cfg.ROBOT_ROTATION_AXIS[None].to(dev)
        zeros_ext = torch.zeros((1, N, N_EXTEND, 3), device=dev)
        lo = self.robot.joints_range[:, 0, None]
        hi = self.robot.joints_range[:, 1, None]

        patience = 0
        for _ in range(max_iter):
            pose_robot = torch.cat([gt_root_rot[None, :, None], axis * dof, zeros_ext], dim=2)
            fk = self.robot.fk_batch(pose_robot, root_trans[None])
            diff = fk["global_translation_extend"][:, :, self.cfg.robot_joint_pick_idx] - target
            # cfg.wrist_pick_idx is empty at 21 DoF, so the geodesic wrist term is dropped exactly as
            # grad_fit_robot.py:231 does for a robot without wrist joints.
            loss = diff.norm(dim=-1).mean() + SMOOTH_WEIGHT * torch.norm(dof[:, 1:] - dof[:, :-1], p=2)
            sched.step(loss)
            if opt.param_groups[0]["lr"] <= MIN_LR:
                patience += 1
            if patience >= MAX_ITER_PATIENCE:
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            dof.data.clamp_(lo, hi)

        dof.data.clamp_(lo, hi)
        with torch.no_grad():
            pose_robot = torch.cat([gt_root_rot[None, :, None], axis * dof, zeros_ext], dim=2)
            fk = self.robot.fk_batch(pose_robot, root_trans[None])
            err = float((fk["global_translation_extend"][:, :, self.cfg.robot_joint_pick_idx] - target)
                        .norm(dim=-1).mean().item())
            # grad_fit_h1.py:219 -- drop the clip so its lowest body sits 8 cm above the floor. No axis
            # permutation: the motion library is in the robot's own z-up frame.
            root_dump = root_trans.clone()
            root_dump[..., 2] -= fk["global_translation"][..., 2].min().item() - 0.08

        entry = self._to_29(pose_robot[0].detach(), dof[0, :, :, 0].detach(), root_dump, gt_root_rot, err)
        del dof, fk, pose_robot
        torch.cuda.empty_cache()
        return entry

    def _to_29(self, pose21, dof21, root_dump, gt_root_rot, err):
        """Re-lay the 21-DoF rows onto the 29-DoF skeleton the env's FK expects (docs/09 §6.13).

        Resolved by body NAME: the right arm shifts three rows between the layouts
        (right_shoulder_pitch_link is non-root 19 at 21 DoF and 22 at 29 DoF), which a hardcoded index
        table gets wrong by exactly that much.
        """
        T = pose21.shape[0]
        pose29 = torch.zeros((T, len(self.names29) + N_EXTEND, 3), dtype=pose21.dtype)
        pose29[:, 0] = pose21[:, 0].cpu()
        dof29 = torch.zeros((T, 29), dtype=dof21.dtype)
        for s, dst in enumerate(self.row_map):
            pose29[:, 1 + dst] = pose21[:, 1 + s].cpu()
            dof29[:, dst] = dof21[:, s].cpu()
        assert not dof29[:, self.unactuated].any(), "non-actuated dof is not identity"
        return dict(
            root_trans_offset=root_dump.squeeze().cpu().numpy(),
            pose_aa=pose29.numpy(),
            dof=dof29.numpy(),
            dof_picked=dof29[:, PICKED_JOINT].numpy(),
            root_rot=sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
            fps=TARGET_FR,
            body_names=self.names29 + list(self.cfg.Extend.extend_link_name),
            dof_names=self.cfg.actuated_joint_names,
            fit_err_m=err,
        )


def body_names(xml_file):
    """Depth-first body walk, matching from_mjcf in robot.py and torch_robot_humanoid_batch.py."""
    names = []

    def visit(node):
        names.append(node.attrib["name"])
        for child in node.findall("body"):
            visit(child)

    visit(ETree.parse(xml_file).getroot().find("worldbody").find("body"))
    return names


def check_upright(library, device="cpu"):
    """Read the library back through the env's OWN FK and config; the one check that catches a rotated world.

    The fit error cannot: a wrong root convention rotates the SMPL target and the robot together, so the fit
    stays self-consistent at a few centimetres. Upright means head_link -- a pelvis + 0.45 m extend -- sits
    well above the pelvis. Measured on a known-bad library the median was -0.000 m; correct is +0.448 m.
    """
    import yaml
    from easydict import EasyDict

    cwd = os.getcwd()
    sys.path.insert(0, str(H2H))
    os.chdir(H2H / "legged_gym")
    from phc.utils.torch_robot_humanoid_batch import Humanoid_Batch

    cfg = EasyDict(yaml.safe_load(open("legged_gym/cfg/cfg_g1/phc/phc_base.yaml")))
    cfg.ROBOT_ROTATION_AXIS = torch.tensor(cfg.ROBOT_ROTATION_AXIS, dtype=torch.float32)
    cfg.Extend = EasyDict(cfg.Extend)
    fk = Humanoid_Batch(cfg)
    os.chdir(cwd)

    kp = {fk.model_names[b]: j for j, b in enumerate(cfg.picked_link)}
    i_pel, i_head = kp["pelvis"], kp["head_link"]
    i_feet = [kp["left_ankle_roll_link"], kp["right_ankle_roll_link"]]

    rise, foot = [], []
    for v in library.values():
        pose = torch.tensor(v["pose_aa"], dtype=torch.float32)[None]
        tr = torch.tensor(v["root_trans_offset"], dtype=torch.float32)[None]
        g = fk.fk_batch(pose, tr)["global_translation_extend"][0]
        rise.append(float(np.median((g[:, i_head, 2] - g[:, i_pel, 2]).numpy())))
        foot.append(float(g[:, i_feet, 2].min()))
    rise, foot = np.array(rise), np.array(foot)
    stats = dict(head_above_pelvis_median=float(np.median(rise)),
                 frac_upright=float(np.mean(rise > 0.2)),
                 lowest_foot_median=float(np.median(foot)))
    assert stats["head_above_pelvis_median"] > 0.25, (
        f"median head-above-pelvis is {stats['head_above_pelvis_median']:.3f} m; upright clips should sit "
        f"near the +0.45 m head extend. A value near zero means the library is in a rotated world frame -- "
        f"check the root convention (R*Q^-1, no pre-rotation)."
    )
    assert stats["lowest_foot_median"] < 0.2, f"median lowest foot {stats['lowest_foot_median']:.3f} m"
    return stats
