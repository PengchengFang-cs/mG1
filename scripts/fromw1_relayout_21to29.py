"""Re-lay a 21-DoF-fitted G1 motion library into the 29-DoF body layout the env's FK expects.

`cfg_g1/phc/phc_base.yaml` is the authoritative spec for how the env reads a motion library, and it
builds its forward kinematics from `resources/robots/g1/xml/g1_29dof.xml` plus three extend links. So
`pose_aa` must carry one row per node of THAT skeleton: 1 root + 29 non-root bodies + 3 extends = 33.
`phc.utils.torch_robot_humanoid_batch.fk_batch` slices `pose[..., :len(self._parents), :]` and then
indexes every parent, so a 27-row array (our 21-DoF layout: 1 + 23 + 3) silently slices short and
blows up as `IndexError: index 0 is out of bounds for dimension 2 with size 0`.

No refit is needed, and this is the part worth being careful about: every one of the 24 bodies the
21-DoF MJCF shares with the 29-DoF one carries byte-identical `pos` and `quat`, so the kinematic chain
through the 21 actuated joints is the same model in both files. The fitted angles therefore transfer
exactly; only their row positions move. The eight joints the 21-DoF robot locks -- waist roll, waist
pitch, and wrist roll/pitch/yaw on each arm -- stay at identity, which is what `picked_joint`
(`[0..12, 15,16,17,18, 22,23,24,25]`) implies the policy can actuate.

Mapping is resolved by body NAME, never by a hardcoded index: the right arm shifts by three rows
between the two layouts (right_shoulder_pitch_link is non-root index 19 at 21 DoF and 22 at 29 DoF),
which is exactly the kind of off-by-three a literal index table invites.
"""
import argparse
import xml.etree.ElementTree as ETree
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
XML29 = REPO / "external/FRoM-W1/H-ACT/human2humanoid/legged_gym/resources/robots/g1/xml/g1_29dof.xml"
# phc_base.yaml: Extend.extend_link_name
N_EXTEND = 3
# phc_base.yaml: picked_joint -- indices into the 29 non-root joints that the policy actuates
PICKED_JOINT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 22, 23, 24, 25]


def body_names(xml_file):
    """Depth-first body walk, matching from_mjcf in both robot.py and torch_robot_humanoid_batch.py."""
    names = []

    def visit(node):
        names.append(node.attrib["name"])
        for child in node.findall("body"):
            visit(child)

    visit(ETree.parse(xml_file).getroot().find("worldbody").find("body"))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names29 = body_names(XML29)
    nonroot29 = names29[1:]
    assert len(nonroot29) == 29, f"expected 29 non-root bodies, got {len(nonroot29)}"
    print(f"29-DoF layout: {len(names29)} bodies + {N_EXTEND} extends -> pose_aa rows = {len(names29) + N_EXTEND}")

    merged = {}
    for p in args.inputs:
        d = joblib.load(p)
        dup = set(d) & set(merged)
        assert not dup, f"inputs overlap on {len(dup)} keys"
        merged.update(d)
        print(f"  {Path(p).name}: {len(d)}")
    print(f"  merged: {len(merged)}")

    out, row_map = {}, None
    for key, v in merged.items():
        src_bodies = v["body_names"]                 # 21-DoF bodies + its own extends
        n_src_nonroot = v["pose_aa"].shape[1] - 1 - N_EXTEND
        src_nonroot = src_bodies[1 : 1 + n_src_nonroot]

        if row_map is None:
            row_map = [nonroot29.index(n) for n in src_nonroot]
            print(f"\nrow map (21-DoF non-root -> 29-DoF non-root):")
            for s, (nm, dst) in enumerate(zip(src_nonroot, row_map)):
                flag = "" if s == dst else f"   <- shifts {dst - s:+d}"
                print(f"  {s:2d} -> {dst:2d}  {nm}{flag}")
            # The eight rows outside picked_joint must end up at identity. Two of them -- waist_roll
            # and the waist-pitch row carried by torso_link -- ARE in row_map, because the 21-DoF MJCF
            # still has those bodies, just without joints; our config gives them a zero rotation axis,
            # so they transfer as identity. The other six are the wrist rows, which have no source row
            # at all. Counting uncovered rows therefore gives 6, not 8; what matters is the result, so
            # assert on the output instead (done per clip below).
            unactuated = [i for i in range(29) if i not in PICKED_JOINT]
            assert len(unactuated) == 8, f"picked_joint leaves {len(unactuated)} rows, expected 8"
            print(f"not actuated ({len(unactuated)}): {[nonroot29[i] for i in unactuated]}")

        T = v["pose_aa"].shape[0]
        pose29 = np.zeros((T, len(names29) + N_EXTEND, 3), dtype=v["pose_aa"].dtype)
        pose29[:, 0] = v["pose_aa"][:, 0]                                   # root rotation
        for s, dst in enumerate(row_map):
            pose29[:, 1 + dst] = v["pose_aa"][:, 1 + s]
        # extend rows stay zero, exactly as grad_fit_robot.py writes them

        dof29 = np.zeros((T, 29), dtype=v["dof"].dtype)
        for s, dst in enumerate(row_map):
            dof29[:, dst] = v["dof_full"][:, s]
        dof_picked = dof29[:, PICKED_JOINT]

        # The 21 actuated angles must survive the move untouched.
        assert np.array_equal(dof_picked, v["dof"]), f"{key}: actuated dof changed during relayout"
        # Everything the policy cannot actuate must be exactly identity, in both representations.
        assert not dof29[:, unactuated].any(), f"{key}: non-actuated dof is not zero"
        assert not pose29[:, [1 + i for i in unactuated]].any(), f"{key}: non-actuated pose_aa is not zero"
        assert not pose29[:, -N_EXTEND:].any(), f"{key}: extend rows are not zero"

        out[key] = {
            "root_trans_offset": v["root_trans_offset"],
            "pose_aa": pose29,
            "dof": dof29,
            "dof_picked": dof_picked,
            "root_rot": v["root_rot"],
            "fps": v["fps"],
            "body_names": names29 + src_bodies[1 + n_src_nonroot :],
            "dof_names": v["dof_names"],
            "fit_err_m": v["fit_err_m"],
        }

    s = out[next(iter(out))]
    print(f"\npose_aa : {s['pose_aa'].shape}   dof: {s['dof'].shape}   dof_picked: {s['dof_picked'].shape}")
    print(f"bodies  : {len(s['body_names'])}")
    errs = np.array([v["fit_err_m"] for v in out.values()])
    frames = np.array([v["dof"].shape[0] for v in out.values()])
    print(f"clips   : {len(out)}   frames: {frames.sum()}   ({frames.sum() / 30 / 60:.1f} min)")
    print(f"fit err : mean {errs.mean():.4f}  median {np.median(errs):.4f}  p95 {np.percentile(errs, 95):.4f}  max {errs.max():.4f}")
    # Grounding puts the lowest body at 0.08 m; a root z near zero would mean the deployment-frame
    # axis permutation leaked in.
    zs = np.array([v["root_trans_offset"][:, 2].min() for v in out.values()])
    print(f"root z  : min {zs.min():.3f}  median {np.median(zs):.3f}  max {zs.max():.3f}")

    # ---- World-frame check, run through the env's own FK and config.
    # The fit error cannot catch a wrong root convention: grad_fit_robot.py pre-rotates the SMPL root
    # by Q = quat(0.5,0.5,0.5,0.5) and conjugates (Q R Q^-1), grad_fit_h1.py does neither (R Q^-1).
    # Mixing them rotates the target and the robot together, so the fit stays self-consistent at a few
    # centimetres while the whole clip sits in a Q-rotated world -- the robot then spawns on its side
    # and every episode terminates on step one. Only reading the output back in the consumer's frame
    # exposes it: upright means head_link (a pelvis + 0.45 m extend) sits well above the pelvis.
    import yaml
    from easydict import EasyDict
    import torch

    H2H = REPO / "external/FRoM-W1/H-ACT/human2humanoid"
    import os, sys
    sys.path.insert(0, str(H2H))
    cwd = os.getcwd()
    os.chdir(H2H / "legged_gym")
    from phc.utils.torch_robot_humanoid_batch import Humanoid_Batch

    pcfg = EasyDict(yaml.safe_load(open("legged_gym/cfg/cfg_g1/phc/phc_base.yaml")))
    pcfg.ROBOT_ROTATION_AXIS = torch.tensor(pcfg.ROBOT_ROTATION_AXIS, dtype=torch.float32)
    pcfg.Extend = EasyDict(pcfg.Extend)
    fk = Humanoid_Batch(pcfg)
    os.chdir(cwd)

    kp = {fk.model_names[b]: j for j, b in enumerate(pcfg.picked_link)}
    i_pel, i_head = kp["pelvis"], kp["head_link"]
    i_feet = [kp["left_ankle_roll_link"], kp["right_ankle_roll_link"]]

    rise, foot_lo = [], []
    for v in out.values():
        pose = torch.tensor(v["pose_aa"], dtype=torch.float32)[None]
        tr = torch.tensor(v["root_trans_offset"], dtype=torch.float32)[None]
        g = fk.fk_batch(pose, tr)["global_translation_extend"][0]
        rise.append(float(np.median((g[:, i_head, 2] - g[:, i_pel, 2]).numpy())))
        foot_lo.append(float(g[:, i_feet, 2].min()))
    rise, foot_lo = np.array(rise), np.array(foot_lo)

    print(f"\nworld-frame check (env FK, phc_base config)")
    print(f"  head above pelvis (m): median {np.median(rise):.3f}  "
          f"frac>0.2 {np.mean(rise > 0.2):.3f}")
    print(f"  lowest foot (m)      : median {np.median(foot_lo):.3f}  "
          f"frac<0.15 {np.mean(foot_lo < 0.15):.3f}")
    assert np.median(rise) > 0.25, (
        f"median head-above-pelvis is {np.median(rise):.3f} m; upright clips should sit near the "
        f"+0.45 m head extend. A value near zero means the library is in a rotated world frame -- "
        f"check the root convention (R*Q^-1, no pre-rotation)."
    )
    assert np.median(foot_lo) < 0.2, (
        f"median lowest foot is {np.median(foot_lo):.3f} m; grounding puts it near 0.08 m."
    )
    print("  OK")

    dst = Path(args.out).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, dst)
    print(f"\nwrote {dst}  ({dst.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
