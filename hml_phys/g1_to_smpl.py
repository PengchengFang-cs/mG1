"""Map G1 link positions onto the 22 SMPL joints, so the G1 rollouts can be scored by the same
text-motion retrieval evaluator we use on the SMPL side.

Why this is needed: the semantic metrics (R-precision, FID) are defined over HumanML3D's 263-d features,
which are built from 22 SMPL joints.  G1 is a 29-DoF robot with a different skeleton, so its motion has to be
expressed in that joint set first.  ADAPT and SENTINEL solve the same problem with TMR; TMR is not available
here, and our Guo-evaluator pipeline (`hml_phys/sim2hml.py` -> `hml_phys/evaluator.py`) already works end to
end, so the robot is mapped into the joint set that pipeline expects.

**What this mapping is and is not.**  It is a geometric correspondence, not a retarget: G1's proportions are
not a human's, and the evaluator was trained on human motion.  Absolute numbers from it are therefore NOT
comparable to published HumanML3D numbers -- the same discipline as docs/06 §2.2b.  They ARE comparable
between two of our own G1 policies, and against a G1 ceiling obtained by pushing the tracker's reference-
following rollouts through this identical path.  Always report that ceiling next to the policy rows.

Direct correspondences (13 of G1's 14 tracked links):
    pelvis -> Pelvis | {left,right}_hip_roll_link -> {L,R}_Hip | {left,right}_knee_link -> {L,R}_Knee
    {left,right}_ankle_roll_link -> {L,R}_Ankle | torso_link -> Spine3
    {left,right}_shoulder_roll_link -> {L,R}_Shoulder | {left,right}_elbow_link -> {L,R}_Elbow
    {left,right}_wrist_yaw_link -> {L,R}_Wrist
Synthesised (G1 has no counterpart):
    Spine1, Spine2  linear along pelvis->torso           Neck, Head  extrapolated above torso
    L/R_Collar      between torso and the shoulders      L/R_Foot    ahead of and below the ankle
The synthesised offsets are expressed as fractions of the robot's own pelvis->torso length, so they scale with
the platform instead of being hard-coded metres.
"""
import numpy as np

# HumanML3D / SMPL joint order (first 22; hands dropped)
SMPL22 = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3",
          "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
          "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]
SMPL_IDX = {n: i for i, n in enumerate(SMPL22)}

# SMPL joint -> the G1 link that carries it (exact substring of the link name).
# These eleven exist on every G1 variant we use.
DIRECT = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip_roll_link", "R_Hip": "right_hip_roll_link",
    "L_Knee": "left_knee_link", "R_Knee": "right_knee_link",
    "L_Ankle": "left_ankle_roll_link", "R_Ankle": "right_ankle_roll_link",
    "L_Shoulder": "left_shoulder_roll_link", "R_Shoulder": "right_shoulder_roll_link",
    "L_Elbow": "left_elbow_link", "R_Elbow": "right_elbow_link",
}

# These three exist on the 29-DoF G1 and NOT on the 21-DoF one. legged_gym loads the asset with
# `collapse_fixed_joints=True`, so the 21-DoF robot's rigid bodies are exactly `pelvis` plus the child
# link of each actuated joint -- 22 bodies, with no torso, no wrist and no head among them (verified
# with gym.get_asset_rigid_body_names on g1_21dof.urdf). When a link is absent the joint is synthesised
# from links that ARE present; see `_synth` for each rule and what it costs.
# Each value is a list of candidate link names, tried in order. The `*_site` / `head_link` names are the
# env's own EXTEND bodies (hml_phys/g1_21dof_config.py:30): the hand sites are the elbow link rotated by
# its OWN frame and offset 0.2 m, and the head is the pelvis offset 0.45 m. When a rollout saved
# `body_pos_ext` those three are present and exact, so they are preferred over anything synthesised from
# positions alone -- the hand in particular cannot be recovered from positions, because a bent elbow puts
# it off the upper arm's line.
OPTIONAL = {
    "Spine3": ["torso_link"],
    "L_Wrist": ["left_hand_site", "left_wrist_yaw_link"],
    "R_Wrist": ["right_hand_site", "right_wrist_yaw_link"],
    "Head": ["head_link"],
}

# The two offsets below are NOT invented here: both are the env's own virtual-point definitions, which
# `hml_phys/g1_21dof_config.py:30` (EXTEND_LINKS) transcribes and the retargeter was fitted against.
HEAD_UP_M = 0.45        # head  = pelvis + 0.45 along the trunk  (extra/extra_base.yaml parents it to body 0)
HAND_FWD_M = 0.20       # hand  = elbow  + 0.20 along the arm    (the teacher's own tracked hand point)


class G1ToSMPL:
    """Build once from the env's body-name list, then call on body positions."""

    def __init__(self, body_names):
        self.body_names = list(body_names)
        self.idx = {}
        for joint, link in DIRECT.items():
            self.idx[joint] = self._resolve(joint, link, required=True)
        self.missing = []
        for joint, cands in OPTIONAL.items():
            hit = next((h for h in (self._resolve(joint, c, required=False) for c in cands)
                        if h is not None), None)
            if hit is None:
                self.missing.append(joint)
            else:
                self.idx[joint] = hit
        self.n_direct = len(self.idx)

    def _resolve(self, joint, link, required):
        hit = [i for i, n in enumerate(self.body_names) if n == link]
        if not hit:                                       # fall back to a unique substring match
            hit = [i for i, n in enumerate(self.body_names) if link in n]
        if len(hit) == 1:
            return hit[0]
        if required:
            raise KeyError(f"link {link!r} for SMPL joint {joint!r}: {len(hit)} matches "
                           f"in {self.body_names}")
        return None

    def __call__(self, bp):
        """bp [..., n_bodies, 3] world link positions (z up) -> [..., 22, 3] SMPL joints, same frame/units."""
        bp = np.asarray(bp)
        out = np.zeros(bp.shape[:-2] + (22, 3), dtype=bp.dtype)
        g = lambda j: bp[..., self.idx[j], :]
        for joint, i in self.idx.items():
            out[..., SMPL_IDX[joint], :] = bp[..., i, :]

        pel = g("Pelvis")
        if "Spine3" in self.missing:
            # No torso body. The shoulder midpoint is the nearest point that is actually MEASURED, and
            # using it keeps the trunk vector real rather than guessed.
            tor = 0.5 * (g("L_Shoulder") + g("R_Shoulder"))
            out[..., SMPL_IDX["Spine3"], :] = tor
        else:
            tor = g("Spine3")
        spine = tor - pel                                  # the robot's own trunk vector: everything scales by it
        trunk = np.linalg.norm(spine, axis=-1, keepdims=True)
        up = np.where(trunk > 1e-6, spine / np.maximum(trunk, 1e-6), np.array([0.0, 0.0, 1.0], bp.dtype))

        for side in ("L", "R"):
            if f"{side}_Wrist" in self.missing:
                # No wrist body, and the 21-DoF G1 has no wrist DoF either, so the hand is a FIXED offset
                # in the elbow link's frame. We only store link POSITIONS, not rotations, so the offset's
                # direction is approximated by continuing the upper arm. That is exact for a straight arm
                # and wrong by up to HAND_FWD_M for a fully bent one. It is a deterministic distortion
                # applied identically to every row of the ladder, and layer 2 of the ladder (the
                # retargeted reference through this same mapper) is what measures its cost.
                sh, el = g(f"{side}_Shoulder"), g(f"{side}_Elbow")
                arm = el - sh
                na = np.linalg.norm(arm, axis=-1, keepdims=True)
                d = np.where(na > 1e-6, arm / np.maximum(na, 1e-6), up)
                out[..., SMPL_IDX[f"{side}_Wrist"], :] = el + d * HAND_FWD_M

        out[..., SMPL_IDX["Spine1"], :] = pel + spine / 3.0
        out[..., SMPL_IDX["Spine2"], :] = pel + spine * 2.0 / 3.0
        # The head is the env's own tracked head point: pelvis + 0.45 along the trunk (EXTEND_LINKS).
        # Using that rather than a fraction of the trunk matters -- STATUS.md §5.6 records that parenting
        # this point to the torso instead of the pelvis moved the retarget target 4.4 cm and biased the
        # fitted DoF. The neck then sits midway between the chest and the head.
        if "Head" in self.missing:
            out[..., SMPL_IDX["Head"], :] = pel + up * HEAD_UP_M
        out[..., SMPL_IDX["Neck"], :] = 0.5 * (tor + out[..., SMPL_IDX["Head"], :])
        # collars: most of the way from the torso towards each shoulder
        for side in ("L", "R"):
            sh = g(f"{side}_Shoulder")
            out[..., SMPL_IDX[f"{side}_Collar"], :] = tor + (sh - tor) * 0.45
        # feet: ahead of and below the ankle, along the pelvis->ankle horizontal direction
        for side in ("L", "R"):
            ank, knee = g(f"{side}_Ankle"), g(f"{side}_Knee")
            fwd = ank - knee
            fwd = fwd - (fwd * up).sum(-1, keepdims=True) * up      # horizontal component of the shank
            nf = np.linalg.norm(fwd, axis=-1, keepdims=True)
            fwd = np.where(nf > 1e-6, fwd / np.maximum(nf, 1e-6), 0.0)
            out[..., SMPL_IDX[f"{side}_Foot"], :] = ank + fwd * trunk * 0.18 - up * trunk * 0.06
        return out

    def describe(self):
        synth = ["Spine1", "Spine2", "Neck", "L_Collar", "R_Collar", "L_Foot", "R_Foot"]
        return (f"G1ToSMPL: {self.n_direct}/22 joints taken directly from links, "
                f"{22 - self.n_direct} synthesised ({', '.join(synth + self.missing)})"
                + (f"; this robot has no {', '.join(OPTIONAL[m][0] for m in self.missing)}, so "
                   f"{', '.join(self.missing)} are synthesised too" if self.missing else "")
                + f"; head = pelvis + {HEAD_UP_M} m, hand = elbow + {HAND_FWD_M} m along the arm")


def g1_body_pos_to_hml263(body_pos, mapper, src_fps=50.0):
    """[T, n_bodies, 3] world-frame (origin-subtracted, z-up) G1 link positions -> ([T'-1, 263], [T', 22, 3]).

    Same tail as the SMPL path (`hml_phys/sim2hml.isaac_body_pos_to_hml263`): rotate z-up -> y-up, resample to
    HumanML3D's 20 fps, then the official `process_file`.  Only the joint sourcing differs, because G1 is not
    an SMPL humanoid.
    """
    from hml_phys.sim2hml import HML_FPS, _Z_UP_TO_Y_UP, joints_to_hml263, resample_time
    j = mapper(np.asarray(body_pos, dtype=np.float64))          # [T, 22, 3] z-up
    yup = resample_time(j @ _Z_UP_TO_Y_UP, src_fps, HML_FPS)
    return joints_to_hml263(yup)
