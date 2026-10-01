"""A 21-DoF G1 retarget config, derived from the policy's own MJCF rather than transcribed by hand.

FRoM-W1's released `retarget/body_retarget/robot_config.py` only has a 29-DoF `G1Config`
(`assets/robot/g1/g1_29dof.xml`, 29 rotation axes including waist roll/pitch and both 3-DoF wrists),
but the released G1 tracking policy runs on 21 DoF (`env_cfg.json`: `g1_21dof.urdf`, `dof_num: 21`).
29 = 21 + waist_roll + waist_pitch + wrist(roll,pitch,yaw) x 2, so 21 DoF is a strict subset -- the
same physical robot with eight joints locked in the URDF.

Two structural consequences of that locking, both handled here:

  * In the 29-DoF MJCF every non-root body carries exactly one joint, so `ROBOT_ROTATION_AXIS` has one
    row per body. In the 21-DoF MJCF `waist_roll_link` and `torso_link` have no joint at all, so bodies
    (24) and joints (21) no longer match. `fk_batch` indexes by body, so the axis table must stay
    body-aligned: the two jointless bodies get a zero axis, which makes `axis * dof` identically zero
    and leaves them locked whatever the optimiser does.
  * `joints_range` comes from the MJCF and therefore has 21 rows while the dof variable has 23. The
    expanded range below re-aligns it so the usual `clamp_` works and pins the jointless rows at 0.

`WRIST_PICK` is empty because a 21-DoF G1 has no wrist joints to orient; `grad_fit_robot.process_data`
already branches on `wrist_pick_idx == []` and drops the geodesic term, so no code change is needed.
The arms stay constrained through the `*_hand_site` extend links, which hang off the elbows.
"""
import xml.etree.ElementTree as ETree

import torch

# Same three virtual links the released 29-DoF config adds; parents are resolved by NAME below,
# because the 21-DoF body list is shorter and the released integer indices (19, 26, 15) do not carry
# over.
EXTEND_LINKS = {
    "left_hand_site": ("left_elbow_link", [0.2, 0, 0]),
    "right_hand_site": ("right_elbow_link", [0.2, 0, 0]),
    "head_link": ("torso_link", [0, 0, 0.45]),
}

# Joint order the policy expects, i.e. the revolute order of g1_21dof.urdf. Asserted against the MJCF
# so a future asset swap cannot silently permute the action vector.
POLICY_DOF_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]


def _walk_mjcf(xml_file):
    """Depth-first body walk matching robot.py's `from_mjcf`, keeping each body's joint (if any)."""
    root = ETree.parse(xml_file).getroot()
    body_root = root.find("worldbody").find("body")
    bodies = []  # (body_name, joint_name|None, axis|None, range|None)

    def visit(node):
        joints = node.findall("joint")
        assert len(joints) <= 1, f"{node.attrib.get('name')} has {len(joints)} joints; expected <= 1"
        if joints:
            j = joints[0]
            axis = [float(v) for v in j.attrib.get("axis", "0 0 1").split()]
            rng = [float(v) for v in j.attrib["range"].split()] if "range" in j.attrib else None
            bodies.append((node.attrib["name"], j.attrib["name"], axis, rng))
        else:
            bodies.append((node.attrib["name"], None, None, None))
        for child in node.findall("body"):
            visit(child)

    visit(body_root)
    return bodies


class G121DOFConfig:
    """Mirrors the attribute surface of the released `G1Config`, built from the 21-DoF MJCF."""

    # Picks are unchanged from the released 29-DoF config: none of them is a wrist link, so all of
    # them exist on the 21-DoF robot too.
    ROBOT_JOINT_PICK = [
        "pelvis",
        "left_hip_pitch_link", "left_knee_link", "left_ankle_pitch_link",
        "right_hip_pitch_link", "right_knee_link", "right_ankle_pitch_link",
        "left_shoulder_pitch_link", "left_elbow_link",
        "right_shoulder_pitch_link", "right_elbow_link",
        "left_hand_site", "right_hand_site", "head_link",
    ]
    SMPL_JOINT_PICK = [
        "Pelvis",
        "L_Hip", "L_Knee", "L_Ankle",
        "R_Hip", "R_Knee", "R_Ankle",
        "L_Shoulder", "L_Elbow",
        "R_Shoulder", "R_Elbow",
        "L_Hand", "R_Hand", "Head",
    ]
    WRIST_PICK = []        # no wrist DoF at 21 -- disables the geodesic wrist term
    SMPL_WRIST_PICK = []

    # The motion library must keep vertical motion (crouching, sitting, jumping), so the root height is
    # grounded per clip rather than pinned. The released 29-DoF config sets FIX_BASE_HEIGHT = True with
    # 0.75 m, which is right for the deployment path and wrong here.
    FIX_BASE_HEIGHT = False
    FIX_BASE_HEIGHT_VALUE = 0.75

    class Extend:
        extend = True
        extend_link_name = list(EXTEND_LINKS)
        extend_local_rotation = [[1, 0, 0, 0]] * len(EXTEND_LINKS)
        extend_local_translation = [EXTEND_LINKS[k][1] for k in EXTEND_LINKS]
        extend_parent_idx = None  # filled in __init__ once the body list is known

    def __init__(self, xml_file, smpl_bone_order_names):
        self.xml_file = xml_file
        bodies = _walk_mjcf(xml_file)
        self.body_names = [b[0] for b in bodies]

        # Rows are bodies, not joints: body 0 is the root and carries the root rotation instead.
        non_root = bodies[1:]
        self.ROBOT_ROTATION_AXIS = torch.tensor(
            [(b[2] if b[2] is not None else [0.0, 0.0, 0.0]) for b in non_root],
            dtype=torch.float32,
        )
        self.JOINT_NUM = len(non_root)

        # Re-align joints_range onto the body rows; locked bodies get [0, 0].
        self.joints_range_expanded = torch.tensor(
            [(b[3] if b[3] is not None else [0.0, 0.0]) for b in non_root],
            dtype=torch.float32,
        )

        # Where the actuated joints sit among the body rows, in MJCF order.
        self.actuated_row_idx = [i for i, b in enumerate(non_root) if b[1] is not None]
        self.actuated_joint_names = [non_root[i][1] for i in self.actuated_row_idx]

        assert self.actuated_joint_names == POLICY_DOF_ORDER, (
            "MJCF joint order does not match the policy's dof order:\n"
            f"  mjcf  : {self.actuated_joint_names}\n"
            f"  policy: {POLICY_DOF_ORDER}"
        )

        self.ROBOT_JOINT_NAMES = self.body_names + self.Extend.extend_link_name
        self.Extend.extend_parent_idx = [
            self.body_names.index(EXTEND_LINKS[k][0]) for k in self.Extend.extend_link_name
        ]

        self.robot_joint_pick_idx = [self.ROBOT_JOINT_NAMES.index(j) for j in self.ROBOT_JOINT_PICK]
        self.smpl_joint_pick_idx = [smpl_bone_order_names.index(j) for j in self.SMPL_JOINT_PICK]
        self.wrist_pick_idx = [self.ROBOT_JOINT_NAMES.index(j) for j in self.WRIST_PICK]
        self.smpl_wrist_pick_idx = [smpl_bone_order_names.index(j) for j in self.SMPL_WRIST_PICK]
