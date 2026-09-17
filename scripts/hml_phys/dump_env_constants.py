"""Extract evaluation-start neutral state and PD offset/scale from a rollout pkl (meta.neutral_state) into env_constants.npz."""
import sys, numpy as np, joblib
src, out = sys.argv[1], sys.argv[2]
m = joblib.load(src)["meta"]["neutral_state"]
assert m is not None and "pd_action_offset" in m, "rollout was recorded before pd constants were added"
np.savez(out, neutral_body_pos=m["body_pos"].astype(np.float32), neutral_dof_pos=m["dof_state"][:, 0].astype(np.float32),
         neutral_root_state=m["root_state"].astype(np.float32), pd_offset=m["pd_action_offset"].astype(np.float32),
         pd_scale=m["pd_action_scale"].astype(np.float32), state_init=str(m["state_init"]))
print("saved", out, "root z", m["root_state"][2], "pd_scale range", m["pd_action_scale"].min(), m["pd_action_scale"].max())
