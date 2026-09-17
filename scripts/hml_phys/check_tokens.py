"""Verify hml_phys.tokens against the UniPhys reference implementation on random windows."""
import sys, numpy as np, joblib
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.tokens import check_against_uniphys, window_tokens
d = joblib.load("data/humanml3d_phys/hml_phys_val.pkl")
rng = np.random.RandomState(0); worst = {}
for _ in range(30):
    i = rng.randint(len(d["name"])); T = len(d["body_pos"][i])
    if T < 48: continue
    s = rng.randint(0, T - 48 + 1); sl = slice(s, s + 48)
    ok, err = check_against_uniphys(d["body_pos"][i][sl], d["dof_state"][i][sl], d["root_state"][i][sl])
    for k, v in err.items(): worst[k] = max(worst.get(k, 0), v)
    root, body = window_tokens(d["body_pos"][i][sl], d["dof_state"][i][sl], d["root_state"][i][sl], d["action"][i][sl])
    assert np.isfinite(root).all() and np.isfinite(body).all()
print("max abs diff vs UniPhys per field:", {k: f"{v:.2e}" for k, v in worst.items()})
# batched implementation vs per-window
from hml_phys.tokens import window_tokens_batch
idx = [i for i in rng.choice(len(d["name"]), 20) if len(d["body_pos"][i]) >= 48]
sl = slice(5, 53)
bp = np.stack([d["body_pos"][i][sl] for i in idx]); ds = np.stack([d["dof_state"][i][sl] for i in idx]); rs = np.stack([d["root_state"][i][sl] for i in idx]); ac = np.stack([d["action"][i][sl] for i in idx])
rb, bb = window_tokens_batch(bp, ds, rs, ac)
diff = max(max(np.abs(rb[j] - window_tokens(bp[j], ds[j], rs[j], ac[j])[0]).max(), np.abs(bb[j] - window_tokens(bp[j], ds[j], rs[j], ac[j])[1]).max()) for j in range(len(idx)))
print(f"batched vs per-window max abs diff: {diff:.2e}")
print("PASS" if max(worst.values()) < 1e-4 and diff < 1e-4 else "FAIL")
