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

# ---- v3: the canonicalisation origin must be honoured by the heading override too
import torch
from hml_phys.tokens import heading_quat, canonicalize, _qbetween, L_HIP, R_HIP, root_to_local_root
from hml_phys.mc_model import PhysPolicyDiT
worst_origin = 0.0
for _ in range(10):
    i = rng.randint(len(d["name"]))
    if len(d["body_pos"][i]) < 60: continue
    bp, dsx, rs = d["body_pos"][i][:60], d["dof_state"][i][:60], d["root_state"][i][:60]
    for origin in [0, 15, 31, 59]:
        cp, _, _ = canonicalize(bp, rs, origin=origin)
        q = heading_quat(cp, origin=origin)
        across = cp[:, L_HIP] - cp[:, R_HIP]; across = across.copy(); across[:, -1] = 0
        across /= np.linalg.norm(across, axis=-1, keepdims=True)
        fwd = np.cross(np.array([[0, 0, 1.0]]), across); fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True)
        qt = _qbetween(fwd, np.tile(np.array([[0, 1.0, 0]]), (len(fwd), 1)))
        ok = np.isfinite(qt).all(-1)
        ok[origin] = False                       # the origin is the degenerate frame the override handles
        ang = np.degrees(2 * np.arccos(np.clip(np.abs((q[ok] * qt[ok]).sum(-1)), 0, 1)))
        worst_origin = max(worst_origin, float(ang.max()))
        assert abs(abs(q[origin][3]) - 1.0) < 1e-6, "origin row must carry the 180-degree yaw"
print(f"heading override at origin: worst non-origin row error {worst_origin:.3f} deg (must be ~0)")

# ---- v3: local root must be identical whether the padding is at the front or at the end
st = np.load("data/humanml3d_phys/token_stats_v3.npz")
m = PhysPolicyDiT(hidden_dim=64, root_depth_double=1, root_depth_single=1, body_depth_double=1, body_depth_single=1,
                  local_root=True, root_stats=(st["root_mean"], st["root_std"]),
                  local_root_stats=(st["local_root_mean"], st["local_root_std"]))
i = 3; bp, dsx, rs, ac = d["body_pos"][i][:60], d["dof_state"][i][:60], d["root_state"][i][:60], d["action"][i][:60]
r, _ = window_tokens(bp, dsx, rs, ac, origin=15)
rn = ((r - st["root_mean"]) / st["root_std"]).astype(np.float32)
fi = np.arange(60) - 16
def run(pad_front, pad_back):
    T = 60 + pad_front + pad_back
    x = np.zeros((T, 15), np.float32); x[pad_front:pad_front + 60] = rn
    v = np.zeros(T, np.float32); v[pad_front:pad_front + 60] = 1
    f = np.concatenate([np.full(pad_front, fi[0]), fi, np.full(pad_back, fi[-1])])
    o = m.to_local_root(torch.from_numpy(x)[None], valid=torch.from_numpy(v)[None],
                        frame_index=torch.from_numpy(f)[None])[0].numpy()
    return o[pad_front:pad_front + 60]
a, b_, c = run(0, 0), run(11, 0), run(0, 7)
print("local root, front-padded vs unpadded: %.2e | back-padded vs unpadded: %.2e" %
      (np.abs(a - b_).max(), np.abs(a - c).max()))
ref = root_to_local_root(r, frame_index=fi)
ref = (ref - st["local_root_mean"]) / st["local_root_std"]
print("torch vs numpy reference (incl. last-row copy): %.2e" % np.abs(a - ref).max())
assert np.abs(a - b_).max() < 1e-5 and np.abs(a - c).max() < 1e-5 and np.abs(a - ref).max() < 1e-4
# batched implementation vs per-window
from hml_phys.tokens import window_tokens_batch
idx = [i for i in rng.choice(len(d["name"]), 20) if len(d["body_pos"][i]) >= 48]
sl = slice(5, 53)
bp = np.stack([d["body_pos"][i][sl] for i in idx]); ds = np.stack([d["dof_state"][i][sl] for i in idx]); rs = np.stack([d["root_state"][i][sl] for i in idx]); ac = np.stack([d["action"][i][sl] for i in idx])
rb, bb = window_tokens_batch(bp, ds, rs, ac)
diff = max(max(np.abs(rb[j] - window_tokens(bp[j], ds[j], rs[j], ac[j])[0]).max(), np.abs(bb[j] - window_tokens(bp[j], ds[j], rs[j], ac[j])[1]).max()) for j in range(len(idx)))
print(f"batched vs per-window max abs diff: {diff:.2e}")
print("PASS" if max(worst.values()) < 1e-4 and diff < 1e-4 and worst_origin < 1e-3 else "FAIL")
