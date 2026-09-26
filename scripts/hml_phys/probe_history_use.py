"""Is the sparse long history actually used? Offline probe, no rollout.

For fixed test windows and a fixed noise/timestep, compare the model's x0 prediction of the executed
action channels under:
  full        : the window as trained
  no_sparse   : the sparse distant rows masked out (valid = 0)  -> removes ~5 s of past
  shuf_sparse : the sparse rows replaced by those of another window -> wrong past, same amount of it
  no_text     : the caption replaced by CLIP("")                -> the signal we know matters
  noise floor : two different noise draws with everything else identical
All differences are RMS over the first K executed action frames, in normalised units.
"""
import argparse, sys, numpy as np, torch
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import PhysWindowDataset, TextCache, load_env_constants, collate, ROOT
from hml_phys.mc_rollout import load_policy
from hml_phys import flow as fl
from hml_phys.tokens import ROOT_DIM

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=512)
ap.add_argument("--K", type=int, default=4); ap.add_argument("--t", type=float, default=0.5)
ap.add_argument("--split", default="test")
args = ap.parse_args()
dev = "cuda"
model, stats, envc, margs, step = load_policy(args.ckpt, device=dev)
tc = TextCache(); env = load_env_constants()
ds = PhysWindowDataset(args.split, H=int(margs["H"]), F=int(margs["F"]), stats_path=margs["stats"], text_cache=tc,
                       env_constants=env, train=False, H_sparse=int(margs["H_sparse"]), L_max=int(margs["L_max"]),
                       alpha=float(margs["alpha"]), randomize_history=False)
is_part = margs.get("arch", "two_stage") == "part"
sel = np.random.RandomState(0).choice(len(ds), args.n, replace=False)
b = collate([ds[int(i)] for i in sel], tc)
b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
B, T, _ = b["root"].shape
nh = int(b["n_hist"][0].item()) if "n_hist" in b else T - int(margs["F"])
g = torch.Generator(device=dev); g.manual_seed(0)
t = torch.full((B,), args.t, device=dev)
noise_r = torch.randn(b["root"].shape, device=dev, generator=g)
noise_b = torch.randn(b["body"].shape, device=dev, generator=g)
zr, _, _ = fl.build_state(b["root"], b["observed_mask"], t, noise=noise_r)
zb, _, _ = fl.build_state(b["body"], b["observed_mask"], t, noise=noise_b)
scal = torch.stack([b["progress"], b["total_len"] / 10.0], -1).float()
empty = tc.index.get("", -1); etok, epool, elen = tc.get(empty)
etok = torch.from_numpy(etok).to(dev)[None].expand(B, -1, -1).contiguous()
epool = torch.from_numpy(epool).to(dev)[None].expand(B, -1).contiguous()
elen = torch.full((B,), int(elen), device=dev)

@torch.no_grad()
def pred(root=None, body=None, valid=None, tok=None, pool=None, ln=None, fi=None, nr=None, nb=None):
    zr_, zb_ = (zr, zb) if nr is None else (fl.build_state(b["root"], b["observed_mask"], t, noise=nr)[0],
                                            fl.build_state(b["body"], b["observed_mask"], t, noise=nb)[0])
    r_, b_ = (root if root is not None else zr_), (body if body is not None else zb_)
    kw = dict(valid=valid if valid is not None else b["valid"],
              frame_index=fi if fi is not None else b["frame_index"])
    tk_, po_, ln_ = (tok if tok is not None else b["text_tokens"],
                     pool if pool is not None else b["text_pooled"],
                     ln if ln is not None else b["text_len"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if is_part:
            x = model(torch.cat([r_, b_], -1), b["observed_mask"], t, tk_, po_, ln_, scal, **kw)
            xb = x[..., ROOT_DIM:]
        else:
            _, xb = model(r_, b_, b["observed_mask"], t, tk_, po_, ln_, scal, **kw)
    return xb.float()[:, nh:nh + args.K, 351:420]           # executed action channels

base = pred()
rms = lambda a, c: float(((a - c) ** 2).mean().sqrt())
# 1. drop the sparse rows
v_nos = b["valid"].clone(); v_nos[:, :nh - int(margs["H"])] = 0.0
# 2. wrong sparse rows (rolled across the batch), same count
zr_sh, zb_sh = zr.clone(), zb.clone()
k = nh - int(margs["H"])
zr_sh[:, :k] = zr.roll(1, 0)[:, :k]; zb_sh[:, :k] = zb.roll(1, 0)[:, :k]
g2 = torch.Generator(device=dev); g2.manual_seed(1)
out = {
    "no_sparse (drop ~5 s of past)": rms(pred(valid=v_nos), base),
    "shuffled sparse (wrong past)": rms(pred(root=zr_sh, body=zb_sh), base),
    "no_text (CLIP empty caption)": rms(pred(tok=etok, pool=epool, ln=elen), base),
    "noise floor (different draw)": rms(pred(nr=torch.randn(b["root"].shape, device=dev, generator=g2),
                                             nb=torch.randn(b["body"].shape, device=dev, generator=g2)), base),
}
print(f"ckpt step {step}, {B} test windows, T={T}, sparse rows={k}, dense={margs['H']}, t={args.t}")
print(f"action scale (RMS of the target itself): {float((b['body'][:, nh:nh+args.K, 351:420] ** 2).mean().sqrt()):.4f}")
for kk, v in out.items():
    print(f"  {kk:34s} RMS delta {v:.4f}")
