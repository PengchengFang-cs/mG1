"""MIND intent arch, offline probe (docs/07 §21.5): how much do the executed actions depend on text, on the intents, on the
sparse history, versus the sampling noise? One pass, fixed test windows, t = 0.5, no rollout.

  no_text      : CLIP('') for the adapter AND the policy (intents recomputed from the empty text)
  swap_intent  : text kept, but the HIP / IIP hidden states come from another window (rolled across the batch)
  chain_intent : intents sampled exactly as in the closed loop (HIP from text, IIP from text + history)
  no_intent    : text kept, intent tokens masked out (a state training never produces: dropped samples lose both)
  no_sparse    : the sparse distant history rows marked invalid
  noise floor  : a different noise draw, everything else identical
RMS over the F_act executed action rows, normalised units.
"""
import argparse, sys, numpy as np, torch
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import TextCache, load_env_constants
from hml_phys.intent_policy_data import IntentPolicyDataset, collate_intent
from hml_phys.mc_rollout import load_policy
from hml_phys import intent_flow as ifl

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=512); ap.add_argument("--t", type=float, default=0.5)
args = ap.parse_args()
dev = "cuda"
model, stats, envc, a, step = load_policy(args.ckpt, device=dev)
tc = TextCache()
ds = IntentPolicyDataset("test", F_act=int(a["F_act"]), H=int(a["H"]), stats_path=a["stats"], text_cache=tc,
                         env_constants=load_env_constants(), train=False, randomize_history=False,
                         H_sparse=int(a["H_sparse"]), L_max=int(a["L_max"]), alpha=float(a["alpha"]))
sel = np.random.RandomState(0).choice(len(ds), args.n, replace=False)
b = collate_intent([ds[int(i)] for i in sel], tc)
b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
B = b["root"].shape[0]
act = ifl.action_channel_mask(dev)
mask, valid, fidx = b["observed_mask"], b["valid"], b["frame_index"]
x0 = ifl.policy_input(torch.cat([b["root"], b["body"]], -1), mask, act)
gen = ifl.generated_elements(mask, valid, act)
scal = torch.stack([b["progress"], b["total_len"] / 10.0], -1).float()
enc = lambda x: (model.vae.encode(x.float())[1] - model.lat_mean) / model.lat_std
t = torch.full((B,), args.t, device=dev); ones = torch.ones(B, device=dev)
g = torch.Generator(device=dev); g.manual_seed(0)
noise = torch.randn(x0.shape, device=dev, generator=g)
g2 = torch.Generator(device=dev); g2.manual_seed(1)
noise2 = torch.randn(x0.shape, device=dev, generator=g2)
e_tok, e_pool, e_len = tc.get(tc.index[""])
empty = (torch.from_numpy(e_tok).to(dev)[None].expand(B, -1, -1).contiguous(), torch.from_numpy(e_pool).to(dev)[None].expand(B, -1).contiguous(),
         torch.full((B,), int(e_len), device=dev))
text = (b["text_tokens"], b["text_pooled"], b["text_len"])


with torch.no_grad():                    # VAE in fp32, outside autocast -- as in training and the closed loop
    I_H_gt, I_I_gt, I_h = enc(b["holi"]), enc(b["fut"]), enc(b["hist"])
s_read = float(a.get("cond_aug_test", 0.75)) if float(a.get("cond_aug", 0.0)) > 0 else 1.0


@torch.no_grad()
def hidden(txt, I_H=None, I_I=None):
    """intent hidden states from ground-truth latents (default) or from given latents, read at the checkpoint's level."""
    I_H = I_H_gt if I_H is None else I_H
    I_I = I_I_gt if I_I is None else I_I
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mem, mv = model.adapter(txt[0].float(), txt[2])
        hH = ifl.intent_hidden(model.hip, I_H, s_read, g, mem=mem, mem_valid=mv)
        hI = ifl.intent_hidden(model.iip, I_I, s_read, g, mem=mem, mem_valid=mv, prefix_latent=I_h, scalars=scal, mem_extra=hH)
    return hH, hI


@torch.no_grad()
def chain_hidden():
    """intents exactly as in the closed loop: HIP sampled from the text, IIP sampled from text + history."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mem_c, mv_c = model.adapter(text[0].float(), text[2]); mem_u, mv_u = model.adapter(empty[0].float(), empty[2])
        I_H = ifl.sample_latent(model.hip, B, mem_c, mv_c, mem_u, mv_u, num_steps=32, cfg_scale=3.5, generator=g, device=dev)
        hH_c = ifl.intent_hidden(model.hip, I_H, s_read, g, mem=mem_c, mem_valid=mv_c)
        hH_u = ifl.intent_hidden(model.hip, I_H, s_read, g, mem=mem_u, mem_valid=mv_u)
        I_I = ifl.sample_latent(model.iip, B, mem_c, mv_c, mem_u, mv_u, num_steps=32, cfg_scale=3.5, generator=g,
                                prefix=I_h, scalars=scal, extra=hH_c, extra_u=hH_u, device=dev)
        hI_c = ifl.intent_hidden(model.iip, I_I, s_read, g, mem=mem_c, mem_valid=mv_c, prefix_latent=I_h, scalars=scal, mem_extra=hH_c)
    return hH_c, hI_c


@torch.no_grad()
def pred(txt=text, hid=None, keep=None, v=valid, nz=noise):
    hH, hI = hidden(txt) if hid is None else hid
    keep = torch.ones(B, dtype=torch.bool, device=dev) if keep is None else keep
    toks, tv = model.intent_tokens(hH, hI, keep)
    z = ifl.build_state_elem(x0, gen, t, noise=nz)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = model.policy(z, mask, t, txt[0], txt[1], txt[2], scal, valid=v, frame_index=fidx, extra_tokens=toks, extra_valid=tv)
    nh = int(b["n_hist"][0])
    return x.float()[:, nh:nh + int(a["F_act"])][..., act.bool()]


rms = lambda p, q: float(((p - q) ** 2).mean().sqrt())
hH, hI = hidden(text)                    # computed once, so every condition shares the same intent read-out
base = pred(hid=(hH, hI))
v_nos = valid.clone(); v_nos[:, :int(b["n_hist"][0]) - int(a["H"])] = 0.0
out = {
    "no_text (CLIP empty caption, intents recomputed)": rms(pred(txt=empty), base),
    "swap_intent (another window's intents)": rms(pred(hid=(hH.roll(1, 0), hI.roll(1, 0))), base),
    "chain_intent (sampled HIP -> IIP, as in the loop)": rms(pred(hid=chain_hidden()), base),
    "no_intent (tokens masked, text kept; unseen state)": rms(pred(hid=(hH, hI), keep=torch.zeros(B, dtype=torch.bool, device=dev)), base),
    "no_sparse (drop the distant history)": rms(pred(hid=(hH, hI), v=v_nos), base),
    "noise floor (different draw)": rms(pred(hid=(hH, hI), nz=noise2), base),
}
print(f"ckpt step {step}, {B} test windows, t={args.t}, action scale {float((x0[gen.bool()] ** 2).mean().sqrt()):.4f}")
for k, v in out.items():
    print(f"  {k:48s} RMS delta {v:.4f}")
