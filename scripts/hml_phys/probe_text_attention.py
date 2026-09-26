"""How much attention mass do motion tokens actually give the text tokens?

Our text is injected by MotionCraft's joint attention: motion and text tokens are concatenated and the
softmax is normalised over BOTH. MIND/SCRIPT instead use cross-attention, where the softmax is normalised
over the text tokens only, so text cannot be starved. This probe measures, per block, the fraction of the
softmax mass that motion queries put on text keys.

Reference points: with T_motion motion tokens and T_text valid text tokens, a uniform-attention model would
put T_text / (T_motion + T_text) of its mass on text. Much less than that means text is being starved.
"""
import argparse, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import PhysWindowDataset, TextCache, load_env_constants, collate
from hml_phys.mc_rollout import load_policy
from hml_phys import flow as fl
from hml_phys.mc_model import _blocks

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=256)
ap.add_argument("--t", type=float, default=0.5); ap.add_argument("--split", default="test")
args = ap.parse_args()
dev = "cuda"
model, stats, envc, margs, step = load_policy(args.ckpt, device=dev)
tc = TextCache(); env = load_env_constants()
# pre-v3 checkpoints have no sparse history; fall back to their contiguous window
ds = PhysWindowDataset(args.split, H=int(margs["H"]), F=int(margs["F"]), stats_path=margs["stats"], text_cache=tc,
                       env_constants=env, train=False, H_sparse=int(margs.get("H_sparse", 0)),
                       L_max=int(margs.get("L_max", margs["H"])), alpha=float(margs.get("alpha", 3.0)),
                       randomize_history=False)
is_v3 = "H_sparse" in margs
is_part = margs.get("arch", "two_stage") == "part"
sel = np.random.RandomState(0).choice(len(ds), args.n, replace=False)
b = collate([ds[int(i)] for i in sel], tc)
b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
B, T, _ = b["root"].shape
text_mode = getattr(model, "text_mode", "joint_tokens") if is_part else "joint_tokens"
sentence_mode = text_mode == "sentence_xattn"
n_text_slots_all = {"joint_tokens": b["text_tokens"].shape[1], "sentence_xattn": 1, "xattn_only": 0}[text_mode]
if n_text_slots_all == 0:
    print("text_mode xattn_only: the joint attention holds no text; only the cross-attention section below applies")

# monkey-patch the attention to record, for motion queries, the softmax mass on the trailing text keys
records = []
orig = _blocks._attention
def spy(q, k, v, key_valid, dropout_p, extra_key_count=0, extra_attn_bias=None):
    key_len = k.shape[2]
    if n_text_slots_all > 0 and key_len == T + n_text_slots_all:   # joint [motion | text] attention (not the cross-attention)
        scale = q.shape[-1] ** -0.5
        scores = (q.float() @ k.float().transpose(-2, -1)) * scale
        if key_valid is not None:
            scores = scores.masked_fill(~key_valid[:, None, None], -1e9)
        p = scores.softmax(-1)                        # [B,H,Nq,key_len]
        n_txt = key_len - T
        mass = p[:, :, :T, T:].sum(-1)                # motion queries -> text keys
        records.append(float(mass.mean()))
    return orig(q, k, v, key_valid, dropout_p, extra_key_count, extra_attn_bias)
_blocks._attention = spy
# the blocks call the module-level _attention, so patching the module attribute is enough

g = torch.Generator(device=dev); g.manual_seed(0)
t = torch.full((B,), args.t, device=dev)
zr, _, _ = fl.build_state(b["root"], b["observed_mask"], t, noise=torch.randn(b["root"].shape, device=dev, generator=g))
zb, _, _ = fl.build_state(b["body"], b["observed_mask"], t, noise=torch.randn(b["body"].shape, device=dev, generator=g))
scal = torch.stack([b["progress"], b["total_len"] / 10.0], -1).float()
fi = b["frame_index"] if is_v3 else torch.arange(T, device=dev)[None].expand(B, T).contiguous()
if is_part and getattr(model, "text_cross_attention", False):
    model.record_xattn, model.xattn_stats = True, []
with torch.no_grad():
    if is_part:
        z = torch.cat([zr, zb], -1)
        model(z, b["observed_mask"], t, b["text_tokens"], b["text_pooled"], b["text_len"], scal,
              valid=b["valid"], frame_index=fi)
    else:
        model(zr, zb, b["observed_mask"], t, b["text_tokens"], b["text_pooled"], b["text_len"], scal,
              valid=b["valid"], frame_index=fi)
_blocks._attention = orig

n_text_slots = b["text_tokens"].shape[1]
n_valid_text = 1.0 if sentence_mode else float(b["text_len"].float().mean())   # padded text slots are masked out of the softmax
uniform = n_valid_text / (T + n_valid_text)
print(f"ckpt step {step} | motion tokens T={T}, valid text tokens {n_valid_text:.1f} of {n_text_slots} slots")
print(f"uniform-attention reference: {100*uniform:.1f}% of the mass would be on text")
r = np.array(records) if records else np.zeros(0)
if is_part and len(r) == 0:
    pass                                             # xattn_only: nothing to report for the joint attention
elif is_part:   # v4: a single shared trunk, every block sees all six parts -> no per-stream split
    dd = [int(x) for x in margs["depth"].split(",")]
    print(f"blocks recorded: {len(records)} (part arch: {dd[0]} double + {dd[1]} single, one shared trunk)")
    for i, x in enumerate(r):
        print(f"  block {i:2d} ({'double' if i < dd[0] else 'single'}): text mass {100*x:6.2f}%")
    print(f"MEAN text attention mass: {100*r.mean():.2f}%  (uniform {100*uniform:.1f}%)  "
          f"| every block feeds all 6 parts, so this is the number to compare against v3's body stream")
else:
    print(f"blocks recorded: {len(records)} (root: {margs['root_depth']} double,single; body: {margs['body_depth']})")
    n_root = sum(int(x) for x in margs["root_depth"].split(","))
    for i, x in enumerate(r):
        print(f"  block {i:2d} ({'root' if i < n_root else 'body'}): text mass {100*x:6.2f}%")
    print(f"MEAN text attention mass: {100*r.mean():.2f}%  (uniform {100*uniform:.1f}%) | "
          f"root stream {100*r[:n_root].mean():.2f}%  body stream {100*r[n_root:].mean():.2f}%")
if is_part and getattr(model, "text_cross_attention", False):
    print("text cross-attention (softmax over text only, after each double block):")
    for i, st in enumerate(model.xattn_stats):
        print(f"  after double block {i}: gate tanh(g) = {st['gate']:+.3e}, |gate*update| / |motion| = {100*st['rel']:.2f}%")
