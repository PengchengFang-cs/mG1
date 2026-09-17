"""CLIP ViT-L/14 text features for every caption in the physics dataset (train/val/test) + rollout items.
Writes data/humanml3d_phys/text_cache_clipL14/{captions.json, tokens.npy [N,50,768] fp16, pooled.npy [N,768] fp32, lengths.npy}.
Same encoder path as MotionCraft FrozenCLIPTextEncoder (token features from the last transformer layer after ln_final,
pooled = EOT token @ text_projection), truncated/padded to 50 tokens.
"""
import argparse, json, os, sys
import numpy as np, joblib, torch
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
import clip

ap = argparse.ArgumentParser()
ap.add_argument("--clip_path", default="checkpoints/clip/ViT-L-14.pt")
ap.add_argument("--max_tokens", type=int, default=50)
ap.add_argument("--out", default="data/humanml3d_phys/text_cache_clipL14")
ap.add_argument("--batch", type=int, default=256)
args = ap.parse_args()

def norm(c):
    return " ".join(c.strip().split())

caps = set()
for sp in ["train", "val", "test"]:
    d = joblib.load(f"data/humanml3d_phys/hml_phys_{sp}.pkl")
    for texts in d["texts"]:
        for t in texts:
            caps.add(norm(t["caption"]))
for f in ["rollout_items_test_random.json", "rollout_items_test.json"]:
    p = os.path.join("data/humanml3d_phys", f)
    if os.path.exists(p):
        for it in json.load(open(p)):
            caps.add(norm(it["caption"]))
caps.add("")  # empty caption = unconditional (text dropout / CFG), encoded like MotionCraft (CLIP of "")
caps = sorted(caps)
print(f"{len(caps)} unique captions")
model, _ = clip.load(args.clip_path, device="cuda", jit=False)
model.eval()
N, L = len(caps), args.max_tokens
os.makedirs(args.out, exist_ok=True)
tok = np.zeros((N, L, model.ln_final.weight.shape[0]), np.float16)  # in RAM; GPFS memmap writes stall (mmapLock)
pooled = np.zeros((N, model.text_projection.shape[1]), np.float32); lengths = np.zeros(N, np.int32)
with torch.no_grad():
    for b in range(0, N, args.batch):
        texts = caps[b:b + args.batch]
        t = clip.tokenize(texts, truncate=True).cuda()
        x = model.token_embedding(t).type(model.dtype) + model.positional_embedding.type(model.dtype)
        x = x.permute(1, 0, 2); x = model.transformer(x); x = x.permute(1, 0, 2)
        x = model.ln_final(x).type(model.dtype)  # [B,77,768] token features
        eot = t.argmax(dim=-1)
        pl = x[torch.arange(x.shape[0]), eot] @ model.text_projection  # pooled [B,768]
        n_valid = (t != 0).sum(-1).clamp(max=L)
        feats = x[:, :L].float().cpu().numpy(); feats[np.arange(L)[None] >= n_valid.cpu().numpy()[:, None]] = 0
        tok[b:b + len(texts)] = feats.astype(np.float16); pooled[b:b + len(texts)] = pl.float().cpu().numpy(); lengths[b:b + len(texts)] = n_valid.cpu().numpy()
        if (b // args.batch) % 20 == 0: print(f"{b}/{N}", flush=True)
np.save(os.path.join(args.out, "tokens.npy"), tok); np.save(os.path.join(args.out, "pooled.npy"), pooled); np.save(os.path.join(args.out, "lengths.npy"), lengths)
json.dump(caps, open(os.path.join(args.out, "captions.json"), "w"))
print("done", args.out, "token dim", tok.shape, "pooled", pooled.shape)
