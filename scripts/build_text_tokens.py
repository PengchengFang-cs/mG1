"""Precompute CLIP ViT-B/32 token-level features for every label in the text dict: {label: (tokens (L,512) float16, length)}.
Run in the uniphys env from UniPhys/ (needs clip). Output: data/text_tokens_clip.pkl"""
import sys, os, joblib, numpy as np, torch
ROOT = "/iridisfs/scratch/pf2m24/projects/motion_rebot"
sys.path.insert(0, os.path.join(ROOT, "UniPhys"))
import clip
labels = sorted(joblib.load(os.path.join(ROOT, "data/text_embedding_dict_clip_merged.pkl")).keys())
model, _ = clip.load("ViT-B/32", device="cuda", jit=False); model.eval()
out = {}
with torch.no_grad():
    for i in range(0, len(labels), 256):
        batch = labels[i:i + 256]
        tok = clip.tokenize(batch, truncate=True).cuda()                      # (B,77)
        x = model.token_embedding(tok).type(model.dtype) + model.positional_embedding.type(model.dtype)
        x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = model.ln_final(x).float()                                          # (B,77,512) token features
        lengths = tok.argmax(dim=-1) + 1                                       # EOT position + 1
        for l, feats, n in zip(batch, x.cpu().numpy(), lengths.cpu().numpy()):
            out[l] = (feats[:int(n)].astype(np.float16), int(n))
joblib.dump(out, os.path.join(ROOT, "data/text_tokens_clip.pkl"))
print("saved", len(out), "labels; max len", max(v[1] for v in out.values()), "example", labels[0], out[labels[0]][0].shape)
