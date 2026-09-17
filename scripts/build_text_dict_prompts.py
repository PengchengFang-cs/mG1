import sys, os, joblib, numpy as np, torch
ROOT = "/iridisfs/scratch/pf2m24/projects/motion_rebot"; sys.path.insert(0, os.path.join(ROOT, "UniPhys"))
from uniphys.utils.clip_utils import load_and_freeze_clip, encode_text
dst = os.path.join(ROOT, "data/text_embedding_dict_clip_merged.pkl"); emb = joblib.load(dst)
prompts = [l.strip() for l in open(os.path.join(ROOT, "data/adapt_eval_prompts.txt")) if l.strip()]
missing = [p for p in prompts if p not in emb]; print("prompts", len(prompts), "missing", len(missing))
if missing:
    m = load_and_freeze_clip(clip_version="ViT-B/32", device="cuda")
    e = encode_text(m, missing).float().cpu().numpy()
    for p, v in zip(missing, e): emb[p] = v.astype(np.float32)
    joblib.dump(emb, dst)
print("dict size", len(emb))
