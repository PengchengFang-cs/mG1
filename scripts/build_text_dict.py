"""Merge UniPhys CLIP text dict with embeddings for any labels in our rollouts that are missing. Run in uniphys env from UniPhys/."""
import sys, os, glob, joblib, numpy as np, torch
ROOT = "/iridisfs/scratch/pf2m24/projects/motion_rebot"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "UniPhys"))
from adapt.data import clean_label
from uniphys.utils.clip_utils import load_and_freeze_clip, encode_text
src = os.path.join(ROOT, "UniPhys/data/babel_state-action-text-pairs/text_embedding_dict_clip.pkl")
dst = os.path.join(ROOT, "data/text_embedding_dict_clip_merged.pkl")
emb = joblib.load(dst) if os.path.exists(dst) else joblib.load(src)
labels = set()
for f in sys.argv[1:]:
    for fp in glob.glob(f):
        for r in joblib.load(fp)["rollouts"]:
            for seg in r["frame_ann"]:
                if len(seg) >= 3: labels.add(clean_label(str(seg[2])))
missing = sorted(l for l in labels if l not in emb)
print("labels", len(labels), "missing", len(missing), missing[:10])
if missing:
    clip_model = load_and_freeze_clip(clip_version="ViT-B/32", device="cuda")
    for i in range(0, len(missing), 256):
        batch = missing[i:i + 256]
        e = encode_text(clip_model, batch).float().cpu().numpy()
        for l, v in zip(batch, e): emb[l] = v.astype(np.float32)
joblib.dump(emb, dst); print("saved", dst, "size", len(emb))
