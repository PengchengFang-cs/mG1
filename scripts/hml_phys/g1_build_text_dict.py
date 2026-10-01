"""Precompute CLIP ViT-L/14 text features for the G1 closed-loop evaluation.

The rollout runs inside the Isaac Lab container, which has no `clip` package (and no network), while the policy
needs exactly the features `hml_phys/text_clip.ClipText` produces: word tokens after `ln_final`, the pooled
EOT projection, and the valid length.  So the features are computed here, outside the container, and the
rollout only loads an npz.

The empty caption is always included under the key "" -- it is the unconditional branch of the classifier-free
guidance and must come from the same encoder, not from zeros.
"""
import argparse, os, sys

import numpy as np

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.text_clip import ClipText

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default="data/g1_eval_prompts.txt")
ap.add_argument("--extra", nargs="*", default=[], help="further captions to include (e.g. BABEL labels)")
ap.add_argument("--babel_labels", default="", help="optional pkl of G1 rollouts; adds every BABEL label seen")
ap.add_argument("--meta_pkl", default="", help="optional motion meta pkl (name -> frame_ann); adds its labels. "
                                               "Needed by the shadow-policy check, which conditions on the "
                                               "label of the motion the tracker is following.")
ap.add_argument("--out", default="data/g1_rollouts/g1_eval_text_clip.npz")
args = ap.parse_args()

texts = [l.strip() for l in open(args.prompts) if l.strip()]
texts += [t for t in args.extra if t.strip()]
if args.meta_pkl:
    import joblib
    for name, m in joblib.load(args.meta_pkl).items():
        for a, b, lab, *_ in m.get("frame_ann", []):
            texts.append(str(lab))
if args.babel_labels:
    import joblib
    obj = joblib.load(args.babel_labels)
    rolls = obj["rollouts"] if isinstance(obj, dict) else obj
    for r in rolls:
        for a, b, lab, *_ in r.get("frame_ann", []):
            texts.append(str(lab))
seen, uniq = set(), []
for t in [""] + texts:                      # "" first, so index 0 is always the unconditional caption
    if t not in seen:
        seen.add(t); uniq.append(t)

enc = ClipText()
tok, pool, ln = [], [], []
for i in range(0, len(uniq), 256):
    a, b, c = enc.encode(uniq[i:i + 256])
    tok.append(a); pool.append(b); ln.append(c)
tok, pool, ln = np.concatenate(tok), np.concatenate(pool), np.concatenate(ln)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez_compressed(args.out, texts=np.array(uniq, dtype=object), tokens=tok.astype(np.float16),
                    pooled=pool.astype(np.float32), length=ln.astype(np.int32))
print(f"{len(uniq)} captions (index 0 = the empty one) -> {args.out}")
print(f"  tokens {tok.shape} {tok.dtype} | pooled {pool.shape} | length {ln.min()}..{ln.max()}")
print(f"  {os.path.getsize(args.out)/1e6:.1f} MB")
