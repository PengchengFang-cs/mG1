"""CLIP ViT-L/14 features for the HumanML3D captions attached to the G1 references.

Same encoder path as the MIND line's `scripts/hml_phys/build_text_cache.py` and MotionCraft's
FrozenCLIPTextEncoder: token features from the last transformer layer after ln_final, pooled = the EOT
token through text_projection, truncated or padded to --max_tokens. Keeping the path identical matters
because MIND's text adapter is a two-layer transformer over exactly these token features; swapping the
encoder would change what the intent predictors are reading.

Output is keyed by CLIP ID so the dataset can look a clip's captions up directly:
    tokens.pkl   {clip_id: (n_caps, max_tokens, 768) float16}
    pooled.pkl   {clip_id: (n_caps, 768) float32}
    captions.json
A clip keeps ALL its captions (HumanML3D gives three); the dataset samples one per window, which is both
MIND's conditioning augmentation and free paraphrase augmentation.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def norm(c):
    return " ".join(str(c).strip().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-json", nargs="+", required=True,
                    help="refs_{split}.text.json from g1e2e_build_references.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-path", default=str(REPO / "checkpoints/clip/ViT-L-14.pt"))
    ap.add_argument("--max-tokens", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    per_clip = {}
    for p in args.text_json:
        for key, rec in json.loads(Path(p).read_text()).items():
            caps = [norm(c) for c in rec["captions"] if norm(c)]
            assert caps, f"clip {key} has no usable caption"
            per_clip[key] = caps
    # The empty caption is the unconditional state: text dropout and classifier-free guidance both need
    # ONE well-defined "no text" input. Zeroing a real caption's features instead leaves a state that
    # depends on that caption's length, so CFG would have nothing fixed to extrapolate from.
    uniq = sorted({c for caps in per_clip.values() for c in caps} | {""})
    print(f"clips {len(per_clip)}   unique captions {len(uniq)}   "
          f"per clip min {min(len(v) for v in per_clip.values())} "
          f"max {max(len(v) for v in per_clip.values())}")

    import clip
    model, _ = clip.load(args.clip_path, device=args.device, jit=False)
    model = model.eval().requires_grad_(False)

    feats, pools = {}, {}
    with torch.no_grad():
        for i in range(0, len(uniq), args.batch):
            chunk = uniq[i:i + args.batch]
            tok = clip.tokenize(chunk, truncate=True).to(args.device)
            x = model.token_embedding(tok).type(model.dtype) + model.positional_embedding.type(model.dtype)
            x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = model.ln_final(x).type(model.dtype)                      # (B, 77, 768)
            eot = x[torch.arange(x.shape[0]), tok.argmax(dim=-1)]
            pooled = (eot @ model.text_projection).float().cpu().numpy()
            lengths = (tok.argmax(dim=-1) + 1).clamp(max=args.max_tokens).cpu().numpy()
            # Zero everything past each caption's own length, as scripts/hml_phys/build_text_cache.py:51
            # does. ln_final emits activations at all 50 positions -- measured mean|x| 0.76, max 8.3 at
            # padding slots -- so leaving them in means any consumer that masks with a different
            # caption's length attends to high-magnitude non-text vectors presented as text.
            xt = x[:, : args.max_tokens].clone()
            idx = torch.arange(args.max_tokens, device=xt.device)[None]
            lt = torch.as_tensor(lengths, device=xt.device)[:, None]
            xt = xt.masked_fill((idx >= lt)[..., None], 0.0)
            tokens = xt.half().cpu().numpy()
            for j, c in enumerate(chunk):
                feats[c] = (tokens[j], int(lengths[j]))
                pools[c] = pooled[j]
            if i % (args.batch * 8) == 0:
                print(f"  {min(i + args.batch, len(uniq))}/{len(uniq)}")

    tok_out, pool_out, len_out = {}, {}, {}
    for key, caps in per_clip.items():
        tok_out[key] = np.stack([feats[c][0] for c in caps])
        len_out[key] = np.array([feats[c][1] for c in caps], np.int64)
        pool_out[key] = np.stack([pools[c] for c in caps]).astype(np.float32)

    tok_out["__uncond__"] = np.stack([feats[""][0]])
    len_out["__uncond__"] = np.array([feats[""][1]], np.int64)
    pool_out["__uncond__"] = np.stack([pools[""]]).astype(np.float32)
    print(f"unconditional CLIP(''): {len_out['__uncond__'][0]} tokens")

    joblib.dump(tok_out, out / "tokens.pkl")
    joblib.dump(len_out, out / "lengths.pkl")
    joblib.dump(pool_out, out / "pooled.pkl")
    (out / "captions.json").write_text(json.dumps(per_clip, ensure_ascii=False, indent=1))
    s = next(iter(tok_out.values()))
    print(f"\ntokens {s.shape} float16   pooled {next(iter(pool_out.values())).shape} float32")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
