"""Merge a FRoM-W1 H-GPT LoRA adapter into Llama-3.1-8B.

Why not use their `lora_merge.py`: it builds the tokenizer from the adapter directory, and the
`tokenizer.json` shipped in every released adapter declares only 515 motion tokens (`<motion_id_0..514>`),
i.e. a 512-entry codebook. The adapter's own weights disagree -- `lm_head.weight` is [130307, 4096], and
130307 - 128256 (Llama-3.1-8B's vocabulary) = 2051 = 2048 + 3, which is the 2048-entry codebook of the only
released VQ-VAE (`vae.quantizer.codebook` is [2048, 1024]). So the shipped tokenizer file is stale and the
weights are the authoritative half.

This script therefore builds the tokenizer the way `hGPT/models/archs/hgpt_lm.py:92` does -- the base
tokenizer plus `<motion_id_i>` for i in range(codebook_size + 3) -- resizes to match, then loads and merges
the adapter. The resulting vocabulary must come out at 130307, which is asserted below.
"""
import argparse, json, os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

W = "/iridisfs/scratch/pf2m24/projects/motion_rebot/external/fromw1_weights"
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=f"{W}/Meta-Llama-3.1-8B")
ap.add_argument("--lora", default=f"{W}/hgpt/motionx/lora/llama-3.1-cot")
ap.add_argument("--out", default=f"{W}/merged/motionx-cot-2k")
ap.add_argument("--codebook", type=int, default=2048, help="must match the VQ-VAE actually used")
ap.add_argument("--dtype", default="bfloat16", help="Llama-3.1-8B ships in bfloat16; fp32 doubles the RAM")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base, legacy=True)
n_base = len(tok)
tok.pad_token = tok.eos_token                                  # hgpt_lm.py does this for decoder-only
added = tok.add_tokens([f"<motion_id_{i}>" for i in range(args.codebook + 3)])
print(f"[merge] base vocab {n_base} + {added} motion tokens -> {len(tok)}")

model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=getattr(torch, args.dtype))
model.resize_token_embeddings(len(tok))
print(f"[merge] resized lm_head to {tuple(model.get_output_embeddings().weight.shape)}")

model = PeftModel.from_pretrained(model, args.lora, torch_dtype=getattr(torch, args.dtype))
print("[merge] adapter loaded; merging")
model = model.merge_and_unload()

got = tuple(model.get_output_embeddings().weight.shape)
assert got[0] == len(tok) == n_base + args.codebook + 3, (
    f"vocabulary mismatch after merge: lm_head {got}, tokenizer {len(tok)}. The adapter was trained for a "
    f"different codebook size than --codebook {args.codebook}.")
print(f"[merge] merged lm_head {got}  OK")

os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=True)
tok.save_pretrained(args.out)
json.dump(dict(base=args.base, lora=args.lora, codebook=args.codebook, vocab=len(tok), dtype=args.dtype),
          open(os.path.join(args.out, "fromw1_merge.json"), "w"), indent=1)
print(f"[merge] wrote {args.out}")
