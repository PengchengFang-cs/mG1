import importlib
for m in ("torch", "numpy", "transformers", "peft", "accelerate", "pytorch_lightning",
          "smplx", "omegaconf", "chumpy", "einops", "safetensors", "trimesh", "scipy"):
    try:
        mod = importlib.import_module(m)
        print("  %-18s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("  %-18s MISSING (%s)" % (m, type(e).__name__))
import torch
print("  cuda available:", torch.cuda.is_available())
