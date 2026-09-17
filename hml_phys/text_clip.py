"""CLIP ViT-L/14 text features exactly as MotionCraft FrozenCLIPTextEncoder (tokens after ln_final, pooled = EOT @ text_projection)."""
import numpy as np, torch
import clip

CLIP_PATH = "/iridisfs/scratch/pf2m24/projects/motion_rebot/checkpoints/clip/ViT-L-14.pt"


class ClipText:
    def __init__(self, path=CLIP_PATH, device="cuda", max_tokens=50):
        self.model, _ = clip.load(path, device=device, jit=False); self.model.eval(); self.device = device; self.L = max_tokens
        self.token_dim = self.model.ln_final.weight.shape[0]; self.pooled_dim = self.model.text_projection.shape[1]

    @torch.no_grad()
    def encode(self, texts):
        """-> tokens [B,L,token_dim] float32 (zero beyond length), pooled [B,pooled_dim] float32, lengths [B] (>=1 for '')"""
        m = self.model
        t = clip.tokenize(list(texts), truncate=True).to(self.device)
        x = m.token_embedding(t).type(m.dtype) + m.positional_embedding.type(m.dtype)
        x = m.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = m.ln_final(x).type(m.dtype)
        pooled = (x[torch.arange(x.shape[0]), t.argmax(dim=-1)] @ m.text_projection).float()
        n_valid = (t != 0).sum(-1).clamp(max=self.L)
        feats = x[:, :self.L].float()
        feats[torch.arange(self.L, device=self.device)[None] >= n_valid[:, None]] = 0
        return feats.cpu().numpy(), pooled.cpu().numpy(), n_valid.cpu().numpy().astype(np.int32)
