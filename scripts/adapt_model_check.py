import sys, os, torch, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adapt.model import DenoiserTransformer
from adapt.diffusion import TokenDiffusion
dev = "cuda"
m = DenoiserTransformer().to(dev); dm = TokenDiffusion(m).to(dev)
print("params (M):", sum(p.numel() for p in m.parameters()) / 1e6)
x0 = torch.randn(8, 20, 125, device=dev); text = torch.randn(8, 512, device=dev)
loss, per = dm.loss(x0, text); print("loss", float(loss), "per-token shape", tuple(per.shape), "hist loss ~", float(per[:, :5].mean()))
loss.backward(); print("backward ok")
m.eval(); t = time.time()
with torch.no_grad():
    for _ in range(20): out = dm.sample(x0[:1, :5], 15, text[:1], steps=2, guidance=2.5)
torch.cuda.synchronize(); print("sample out", tuple(out.shape), " ms/step(2 DDIM, CFG)", round((time.time() - t) / 20 * 1000, 1))
