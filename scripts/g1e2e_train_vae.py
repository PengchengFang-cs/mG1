"""Train the intent VAE on G1 proprio sequences (stage 1 of two).

MIND's intent mechanism needs a frozen VAE over STATE sequences before anything else: the holistic and
immediate intent predictors operate in its latent space, not on raw states. The architecture here is the
MIND line's `hml_phys/intent_vae.py` unchanged -- causal 1D-conv encoder/decoder, temporal downsampling 4,
optimal-sigma reconstruction (MotionStreamer's ReConsLoss) plus lambda_KL -- with only the input width
changed, 366-d SMPL state to our 51-d G1 proprio.

Selection split: this dataset has train and test only, so per project CLAUDE.md §1 test IS the evaluation
split and is used normally. Statistics come from train and are reused for test, never recomputed.

  srun --jobid=<id> --overlap --ntasks=1 bash -lc '
    source scripts/activate_h2h.sh
    python scripts/g1e2e_train_vae.py --out outputs/g1e2e/vae --steps 60000'
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/g1_e2e/rollouts_train.pkl")
    ap.add_argument("--rollouts-eval", default="data/g1_e2e/rollouts_test.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=16, help="sequence length; MIND uses 16")
    ap.add_argument("--gen-hz", type=int, default=25, help="generation rate; 16 frames = 0.64 s at 25 Hz (20 Hz does not divide 50)")
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--down-t", type=int, default=2, help="2 stride-2 stages = 4x temporal downsampling")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dilation", type=int, default=3)
    ap.add_argument("--kl", type=float, default=1e-5, help="lambda_KL (MIND appendix A)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from hml_phys.g1e2e_data import G1E2EStateSeq, PROPRIO_DIM, compute_stats
    from hml_phys.intent_vae import IntentVAE, sigma_vae_nll

    down = args.down_t ** 2
    tr = G1E2EStateSeq(args.rollouts, args.L, gen_hz=args.gen_hz, max_clips=args.max_clips, down=down)
    # test reuses train's statistics; recomputing them per split would leak the eval distribution in
    te = G1E2EStateSeq(args.rollouts_eval, args.L, gen_hz=args.gen_hz, down=down,
                       stats=(np.concatenate([tr.mean, np.zeros(21, np.float32)]),
                              np.concatenate([tr.std, np.ones(21, np.float32)])))
    print(f"train {len(tr.clips)} clips / {len(tr)} windows   test {len(te.clips)} / {len(te)}")
    print(f"L {args.L} at {args.gen_hz} Hz = {args.L / args.gen_hz:.2f} s   latent {args.latent}   "
          f"downsampling {down} -> {args.L // down} latent frames")
    np.savez(out / "stats.npz", mean=tr.mean, std=tr.std, mean_full=tr.mean_full,
             std_full=tr.std_full, gen_hz=args.gen_hz, L=args.L)

    dev = torch.device(args.device)
    model = IntentVAE(input_dim=PROPRIO_DIM, width=args.width, down_t=args.down_t, stride_t=2,
                      depth=args.depth, dilation_growth_rate=args.dilation, latent_dim=args.latent).to(dev)
    print(f"IntentVAE {model.num_params() / 1e6:.1f} M params, input_dim {PROPRIO_DIM}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=2,
                          drop_last=shuffle, persistent_workers=True)

    tl, el = loader(tr, True), loader(te, False)

    def evaluate():
        model.eval()
        rec, kl, n = 0.0, 0.0, 0
        with torch.no_grad():
            for b in el:
                x = b["x"].to(dev)
                xr, mu, lv = model(x)
                rec += float(((xr - x) ** 2).mean()) * x.shape[0]
                kl += float((-0.5 * (1 + lv - mu ** 2 - lv.exp())).sum(-1).mean()) * x.shape[0]
                n += x.shape[0]
        model.train()
        return rec / n, kl / n

    hist, best, step, t0 = [], float("inf"), 0, time.time()
    while step < args.steps:
        for b in tl:
            if step >= args.steps:
                break
            x = b["x"].to(dev)
            xr, mu, lv = model(x)
            rec = sigma_vae_nll(xr, x) / x.numel()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).sum(-1).mean()
            loss = rec + args.kl * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"step {step} loss={float(loss):.4f} rec={float(rec):.4f} kl={float(kl):.2f} "
                      f"{(time.time() - t0) / step * 1000:.0f}ms/it", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                e_rec, e_kl = evaluate()
                hist.append(dict(step=step, test_mse=e_rec, test_kl=e_kl))
                print(f"  [eval] step {step} test MSE {e_rec:.5f}  KL {e_kl:.2f}", flush=True)
                ck = dict(model=model.state_dict(), args=vars(args), step=step,
                          mean=tr.mean, std=tr.std, mean_full=tr.mean_full, std_full=tr.std_full,
                          input_dim=PROPRIO_DIM)
                torch.save(ck, out / "latest.pt")
                if e_rec < best:
                    best = e_rec
                    torch.save(ck, out / "best.pt")
                    print(f"  [eval] new best test MSE {best:.5f}", flush=True)
                (out / "history.json").write_text(json.dumps(hist, indent=1))

    print(f"done, {step} steps in {(time.time() - t0) / 60:.1f} min, best test MSE {best:.5f}")


if __name__ == "__main__":
    main()
