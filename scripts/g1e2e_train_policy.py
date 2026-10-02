"""Train the MIND intent architecture end to end on G1 (stage 2 of two).

Three heads, equal weights, as MIND eq. 5:
  HIP    text -> holistic intent latents
  IIP    text + observed history intent + HIP hidden states -> immediate intent latents
  policy the action rows of the future, conditioned on the history token, the text, and the two intent
         predictors' HIDDEN STATES (not their latents -- MIND reads the hidden states, docs/07 §21.4)

Everything structural comes from the MIND line unchanged: `hml_phys/intent_model.IntentPolicy`, the frozen
intent VAE from stage 1, x0 prediction with the velocity-space loss and logit-normal t, text dropout 0.1
with one mask shared across the three heads. Two things are G1's own:

  * the policy backbone is FLAT. `PartPhysPolicyDiT` takes `part_dims` precisely so a flat input vector can
    stand in for the physics token's body groups, so G1 passes [72]. G1's 21 joints would need a body-part
    split defined from scratch, and the MIND line already measured that part structure does not pay.
  * the losses come from `hml_phys/g1e2e_flow.py`, the same operations over the 72-d layout.

Selection split: train and test only, so per CLAUDE.md §1 test IS the evaluation split. The periodic
test-split denoising loss picks the checkpoint; the number that gets reported must come from the closed
loop, not from this loss.

  srun --jobid=<id> --overlap --ntasks=1 bash -lc '
    source scripts/activate_h2h.sh
    python scripts/g1e2e_train_policy.py --vae outputs/g1e2e/vae/best.pt --out outputs/g1e2e/policy'
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
    ap.add_argument("--text-cache", default="data/g1_e2e/text_clipL14")
    ap.add_argument("--vae", required=True, help="frozen intent VAE from g1e2e_train_vae.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-hz", type=int, default=25)
    ap.add_argument("--H", type=int, default=16, help="history rows; 16 at 25 Hz = 0.64 s")
    ap.add_argument("--F", type=int, default=4, help="generated action rows")
    ap.add_argument("--obs-future", default="first", choices=["none", "first", "all"])
    ap.add_argument("--hidden", type=int, default=576, help="policy width; divisible by heads and parts")
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--depth-double", type=int, default=3)
    ap.add_argument("--depth-single", type=int, default=6)
    ap.add_argument("--intent-dim", type=int, default=384)
    ap.add_argument("--intent-heads", type=int, default=6)
    ap.add_argument("--intent-depth", type=int, default=4)
    ap.add_argument("--text-drop", type=float, default=0.1)
    ap.add_argument("--cond-aug", type=float, default=0.5,
                    help="MIND: intent hidden states read at s ~ U(cond_aug, 1) in training")
    ap.add_argument("--cond-aug-test", type=float, default=0.75, help="MIND's fixed test-time s")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--window-stride", type=int, default=2)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    from hml_phys.g1e2e_data import G1E2EWindows, TOKEN_DIM, PROPRIO_DIM, describe
    from hml_phys.g1e2e_flow import (action_channel_mask, build_state_elem, generated_elements,
                                     latent_loss, observed_mask, policy_input, policy_loss, velocity_pair)
    from hml_phys.intent_flow import intent_hidden
    from hml_phys.intent_model import IntentPolicy
    from hml_phys.intent_vae import IntentVAE
    from hml_phys import flow as fl

    tc = Path(args.text_cache).resolve()
    tok_path, len_path = tc / "tokens.pkl", tc / "lengths.pkl"
    assert tok_path.exists(), f"{tok_path} missing -- run scripts/g1e2e_build_text_cache.py first"

    # Normalisation comes FROM the VAE checkpoint, never recomputed here: the frozen latents only mean
    # what stage 1 taught them to mean if both stages scale the input identically.
    vck = torch.load(args.vae, map_location="cpu")
    assert "mean_full" in vck, (
        f"{args.vae} predates the shared-statistics change; retrain the VAE with the current script")
    stats = (vck["mean_full"].astype(np.float32), vck["std_full"].astype(np.float32))
    assert stats[0].shape[0] == TOKEN_DIM, f"VAE stats are {stats[0].shape[0]}-d, token is {TOKEN_DIM}"
    tr = G1E2EWindows(args.rollouts, tok_path, args.H, args.F, gen_hz=args.gen_hz,
                      stride=args.window_stride, obs_future=args.obs_future, max_clips=args.max_clips,
                      stats=stats, seed=args.seed)
    te = G1E2EWindows(args.rollouts_eval, tok_path, args.H, args.F, gen_hz=args.gen_hz,
                      stride=max(1, args.H), obs_future=args.obs_future,
                      stats=stats, seed=args.seed + 1)
    print(f"train {json.dumps(describe(tr))}")
    print(f"test  {json.dumps(describe(te))}")
    np.savez(out / "stats.npz", mean=tr.mean, std=tr.std, gen_hz=args.gen_hz, H=args.H, F=args.F)

    # frozen intent VAE
    ck = vck
    va = ck["args"]
    vae = IntentVAE(input_dim=ck["input_dim"], width=va["width"], down_t=va["down_t"], stride_t=2,
                    depth=va["depth"], dilation_growth_rate=va["dilation"], latent_dim=va["latent"]).to(dev)
    vae.load_state_dict(ck["model"])
    vae.eval().requires_grad_(False)
    assert ck["input_dim"] == PROPRIO_DIM, f"VAE expects {ck['input_dim']}-d state, G1 proprio is {PROPRIO_DIM}"
    assert va["gen_hz"] == args.gen_hz, (
        f"the VAE was trained at {va['gen_hz']} Hz and this run uses {args.gen_hz}; the latents encode "
        f"a different amount of real time at a different rate")
    n_lat = args.H // vae.down
    assert args.H % vae.down == 0, f"H={args.H} must be divisible by the VAE downsampling {vae.down}"
    print(f"frozen IntentVAE: input {ck['input_dim']}, latent {vae.latent_dim}, down {vae.down} "
          f"-> {n_lat} latent frames from {args.H} history rows")

    policy_kw = dict(hidden_dim=args.hidden, num_heads=args.heads, depth_double=args.depth_double,
                     depth_single=args.depth_single, text_mode="sentence_xattn",
                     text_cross_attention=True, n_scalar_cond=2, part_dims=[TOKEN_DIM])
    model = IntentPolicy(policy_kw, args.intent_dim, args.intent_heads, args.intent_depth,
                         text_token_dim=768).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"IntentPolicy {n_par / 1e6:.1f} M params   flat part_dims=[{TOKEN_DIM}]")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    act_mask = action_channel_mask(dev)
    lens = __import__("joblib").load(len_path)

    def batch_to_dev(b):
        x = b["x"].to(dev)
        text = b["text"].to(dev).float()
        text_len = torch.tensor([lens[k][0] for k in b["key"]], device=dev)
        return x, b["holi"].to(dev), text, text_len, b["scal"].to(dev)

    def hidden_level(B, train):
        """Noise level at which the intent hidden states are read. MIND augments the conditioning with
        s ~ U(s_min, 1) in training and uses a fixed 0.75 at test time; s = 1 is the clean read-out."""
        if args.cond_aug <= 0:
            return 1.0
        if not train:
            return float(args.cond_aug_test)
        return args.cond_aug + (1.0 - args.cond_aug) * torch.rand(B, device=dev)

    def losses(x, holi, text, text_len, scal, train=True):
        B, T, _ = x.shape
        obs = observed_mask(B, args.H, T, dev)
        gen = generated_elements(obs, act_mask)
        x_in = policy_input(x, obs, act_mask)

        # The frozen VAE encodes the observed history into the intent latents the two predictors live in.
        with torch.no_grad():
            lat_hist, _, _ = vae.encode(x[:, :args.H, :PROPRIO_DIM])

        # One text-dropout mask shared by all three heads, so a dropped sample is unconditional everywhere.
        keep = torch.rand(B, device=dev) >= (args.text_drop if train else 0.0)
        txt = text * keep[:, None, None]
        mem, mem_valid = model.adapter(txt, text_len)
        mem_valid = mem_valid & keep[:, None]

        # HIP: text -> holistic intent. Its target is the WHOLE CLIP resampled to H rows (the dataset's
        # `holistic` field), because that is what a caption describes -- not the current window. It also
        # keeps the latent count at what the intent DiT's positional embedding is sized for.
        with torch.no_grad():
            lat_holi, _, _ = vae.encode(holi)
        tH = fl.sample_t(B, dev)
        zH, _ = build_state_elem(lat_holi, torch.ones_like(lat_holi), tH)
        xH, _ = model.hip(zH, tH, mem, mem_valid)
        l_hip = latent_loss(*velocity_pair(xH, lat_holi, zH, tH))

        # IIP: text + the observed history latents (as a prefix) + HIP's hidden states (as extra memory).
        hH = intent_hidden(model.hip, lat_holi, hidden_level(B, train), mem=mem, mem_valid=mem_valid)
        tI = fl.sample_t(B, dev)
        zI, _ = build_state_elem(lat_hist, torch.ones_like(lat_hist), tI)
        xI, _ = model.iip(zI, tI, mem, mem_valid, prefix_latent=lat_hist, scalars=scal, mem_extra=hH)
        l_iip = latent_loss(*velocity_pair(xI, lat_hist, zI, tI))

        # Policy: generate the future ACTION rows, reading the two predictors' HIDDEN STATES (not their
        # latents) as extra tokens in its text stream (docs/07 §21.4-2).
        hI = intent_hidden(model.iip, lat_hist, hidden_level(B, train), mem=mem, mem_valid=mem_valid,
                           prefix_latent=lat_hist, scalars=scal, mem_extra=hH)
        toks, tval = model.intent_tokens(hH, hI, keep)
        t = fl.sample_t(B, dev)
        z, _ = build_state_elem(x_in, gen, t)
        x0_hat = model.policy(z, obs, t, txt, txt.mean(1), text_len, scal,
                              extra_tokens=toks, extra_valid=tval)
        l_act = policy_loss(*velocity_pair(x0_hat, x, z, t), gen)
        return l_hip, l_iip, l_act

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=4,
                          drop_last=shuffle, persistent_workers=True)

    tl, el = loader(tr, True), loader(te, False)

    def evaluate():
        model.eval()
        acc, n = np.zeros(3), 0
        with torch.no_grad():
            for b in el:
                x, holi, text, tlen, scal = batch_to_dev(b)
                ls = losses(x, holi, text, tlen, scal, train=False)
                acc += np.array([float(v) for v in ls]) * x.shape[0]
                n += x.shape[0]
        model.train()
        return acc / n

    hist, best, step, t0 = [], float("inf"), 0, time.time()
    while step < args.steps:
        for b in tl:
            if step >= args.steps:
                break
            x, holi, text, tlen, scal = batch_to_dev(b)
            l_hip, l_iip, l_act = losses(x, holi, text, tlen, scal)
            loss = l_hip + l_iip + l_act                      # MIND eq. 5, equal weights
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"step {step} loss={float(loss):.4f} hip={float(l_hip):.4f} "
                      f"iip={float(l_iip):.4f} act={float(l_act):.4f} gn={float(gn):.2f} "
                      f"{(time.time() - t0) / step * 1000:.0f}ms/it", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                e = evaluate()
                hist.append(dict(step=step, hip=e[0], iip=e[1], act=e[2], sum=float(e.sum())))
                print(f"  [eval] step {step} hip {e[0]:.4f} iip {e[1]:.4f} act {e[2]:.4f}", flush=True)
                ck_out = dict(model=model.state_dict(), args=vars(args), step=step,
                              mean=tr.mean, std=tr.std, policy_kw=policy_kw, vae=str(args.vae))
                torch.save(ck_out, out / "latest.pt")
                # Selection is on the ACTION loss alone: HIP overfits long before the policy does, so a
                # sum-based criterion would start picking checkpoints for the wrong reason.
                if e[2] < best:
                    best = e[2]
                    torch.save(ck_out, out / "best.pt")
                    print(f"  [eval] new best test action loss {best:.4f}", flush=True)
                (out / "history.json").write_text(json.dumps(hist, indent=1))

    print(f"done, {step} steps in {(time.time() - t0) / 60:.1f} min, best test action loss {best:.4f}")


if __name__ == "__main__":
    main()
