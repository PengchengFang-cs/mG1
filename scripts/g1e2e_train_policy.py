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
import math
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
    ap.add_argument("--p-rest", type=float, default=0.1,
                    help="share of start-rest windows: history = 16 copies of the clip's first frame "
                         "with zero velocities and the hold action, future = the clip's opening. The "
                         "closed loop starts exactly there and the recorded data contains no such window "
                         "(dataset.py:19-21). Needs default_dof_pos in the rollouts meta.")
    ap.add_argument("--p-holi-exact", type=float, default=0.25,
                    help="probability that the holistic target is the exact test-time span rather than a "
                         "random sub-span (intent_data.holistic_crop_rows)")
    ap.add_argument("--max-mpjpe", type=float, default=0.0,
                    help="drop clips whose mean body tracking error exceeds this (m). 0 = keep all, which "
                         "is what the runs so far did: the recorder only removed the clips where the "
                         "tracker FELL, so a clip followed at 0.49 m average still carries its caption.")
    ap.add_argument("--hidden", type=int, default=576, help="policy width; divisible by heads and parts")
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--depth-double", type=int, default=3)
    ap.add_argument("--depth-single", type=int, default=6)
    ap.add_argument("--ablate-intent", default="none", choices=["none", "all", "hip", "iip"],
                    help="TRAIN an ablation of the MIND intent mechanism, to find out whether it earns "
                         "its place. 'all': the action policy never sees intent tokens, so it is "
                         "conditioned on text and proprio alone, and the HIP/IIP losses are dropped "
                         "since nothing consumes them. 'hip': the policy sees the holistic half only. "
                         "'iip': the policy sees the immediate half only -- note this does NOT remove "
                         "HIP from the architecture, because IIP takes HIP's hidden states as its "
                         "memory (mem_extra=hH), so 'iip' means 'the policy reads only the immediate "
                         "tokens', not 'HIP is gone'. Removing HIP outright is a deeper change.")
    ap.add_argument("--intent-dim", type=int, default=384)
    ap.add_argument("--intent-heads", type=int, default=6)
    ap.add_argument("--intent-depth", type=int, default=4)
    ap.add_argument("--intent-mlp", type=float, default=1.5,
                    help="SwiGLU ratio in HIP/IIP. 1.5 is the reference (train_intent_policy.py:32), "
                         "chosen to keep the total under 100M; the class default is 4.0.")
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--ema-every", type=int, default=10,
                    help="0 disables EMA. The reference rolls out EMA weights (mc_rollout.py:25,31) and "
                         "its reproduced R@1 0.4117 is an EMA number.")
    ap.add_argument("--select", default="chain", choices=["chain", "act"],
                    help="which test loss picks best.pt. 'chain' = the action loss when the intents come "
                         "from the test-time SAMPLING chain, i.e. what the closed loop actually feeds the "
                         "policy (train_intent_policy.py:162,246). 'act' = the old behaviour, measured "
                         "with GROUND-TRUTH intents: lat_fut latent frame 0 encodes exactly the F rows "
                         "whose actions are being predicted, so it selects the checkpoint that leans "
                         "hardest on an oracle that does not exist at inference.")
    ap.add_argument("--chain-steps", type=int, default=10)
    ap.add_argument("--chain-cfg", type=float, default=2.5)
    ap.add_argument("--eval-batches", type=int, default=20,
                    help="test batches per evaluation. The chain criterion runs a full 10-step HIP and "
                         "IIP sample per batch, so scoring the whole test split every eval would dominate "
                         "the run; the loader is unshuffled, so a fixed prefix stays deterministic.")
    ap.add_argument("--text-drop", type=float, default=0.1)
    ap.add_argument("--cond-aug", type=float, default=0.5,
                    help="MIND: intent hidden states read at s ~ U(cond_aug, 1) in training")
    ap.add_argument("--cond-aug-test", type=float, default=0.75, help="MIND's fixed test-time s")
    ap.add_argument("--lat-stat-rows", type=int, default=200000,
                    help="latent rows sampled to measure the intent-latent normalisation")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--intent-target", default="proprio", choices=["proprio", "ref"],
                    help="what HIP and IIP are trained to predict. 'proprio' (the default so far) is the "
                         "encoding of the next H / whole-clip PROPRIO rows -- what the body ends up at. "
                         "'ref' is the encoding of the teacher's 27-d reference block, i.e. the control "
                         "TARGET the teacher was tracking, which is closer to what 'intent' means and is "
                         "the one input the teacher had that the first recording threw away. Needs "
                         "--vae-ref and rollouts recorded with ref_obs. The history prefix stays on the "
                         "proprio VAE either way: the history is observable, the reference never is.")
    ap.add_argument("--vae-ref", default="",
                    help="frozen VAE over the 27-d reference block, from g1e2e_train_vae.py --field ref")
    ap.add_argument("--lr-schedule", default="const", choices=["const", "cosine"],
                    help="cosine: linear warm-up then half-cosine decay to lr*lr_final_ratio by --steps, "
                         "the same form the reference uses (train_intent_policy.py:102-114). The G1 side "
                         "only ever ran const, and on the full data `act_chain` bottoms at step 10000 and "
                         "then rises monotonically to 100k -- the run drifts past its own optimum.")
    ap.add_argument("--lr-final-ratio", type=float, default=0.01)
    ap.add_argument("--lr-warmup", type=int, default=2000,
                    help="reference default; MoGeFlow / MoMask / MotionStreamer all use 2000")
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
    from hml_phys.intent_flow import intent_hidden, sample_latent
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
    # The dataset takes the cache DIRECTORY: a caption's mask is only correct against its own length and
    # the policy needs the real pooled vector, so tokens / lengths / pooled all travel together.
    tr = G1E2EWindows(args.rollouts, tc, args.H, args.F, gen_hz=args.gen_hz,
                      stride=args.window_stride, obs_future=args.obs_future, max_clips=args.max_clips,
                      stats=stats, seed=args.seed, p_rest=args.p_rest, holi_aug=True,
                      p_holi_exact=args.p_holi_exact, max_mpjpe=args.max_mpjpe,
                      want_ref=(args.intent_target == "ref"))
    # The test set takes NO augmentation: the selection criterion must only move when the weights move.
    te = G1E2EWindows(args.rollouts_eval, tc, args.H, args.F, gen_hz=args.gen_hz,
                      stride=max(1, args.H), obs_future=args.obs_future,
                      stats=stats, seed=args.seed + 1, deterministic_caption=True,
                      p_rest=0.0, holi_aug=False, max_mpjpe=args.max_mpjpe,
                      want_ref=(args.intent_target == "ref"))
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

    # Optional second frozen VAE over the teacher's reference block, used ONLY to define what the intent
    # predictors aim at (--intent-target ref). Its latent shape must match the proprio VAE's, because the
    # IIP conditions on a proprio-latent prefix and predicts in the same latent geometry.
    tvae = None
    if args.intent_target == "ref":
        assert args.vae_ref, "--intent-target ref needs --vae-ref"
        tck = torch.load(args.vae_ref, map_location="cpu")
        ta_ = tck["args"]
        tvae = IntentVAE(input_dim=tck["input_dim"], width=ta_["width"], down_t=ta_["down_t"], stride_t=2,
                         depth=ta_["depth"], dilation_growth_rate=ta_["dilation"],
                         latent_dim=ta_["latent"]).to(dev)
        tvae.load_state_dict(tck["model"])
        tvae.eval().requires_grad_(False)
        ref_field = tck.get("field")
        assert ref_field == "ref", (
            f"--vae-ref must be a VAE trained with --field ref; this one says {ref_field!r}")
        assert tvae.down == vae.down and tvae.latent_dim == vae.latent_dim, (
            f"ref VAE latent geometry {tvae.down}/{tvae.latent_dim} differs from the proprio VAE's "
            f"{vae.down}/{vae.latent_dim}; the IIP predicts in one geometry only")
        # Look the key up OUTSIDE the f-string. An earlier version built it with chr() to dodge quoting
        # inside a shell heredoc, which produced the literal key "'input_dim'" (quotes included) and
        # killed both reference-target arms with KeyError at load time, after the ref VAE had trained.
        ref_in = tck["input_dim"]
        print(f"ref VAE: input {ref_in}, latent {tvae.latent_dim}, down {tvae.down}")
    assert ck["input_dim"] == PROPRIO_DIM, f"VAE expects {ck['input_dim']}-d state, G1 proprio is {PROPRIO_DIM}"
    assert va["gen_hz"] == args.gen_hz, (
        f"the VAE was trained at {va['gen_hz']} Hz and this run uses {args.gen_hz}; the latents encode "
        f"a different amount of real time at a different rate")
    n_lat = args.H // vae.down
    assert args.H % vae.down == 0, f"H={args.H} must be divisible by the VAE downsampling {vae.down}"
    from hml_phys.intent_model import N_LAT, D_LAT
    assert n_lat == N_LAT and vae.latent_dim == D_LAT, (
        f"IntentDiT builds its positional embedding and input projection from the module-level "
        f"N_LAT={N_LAT}, D_LAT={D_LAT} (intent_model.py:26), but this run gives {n_lat} latent frames of "
        f"{vae.latent_dim} dims")
    print(f"frozen IntentVAE: input {ck['input_dim']}, latent {vae.latent_dim}, down {vae.down} "
          f"-> {n_lat} latent frames from {args.H} history rows")

    policy_kw = dict(hidden_dim=args.hidden, num_heads=args.heads, depth_double=args.depth_double,
                     depth_single=args.depth_single, text_mode="sentence_xattn",
                     text_cross_attention=True, n_scalar_cond=2, part_dims=[TOKEN_DIM])
    # intent_mlp is the 5th positional argument (intent_model.py:115) and defaults to 4.0. Omitting it
    # took that default instead of the reference's validated 1.5, giving 126.1 M params against the
    # reference's ratio -- a 2.67x wider SwiGLU in both HIP and IIP, on 1/12 of the data. Measured on
    # this model: 126.1 M at 4.0 vs 117.2 M at 1.5 (logs/g1e2e_policy.log:4, g1e2e_policy2.log:4). The
    # reference's own 99.84 M is a different model (its own part_dims) and is NOT this model's target.
    model = IntentPolicy(policy_kw, args.intent_dim, args.intent_heads, args.intent_depth,
                         args.intent_mlp, text_token_dim=768).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"IntentPolicy {n_par / 1e6:.1f} M params   flat part_dims=[{TOKEN_DIM}]")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    act_mask = action_channel_mask(dev)

    def set_lr(step):
        """Constant, or linear warm-up then half-cosine decay to lr*lr_final_ratio by --steps.

        Transcribed from train_intent_policy.py:102-114. `step` is the step ABOUT TO BE TAKEN, so this is
        called after the optimiser step counter increments, as the reference does."""
        if args.lr_schedule == "const":
            return args.lr
        if step <= args.lr_warmup:
            lr = args.lr * step / max(args.lr_warmup, 1)
        else:
            prog = min(1.0, (step - args.lr_warmup) / max(args.steps - args.lr_warmup, 1))
            lr = args.lr * (args.lr_final_ratio
                            + (1 - args.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr
        return lr
    # How many future rows keep their proprio. This has to agree with the dataset's obs_future, and it is
    # what `policy_input` was silently overriding: row H is the state the loop is in when it plans.
    n_fut_prop = {"none": 0, "first": 1, "all": args.F}[args.obs_future]
    print(f"obs_future={args.obs_future}: the policy sees the proprio of {n_fut_prop} future row(s); "
          f"row H is the state its first action is applied in")
    ema = ({k: v.detach().clone().float() for k, v in model.state_dict().items()}
           if args.ema_every > 0 else None)

    # Intent-latent normalisation. Flow matching mixes the target with eps ~ N(0,1), so the targets have
    # to be unit-scale; the validated pipeline has a dedicated step for this
    # (scripts/hml_phys/compute_intent_latent_stats.py, applied at train_intent_policy.py:130-133).
    # Measured here rather than assumed, over a bounded sample of training windows.
    import joblib
    lat_stat_path = out / "intent_latent_stats.npz"
    if lat_stat_path.exists():
        z = np.load(lat_stat_path)
        lat_mean = torch.tensor(z["mean"], device=dev)
        lat_std = torch.tensor(z["std"], device=dev)
    else:
        from torch.utils.data import DataLoader as _DL
        acc = []
        # shuffle=True: `tr.windows` is built clip by clip in sorted key order, so shuffle=False stopped
        # after the alphabetically first ~11% of clips and measured the holistic statistics on ~200 of
        # 1816 holistic vectors. The reference shuffles (compute_intent_latent_stats.py:21-22).
        _g = torch.Generator().manual_seed(args.seed)
        with torch.no_grad():
            for b in _DL(tr, batch_size=256, shuffle=True, num_workers=2, generator=_g):
                for fld in ("x", "fut", "holi"):
                    v = b[fld].to(dev)
                    # Only the H HISTORY rows, which is the only thing `losses` ever encodes from `x`.
                    # Encoding all H+F rows added a 5th latent frame built from the zeroed future proprio
                    # -- a sequence that never exists -- to the statistics.
                    v = v[:, :args.H, :PROPRIO_DIM] if fld == "x" else v
                    _, mu, _ = vae.encode(v)
                    acc.append(mu.reshape(-1, mu.shape[-1]).cpu())
                if sum(a.shape[0] for a in acc) > args.lat_stat_rows:
                    break
        A = torch.cat(acc)
        lat_mean = A.mean(0).to(dev)
        lat_std = A.std(0).clamp_min(1e-3).to(dev)
        np.savez(lat_stat_path, mean=lat_mean.cpu().numpy(), std=lat_std.cpu().numpy(),
                 rows=int(A.shape[0]))
    print(f"intent latents: |mean| {float(lat_mean.abs().mean()):.3f}  "
          f"std {float(lat_std.mean()):.3f} (min {float(lat_std.min()):.3f}, "
          f"max {float(lat_std.max()):.3f})")

    def norm_lat(v):
        return (v - lat_mean) / lat_std

    # The REF latents need their own normaliser: they come from a different encoder over a different
    # quantity, so the proprio VAE's latent statistics do not describe them. Same bounded, shuffled
    # sample, same cache-in-the-output-directory discipline.
    tlat_mean = tlat_std = None
    if tvae is not None:
        tpath = out / "ref_latent_stats.npz"
        if tpath.exists():
            z = np.load(tpath)
            tlat_mean = torch.tensor(z["mean"], device=dev)
            tlat_std = torch.tensor(z["std"], device=dev)
        else:
            from torch.utils.data import DataLoader as _DL
            acc = []
            _g2 = torch.Generator().manual_seed(args.seed + 7)
            with torch.no_grad():
                for b in _DL(tr, batch_size=256, shuffle=True, num_workers=2, generator=_g2):
                    for fld in ("fut_ref", "holi_ref"):
                        _, mu, _ = tvae.encode(b[fld].to(dev))
                        acc.append(mu.reshape(-1, mu.shape[-1]).cpu())
                    if sum(a.shape[0] for a in acc) > args.lat_stat_rows:
                        break
            A = torch.cat(acc)
            tlat_mean = A.mean(0).to(dev)
            tlat_std = A.std(0).clamp_min(1e-3).to(dev)
            np.savez(tpath, mean=tlat_mean.cpu().numpy(), std=tlat_std.cpu().numpy(),
                     rows=int(A.shape[0]))
        print(f"ref latents: |mean| {float(tlat_mean.abs().mean()):.3f}  "
              f"std {float(tlat_std.mean()):.3f} (min {float(tlat_std.min()):.3f}, "
              f"max {float(tlat_std.max()):.3f})")

    def norm_tlat(v):
        return (v - tlat_mean) / tlat_std

    # The unconditional text state is CLIP(''), cached once, not a zeroed caption: zeroing leaves a state
    # that depends on the dropped caption's length, so classifier-free guidance has no fixed point to
    # extrapolate from (and the policy's own mask, rebuilt from the real length in part_model.py:160,
    # would still attend over that many slots of LayerNorm(0)-derived constants).
    _tok_all = joblib.load(tok_path)
    _len_all = joblib.load(tc / "lengths.pkl")
    _pool_all = joblib.load(tc / "pooled.pkl")
    assert "__uncond__" in _tok_all, (
        f"{tok_path} has no CLIP('') entry; rebuild it with the current scripts/g1e2e_build_text_cache.py")
    unc_tok = torch.tensor(_tok_all["__uncond__"][0], dtype=torch.float32, device=dev)
    unc_len = int(_len_all["__uncond__"][0])
    unc_pool = torch.tensor(_pool_all["__uncond__"][0], dtype=torch.float32, device=dev)
    del _tok_all, _len_all, _pool_all
    print(f"unconditional CLIP(''): {unc_len} tokens")

    def batch_to_dev(b):
        # fut_ref / holi_ref are present only when the dataset was built with want_ref; they are the
        # --intent-target ref targets and are None otherwise.
        return (b["x"].to(dev), b["fut"].to(dev), b["holi"].to(dev), b["text"].to(dev).float(),
                b["text_pooled"].to(dev).float(), b["text_len"].to(dev), b["scal"].to(dev),
                b["fut_ref"].to(dev) if "fut_ref" in b else None,
                b["holi_ref"].to(dev) if "holi_ref" in b else None)

    def hidden_level(B, train, gen=None):
        """Noise level at which the intent hidden states are read. MIND augments the conditioning with
        s ~ U(s_min, 1) in training and uses a fixed 0.75 at test time; s = 1 is the clean read-out."""
        if args.cond_aug <= 0:
            return 1.0
        if not train:
            return float(args.cond_aug_test)
        return args.cond_aug + (1.0 - args.cond_aug) * torch.rand(B, device=dev, generator=gen)

    def losses(x, fut, holi, text, text_pooled, text_len, scal, fut_ref=None, holi_ref=None,
               train=True, gen=None):
        B, T, _ = x.shape
        obs = observed_mask(B, args.H, T, dev)
        gmask = generated_elements(obs, None, act_mask)
        x_in = policy_input(x, obs, act_mask, n_future_proprio=n_fut_prop)

        # The POSTERIOR MEAN, not a sample: the reference encodes with `_, mu, _ = vae.encode(...)`
        # (train_intent_policy.py:132). CausalEncoder's first return is mu + randn*sigma, and that draw is
        # not generator-controlled, so using it would also inject unseeded noise into the eval loss.
        with torch.no_grad():
            # The HISTORY prefix always comes from the proprio VAE: the history is what the loop can
            # actually observe. Only the TARGETS change with --intent-target.
            _, mu_hist, _ = vae.encode(x[:, :args.H, :PROPRIO_DIM])
            lat_hist = norm_lat(mu_hist)
            if tvae is None:
                _, mu_fut, _ = vae.encode(fut)
                _, mu_holi, _ = vae.encode(holi)
                lat_fut, lat_holi = norm_lat(mu_fut), norm_lat(mu_holi)
            else:
                # --intent-target ref: the intents aim at the teacher's own CONTROL TARGET rather than at
                # the proprio the robot will happen to reach. Closer to what "intent" means -- what the
                # motion is trying to do, not what the body ends up at -- and it is the one input the
                # teacher had that our recording used to throw away. It stays a target only: at inference
                # there is no reference at any time, which is the task.
                _, mu_fut, _ = tvae.encode(fut_ref)
                _, mu_holi, _ = tvae.encode(holi_ref)
                lat_fut, lat_holi = norm_tlat(mu_fut), norm_tlat(mu_holi)

        # One dropout mask shared by all three heads. A dropped sample is replaced by the cached CLIP('')
        # tokens, length AND pooled vector, giving a single well-defined unconditional state.
        keep = torch.rand(B, device=dev, generator=gen) >= (args.text_drop if train else 0.0)
        k3 = keep[:, None, None]
        txt = torch.where(k3, text, unc_tok[None].expand_as(text))
        pooled = torch.where(keep[:, None], text_pooled, unc_pool[None].expand_as(text_pooled))
        tlen = torch.where(keep, text_len, torch.full_like(text_len, unc_len))
        mem, mem_valid = model.adapter(txt, tlen)

        # HIP: text -> holistic intent, the WHOLE CLIP resampled to H rows (what a caption describes).
        tH = fl.sample_t(B, dev, generator=gen)
        zH, _ = build_state_elem(lat_holi, torch.ones_like(lat_holi), tH, generator=gen)
        xH, _ = model.hip(zH, tH, mem, mem_valid)
        l_hip = latent_loss(*velocity_pair(xH, lat_holi, zH, tH))

        # IIP: predict the IMMEDIATE intent -- the encoding of the next H state rows -- from text, the
        # history latents as a clean PREFIX, and HIP's hidden states as extra memory. Target and prefix
        # must differ: with both set to the history latents the head copies a token it is handed in the
        # same sequence (IntentDiT with prefix=True reads its output from tokens n_lat..2*n_lat-1), so it
        # learns an identity map, l_iip collapses to ~0 and contributes no gradient, and `hI` -- one of
        # the two signals MIND's policy reads -- becomes the hidden state of that identity.
        hH = intent_hidden(model.hip, lat_holi, hidden_level(B, train, gen), generator=gen,
                           mem=mem, mem_valid=mem_valid)
        tI = fl.sample_t(B, dev, generator=gen)
        zI, _ = build_state_elem(lat_fut, torch.ones_like(lat_fut), tI, generator=gen)
        xI, _ = model.iip(zI, tI, mem, mem_valid, prefix_latent=lat_hist, scalars=scal, mem_extra=hH)
        l_iip = latent_loss(*velocity_pair(xI, lat_fut, zI, tI))

        # Policy: generate the future ACTION rows, reading both predictors' HIDDEN STATES (not their
        # latents) as extra tokens in its text stream, and the cache's real CLIP pooled vector as the
        # sentence-level text -- `text_mode="sentence_xattn"` makes that the only sentence signal, and a
        # mean over the 50 token slots is not the projection-space embedding the policy expects.
        hI = intent_hidden(model.iip, lat_fut, hidden_level(B, train, gen), generator=gen,
                           mem=mem, mem_valid=mem_valid, prefix_latent=lat_hist, scalars=scal,
                           mem_extra=hH)
        toks, tval = model.intent_tokens(hH, hI, keep)
        tval = mask_intent(tval)
        t = fl.sample_t(B, dev, generator=gen)
        z, _ = build_state_elem(x_in, gmask, t, generator=gen)
        x0_hat = model.policy(z, obs, t, txt, pooled, tlen, scal,
                              extra_tokens=toks, extra_valid=tval)
        l_act = policy_loss(*velocity_pair(x0_hat, x_in, z, t), gmask)
        return l_hip, l_iip, l_act

    def mask_intent(tval):
        """Hide part of the intent stream from the action policy. The 8 tokens are 4 holistic then 4
        immediate (intent_model.py:131), so the split is down the middle."""
        if args.ablate_intent == "none":
            return tval
        half = tval.shape[1] // 2
        m = torch.zeros_like(tval)
        if args.ablate_intent == "hip":
            m[:, :half] = tval[:, :half]
        elif args.ablate_intent == "iip":
            m[:, half:] = tval[:, half:]
        return m

    def total_loss(l_hip, l_iip, l_act):
        """MIND eq. 5, equal weights -- minus any predictor the ablation has disconnected, whose loss
        would otherwise train a module nothing reads."""
        if args.ablate_intent == "all":
            return l_act
        if args.ablate_intent == "hip":
            return l_hip + l_act          # IIP disconnected from the policy
        return l_hip + l_iip + l_act      # 'iip' keeps both: IIP needs HIP as its memory

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=4,
                          drop_last=shuffle, persistent_workers=True)

    tl, el = loader(tr, True), loader(te, False)

    @torch.no_grad()
    def chain_action_loss(x, fut_unused, holi_unused, text, text_pooled, text_len, scal, gen):
        """The action loss when the intents come from the TEST-TIME SAMPLING CHAIN, not from the VAE.

        This is the reference's selection criterion (train_intent_policy.py:162-191, 246: "select on what
        the closed loop actually feeds the policy"). It matters here more than it does there: `lat_fut` is
        the encoding of rows H..2H-1, and the VAE downsamples causally by 4, so its FIRST latent frame
        encodes exactly rows H..H+F-1 -- the F rows whose actions the policy is being trained to emit.
        Selecting on the ground-truth-intent loss therefore picks the checkpoint that leans hardest on an
        oracle which, at inference, is replaced by a 10-step sample from text + history.
        """
        B, T, _ = x.shape
        obs = observed_mask(B, args.H, T, dev)
        gmask = generated_elements(obs, None, act_mask)
        x_in = policy_input(x, obs, act_mask, n_future_proprio=n_fut_prop)
        _, mu_hist, _ = vae.encode(x[:, :args.H, :PROPRIO_DIM])
        lat_hist = norm_lat(mu_hist)
        s_t = 1.0 if args.cond_aug <= 0 else float(args.cond_aug_test)

        mem_c, mv_c = model.adapter(text, text_len)
        mem_u, mv_u = model.adapter(unc_tok[None].expand_as(text),
                                    torch.full_like(text_len, unc_len))
        I_H = sample_latent(model.hip, B, mem_c, mv_c, mem_u, mv_u, num_steps=args.chain_steps,
                            cfg_scale=args.chain_cfg, generator=gen, device=dev)
        hH_c = intent_hidden(model.hip, I_H, s_t, gen, mem=mem_c, mem_valid=mv_c)
        hH_u = intent_hidden(model.hip, I_H, s_t, gen, mem=mem_u, mem_valid=mv_u)
        I_I = sample_latent(model.iip, B, mem_c, mv_c, mem_u, mv_u, num_steps=args.chain_steps,
                            cfg_scale=args.chain_cfg, generator=gen, prefix=lat_hist, scalars=scal,
                            extra=hH_c, extra_u=hH_u, device=dev)
        hI_c = intent_hidden(model.iip, I_I, s_t, gen, mem=mem_c, mem_valid=mv_c,
                             prefix_latent=lat_hist, scalars=scal, mem_extra=hH_c)
        toks, tval = model.intent_tokens(hH_c, hI_c, torch.ones(B, dtype=torch.bool, device=dev))
        tval = mask_intent(tval)
        t = fl.sample_t(B, dev, generator=gen)
        z, _ = build_state_elem(x_in, gmask, t, generator=gen)
        x0_hat = model.policy(z, obs, t, text, text_pooled, text_len, scal,
                              extra_tokens=toks, extra_valid=tval)
        return policy_loss(*velocity_pair(x0_hat, x_in, z, t), gmask)

    def evaluate():
        """Seeded and deterministic: the criterion must not move because the noise draws moved.

        The reference pins a generator for exactly this reason (train_intent_policy.py:232-237). Here the
        test dataset also fixes its caption choice (deterministic_caption=True), so the only thing that
        changes between evaluations is the weights. Measured on the EMA weights when EMA is on, because
        those are the weights the closed loop rolls out (mc_rollout.py:25,31)."""
        swapped = None
        if ema is not None:
            swapped = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: v.to(swapped[k].dtype) for k, v in ema.items()})
        model.eval()
        eg = torch.Generator(device=dev).manual_seed(args.seed)
        acc, n = np.zeros(4), 0
        with torch.no_grad():
            for nb, b in enumerate(el):
                if args.eval_batches and nb >= args.eval_batches:
                    break
                x, fut, holi, text, pooled, tlen, scal, fut_ref, holi_ref = batch_to_dev(b)
                ls = losses(x, fut, holi, text, pooled, tlen, scal, fut_ref, holi_ref, train=False, gen=eg)
                ch = chain_action_loss(x, fut, holi, text, pooled, tlen, scal, eg)
                acc += np.array([float(v) for v in ls] + [float(ch)]) * x.shape[0]
                n += x.shape[0]
        model.train()
        if swapped is not None:
            model.load_state_dict(swapped)
        return acc / n

    hist, best, step, t0 = [], float("inf"), 0, time.time()
    while step < args.steps:
        for b in tl:
            if step >= args.steps:
                break
            x, fut, holi, text, pooled, tlen, scal, fut_ref, holi_ref = batch_to_dev(b)
            l_hip, l_iip, l_act = losses(x, fut, holi, text, pooled, tlen, scal, fut_ref, holi_ref)
            loss = total_loss(l_hip, l_iip, l_act)
            if not torch.isfinite(loss):
                raise SystemExit(
                    f"non-finite loss at step {step}: hip {float(l_hip)} iip {float(l_iip)} "
                    f"act {float(l_act)}. Aborting rather than overwriting latest.pt with NaN weights "
                    f"and burning the remaining steps on a criterion that can never improve again.")
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            cur_lr = set_lr(step + 1)       # the schedule applies to the step about to be taken
            step += 1
            if ema is not None and step % args.ema_every == 0:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(args.ema_decay).add_(v.float(), alpha=1 - args.ema_decay)
                        else:
                            ema[k].copy_(v)
            if step % args.log_every == 0:
                print(f"step {step} loss={float(loss):.4f} hip={float(l_hip):.4f} "
                      f"iip={float(l_iip):.4f} act={float(l_act):.4f} gn={float(gn):.2f} "
                      f"lr={cur_lr:.2e} {(time.time() - t0) / step * 1000:.0f}ms/it", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                e = evaluate()
                hist.append(dict(step=step, hip=e[0], iip=e[1], act=e[2], act_chain=e[3],
                                 sum=float(e[:3].sum())))
                print(f"  [eval] step {step} hip {e[0]:.4f} iip {e[1]:.4f} act {e[2]:.4f} "
                      f"act_chain {e[3]:.4f}", flush=True)
                # Ship the EMA weights as `model`, since those are the ones the closed loop rolls out;
                # the raw iterate goes alongside so a run can be continued or compared.
                shipped = ({k: v.to(model.state_dict()[k].dtype) for k, v in ema.items()}
                           if ema is not None else model.state_dict())
                ck_out = dict(model=shipped, raw=model.state_dict() if ema is not None else None,
                              args=vars(args), step=step, mean=tr.mean, std=tr.std,
                              policy_kw=policy_kw, vae=str(args.vae),
                              intent_latent_mean=lat_mean.cpu().numpy(),
                              intent_latent_std=lat_std.cpu().numpy())
                torch.save(ck_out, out / "latest.pt")
                # Selection is on the chain action loss by default: HIP overfits long before the policy
                # does (so a sum-based criterion picks checkpoints for the wrong reason), and the
                # ground-truth-intent action loss cannot see the degradation that matters at inference.
                crit = e[3] if args.select == "chain" else e[2]
                if crit < best:
                    best = crit
                    torch.save(ck_out, out / "best.pt")
                    print(f"  [eval] new best test {args.select} loss {best:.4f}", flush=True)
                (out / "history.json").write_text(json.dumps(hist, indent=1))

    print(f"done, {step} steps in {(time.time() - t0) / 60:.1f} min, best test {args.select} loss {best:.4f}")


if __name__ == "__main__":
    main()
