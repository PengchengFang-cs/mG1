"""DGPO on the flow action policy: group preference optimisation, no residual, no critic.

WHY THIS AND NOT RESIDUAL PPO. The residual existed for one reason: PPO needs a tractable per-step
log-probability, and getting one through a 10-step denoising chain means diffusion-RL machinery, so we
froze the flow policy and made the trainable part a small Gaussian. That decision cost us the thing we
care about -- the residual maximises the teacher's tracking reward, which has no text term, and R@1
fell from 0.33 to 0.16 with 21-joint authority (STATUS.md §5.13b).

DGPO (group preference, `realizability_floor/vendor/DGPO`) removes the reason. It needs only each
sample's DENOISING LOSS, never a log-probability, so the flow policy itself becomes trainable and the
deterministic ODE sampler we already use needs no conversion to an SDE -- which is what Flow-GRPO
(arXiv 2505.05470) requires and what MixGRPO's sliding window then tries to claw back.

    per group of G rollouts sharing a caption, with group-relative advantages A_g:
      one shared noise level t and one shared epsilon across the group
      dsm_g      = the policy's denoising loss on ITS OWN generated action chunk
      ref_dsm_g  = the same under the EMA reference
      w          = sigmoid( sum_g A_g * beta_dpo * (dsm_g - ref_dsm_g) / G )      [detached]
      loss       = mean_g( w * A_g * dsm_g )  +  anchor

A_g > 0 pushes that sample's denoising loss DOWN (fit the good rollout); A_g < 0 pushes it UP.

THE GROUP IS G ROLLOUTS, NOT G IMAGES. In text-to-image a sample is scorable the moment it exists. An
action chunk is not: its quality only appears once executed. So a group is G complete rollouts of the
same caption from the same start state, differing only in the policy's sampling noise, and the reward
is episodic -- which is the shape our rewards already have (did it fall; does the executed motion match
the caption). Densifying an episodic reward is what produced the window bug in §5.17.

Groups are built by duplicating each clip G times in the motion library, consecutively, so the env's
own `begin_seq_motion_samples` / `forward_motion_samples` assign env j to clip j//G with no surgery,
and each iteration advances to the next 512/G captions.

MANIFOLD DRIFT is the known failure mode of exactly this objective. "Manifold Drift in Flow Preference
Optimization" (arXiv 2608.20011) shows FlowDPO's loss is the winner's flow-matching error minus the
loser's, and that subtracting the loser term is what pushes terminal samples off the pretrained data
manifold; DGPO's `A_g < 0` branch is that term. Their fix is a winner-side anchor. Ours is two things:
a plain denoising loss on RECORDED data mixed in (`--w-anchor`), which is the same anchor the DGPO
recipe already suggested as an optional small term and which is promoted here to a first-class one;
and an EMA reference rather than a fixed one, because three independent results say a fixed-reference
KL is not enough (it changes the optimisation timescale, it fails under heavy-tailed reward error, and
its coefficient is sharply sensitive).

REWARD, with the Stage 0 guards (STATUS.md §5.18):
    survival   -1 if it fell, plus the fraction of the window survived
    semantic   min over TWO retrieval models trained on DISJOINT clip slices -- a minimum, because
               averaging an ensemble is not conservative (Coste et al., arXiv 2310.02743)
    each term is normalised by its own standard deviation before being summed, because GRPO-style
    objectives optimise the highest-variance reward and ignore the rest (MO-GRPO, arXiv 2509.22047)
A third retrieval model, trained on the third slice, is the judge: it never enters the reward, and the
manifold monitor runs in ITS embedding space, because in the reward model's own space the outlier
statistic was measured to be useless (§5.18).

Run on a compute node, inside a persistent Slurm step.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
H2H = REPO / "external/FRoM-W1/H-ACT/human2humanoid"
LEGGED_GYM = H2H / "legged_gym"
CFG_DIR = LEGGED_GYM / "legged_gym/cfg/cfg_g1"
CONTROL_HZ = 50
REF_DIM = 27


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="the BC checkpoint DGPO starts from")
    ap.add_argument("--refs", required=True, help="the reference library the captions come from")
    ap.add_argument("--text-cache", required=True)
    ap.add_argument("--tmr", required=True,
                    help="comma-separated retrieval models for the REWARD; their minimum is used")
    ap.add_argument("--tmr-judge", required=True,
                    help="a retrieval model that never enters the reward; the monitor runs in its space")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=8, help="rollouts per caption")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--n-clips", type=int, default=512,
                    help="distinct captions cycled through; each is duplicated --group times")
    ap.add_argument("--episode-steps", type=int, default=150,
                    help="control steps per rollout. 150 = 3 s, above the retrieval model's 2 s "
                         "minimum, and it keeps one DGPO update near a minute.")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta-dpo", type=float, default=30.0, help="DGPO recipe: 10-100 with a frozen ref")
    ap.add_argument("--adv-clip", type=float, default=5.0)
    ap.add_argument("--ema", type=float, default=0.995,
                    help="EMA rate for the reference. GARDO (arXiv 2512.24138) rolls the reference "
                         "rather than fixing it, because a fixed reference's penalty grows as the "
                         "policy improves and eventually swamps the RL term.")
    ap.add_argument("--w-anchor", type=float, default=0.3,
                    help="weight on the WINNER-SIDE ANCHOR against manifold drift: a plain denoising "
                         "loss on the samples with positive advantage, weighted by (1-t)^2. This is "
                         "ThermoDPO's structure (arXiv 2608.20011) -- that paper proves its objective "
                         "reduces to rejection-sampling fine-tuning as its temperature goes to zero, "
                         "and its experiments use the (1-t)^2 reweighting because the plain t^2 factor "
                         "weakens the anchor exactly where it is needed, near t = 0. It needs no extra "
                         "data: the winners of the group ARE the anchor. 0 disables it, which is the "
                         "ablation that shows whether it was needed.")
    ap.add_argument("--keep-per-ep", type=int, default=4,
                    help="planning steps sampled per rollout for the DGPO update. A rollout contains "
                         "~75 generated chunks and they all share the episode's reward; keeping a few "
                         "is the same kind of subsampling DGPO already applies to the noise level.")
    ap.add_argument("--w-surv", type=float, default=1.0)
    ap.add_argument("--w-sem", type=float, default=1.0)
    ap.add_argument("--num-steps", type=int, default=10, help="flow sampling steps")
    ap.add_argument("--cfg-action", type=float, default=1.0)
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args, overrides = ap.parse_known_args()
    bad = [o for o in overrides if o.startswith("-")]
    assert not bad, f"unrecognised option(s) {bad}; hydra overrides are key=value, not flags"
    assert args.num_envs % args.group == 0, "--num-envs must be a multiple of --group"

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    refs = Path(args.refs).resolve()
    policy_path = Path(args.policy).resolve()
    text_cache = Path(args.text_cache).resolve()
    tmr_paths = [str(Path(p).resolve()) for p in args.tmr.split(",")]
    judge_path = str(Path(args.tmr_judge).resolve())
    for p in [refs, policy_path] + tmr_paths + [judge_path]:
        assert Path(p).exists(), p
    assert text_cache.is_dir(), text_cache

    # The duplicated library is built BEFORE the chdir, while relative paths still mean what they say.
    import joblib
    dup = refs.parent / f"{refs.stem}.dup{args.group}x{args.n_clips}.pkl"
    if not dup.exists():
        print(f"building the duplicated library {dup.name} ...", flush=True)
        lib = joblib.load(refs)
        keys = list(lib)[:args.n_clips]
        d = {}
        for k in keys:
            for g in range(args.group):
                e = dict(lib[k])
                e["base_key"] = k
                d[f"{k}#g{g}"] = e
        joblib.dump(d, dup)
        print(f"  {len(keys)} clips x {args.group} = {len(d)} entries", flush=True)
    os.chdir(LEGGED_GYM)
    sys.path.insert(0, str(H2H))
    sys.path.insert(0, str(REPO))

    from isaacgym import gymapi          # noqa: E402  (before torch, on purpose)
    import numpy as np                   # noqa: E402
    import torch                         # noqa: E402
    import hydra                         # noqa: E402
    from omegaconf import OmegaConf       # noqa: E402
    from easydict import EasyDict         # noqa: E402
    import legged_gym.envs                # noqa: E402,F401
    from legged_gym.utils import task_registry   # noqa: E402

    from hml_phys.g1e2e_data import PROPRIO_DIM, ACTION_DIM, TOKEN_DIM          # noqa: E402
    from hml_phys.g1e2e_flow import (action_channel_mask, build_state_elem,      # noqa: E402
                                     generated_elements, observed_mask,
                                     policy_loss, velocity_pair)
    from hml_phys.intent_model import IntentPolicy                              # noqa: E402
    from hml_phys.intent_vae import IntentVAE                                   # noqa: E402
    from hml_phys.intent_flow import (intent_hidden, sample_actions,            # noqa: E402
                                      sample_latent)
    from hml_phys import flow as fl                                             # noqa: E402

    def dsm_per_sample(v_hat, v, gmask):
        """hml_phys.g1e2e_flow.policy_loss reduces to a scalar; DGPO needs one number per sample."""
        num = ((v_hat - v) ** 2 * gmask).flatten(1).sum(1)
        return num / gmask.flatten(1).sum(1).clamp_min(1.0)
    from hml_phys.g1_tmr import TMR_FPS, load_tmr                               # noqa: E402

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ck = torch.load(policy_path, map_location="cpu")
    ta = ck["args"]
    gen_hz, H, F = ta["gen_hz"], ta["H"], ta["F"]
    hold = CONTROL_HZ // gen_hz
    n_fut_prop = {"none": 0, "first": 1, "all": F}[ta.get("obs_future", "first")]

    with hydra.initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg_h = hydra.compose(config_name="config_eval", overrides=[
            f"motion.motion_file={dup}", f"num_envs={args.num_envs}", f"sim_device={args.device}",
            "headless=True", "use_wandb=False",
            "asset.terminate_by_ref_motion_distance=False",   # no reference at deployment, so none here
            "asset.terminate_by_1time_motion=True",
            "motion.resample_motions_for_envs=False",
            "rewards.penalty_curriculum=False",
            *overrides])
    cfg = EasyDict(OmegaConf.to_container(cfg_h, resolve=True))
    cfg.physics_engine = gymapi.SIM_PHYSX
    n_obs = int(cfg.env.num_observations)
    n_hist = (int(cfg.env.short_history_length) * (int(cfg.extra.dof_num) * 3 + 6)
              if cfg.env.add_short_history else 0)
    assert n_obs == 48 + REF_DIM + ACTION_DIM + n_hist, f"observation layout moved: {n_obs}"
    env, _ = task_registry.make_env_hydra(name=cfg.task, hydra_cfg=cfg, env_cfg=cfg)
    dev = env.device
    B = env.num_envs
    G = args.group
    NG = B // G
    print(f"env: {B} envs = {NG} captions x {G} rollouts; episode {args.episode_steps} control steps "
          f"({args.episode_steps / CONTROL_HZ:.1f} s); reference-distance termination OFF", flush=True)

    # ---- the policy: only `policy` trains; the intent path and the VAE stay frozen ---------------
    stats = np.load(policy_path.parent / "stats.npz")
    mean = torch.tensor(stats["mean"], device=dev)
    std = torch.tensor(stats["std"], device=dev)
    vae_path = Path(ta["vae"])
    if not vae_path.is_absolute():
        vae_path = REPO / vae_path
    vck = torch.load(vae_path, map_location="cpu")
    va = vck["args"]
    vae = IntentVAE(input_dim=vck["input_dim"], width=va["width"], down_t=va["down_t"], stride_t=2,
                    depth=va["depth"], dilation_growth_rate=va["dilation"],
                    latent_dim=va["latent"]).to(dev)
    vae.load_state_dict(vck["model"])
    vae.eval().requires_grad_(False)

    def build_policy():
        m = IntentPolicy(ck["policy_kw"], ta["intent_dim"], ta["intent_heads"], ta["intent_depth"],
                         ta.get("intent_mlp", 4.0), text_token_dim=768).to(dev)
        m.load_state_dict(ck["model"])
        return m

    model = build_policy()
    model.eval()
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("policy."))
    ref = build_policy()                       # the EMA reference
    ref.eval().requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.1f} M of "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M (the action policy only); "
          f"EMA reference at {args.ema}", flush=True)

    # ---- reward: min over disjoint-slice models; judge kept apart --------------------------------
    rew_tmrs, rew_norm = [], []
    for p in tmr_paths:
        m, mu, sd, meta = load_tmr(p, dev)
        rew_tmrs.append(m)
        rew_norm.append((torch.tensor(mu, device=dev), torch.tensor(sd, device=dev)))
        print(f"  reward model {Path(p).parent.name}: held-out R@1 {meta.get('r1', float('nan')):.4f}",
              flush=True)
    judge, j_mu, j_sd, j_meta = load_tmr(judge_path, dev)
    j_mu, j_sd = torch.tensor(j_mu, device=dev), torch.tensor(j_sd, device=dev)
    print(f"  judge {Path(judge_path).parent.name}: held-out R@1 "
          f"{j_meta.get('r1', float('nan')):.4f} -- never in the reward", flush=True)
    assert len(rew_tmrs) >= 2, (
        "the reward needs at least two models so a minimum means something; one model is the "
        "unprotected setting the literature warns against")

    # ---- text: the policy's CLIP cache, and the retrieval models' GloVe+POS ----------------------
    tok = joblib.load(text_cache / "tokens.pkl")
    lens = joblib.load(text_cache / "lengths.pkl")
    pool = joblib.load(text_cache / "pooled.pkl")
    from hml_phys.evaluator import GLOVE_DIR, read_split, read_texts                # noqa: E402
    from hml_phys.t2m.word_vectorizer import WordVectorizer                          # noqa: E402
    wv = WordVectorizer(GLOVE_DIR, "our_vab")
    pos_by_key = {}
    for sp in ("train", "test", "val"):
        try:
            names = read_split(sp)
        except Exception:
            continue
        for nm in names:
            try:
                pos_by_key.setdefault(nm, [t["tokens"] for t in read_texts(nm)])
            except Exception:
                pass
    MTL = 20

    def enc_pos(tk):
        if len(tk) < MTL:
            tk = ["sos/OTHER"] + list(tk) + ["eos/OTHER"]
            sl = len(tk)
            tk = tk + ["unk/OTHER"] * (MTL + 2 - sl)
        else:
            tk = ["sos/OTHER"] + list(tk[:MTL]) + ["eos/OTHER"]
            sl = len(tk)
        e, o = zip(*[wv[t] for t in tk])
        return np.stack(e).astype(np.float32), np.stack(o).astype(np.float32), sl

    env.cfg.env.test = True
    env.begin_seq_motion_samples()
    act_mask = action_channel_mask(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    W20 = max(int(round(args.episode_steps / CONTROL_HZ * TMR_FPS)), 8)

    hist_json, t0 = [], time.time()
    for it in range(1, args.iters + 1):
        if it > 1:
            env.forward_motion_samples()      # next NG captions, all envs reset
        lib_ids = env._motion_lib._curr_motion_ids.clone()
        keys = [str(k) for k in env._motion_lib._motion_data_keys[lib_ids.cpu().numpy()]]
        base = [k.split("#g")[0] for k in keys]
        for j in range(NG):                   # the duplication must line up with the grouping
            assert len(set(base[j * G:(j + 1) * G])) == 1, f"group {j} is not one caption: {base[j*G:(j+1)*G]}"

        text = torch.stack([torch.tensor(tok[k][0], dtype=torch.float32) for k in base]).to(dev)
        pooled = torch.stack([torch.tensor(pool[k][0], dtype=torch.float32) for k in base]).to(dev)
        tlen = torch.tensor([int(lens[k][0]) for k in base], device=dev)
        text_u = torch.tensor(tok["__uncond__"][0], dtype=torch.float32, device=dev)[None].expand_as(text)
        pooled_u = torch.tensor(pool["__uncond__"][0], dtype=torch.float32, device=dev)[None].expand_as(pooled)
        tlen_u = torch.full_like(tlen, int(lens["__uncond__"][0]))
        with torch.inference_mode():
            mem_c, mv_c = model.adapter(text, tlen)
            mem_u, mv_u = model.adapter(text_u, tlen_u)
        s_read = float(ta["cond_aug_test"])
        lat_st = np.load(policy_path.parent / "intent_latent_stats.npz")
        lat_mean = torch.tensor(lat_st["mean"], device=dev)
        lat_std = torch.tensor(lat_st["std"], device=dev)
        dur_s = env._motion_lib.get_motion_length().clone().to(dev).float()
        n_env = (dur_s * CONTROL_HZ).ceil().long().clamp_min(1)

        # the retrieval models' text embeddings for this window, one caption per env
        ok = torch.tensor([k in pos_by_key for k in base], device=dev)
        we, po, cl = zip(*[enc_pos(pos_by_key.get(k, [["unk/OTHER"]])[0]) for k in base])
        cl = np.asarray(cl)
        order = np.argsort(-cl, kind="stable")
        inv = torch.tensor(np.argsort(order), device=dev)
        wet = torch.tensor(np.stack(we)[order], device=dev)
        pot = torch.tensor(np.stack(po)[order], device=dev)
        clt = torch.tensor(cl[order], device=dev).long()
        with torch.inference_mode():
            rew_text = [m.encode_text(wet, pot, clt)[inv].clone() for m in rew_tmrs]
            judge_text = judge.encode_text(wet, pot, clt)[inv].clone()

        # ---- roll the group out -------------------------------------------------------------------
        obs, _ = env.reset()
        hist = torch.zeros(B, H, TOKEN_DIM, device=dev)

        def read_prop():
            return torch.cat([env.base_lin_vel, env.base_ang_vel, env.projected_gravity,
                              env.dof_pos, env.dof_vel], dim=-1)

        hold_a = (env.dof_pos - env.default_dof_pos) / float(cfg.control.action_scale)
        hist[:, :, :PROPRIO_DIM] = ((read_prop() - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM])[:, None, :]
        hist[:, :, PROPRIO_DIM:] = ((hold_a - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:])[:, None, :]
        with torch.inference_mode():
            I_H = sample_latent(model.hip, B, mem_c, mv_c, mem_u, mv_u, num_steps=args.num_steps,
                                cfg_scale=args.cfg_scale, generator=gen, device=dev)
            hH_c = intent_hidden(model.hip, I_H, s_read, gen, mem=mem_c, mem_valid=mv_c)
            hH_u = intent_hidden(model.hip, I_H, s_read, gen, mem=mem_u, mem_valid=mv_u)

        alive = torch.ones(B, dtype=torch.bool, device=dev)
        fall_step = torch.full((B,), args.episode_steps, dtype=torch.long, device=dev)
        prop_buf = torch.zeros(B, args.episode_steps, PROPRIO_DIM, device=dev)
        n_plan = (args.episode_steps + hold - 1) // hold
        keep = sorted(np.random.RandomState(args.seed + it).choice(
            n_plan, size=min(args.keep_per_ep, n_plan), replace=False).tolist())
        kept = []       # (x_obs, obs_m, gmask, toks, scal, x0) for the DGPO update

        step, plan_i = 0, 0
        while step < args.episode_steps:
            with torch.inference_mode():
                x_obs = torch.zeros(B, H + F, TOKEN_DIM, device=dev)
                x_obs[:, :H] = hist
                obs_m = observed_mask(B, H, H + F, dev)
                gmask = generated_elements(obs_m, None, act_mask)
                if n_fut_prop > 0:
                    x_obs[:, H, :PROPRIO_DIM] = (read_prop() - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM]
                _, mu_h, _ = vae.encode(hist[:, :, :PROPRIO_DIM])
                lat_h = (mu_h - lat_mean) / lat_std
                scal = torch.stack([(torch.full((B,), float(step), device=dev)
                                     / n_env.float()).clamp(max=1.0), dur_s / 10.0], -1)
                I_I = sample_latent(model.iip, B, mem_c, mv_c, mem_u, mv_u, num_steps=args.num_steps,
                                    cfg_scale=args.cfg_scale, generator=gen, prefix=lat_h,
                                    scalars=scal, extra=hH_c, extra_u=hH_u, device=dev)
                hI_c = intent_hidden(model.iip, I_I, s_read, gen, mem=mem_c, mem_valid=mv_c,
                                     prefix_latent=lat_h, scalars=scal, mem_extra=hH_c)
                toks, _ = model.intent_tokens(hH_c, hI_c,
                                              torch.ones(B, dtype=torch.bool, device=dev))
                x0 = sample_actions(model.policy, x_obs, obs_m, gmask, (text, pooled, tlen),
                                    (text_u, pooled_u, tlen_u), scal, toks,
                                    num_steps=args.num_steps, cfg_scale=args.cfg_action,
                                    generator=gen)
            if plan_i in keep:
                kept.append(tuple(t.clone() for t in (x_obs, obs_m, gmask, toks, scal, x0)))
            a_raw = x0[:, H, PROPRIO_DIM:] * std[PROPRIO_DIM:] + mean[PROPRIO_DIM:]
            prop_in = read_prop()
            applied = a_raw
            for _ in range(hold):
                if step >= args.episode_steps:
                    break
                prop_buf[:, step] = read_prop()
                obs, _, _, dones, _ = env.step(a_raw.detach())
                newly = dones.bool() & alive
                fall_step[newly] = step
                alive &= ~newly
                step += 1
            hist = torch.roll(hist, -1, dims=1)
            hist[:, -1, :PROPRIO_DIM] = (prop_in - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM]
            hist[:, -1, PROPRIO_DIM:] = (applied - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:]
            plan_i += 1

        # ---- episodic reward ----------------------------------------------------------------------
        fell = ~alive
        r_surv = fall_step.float() / args.episode_steps - fell.float()
        feat = torch.nn.functional.interpolate(prop_buf.transpose(1, 2), size=W20, mode="linear",
                                               align_corners=True).transpose(1, 2).contiguous()
        sims = []
        with torch.inference_mode():
            for m, (mu, sd), te in zip(rew_tmrs, rew_norm, rew_text):
                me = m.encode_motion((feat - mu) / sd,
                                     torch.full((B,), W20, device=dev, dtype=torch.long))
                sims.append(-(te - me).norm(dim=-1))
            jm = judge.encode_motion((feat - j_mu) / j_sd,
                                     torch.full((B,), W20, device=dev, dtype=torch.long))
            judge_sim = -(judge_text - jm).norm(dim=-1)
        r_sem = torch.stack(sims, 0).min(0).values          # conservative aggregation
        r_sem = torch.where(ok, r_sem, r_sem.mean().expand_as(r_sem))

        # variance-normalise each term before summing (MO-GRPO)
        nrm = lambda v: v / v.std().clamp_min(1e-6)
        r = args.w_surv * nrm(r_surv) + args.w_sem * nrm(r_sem)
        rg = r.view(NG, G)
        adv = ((rg - rg.mean(1, keepdim=True)) / rg.std(1, keepdim=True).clamp_min(1e-6)) \
            .clamp(-args.adv_clip, args.adv_clip).reshape(-1)
        live = rg.std(1) > 1e-6                              # a group with no spread teaches nothing
        adv = adv * live.repeat_interleave(G).float()

        # ---- the DGPO update ---------------------------------------------------------------------
        # One shared noise level AND one shared epsilon across the WHOLE batch, per the DGPO recipe
        # (`use_shared_noise`): the group's samples must be compared at the same point of the schedule.
        t_shared = fl.sample_t(1, dev, generator=gen).expand(B)
        ones_tok = torch.ones(B, kept[0][3].shape[1], dtype=torch.bool, device=dev)
        zs = []
        for (x_obs, obs_m, gmask, toks, scal, x0) in kept:
            z, _ = build_state_elem(x0, gmask, t_shared, generator=gen)
            zs.append(z)

        # Pass 1, no grad: dsm and the reference's dsm, to form the detached group weight `w` and to
        # decide the winners. Done separately so pass 2 can accumulate gradients one kept step at a
        # time -- four simultaneous forward/backward passes at batch 512 does not fit.
        with torch.inference_mode():
            d_ng, d_ref = [], []
            for z, (x_obs, obs_m, gmask, toks, scal, x0) in zip(zs, kept):
                xh = model.policy(z, obs_m, t_shared, text, pooled, tlen, scal,
                                  extra_tokens=toks, extra_valid=ones_tok)
                xr = ref.policy(z, obs_m, t_shared, text, pooled, tlen, scal,
                                extra_tokens=toks, extra_valid=ones_tok)
                d_ng.append(dsm_per_sample(*velocity_pair(xh, x0, z, t_shared), gmask))
                d_ref.append(dsm_per_sample(*velocity_pair(xr, x0, z, t_shared), gmask))
        dsm_ng = torch.stack(d_ng, 0).mean(0).clone()
        ref_dsm = torch.stack(d_ref, 0).mean(0).clone()
        w = torch.sigmoid((adv * args.beta_dpo * (dsm_ng - ref_dsm)).view(NG, G).mean(1))
        wrep = w.repeat_interleave(G)
        win = (adv > 0).float()
        # ThermoDPO-weighted: (1-t)^2 on the winner reconstruction term.
        anchor_w = (1.0 - t_shared) ** 2

        # Pass 2, with grad, accumulated over the kept steps.
        opt.zero_grad(set_to_none=True)
        acc_dgpo = acc_anchor = 0.0
        for z, (x_obs, obs_m, gmask, toks, scal, x0) in zip(zs, kept):
            xh = model.policy(z, obs_m, t_shared, text, pooled, tlen, scal,
                              extra_tokens=toks, extra_valid=ones_tok)
            d = dsm_per_sample(*velocity_pair(xh, x0, z, t_shared), gmask)
            l_dgpo = (wrep * adv * d).mean()
            l_anch = (win * anchor_w * d).sum() / win.sum().clamp_min(1.0)
            (l_dgpo + args.w_anchor * l_anch).div(len(kept)).backward()
            acc_dgpo += float(l_dgpo) / len(kept)
            acc_anchor += float(l_anch) / len(kept)
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        dsm, loss_dgpo, loss_anchor = dsm_ng, acc_dgpo, acc_anchor
        loss = loss_dgpo + args.w_anchor * loss_anchor
        with torch.no_grad():
            for pm, pr in zip(model.policy.parameters(), ref.policy.parameters()):
                pr.mul_(args.ema).add_(pm, alpha=1 - args.ema)

        m = dict(iter=it, captions=NG, fall=float(fell.float().mean()),
                 surv=float((fall_step.float() / args.episode_steps).mean()),
                 r_sem=float(r_sem.mean()), judge_sim=float(judge_sim.mean()),
                 adv_abs=float(adv.abs().mean()), live_groups=int(live.sum()),
                 dsm=float(dsm.mean()), ref_dsm=float(ref_dsm.mean()), w=float(w.mean()),
                 loss=loss, loss_dgpo=loss_dgpo, loss_anchor=loss_anchor,
                 gn=float(gn), minutes=(time.time() - t0) / 60)
        hist_json.append(m)
        (out / "history.json").write_text(json.dumps(hist_json, indent=1))
        print(f"it {it} fall {m['fall']:.3f} surv {m['surv']:.3f} r_sem {m['r_sem']:.3f} "
              f"judge {m['judge_sim']:.3f} w {m['w']:.3f} dsm {m['dsm']:.4f} "
              f"anch {m['loss_anchor']:.4f} gn {m['gn']:.2f} live {m['live_groups']}/{NG} "
              f"{m['minutes']:.1f}min", flush=True)

        timed = args.max_hours and (time.time() - t0) / 3600 >= args.max_hours
        if it % args.save_every == 0 or it == args.iters or timed:
            torch.save(dict(model=model.state_dict(), policy_kw=ck["policy_kw"], args=ta,
                            dgpo_args=vars(args), iter=it), out / "latest.pt")
        if timed:
            print(f"stopping at iteration {it}: --max-hours {args.max_hours}", flush=True)
            break

    print(f"done, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
