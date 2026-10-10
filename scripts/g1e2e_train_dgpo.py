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

THE GROUP IS G ROLLOUTS IN ONE ENV SLOT, NOT G ENVS. In text-to-image a sample is scorable the moment
it exists. An action chunk is not: its quality only appears once executed. So a group is G complete
rollouts of the same caption, and the reward is episodic -- which is the shape our rewards already have
(did it fall; does the executed motion match the caption). Densifying an episodic reward is what
produced the window defect in §5.17.

The members must differ ONLY in the policy's sampling noise, or the group-relative advantage compares
different problems. Putting G envs on one caption does NOT satisfy that, and a review caught it: with
`terrain.curriculum False`, `legged_robot.py:2878-2903` draws an independent random terrain row per env
and `terrain.py:81-100` gives each patch an independently drawn type and difficulty, so eight envs land
on eight different surfaces (the chance all eight share a type is about 1.7%); and `config_eval.yaml:63`
misspells `andomize_base_com`, so `randomize_base_com` stays True and every env carries its own
U(-0.1, 0.1) m torso centre-of-mass bias. Both are drawn once at sim creation and never redrawn, so the
same env slots would be punished in every iteration -- a constant per-slot term of the same order as the
signal.

So a group is G SEQUENTIAL PASSES THROUGH THE SAME ENV SLOT: identical terrain, identical centre of
mass, identical reset state, differing only in the sampling generator. 512 envs then give 512 groups per
update instead of 64, at G times the rollout cost -- the same cost per group, with groups that are
actually controlled.

MANIFOLD DRIFT is the known failure mode of exactly this objective. "Manifold Drift in Flow Preference
Optimization" (arXiv 2608.20011) shows FlowDPO's loss is the winner's flow-matching error minus the
loser's, and that subtracting the loser term is what pushes terminal samples off the pretrained data
manifold; DGPO's `A_g < 0` branch is that term. Their fix is a winner-side anchor. Ours is two things:
a plain denoising loss on RECORDED data mixed in (`--w-anchor`), which is the same anchor the DGPO
recipe already suggested as an optional small term and which is promoted here to a first-class one;
an EMA reference rather than a fixed one, because three independent results say a fixed-reference KL is
not enough (it changes the optimisation timescale, it fails under heavy-tailed reward error, and its
coefficient is sharply sensitive); and -- because both of those tethers are self-referential, the anchor
being the policy's own winners and the EMA reference following the policy within a few hundred
iterations -- a small pull toward the FROZEN starting checkpoint, which is the one thing here that
actually represents the recorded data, having been trained on all 136k trajectories.

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
    ap.add_argument("--group", type=int, default=4,
                    help="rollouts per caption, run as G SEQUENTIAL PASSES THROUGH THE SAME ENV SLOT")
    ap.add_argument("--num-envs", type=int, default=512)

    ap.add_argument("--episode-steps", type=int, default=300,
                    help="control steps per rollout. Measured: the 512 captions have a median clip of "
                         "7.80 s = 390 control steps, so 300 covers 77%% of the median and gives the "
                         "progress scalar most of its range, and 300 steps is 120 frames at the "
                         "retrieval model's 20 Hz -- inside the [40, 196] range it was trained on. The "
                         "first attempt used 150, which covered 38%% of a median clip and left 484 of "
                         "512 clips unable to finish inside the window, so the survival term barely "
                         "varied and three quarters of the phase range was never trained.")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--loser-weight", type=float, default=1.0,
                    help="weight on the NEGATIVE-advantage branch. 1.0 is DGPO as specified; 0.0 keeps "
                         "only the winners, which is rejection-sampling fine-tuning. v1 collapsed with "
                         "this at 1.0: the fall rate went from 0.059 to 0.998 by iteration 11 while "
                         "NEITHER reward improved, so it was not reward hacking but drift. The cause is "
                         "that the loser branch MAXIMISES a denoising loss, which is unbounded above -- "
                         "the frozen reference's loss on the policy's own samples rose 20-fold, and the "
                         "manifold monitor went from 3.54 to 4.25, where STATUS.md §5.18 puts the "
                         "already-hacked configurations. ThermoDPO reports RFT as a strong baseline on "
                         "several held-out metrics, and RFT has no drift driver at all, so 0.0 is where "
                         "to start; whether the preference branch earns its place is then a clean "
                         "ablation rather than an assumption.")
    ap.add_argument("--epochs", type=int, default=4,
                    help="gradient passes over one batch of rollouts. The rollout is 15 of the 16.7 "
                         "minutes an iteration takes, so extra passes are nearly free, and without "
                         "them 11 hours buys only ~40 gradient steps for an 86M-parameter policy. The "
                         "reuse is UNCORRECTED off-policy: `w` and the advantages are fixed from the "
                         "rollout, and the reference implementation offers an optional PPO-style clip "
                         "on exp(-dsm + dsm_old) for exactly this, which is not implemented here. Kept "
                         "small, and bounded by lr 1e-5 with gradient clipping.")
    ap.add_argument("--beta-dpo", type=float, default=30.0, help="DGPO recipe: 10-100 with a frozen ref")
    ap.add_argument("--adv-clip", type=float, default=5.0)
    ap.add_argument("--live-frac", type=float, default=0.1,
                    help="a group is skipped when its reward spread is below this fraction of the "
                         "batch spread. An absolute 1e-6 floor never fires on continuous retrieval "
                         "distances, so a group whose members are all equally good would otherwise "
                         "contribute a FULL-magnitude gradient derived from float noise.")
    ap.add_argument("--w-bc", type=float, default=0.05,
                    help="pull toward the FROZEN starting checkpoint in x0 space. This is the only "
                         "term here that tethers the policy to the recorded data -- the winner anchor's "
                         "target is the policy's own sample, so it moves with the policy. Against "
                         "manifold drift, which is the documented failure of exactly this objective, a "
                         "self-referential tether is no tether.")
    ap.add_argument("--w-anchor", type=float, default=0.3,
                    help="weight on the WINNER-SIDE ANCHOR against manifold drift: a plain denoising "
                         "loss on the samples with positive advantage, weighted by (1-t)^2. This is "
                         "ThermoDPO's structure (arXiv 2608.20011) -- that paper proves its objective "
                         "reduces to rejection-sampling fine-tuning as its temperature goes to zero, "
                         "and its experiments reweight the winner term so the anchor is not weakened "
                         "at the CLEAN end of the schedule. That paper writes (1-t)^2 in the SD3 "
                         "convention where t = 0 is clean; this repo uses the opposite convention "
                         "(hml_phys/flow.py:2, t = 1 clean), so the same thing is t^2 HERE. Writing "
                         "(1-t)^2 put 0.87 of the weight at the noisy end and 0.07 at the clean end -- "
                         "backwards, and weakest exactly where terminal-sample drift appears. It needs no extra "
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
    ap.add_argument("--manifold-from", default="",
                    help="recorded rollouts defining the data manifold. The monitor is the mean "
                         "distance to the k nearest recorded motions, computed in the JUDGE's "
                         "embedding space, because STATUS.md §5.18 measured that the outlier statistic "
                         "carries no signal in the reward model's own space. Empty disables it, which "
                         "leaves the documented failure mode of this objective undetected for the "
                         "whole run.")
    ap.add_argument("--manifold-n", type=int, default=1500)
    ap.add_argument("--manifold-knn", type=int, default=10)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/latest.pt")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args, overrides = ap.parse_known_args()
    bad = [o for o in overrides if o.startswith("-")]
    assert not bad, f"unrecognised option(s) {bad}; hydra overrides are key=value, not flags"


    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # The evaluation script reads the normalisers from the CHECKPOINT'S OWN directory, so a checkpoint
    # written anywhere else is unloadable without them. v2rft's closed loop died on exactly this after
    # the training finished.
    import shutil
    for _n in ("stats.npz", "intent_latent_stats.npz"):
        _src = Path(args.policy).resolve().parent / _n
        if _src.exists() and not (out / _n).exists():
            shutil.copy2(_src, out / _n)
    refs = Path(args.refs).resolve()
    policy_path = Path(args.policy).resolve()
    text_cache = Path(args.text_cache).resolve()
    tmr_paths = [str(Path(p).resolve()) for p in args.tmr.split(",")]
    judge_path = str(Path(args.tmr_judge).resolve())
    for p in [refs, policy_path] + tmr_paths + [judge_path]:
        assert Path(p).exists(), p
    assert text_cache.is_dir(), text_cache

    import joblib

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
                                     generated_elements, observed_mask, velocity_pair)
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
            f"motion.motion_file={refs}", f"num_envs={args.num_envs}", f"sim_device={args.device}",
            "headless=True", "use_wandb=False",
            "asset.terminate_by_ref_motion_distance=False",   # no reference at deployment, so none here
            # OFF, as in the evaluation script: with it on, a clip simply ENDING is routed into
            # `dones` and would be counted as a fall. 27 of the 512 clips are shorter than 3 s.
            "asset.terminate_by_1time_motion=False",
            # The typo `andomize_base_com` in config_eval.yaml:63 leaves base-CoM randomisation ON, so
            # every env carries its own +-0.1 m torso centre-of-mass bias for the whole run. Harmless
            # when each env is its own measurement; fatal when envs are compared to each other.
            "domain_rand.randomize_base_com=False",
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
    # FROZEN, not an EMA. The recipe specifies beta_dpo 10-100 "with a frozen reference", and the
    # reference implementation freezes it by default with EMA only as an option. A reference that
    # tracks the policy drives `dsm - ref_dsm` toward zero, which pins `w` at sigmoid(0) = 0.5 and
    # makes the whole objective plain advantage-weighted denoising with --beta-dpo inert. It is also
    # the only thing here that represents the recorded data, having been trained on all 136k
    # trajectories, so freezing it is what gives --w-bc something real to pull toward.
    ref = build_policy()
    ref.eval().requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.1f} M of "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M (the action policy only); "
          f"frozen reference", flush=True)

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

    s_read = float(ta["cond_aug_test"])
    _lat = np.load(policy_path.parent / "intent_latent_stats.npz")
    lat_mean = torch.tensor(_lat["mean"], device=dev)
    lat_std = torch.tensor(_lat["std"], device=dev)
    env.cfg.env.test = True
    env.begin_seq_motion_samples()
    act_mask = action_channel_mask(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    W20 = max(int(round(args.episode_steps / CONTROL_HZ * TMR_FPS)), 8)

    # ---- the manifold monitor, in the JUDGE's space ---------------------------------------------
    # STATUS.md §5.18 measured this: mean k-NN distance to recorded robot motion tracks R@1 monotonically
    # in both embedding spaces, while the outlier fraction carries signal ONLY in the judge's space.
    # The monitor therefore lives here, computed on the same window as the reward but with the model
    # that never enters it.
    man_ref = None
    if args.manifold_from:
        rec = joblib.load(args.manifold_from)
        fs = []
        for k, v in list(rec.items())[:args.manifold_n]:
            f = np.asarray(v["proprio"], dtype=np.float32)
            if len(f) >= 100:
                fs.append(f)
        embs = []
        with torch.inference_mode():
            for b in range(0, len(fs), 128):
                chunk = fs[b:b + 128]
                mx = min(max(len(f) for f in chunk), 10 * W20)
                buf = torch.zeros(len(chunk), W20, PROPRIO_DIM, device=dev)
                for q, f in enumerate(chunk):
                    t_ = torch.tensor(f[:mx], device=dev)[None].transpose(1, 2)
                    buf[q] = torch.nn.functional.interpolate(
                        t_, size=W20, mode="linear", align_corners=True)[0].T
                embs.append(judge.encode_motion((buf - j_mu) / j_sd,
                                                torch.full((len(chunk),), W20, device=dev,
                                                           dtype=torch.long)).clone())
        man_ref = torch.cat(embs, 0)
        print(f"manifold reference: {man_ref.shape[0]} recorded motions in the judge's space", flush=True)

    hist_json, t0, it0 = [], time.time(), 0
    if args.resume and (out / "latest.pt").exists():
        rck = torch.load(out / "latest.pt", map_location=dev)
        model.load_state_dict(rck["model"])
        if "opt" in rck:
            opt.load_state_dict(rck["opt"])
        it0 = int(rck.get("iter", 0))
        print(f"resumed at iteration {it0}", flush=True)

    def read_prop():
        return torch.cat([env.base_lin_vel, env.base_ang_vel, env.projected_gravity,
                          env.dof_pos, env.dof_vel], dim=-1)

    for it in range(it0 + 1, args.iters + 1):
        if it > 1:
            env.forward_motion_samples()      # the next B captions; all envs reset
        lib_ids = env._motion_lib._curr_motion_ids.clone()
        n_uni = env._motion_lib._num_unique_motions
        assert torch.equal(lib_ids, (torch.arange(B, device=lib_ids.device) + env.start_idx) % n_uni), (
            "env j is not library entry j; the per-env caption mapping the whole method rests on is "
            "broken (motion_lib_base.py:280-286)")
        base = [str(k) for k in env._motion_lib._motion_data_keys[lib_ids.cpu().numpy()]]

        text = torch.stack([torch.tensor(tok[k][0], dtype=torch.float32) for k in base]).to(dev)
        pooled = torch.stack([torch.tensor(pool[k][0], dtype=torch.float32) for k in base]).to(dev)
        tlen = torch.tensor([int(lens[k][0]) for k in base], device=dev)
        text_u = torch.tensor(tok["__uncond__"][0], dtype=torch.float32, device=dev)[None].expand_as(text)
        pooled_u = torch.tensor(pool["__uncond__"][0], dtype=torch.float32, device=dev)[None].expand_as(pooled)
        tlen_u = torch.full_like(tlen, int(lens["__uncond__"][0]))
        with torch.inference_mode():
            mem_c, mv_c = model.adapter(text, tlen)
            mem_u, mv_u = model.adapter(text_u, tlen_u)
        dur_s = env._motion_lib.get_motion_length().clone().to(dev).float()
        assert dur_s.shape[0] == B, f"{dur_s.shape[0]} motion lengths for {B} envs"
        n_env = (dur_s * CONTROL_HZ).ceil().long().clamp_min(1)

        # A caption whose POS tokens cannot be read gets NO semantic reward; its env still contributes
        # a survival signal. A review measured this set as empty over the first 512 clips, but the
        # caption window advances every iteration and 009831 fails `read_texts`, so it is not empty --
        # and asserting here refused the whole run over one clip.
        has_pos = torch.tensor([k in pos_by_key for k in base], device=dev)
        if not bool(has_pos.all()):
            miss = [k for k in base if k not in pos_by_key]
            print(f"  {len(miss)} of {B} captions have no POS tokens (e.g. {miss[:3]}); "
                  f"those envs get survival only", flush=True)
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

        # ---- G sequential passes through the SAME env slots -------------------------------------
        R_surv = torch.zeros(G, B, device=dev)
        R_sem = torch.zeros(G, B, device=dev)
        keptG, ep_fall, ep_vratio, ep_man = [], [], [], []
        n_plan = (args.episode_steps + hold - 1) // hold
        keep = sorted(np.random.RandomState(args.seed + it).choice(
            n_plan, size=min(args.keep_per_ep, n_plan), replace=False).tolist())

        for g in range(G):
            gg = torch.Generator(device=dev).manual_seed(args.seed * 100003 + it * 97 + g)
            obs, _ = env.reset()
            hist = torch.zeros(B, H, TOKEN_DIM, device=dev)
            hold_a = (env.dof_pos - env.default_dof_pos) / float(cfg.control.action_scale)
            p0 = read_prop()
            hist[:, :, :PROPRIO_DIM] = ((p0 - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM])[:, None, :]
            hist[:, :, PROPRIO_DIM:] = ((hold_a - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:])[:, None, :]
            with torch.inference_mode():
                I_H = sample_latent(model.hip, B, mem_c, mv_c, mem_u, mv_u, num_steps=args.num_steps,
                                    cfg_scale=args.cfg_scale, generator=gg, device=dev)
                hH_c = intent_hidden(model.hip, I_H, s_read, gg, mem=mem_c, mem_valid=mv_c)
                hH_u = intent_hidden(model.hip, I_H, s_read, gg, mem=mem_u, mem_valid=mv_u)

            alive = torch.ones(B, dtype=torch.bool, device=dev)
            fall_step = torch.full((B,), args.episode_steps, dtype=torch.long, device=dev)
            prop_buf = torch.zeros(B, args.episode_steps, PROPRIO_DIM, device=dev)
            kept = []
            step, plan_i = 0, 0
            while step < args.episode_steps:
                with torch.inference_mode():
                    x_obs = torch.zeros(B, H + F, TOKEN_DIM, device=dev)
                    x_obs[:, :H] = hist
                    obs_m = observed_mask(B, H, H + F, dev)
                    gmask = generated_elements(obs_m, None, act_mask)
                    # Only row H is executed (K = 1). The episodic reward cannot speak for rows H+1:,
                    # so they are excluded from the objective rather than given credit they did not earn.
                    gmask = gmask.clone()
                    gmask[:, H + 1:] = 0
                    if n_fut_prop > 0:
                        x_obs[:, H, :PROPRIO_DIM] = (read_prop() - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM]
                    _, mu_h, _ = vae.encode(hist[:, :, :PROPRIO_DIM])
                    lat_h = (mu_h - lat_mean) / lat_std
                    scal = torch.stack([(torch.full((B,), float(step), device=dev)
                                         / n_env.float()).clamp(max=1.0), dur_s / 10.0], -1)
                    I_I = sample_latent(model.iip, B, mem_c, mv_c, mem_u, mv_u,
                                        num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                                        generator=gg, prefix=lat_h, scalars=scal,
                                        extra=hH_c, extra_u=hH_u, device=dev)
                    hI_c = intent_hidden(model.iip, I_I, s_read, gg, mem=mem_c, mem_valid=mv_c,
                                         prefix_latent=lat_h, scalars=scal, mem_extra=hH_c)
                    toks, _ = model.intent_tokens(hH_c, hI_c,
                                                  torch.ones(B, dtype=torch.bool, device=dev))
                    x0 = sample_actions(model.policy, x_obs, obs_m, gmask, (text, pooled, tlen),
                                        (text_u, pooled_u, tlen_u), scal, toks,
                                        num_steps=args.num_steps, cfg_scale=args.cfg_action,
                                        generator=gg)
                if plan_i in keep:
                    # `alive` is stored with the sample: a chunk generated after this env's own
                    # termination was produced from a history that straddles the auto-reset, and it
                    # must not carry the episode's advantage.
                    kept.append((obs_m.clone(), gmask.clone(), toks.clone(), scal.clone(),
                                 x0.clone(), alive.clone()))
                a_raw = x0[:, H, PROPRIO_DIM:] * std[PROPRIO_DIM:] + mean[PROPRIO_DIM:]
                prop_in = read_prop()
                for _ in range(hold):
                    if step >= args.episode_steps:
                        break
                    prop_buf[:, step] = read_prop()
                    obs, _, _, dones, _ = env.step(a_raw.detach())
                    # A fall only counts INSIDE the clip's own length, exactly as the evaluation
                    # script scores it; `terminate_by_1time_motion` is off so a clip ending is not a
                    # done at all, but a short clip's env keeps stepping past its end.
                    newly = dones.bool() & alive & (torch.full((B,), step, device=dev) < n_env)
                    fall_step[newly] = step
                    alive &= ~newly
                    step += 1
                hist = torch.roll(hist, -1, dims=1)
                hist[:, -1, :PROPRIO_DIM] = (prop_in - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM]
                hist[:, -1, PROPRIO_DIM:] = (a_raw - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:]
                plan_i += 1

            # ---- this pass's episodic rewards, on each env's OWN valid prefix -------------------
            n_valid = torch.minimum(fall_step, torch.minimum(n_env, torch.full_like(n_env,
                                                                                   args.episode_steps)))
            R_surv[g] = n_valid.float() / args.episode_steps - (~alive).float()
            m20 = (n_valid.float() / CONTROL_HZ * TMR_FPS).round().long().clamp(40, W20)
            feat = torch.zeros(B, W20, PROPRIO_DIM, device=dev)
            for j in range(B):                       # each env resamples its own prefix, not the pad
                nv = int(n_valid[j])
                if nv < 8:
                    continue
                src = prop_buf[j, :nv].T[None]
                feat[j, :int(m20[j])] = torch.nn.functional.interpolate(
                    src, size=int(m20[j]), mode="linear", align_corners=True)[0].T
            # Below 40 frames the encoder is out of distribution; without POS tokens there is no
            # text side to compare against. Either way the semantic term is absent, not zero.
            sc = (n_valid >= 100) & has_pos
            ml = m20.clone()
            srt = torch.argsort(ml, descending=True)
            isrt = torch.argsort(srt)
            with torch.inference_mode():
                sims = []
                for m, (mu, sd), te in zip(rew_tmrs, rew_norm, rew_text):
                    me = m.encode_motion(((feat - mu) / sd)[srt], ml[srt])[isrt]
                    sims.append(-(te - me).norm(dim=-1))
                jm = judge.encode_motion(((feat - j_mu) / j_sd)[srt], ml[srt])[isrt]
                jsim = -(judge_text - jm).norm(dim=-1)
                if man_ref is not None:
                    ep_man.append(float(torch.cdist(jm, man_ref).topk(
                        args.manifold_knn, dim=1, largest=False).values.mean()))
            R_sem[g] = torch.stack(sims, 0).min(0).values      # conservative aggregation
            R_sem[g] = torch.where(sc, R_sem[g], torch.full_like(R_sem[g], float("nan")))
            keptG.append(kept)
            ep_fall.append(float((~alive).float().mean()))
            v_act = (prop_buf[:, :, :2].norm(dim=-1).sum(1) / n_valid.clamp_min(1).float())
            ep_vratio.append(float(v_act.mean()))
            ep_jsim = float(jsim[sc].mean()) if bool(sc.any()) else float("nan")

        # ---- rewards -> within-group normalisation -> advantage ---------------------------------
        # Each term is divided by the pooled std of its WITHIN-GROUP residuals, not of the batch.
        # A per-caption offset cancels when the group is centred but inflates a batch std, so a batch
        # std silently shrinks whichever term varies most between captions -- which is exactly the
        # semantic term. And the group is centred WITHOUT a second standardisation, because dividing
        # each group by its own spread erases the weights entirely.
        def within(v):
            r = v - v.nanmean(0, keepdim=True)
            sd = r[~torch.isnan(r)].std()
            return r / sd.clamp_min(1e-3), sd

        a_surv, sd_surv = within(R_surv)
        a_sem, sd_sem = within(torch.nan_to_num(R_sem, nan=float("nan")))
        a_sem = torch.nan_to_num(a_sem, nan=0.0)              # unscoreable envs get no semantic signal
        use_surv = float(sd_surv) > 1e-3
        use_sem = float(sd_sem) > 1e-3
        adv = (args.w_surv * a_surv if use_surv else 0.0) + (args.w_sem * a_sem if use_sem else 0.0)
        if not torch.is_tensor(adv):
            print(f"it {it}: neither reward term has within-group spread; skipping", flush=True)
            continue
        spread = (R_surv.std(0) * (args.w_surv / max(float(sd_surv), 1e-3))
                  + a_sem.std(0) * args.w_sem)
        live = spread > args.live_frac * spread.mean().clamp_min(1e-9)
        adv = adv.clamp(-args.adv_clip, args.adv_clip) * live[None, :].float()
        # The loser branch is what drove v1 off the manifold; this is the knob that removes it.
        adv = adv.clamp_min(0) + args.loser_weight * adv.clamp_max(0)

        # ---- the DGPO update ---------------------------------------------------------------------
        # One shared noise level across everything, and ONE SHARED EPSILON PER GROUP: the reference
        # implementation draws one noise tensor per group and indexes it, so the members differ only in
        # their own x0. Here a group is the G passes of one env slot, so epsilon is shared across g.
        t_shared = fl.sample_t(1, dev, generator=gen).expand(B)
        K = len(keptG[0])
        eps = [torch.randn(keptG[0][k][4].shape, device=dev, generator=gen) for k in range(K)]
        ones_tok = torch.ones(B, keptG[0][0][2].shape[1], dtype=torch.bool, device=dev)

        zs, alive_m = [], []
        for g in range(G):
            zg, ag = [], []
            for k in range(K):
                obs_m, gmask, toks, scal, x0, al = keptG[g][k]
                z, _ = build_state_elem(x0, gmask, t_shared, noise=eps[k])
                zg.append(z)
                ag.append(al.float())
            zs.append(zg)
            alive_m.append(ag)

        # Pass 1, no grad: dsm under the policy and under the FROZEN reference, for the group weight.
        with torch.inference_mode():
            dn = torch.zeros(G, B, device=dev)
            dr = torch.zeros(G, B, device=dev)
            for g in range(G):
                for k in range(K):
                    obs_m, gmask, toks, scal, x0, al = keptG[g][k]
                    z = zs[g][k]
                    xh = model.policy(z, obs_m, t_shared, text, pooled, tlen, scal,
                                      extra_tokens=toks, extra_valid=ones_tok)
                    xr = ref.policy(z, obs_m, t_shared, text, pooled, tlen, scal,
                                    extra_tokens=toks, extra_valid=ones_tok)
                    dn[g] += dsm_per_sample(*velocity_pair(xh, x0, z, t_shared), gmask) / K
                    dr[g] += dsm_per_sample(*velocity_pair(xr, x0, z, t_shared), gmask) / K
        dsm_ng, ref_dsm = dn.clone(), dr.clone()
        w = torch.sigmoid((adv * args.beta_dpo * (dsm_ng - ref_dsm)).mean(0))     # [B], one per group
        # ThermoDPO-weighted, in THIS repo's convention (t = 1 clean): t^2, so the winner anchor is
        # strongest at the clean end, which is where terminal-sample drift appears.
        anchor_w = t_shared ** 2
        win = (adv > 0).float()

        acc = dict(dgpo=0.0, anch=0.0, bc=0.0)
        nstep = G * K
        for _ep in range(args.epochs):
          opt.zero_grad(set_to_none=True)
          for g in range(G):
              for k in range(K):
                  obs_m, gmask, toks, scal, x0, al = keptG[g][k]
                  z = zs[g][k]
                  alf = alive_m[g][k]
                  xh = model.policy(z, obs_m, t_shared, text, pooled, tlen, scal,
                                    extra_tokens=toks, extra_valid=ones_tok)
                  d = dsm_per_sample(*velocity_pair(xh, x0, z, t_shared), gmask)
                  l_dgpo = (w * adv[g] * alf * d).mean()
                  l_anch = ((win[g] * alf) * anchor_w * d).sum() / (win[g] * alf).sum().clamp_min(1.0)
                  if args.w_bc:
                      with torch.inference_mode():
                          xb = ref.policy(z, obs_m, t_shared, text, pooled, tlen, scal,
                                          extra_tokens=toks, extra_valid=ones_tok)
                      l_bc = (((xh - xb.clone()) ** 2) * gmask).sum() / gmask.sum().clamp_min(1.0)
                  else:
                      l_bc = torch.zeros((), device=dev)
                  (l_dgpo + args.w_anchor * l_anch + args.w_bc * l_bc).div(nstep).backward()
                  acc["dgpo"] += float(l_dgpo) / nstep / args.epochs
                  acc["anch"] += float(l_anch) / nstep / args.epochs
                  acc["bc"] += float(l_bc) / nstep / args.epochs
          gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
          opt.step()

        m = dict(iter=it, captions=B, group=G, t=float(t_shared[0]),
                 fall=float(np.mean(ep_fall)), surv=float(R_surv.mean()),
                 r_sem=float(torch.nan_to_num(R_sem, nan=0.0).sum() / torch.isfinite(R_sem).sum()),
                 judge_sim=ep_jsim, manifold_knn=(float(np.mean(ep_man)) if ep_man else float("nan")),
                 v_act=float(np.mean(ep_vratio)),
                 sd_surv=float(sd_surv), sd_sem=float(sd_sem),
                 use_surv=use_surv, use_sem=use_sem,
                 adv_abs=float(adv.abs().mean()), adv_pos=float((adv > 0).float().mean()),
                 adv_neg=float((adv < 0).float().mean()), live_groups=int(live.sum()),
                 dsm=float(dsm_ng.mean()), ref_dsm=float(ref_dsm.mean()),
                 w=float(w.mean()), w_std=float(w.std()),
                 loss_dgpo=acc["dgpo"], loss_anchor=acc["anch"], loss_bc=acc["bc"],
                 gn=float(gn), epochs=args.epochs, minutes=(time.time() - t0) / 60)
        hist_json.append(m)
        (out / "history.json").write_text(json.dumps(hist_json, indent=1))
        print(f"it {it} fall {m['fall']:.3f} r_sem {m['r_sem']:.3f} judge {m['judge_sim']:.3f} "
              f"knn {m['manifold_knn']:.3f} w {m['w']:.3f}+-{m['w_std']:.3f} dsm {m['dsm']:.4f}/"
              f"{m['ref_dsm']:.4f} sd(surv,sem) {m['sd_surv']:.3f},{m['sd_sem']:.3f} "
              f"live {m['live_groups']}/{B} gn {m['gn']:.2f} {m['minutes']:.1f}min", flush=True)

        timed = args.max_hours and (time.time() - t0) / 3600 >= args.max_hours
        if it % args.save_every == 0 or it == args.iters or timed:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            policy_kw=ck["policy_kw"], args=ta, dgpo_args=vars(args), iter=it),
                       out / "latest.pt")
            torch.save(dict(model=model.state_dict(), policy_kw=ck["policy_kw"], args=ta, iter=it),
                       out / f"iter_{it}.pt")      # keeping snapshots removes the no-recovery trap
        if timed:
            print(f"stopping at iteration {it}: --max-hours {args.max_hours}", flush=True)
            break

    print(f"done, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
