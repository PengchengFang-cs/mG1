"""Closed-loop evaluation of the end-to-end G1 policy: text in, actions out, no reference.

This is the number that counts. The training losses say how well the policy imitates the tracker at
states the tracker visited; only a closed loop says whether it can hold itself up at the states it
visits ITSELF, where a behaviour-cloned policy has never been supervised.

The env is FRoM-W1's teleop tracking env, but driven without a reference:
  * reference-based termination is switched off (`terminate_by_ref_motion_distance`,
    `terminate_by_1time_motion`) because there is no reference to be far from or to run out of;
  * contact and tilt termination stay on -- those are the fall criteria;
  * a reference library is still passed because the env needs one to construct, and its observations are
    never read: our policy consumes its own 51-d proprio straight from the env's buffers.

Generation runs at `gen_hz` and the robot at 50 Hz. The policy emits F action rows per plan, each held
for 50/gen_hz control steps (zero-order hold). Holding rather than interpolating is the conservative
choice: interpolation would smooth the action stream in a way the recorded teacher data never saw, and
would flatter the policy for a reason that has nothing to do with what it learned.

PROTOCOL. A behaviour-cloned policy is asked whether it can EXECUTE what it was taught, so the default
protocol is in-distribution on both axes:
  prompts    the captions of our own training pool (`--refs data/g1_e2e/refs_train_*.pkl`), the way
             ADAPT scores its BC ablation on the 130-command pool drawn from its own training skill
             vocabulary, holding out exactly one command ("jog") for a separate generalisation test.
  horizon    each env runs for ITS OWN clip's duration (`--episode motion`), which is both what the
             caption describes and what the duration scalar was trained on.
  history    initialised from the real state at reset (`--hist-init rest`), not the dataset mean.
Held-out captions (`--refs data/g1_e2e/refs_test.pkl`) and a fixed 20 s horizon regardless of prompt
(`--episode fixed`) are a DIFFERENT and strictly harder question -- generalisation and RL-style
robustness. Those belong with the on-policy stage, not with pure BC, and must be reported as such.

Reported, with the convention stated (CLAUDE.md §2 and §4 -- truncate-fallen, single rollout, single
computation):
  physical  fall rate, duration completion (fraction of the 20 s episode survived)
  semantic  R@1/R@2/R@3 and FID through the Guo evaluator, via G1 links -> 22 SMPL joints
            (`hml_phys/g1_to_smpl.py`) -> 263-d features -> `hml_phys/evaluator.py`

Run on a compute node:
  srun --jobid=<id> --overlap --ntasks=1 bash -lc '
    source scripts/activate_h2h.sh
    python scripts/g1e2e_eval_closed_loop.py --policy outputs/g1e2e/policy/best.pt \
      --out outputs/g1e2e/eval_test.json'
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
LOG_ROOT = LEGGED_GYM / "logs/robot:teleop"
CONTROL_HZ = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="checkpoint from g1e2e_train_policy.py")
    ap.add_argument("--refs", default="data/g1_e2e/refs_train_sub2000.pkl",
                    help="the PROMPT POOL. In the pure-BC stage this is the TRAINING pool (CLAUDE.md "
                         "§11): a supervised imitation model is asked whether the skills it was taught "
                         "can be executed, not whether it generalises. Held-out captions, a fixed "
                         "robustness horizon and disturbance rejection are on-policy questions and are "
                         "not evidence about an SFT model. The env also needs a library to construct, "
                         "but its observations are never read -- the policy consumes its own proprio.")
    ap.add_argument("--text-cache", default="data/g1_e2e/text_clipL14")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode", choices=["motion", "fixed"], default="motion",
                    help="'motion': each env runs for ITS OWN clip's duration -- what the caption "
                         "describes, and what the duration scalar was trained on. 'fixed': every env "
                         "runs --episode-s seconds regardless of the prompt, an RL-style robustness "
                         "horizon that belongs with PPO, not with a behaviour-cloned policy.")
    ap.add_argument("--episode-s", type=float, default=20.0,
                    help="episode length for --episode fixed; unused in 'motion' mode")
    ap.add_argument("--hold-action", choices=["hold", "zero"], default="hold",
                    help="attribution knob for --hist-init rest: 'hold' writes (dof_pos-default)/scale, "
                         "the action whose PD target is the pose the robot is in; 'zero' writes a raw 0, "
                         "which commands the DEFAULT pose and is what the pre-fix loop did.")
    ap.add_argument("--holi-per-plan", action="store_true",
                    help="attribution knob: re-draw the holistic intent inside every plan, as the "
                         "pre-fix loop did, instead of once per episode (mc_rollout.py:160-169).")
    ap.add_argument("--weights", choices=["model", "raw"], default="model",
                    help="which weight set in the checkpoint to roll out. 'model' is the EMA (what the "
                         "reference rolls out, mc_rollout.py:25,31); 'raw' is the AdamW iterate at the "
                         "same step, stored alongside so the EMA can be attributed without retraining. "
                         "A checkpoint written before EMA existed has no 'raw' and only accepts 'model'.")
    ap.add_argument("--n-future-proprio", type=int, default=-1,
                    help="override how many future rows keep their proprio; -1 = take it from the "
                         "checkpoint's obs_future, which is the only setting that matches training. "
                         "Forcing 0 on a model TRAINED with 1 does not reproduce a model trained "
                         "without it -- it denies the policy an input it was fitted to use, so the "
                         "result bounds how much the policy relies on the current state, nothing more.")
    ap.add_argument("--hist-init", choices=["rest", "zeros"], default="rest",
                    help="'rest': fill the history with the real proprio at reset and a zero action, "
                         "i.e. the robot holding the pose it starts in -- a state the data contains. "
                         "'zeros' writes the dataset MEAN into every history row, a state the robot is "
                         "never in; kept only to reproduce the earlier numbers.")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=10, help="flow sampling steps")
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--cfg-action", type=float, default=-1.0,
                    help="guidance scale for the ACTION policy, separately from the intent predictors. "
                         "-1 = use --cfg-scale for both, which is what the first runs did. MIND applies "
                         "guidance to the intent LATENTS; on raw joint actions the same scale pushes the "
                         "output away from the unconditional distribution, i.e. it systematically "
                         "inflates joint targets. Measured: shadow row-0 NMSE 0.4175 at 2.5 vs 0.1476 at "
                         "1.0, same process, same clips.")
    ap.add_argument("--K", type=int, default=2,
                    help="action rows executed per plan before replanning; 0 = all F. The validated MIND "
                         "configuration executes 2 of 4 (STATUS.md 3). K=F leaves the last row 160 ms "
                         "stale: measured per-row shadow NMSE 0.1476/0.2172/0.2912/0.3568.")
    ap.add_argument("--warmup-tracker", action="store_true",
                    help="let FRoM-W1's tracker drive the first H generation frames, so the history "
                         "buffer starts from real states consistent with the robot's actual pose instead "
                         "of zeros. A diagnostic: it separates a cold-start distribution shift from the "
                         "policy being unable to hold itself up at all.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hold", type=int, default=0,
                    help="control steps each action is held; 0 = 50/gen_hz, the rate the policy plans at")
    ap.add_argument("--teacher", action="store_true",
                    help="diagnostic: the TRACKER drives the whole episode, in this same env and with its "
                         "action held for --hold control steps. Run it at hold=1 and at the policy's hold "
                         "to separate 'behaviour cloning cannot hold the robot up' from 'nothing can hold "
                         "the robot up at this action rate' -- the teacher's own actions, subsampled and "
                         "held, are not the teacher's behaviour.")
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="teacher mode only: add white noise of this RELATIVE size to the teacher's "
                         "action (sigma = value x the per-joint action std of the training data). It "
                         "turns the shadow NMSE into a survival number: a policy with shadow NMSE n has "
                         "a relative action RMS error of sqrt(n), so passing sqrt(n) here asks what the "
                         "plant does under an error of exactly our size, without our policy's structure.")
    ap.add_argument("--action-noise-sweep", default="",
                    help="teacher mode: comma-separated relative sizes run in ONE process, e.g. "
                         "'0,0.1,0.2,0.45'. Overrides --action-noise.")
    ap.add_argument("--ablate-sweep", default="",
                    help="shadow mode: comma-separated input ablations run in ONE process, from "
                         "none,proprio,action,text. Each blanks that channel to the dataset mean, so the "
                         "NMSE says what the policy is actually using.")
    ap.add_argument("--cfg-sweep", default="",
                    help="shadow mode: comma-separated guidance scales run in ONE process, e.g. "
                         "'1.0,1.5,2.5'. Overrides --cfg-scale.")
    ap.add_argument("--shadow", action="store_true",
                    help="diagnostic: the tracker drives every step, so states stay on-distribution, "
                         "while the policy predicts the same actions WITHOUT executing them. A low error "
                         "means the imitation is sound and the failure is closed-loop compounding; a high "
                         "error means the policy is being fed something training never showed it.")
    args, overrides = ap.parse_known_args()
    # parse_known_args hands anything unrecognised to hydra as an override, where a leading "--" is a
    # lexer error rather than an "unknown flag" message. Catch our own typos here instead.
    bad = [o for o in overrides if o.startswith("-")]
    assert not bad, f"unrecognised option(s) {bad}; hydra overrides are key=value, not flags"

    # Resolve every path BEFORE the chdir below: the env's asset path is relative and only works from
    # legged_gym/, so after that point a relative --policy or --refs silently points into their tree.
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    refs = Path(args.refs).resolve()
    policy_path = Path(args.policy).resolve()
    text_cache = Path(args.text_cache).resolve()
    assert policy_path.exists(), policy_path
    assert refs.exists(), refs
    assert text_cache.is_dir(), text_cache

    assert (LEGGED_GYM / "resources/robots/g1/urdf/g1_21dof.urdf").exists(), "21-DoF asset missing"
    os.chdir(LEGGED_GYM)
    sys.path.insert(0, str(H2H))
    sys.path.insert(0, str(REPO))

    from isaacgym import gymapi          # noqa: E402  (before torch, on purpose)
    import numpy as np                   # noqa: E402
    import torch                         # noqa: E402
    import joblib                        # noqa: E402
    import hydra                         # noqa: E402
    from omegaconf import OmegaConf       # noqa: E402
    from easydict import EasyDict         # noqa: E402
    import legged_gym.envs                # noqa: E402,F401
    from legged_gym.utils import task_registry   # noqa: E402

    from hml_phys.g1e2e_data import PROPRIO_DIM, ACTION_DIM, TOKEN_DIM   # noqa: E402
    from hml_phys.g1e2e_flow import (action_channel_mask, generated_elements,   # noqa: E402
                                     observed_mask)
    from hml_phys.intent_model import IntentPolicy                        # noqa: E402
    from hml_phys.intent_vae import IntentVAE                             # noqa: E402
    from hml_phys.intent_flow import (intent_hidden, sample_actions,      # noqa: E402
                                      sample_latent)

    ck = torch.load(policy_path, map_location="cpu")
    ta = ck["args"]
    gen_hz, H, F = ta["gen_hz"], ta["H"], ta["F"]
    hold = args.hold or CONTROL_HZ // gen_hz
    K = args.K or F
    # Mirror the checkpoint's obs_future, so the loop shows the policy exactly the rows training did.
    n_fut_prop = ({"none": 0, "first": 1, "all": F}[ta.get("obs_future", "first")]
                  if args.n_future_proprio < 0 else args.n_future_proprio)
    print(f"policy: gen {gen_hz} Hz, H {H}, F {F}, executing K={K} rows, hold {hold} control steps each")

    # The env must not terminate on a reference it is not following; contact and tilt termination stay on.
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg_h = hydra.compose(config_name="config_eval", overrides=[
            f"motion.motion_file={refs}", f"num_envs={args.num_envs}", f"sim_device={args.device}",
            "headless=True", "use_wandb=False",
            "asset.terminate_by_ref_motion_distance=False",
            "asset.terminate_by_1time_motion=False",
            # In 'motion' mode the horizon is per clip and enforced by this script; the env's own cap is
            # only there to stop it auto-resetting under us, so it is set well past the longest clip.
            f"env.episode_length_s={args.episode_s if args.episode == 'fixed' else 60.0}",
            *overrides])
    cfg = EasyDict(OmegaConf.to_container(cfg_h, resolve=True))
    cfg.physics_engine = gymapi.SIM_PHYSX
    assert cfg.asset.terminate_by_ref_motion_distance is False
    assert cfg.asset.terminate_by_1time_motion is False
    print(f"termination: contacts {cfg.asset.terminate_after_contacts_on}, "
          f"gravity {cfg.asset.terminate_by_gravity}, reference OFF")

    env, _ = task_registry.make_env_hydra(name=cfg.task, hydra_cfg=cfg, env_cfg=cfg)
    dev = env.device
    tracker = None
    if args.warmup_tracker or args.teacher:
        runner, _ = task_registry.make_alg_runner(env=env, name=cfg.task, args=cfg, train_cfg=cfg.train)
        tracker = runner.get_inference_policy(device=dev)
    # Sequential assignment, always. Env i then holds library motion i, so the caption we feed, the pose
    # the env resets into and the clip duration we score against all describe the SAME clip. Under the
    # default random sampling `_motion_data_keys[:num_envs]` is not the batch the envs actually hold, and
    # the caption would be paired with an unrelated start pose.
    env.cfg.env.test = True
    env.begin_seq_motion_samples()

    stats = np.load(policy_path.parent / "stats.npz")
    mean = torch.tensor(stats["mean"], device=dev)
    std = torch.tensor(stats["std"], device=dev)

    vae_path = Path(ta["vae"])
    if not vae_path.is_absolute():
        vae_path = REPO / vae_path
    vck = torch.load(vae_path, map_location="cpu")
    va = vck["args"]
    vae = IntentVAE(input_dim=vck["input_dim"], width=va["width"], down_t=va["down_t"], stride_t=2,
                    depth=va["depth"], dilation_growth_rate=va["dilation"], latent_dim=va["latent"]).to(dev)
    vae.load_state_dict(vck["model"])
    vae.eval().requires_grad_(False)

    # intent_mlp from the CHECKPOINT, never the class default. It is the 5th positional argument
    # (intent_model.py:115) and defaults to 4.0, so omitting it built a 1536-wide SwiGLU against a
    # checkpoint trained at 1.5 -> 576, and load_state_dict failed with a wall of size mismatches.
    # A checkpoint that predates the flag keeps the old 4.0 it was actually trained with.
    model = IntentPolicy(ck["policy_kw"], ta["intent_dim"], ta["intent_heads"], ta["intent_depth"],
                         ta.get("intent_mlp", 4.0), text_token_dim=768).to(dev)
    sd = ck["model"] if args.weights == "model" else ck.get("raw")
    assert sd is not None, f"--weights raw: this checkpoint has no raw iterate (keys: {sorted(ck)})"
    model.load_state_dict(sd)
    print(f"weights: {args.weights}, checkpoint step {ck.get('step')}")
    model.eval().requires_grad_(False)

    tc = text_cache
    tok = joblib.load(tc / "tokens.pkl")
    lens = joblib.load(tc / "lengths.pkl")
    lib = env._motion_lib
    ids = lib._curr_motion_ids.clone()
    keys = [str(k) for k in lib._motion_data_keys[ids.cpu().numpy()]]
    assert all(k in tok for k in keys), "some env clips have no cached caption"
    pool = joblib.load(tc / "pooled.pkl")
    text = torch.stack([torch.tensor(tok[k][0], dtype=torch.float32) for k in keys]).to(dev)
    pooled = torch.stack([torch.tensor(pool[k][0], dtype=torch.float32) for k in keys]).to(dev)
    tlen = torch.tensor([int(lens[k][0]) for k in keys], device=dev)
    # The unconditional state, cached at training time, broadcast to the batch.
    text_u = torch.tensor(tok["__uncond__"][0], dtype=torch.float32, device=dev)[None].expand_as(text)
    pooled_u = torch.tensor(pool["__uncond__"][0], dtype=torch.float32, device=dev)[None].expand_as(pooled)
    tlen_u = torch.full_like(tlen, int(lens["__uncond__"][0]))
    mem_c, mv_c = model.adapter(text, tlen)
    mem_u, mv_u = model.adapter(text_u, tlen_u)
    s_read = float(ta["cond_aug_test"])
    lat_st = np.load(policy_path.parent / "intent_latent_stats.npz")
    lat_mean = torch.tensor(lat_st["mean"], device=dev)
    lat_std = torch.tensor(lat_st["std"], device=dev)
    print(f"prompts: {len(keys)} clips, one caption each; CLIP('') {int(lens['__uncond__'][0])} tokens; "
          f"intent-hidden read at s={s_read}")

    act_mask = action_channel_mask(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    B = env.num_envs
    # Per-env horizon, in the env's own control steps. No motion_ids passed: the library's per-motion
    # tensors are LOCAL to the loaded batch and indexed by env, while _curr_motion_ids are global ids
    # into the full library -- passing them in triggers a device-side assert (g1e2e_record_rollouts.py).
    secs = lib.get_motion_length().clone()
    assert secs.shape[0] == B, (
        f"library returned {secs.shape[0]} motion lengths for {B} envs; these tensors are per-env and "
        f"local to the loaded batch")
    if args.episode == "fixed":
        secs = torch.full_like(secs, float(args.episode_s))
    dur_s = secs.to(dev).float()
    n_env = (dur_s * CONTROL_HZ).ceil().long().clamp_min(1)
    n_ctrl = int(n_env.max().item())
    print(f"episode mode '{args.episode}': clip durations {float(dur_s.min()):.2f}-{float(dur_s.max()):.2f} s "
          f"(mean {float(dur_s.mean()):.2f}), batch horizon {n_ctrl} steps; history init '{args.hist_init}'")

    hist = torch.zeros(B, H, TOKEN_DIM, device=dev)
    body_pos = []
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    # "Never fell" is encoded as reaching the env's OWN horizon, not the batch's.
    fall_step = n_env.clone()

    def read_prop():
        return torch.cat([env.base_lin_vel, env.base_ang_vel, env.projected_gravity,
                          env.dof_pos, env.dof_vel], dim=-1)

    def push_hist(prop, action_raw):
        """One row per executed action: (the state the action was applied IN, that action). Both real.

        This has to be the recorder's pairing exactly. `g1e2e_record_rollouts.py` stores proprio BEFORE
        the step -- "the state the action is applied in" -- so a training token is (s_i, a_i). Two earlier
        versions of this loop got it wrong: the first rolled twice per action and split the two channels
        across separate rows (half the conditioning was the dataset mean); the second read proprio AFTER
        the hold, writing (s_{i+1}, a_i) and so putting every row's state one generation step -- 40 ms --
        ahead of the action it is paired with. The shadow check could not catch either, because it reads
        the history through this same function.
        """
        nonlocal hist
        hist = torch.roll(hist, -1, dims=1)
        hist[:, -1, :PROPRIO_DIM] = (prop - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM]
        hist[:, -1, PROPRIO_DIM:] = (action_raw - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:]


    obs, _ = env.reset()
    if args.hist_init == "rest":
        # The real state at t=0, held for all H rows, paired with the HOLD ACTION -- the action whose PD
        # target is the joint angles the robot is already in: `target = default_dof_pos + 0.25 * EMA(a)`
        # (legged_robot.py:1731), so holding the current pose needs `a = (dof_pos - default)/0.25`, which
        # is the reference's `tokens.hold_action` (tokens.py:147-148, used at mc_rollout.py:174).
        # A raw action of 0 -- what this used to write -- commands the DEFAULT pose, not the pose just
        # read, so all 16 rows asserted a large commanded-vs-actual joint error that the recorded data
        # only ever contains during tracking failures.
        prop0 = read_prop()
        hold_a = ((env.dof_pos - env.default_dof_pos) / float(cfg.control.action_scale)
                  if args.hold_action == "hold" else torch.zeros_like(env.dof_pos))
        hist[:, :, :PROPRIO_DIM] = ((prop0 - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM])[:, None, :]
        hist[:, :, PROPRIO_DIM:] = ((hold_a - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:])[:, None, :]
    t0 = time.time()
    step = 0

    shadow_se, shadow_var, shadow_n = 0.0, 0.0, 0
    if tracker is not None and not args.teacher:
        # Warm-up: the tracker tracks the reference for H generation frames, filling the history from the
        # SAME buffers the closed loop reads, so the handover state is real and self-consistent.
        n_warm = H * hold
        for w in range(n_warm):
            with torch.inference_mode():
                a_tr = tracker(obs.detach())
            if w % hold == 0:
                prop_in = read_prop()        # the state this generation step's action is applied in
                a_in = a_tr.detach()
            obs, _, _, dones, _ = env.step(a_tr.detach())
            newly = dones.bool() & alive & (step < n_env)
            fall_step[newly] = step
            alive &= ~newly
            body_pos.append(env._rigid_body_pos.detach().clone().cpu())
            step += 1
            if w % hold == hold - 1:
                push_hist(prop_in, a_in)
        print(f"warm-up: tracker drove {n_warm} control steps ({n_warm / CONTROL_HZ:.2f} s); "
              f"{int((~alive).sum())}/{B} already down, handing over to the policy", flush=True)

    ablate = "none"
    holi = [None, None]        # (hH_c, hH_u): the episode's holistic intent hidden states

    def sample_holistic():
        """Draw the holistic intent for a fresh episode. Called once after every env reset."""
        nonlocal holi
        if ablate == "text":
            mC, vC, cfgv = mem_u, mv_u, 1.0
        else:
            mC, vC, cfgv = mem_c, mv_c, args.cfg_scale
        with torch.inference_mode():
            I_H = sample_latent(model.hip, B, mC, vC, mem_u, mv_u, num_steps=args.num_steps,
                                cfg_scale=cfgv, generator=gen, device=dev)
            holi = [intent_hidden(model.hip, I_H, s_read, gen, mem=mC, mem_valid=vC),
                    intent_hidden(model.hip, I_H, s_read, gen, mem=mem_u, mem_valid=mv_u)]

    def plan():
        """One planning pass: HIP -> IIP -> policy, returning the x0 window. Shared by the closed
        loop and the shadow check so the two cannot drift apart.

        `ablate` blanks one input channel to the dataset mean (0 in normalised units) so a shadow run
        can report what the policy is actually using. A persistence predictor -- repeat the previous
        action, which is the last row of the history's action channel -- reaches NMSE 0.1581 on the same
        clips, better than the policy's 0.1948, so "the policy imitates well" is not established and
        which channel carries its signal is the question that matters.
        """
        x_obs = torch.zeros(B, H + F, TOKEN_DIM, device=dev)
        h = hist
        if ablate == "proprio":
            h = hist.clone()
            h[:, :, :PROPRIO_DIM] = 0.0
        elif ablate == "action":
            h = hist.clone()
            h[:, :, PROPRIO_DIM:] = 0.0
        x_obs[:, :H] = h
        # Row H's proprio is the state this plan's first action is applied in -- the state the robot is
        # in RIGHT NOW. Training keeps it (obs_future="first"), so the loop must supply it; leaving it
        # zero made the policy a feedforward predictor with 40-160 ms of dead time. `n_fut_prop` mirrors
        # the checkpoint's own obs_future so the two cannot drift apart.
        if n_fut_prop > 0 and ablate != "proprio":
            prop_now = read_prop()
            x_obs[:, H, :PROPRIO_DIM] = (prop_now - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM]
        obs_m = observed_mask(B, H, H + F, dev)
        g = generated_elements(obs_m, None, act_mask)
        _, mu_hist, _ = vae.encode(h[:, :, :PROPRIO_DIM])
        lat_hist = (mu_hist - lat_mean) / lat_std
        # Exactly the pair training used: progress through THIS clip, and the clip's own duration in
        # seconds / 10 (hml_phys/g1e2e_data.py:143-151). The earlier run passed episode_s/10 = 2.0 to
        # every env while training only ever saw a real clip length (~0.2-1.0) -- off-distribution
        # conditioning fed to all three modules, independent of which prompt pool was used.
        scal = torch.stack([(step / n_env.float()).clamp(max=1.0), dur_s / 10.0], -1)

        # Classifier-free guidance needs the SAME unconditional state training used: CLIP(''), and
        # for the policy no intent tokens at all (intent_flow.sample_actions). Passing the
        # conditional memory as both branches would make the guidance term identically zero.
        # Ablating the text means conditioning on CLIP('') everywhere, which also makes guidance a
        # no-op -- so the scale drops to 1.0 rather than amplifying the gap between two identical
        # branches.
        if ablate == "text":
            mC, vC, txtC, cfgv = mem_u, mv_u, (text_u, pooled_u, tlen_u), 1.0
        else:
            mC, vC, txtC, cfgv = mem_c, mv_c, (text, pooled, tlen), args.cfg_scale
        # The HOLISTIC intent is the plan for the whole clip and is drawn ONCE per episode, from the text
        # alone (mc_rollout.py:160-169, "one holistic intent per episode, from the text alone, MIND 4.3").
        # Re-drawing it inside every plan moved the long-horizon conditioning to a different mode every
        # 160 ms, and made hI -- whose memory is hH_c -- incoherent with the previous chunk.
        if args.holi_per_plan:
            sample_holistic()        # pre-fix behaviour: a fresh holistic intent every plan
        assert holi[0] is not None, "sample_holistic() must run once after each env reset"
        hH_c, hH_u = holi
        I_I = sample_latent(model.iip, B, mC, vC, mem_u, mv_u, num_steps=args.num_steps,
                            cfg_scale=cfgv, generator=gen, prefix=lat_hist, scalars=scal,
                            extra=hH_c, extra_u=hH_u, device=dev)
        hI_c = intent_hidden(model.iip, I_I, s_read, gen, mem=mC, mem_valid=vC,
                             prefix_latent=lat_hist, scalars=scal, mem_extra=hH_c)
        toks, _ = model.intent_tokens(hH_c, hI_c, torch.ones(B, dtype=torch.bool, device=dev))
        x0 = sample_actions(model.policy, x_obs, obs_m, g,
                            txtC, (text_u, pooled_u, tlen_u),
                            scal, toks, num_steps=args.num_steps, cfg_scale=(cfgv if args.cfg_action < 0 else args.cfg_action),
                            generator=gen)

        return x0

    if args.teacher:
        # The teacher's own actions, in the policy's env, at the policy's action rate. This is the
        # control-rate control experiment: it asks whether ANY controller survives when its output is
        # held for `hold` steps, before asking whether ours imitates well enough.
        a_std = std[PROPRIO_DIM:]
        sigmas = [float(s) for s in args.action_noise_sweep.split(",")] if args.action_noise_sweep \
            else [args.action_noise]
        print(f"teacher mode: tracker drives every step, action held {hold} control step(s) "
              f"(= {CONTROL_HZ / hold:.1f} Hz effective); relative action noise {sigmas} "
              f"(sigma = value x per-joint action std). Setting up the env costs far more than the "
              f"rollout here, so the whole sweep runs in one process.", flush=True)
        rows = []
        for sigma in sigmas:
            if rows:                      # a fresh episode for each sigma, same clips, same order
                env.begin_seq_motion_samples()
                obs, _ = env.reset()
            alive = torch.ones(B, dtype=torch.bool, device=dev)
            fall_step = n_env.clone()
            step, ts = 0, time.time()
            while step < n_ctrl:
                with torch.inference_mode():
                    a_tr = tracker(obs.detach())
                    if sigma > 0:
                        # Fresh noise per generation step, held with the action, so its spectrum matches
                        # a policy that makes an independent error each time it plans.
                        a_tr = a_tr + sigma * a_std * torch.randn(
                            a_tr.shape, generator=gen, device=dev, dtype=a_tr.dtype)
                for _ in range(hold):
                    if step >= n_ctrl:
                        break
                    obs, _, _, dones, _ = env.step(a_tr.detach())
                    newly = dones.bool() & alive & (step < n_env)
                    fall_step[newly] = step
                    alive &= ~newly
                    step += 1
            fall_rate = float((~alive).float().mean())
            duration = float((fall_step.float() / n_env.float()).mean())
            rows.append(dict(action_noise=sigma, fall_rate=fall_rate, duration_completion=duration,
                             n_fell=int((~alive).sum())))
            print(f"\nnoise {sigma:.3f}   fall rate {fall_rate:.4f}   "
                  f"duration completion {duration:.4f}   {time.time() - ts:.0f}s", flush=True)
        out.write_text(json.dumps(dict(mode="teacher", hold=hold, effective_hz=CONTROL_HZ / hold,
                                       refs=str(refs), n_envs=B, episode=args.episode,
                                       clip_s_mean=float(dur_s.mean()), rows=rows),
                                  indent=2))
        print(f"wrote {out}")
        return

    if args.shadow:
        assert tracker is not None, "--shadow needs --warmup-tracker: the tracker has to drive"
        # Per-ROW error, not just the first row. The closed loop executes K rows per plan, so rows
        # H+1..H+F-1 are applied open-loop 40..160 ms after the state they were planned from. Measuring
        # only row H says nothing about them, and if their error grows with the offset then the chunk
        # length K -- not the imitation quality -- is what kills the closed loop.
        # Also a CFG sweep: guidance 2.5 is MIND's setting for the intent LATENTS; applying it to raw
        # joint actions pushes them away from the unconditional mean and may simply inflate them.
        cfgs = [float(x) for x in args.cfg_sweep.split(",")] if args.cfg_sweep else [args.cfg_scale]
        abls = args.ablate_sweep.split(",") if args.ablate_sweep else ["none"]
        variants = [(ab, c) for ab in abls for c in cfgs]
        cfg_rows = []
        for ablate, cfg_val in variants:
            args.cfg_scale = cfg_val
            if cfg_rows:
                env.begin_seq_motion_samples()
                obs, _ = env.reset()
                hist = torch.zeros(B, H, TOKEN_DIM, device=dev)
                step = 0
                for w in range(H * hold):           # refill the history exactly as the warm-up does
                    with torch.inference_mode():
                        a_tr = tracker(obs.detach())
                    if w % hold == 0:
                        prop_in, a_in = read_prop(), a_tr.detach()
                    obs, _, _, _, _ = env.step(a_tr.detach())
                    step += 1
                    if w % hold == hold - 1:
                        push_hist(prop_in, a_in)
            sample_holistic()        # one holistic intent per episode, per variant
            se = torch.zeros(F, dtype=torch.float64)
            var = torch.zeros(F, dtype=torch.float64)
            cnt = torch.zeros(F, dtype=torch.float64)
            pending = []            # [(rows_for_the_next_F_gen_steps, how many are already consumed)]
            ts = time.time()
            while step < n_ctrl:
                with torch.inference_mode():
                    pending.append([plan()[:, H:H + F, PROPRIO_DIM:], 0])
                    a_tr = tracker(obs.detach())
                    true_n = (a_tr - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:]
                    for p in pending:
                        k = p[1]
                        se[k] += float(((p[0][:, k] - true_n) ** 2).sum())
                        var[k] += float((true_n ** 2).sum())
                        cnt[k] += true_n.numel()
                        p[1] += 1
                    pending = [p for p in pending if p[1] < F]
                prop_in = read_prop()
                for _ in range(hold):
                    if step >= n_ctrl:
                        break
                    obs, _, _, dones, _ = env.step(a_tr.detach())
                    step += 1
                push_hist(prop_in, a_tr.detach())
            per_row = (se / var.clamp_min(1e-9)).tolist()
            nmse = float(se.sum() / var.sum().clamp_min(1e-9))
            cfg_rows.append(dict(cfg_scale=cfg_val, ablate=ablate, nmse_row0=per_row[0],
                                 nmse_all_rows=nmse, nmse_per_row=per_row, n_elements=int(cnt.sum())))
            print(f"\nablate {ablate:<8} cfg {cfg_val:.2f}   row-0 NMSE {per_row[0]:.4f}   "
                  f"all-rows {nmse:.4f}   per row {[f'{v:.4f}' for v in per_row]}   "
                  f"{time.time() - ts:.0f}s", flush=True)
        shadow_se, shadow_n = 0.0, 0
        nmse = cfg_rows[0]["nmse_row0"]
        out.write_text(json.dumps(dict(mode="shadow", nmse=nmse, rows=cfg_rows,
                                       policy=str(policy_path)), indent=2))
        print(f"wrote {out}")
        return


    sample_holistic()            # one holistic intent per episode (mc_rollout.py:160-169)
    while step < n_ctrl:

        with torch.inference_mode():
            x0 = plan()
            a_n = x0[:, H:H + K, PROPRIO_DIM:]
            actions = a_n * std[PROPRIO_DIM:] + mean[PROPRIO_DIM:]

        for k in range(K):
            if step >= n_ctrl:
                break
            a = actions[:, k]
            prop_in = read_prop()           # the state this action is applied in -- the recorder's pairing
            for _ in range(hold):
                if step >= n_ctrl:
                    break
                _, _, _, dones, _ = env.step(a)
                # Only a fall WITHIN the clip counts. Short-clip envs keep stepping to the batch horizon
                # because isaacgym steps them together; what they do after their own motion has ended is
                # not part of what the caption asked for.
                newly = dones.bool() & alive & (step < n_env)
                fall_step[newly] = step
                alive &= ~newly
                # legged_robot.py:2352 keeps this already shaped [num_envs, num_bodies, 3]; the raw
                # 13-wide state tensor is _rigid_body_state and is not reshaped per env.
                body_pos.append(env._rigid_body_pos.detach().clone().cpu())
                step += 1
            push_hist(prop_in, actions[:, k])

    fall_rate = float((~alive).float().mean())
    duration = float((fall_step.float() / n_env.float()).mean())
    print(f"\nfall rate {fall_rate:.4f}   duration completion {duration:.4f}   "
          f"{time.time() - t0:.0f}s for {n_ctrl} control steps")

    bp = torch.stack(body_pos, 1).numpy()      # [B, T, n_bodies, 3]
    np.savez(out.with_suffix(".bodypos.npz"), body_pos=bp.astype(np.float16),
             fall_step=fall_step.cpu().numpy(), horizon=n_env.cpu().numpy(), keys=np.array(keys))
    res = dict(policy=str(policy_path), refs=str(refs), warmup_tracker=bool(args.warmup_tracker),
               n_envs=B, episode=args.episode, episode_s=(args.episode_s if args.episode == "fixed" else None),
               clip_s_mean=float(dur_s.mean()), hist_init=args.hist_init, weights=args.weights, ckpt_step=ck.get("step"),
               hold_action=args.hold_action, holi_per_plan=bool(args.holi_per_plan),
               cfg_action=(args.cfg_scale if args.cfg_action < 0 else args.cfg_action),
               hold=hold, n_future_proprio=n_fut_prop, ablate=ablate,
               gen_hz=gen_hz, H=H, F=F, K=K,
               num_steps=args.num_steps, cfg_scale=args.cfg_scale, seed=args.seed,
               fall_rate=fall_rate, duration_completion=duration, n_fell=int((~alive).sum()),
               note=("Single rollout, single computation (CLAUDE.md §4). Physical metrics only; the "
                     "semantic metrics are computed separately from the saved body positions through "
                     "hml_phys/g1_to_smpl.py -> the Guo evaluator."))
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out} and {out.with_suffix('.bodypos.npz').name}")


if __name__ == "__main__":
    main()
