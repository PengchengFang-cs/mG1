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
    ap.add_argument("--refs", default="data/g1_e2e/refs_test.pkl",
                    help="only to construct the env; its observations are not used")
    ap.add_argument("--text-cache", default="data/g1_e2e/text_clipL14")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode-s", type=float, default=20.0, help="episode length, as the protocol fixes it")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=10, help="flow sampling steps")
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--K", type=int, default=0, help="action rows executed per plan; 0 = all F")
    ap.add_argument("--warmup-tracker", action="store_true",
                    help="let FRoM-W1's tracker drive the first H generation frames, so the history "
                         "buffer starts from real states consistent with the robot's actual pose instead "
                         "of zeros. A diagnostic: it separates a cold-start distribution shift from the "
                         "policy being unable to hold itself up at all.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
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
    hold = CONTROL_HZ // gen_hz
    K = args.K or F
    print(f"policy: gen {gen_hz} Hz, H {H}, F {F}, executing K={K} rows, hold {hold} control steps each")

    # The env must not terminate on a reference it is not following; contact and tilt termination stay on.
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg_h = hydra.compose(config_name="config_eval", overrides=[
            f"motion.motion_file={refs}", f"num_envs={args.num_envs}", f"sim_device={args.device}",
            "headless=True", "use_wandb=False",
            "asset.terminate_by_ref_motion_distance=False",
            "asset.terminate_by_1time_motion=False",
            f"env.episode_length_s={args.episode_s}",
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
    if args.warmup_tracker:
        runner, _ = task_registry.make_alg_runner(env=env, name=cfg.task, args=cfg, train_cfg=cfg.train)
        tracker = runner.get_inference_policy(device=dev)
        env.begin_seq_motion_samples()      # the tracker needs the reference to advance

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

    model = IntentPolicy(ck["policy_kw"], ta["intent_dim"], ta["intent_heads"], ta["intent_depth"],
                         text_token_dim=768).to(dev)
    model.load_state_dict(ck["model"])
    model.eval().requires_grad_(False)

    tc = text_cache
    tok = joblib.load(tc / "tokens.pkl")
    lens = joblib.load(tc / "lengths.pkl")
    keys = [str(k) for k in env._motion_lib._motion_data_keys[: args.num_envs]]
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
    n_ctrl = int(round(args.episode_s * CONTROL_HZ))
    B = env.num_envs

    hist = torch.zeros(B, H, TOKEN_DIM, device=dev)
    body_pos = []
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    fall_step = torch.full((B,), n_ctrl, dtype=torch.long, device=dev)

    def push_hist(action_raw):
        """One row per executed action: (the state reached, the action that was applied), both real.

        A training token is (proprio_i, action_i) with both channels filled. The first version of this
        loop rolled twice per action and wrote the two halves into separate rows, so every other history
        row carried zeroed proprio and the rest a stale action slot -- half the conditioning was the
        dataset mean. Proprio is read AFTER the step, pairing s_{t+1} with the a_{t+1} that produced it.
        """
        nonlocal hist
        prop = torch.cat([env.base_lin_vel, env.base_ang_vel, env.projected_gravity,
                          env.dof_pos, env.dof_vel], dim=-1)
        hist = torch.roll(hist, -1, dims=1)
        hist[:, -1, :PROPRIO_DIM] = (prop - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM]
        hist[:, -1, PROPRIO_DIM:] = (action_raw - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:]


    env.cfg.env.test = True
    obs, _ = env.reset()
    t0 = time.time()
    step = 0

    shadow_se, shadow_var, shadow_n = 0.0, 0.0, 0
    if tracker is not None:
        # Warm-up: the tracker tracks the reference for H generation frames, filling the history from the
        # SAME buffers the closed loop reads, so the handover state is real and self-consistent.
        n_warm = H * hold
        for w in range(n_warm):
            with torch.inference_mode():
                a_tr = tracker(obs.detach())
            obs, _, _, dones, _ = env.step(a_tr.detach())
            newly = dones.bool() & alive
            fall_step[newly] = step
            alive &= ~newly
            body_pos.append(env._rigid_body_pos.detach().clone().cpu())
            step += 1
            if w % hold == hold - 1:
                push_hist(a_tr.detach())
        print(f"warm-up: tracker drove {n_warm} control steps ({n_warm / CONTROL_HZ:.2f} s); "
              f"{int((~alive).sum())}/{B} already down, handing over to the policy", flush=True)

    def plan():
        """One planning pass: HIP -> IIP -> policy, returning the x0 window. Shared by the closed
        loop and the shadow check so the two cannot drift apart."""
        x_obs = torch.zeros(B, H + F, TOKEN_DIM, device=dev)
        x_obs[:, :H] = hist
        obs_m = observed_mask(B, H, H + F, dev)
        g = generated_elements(obs_m, None, act_mask)
        _, mu_hist, _ = vae.encode(hist[:, :, :PROPRIO_DIM])
        lat_hist = (mu_hist - lat_mean) / lat_std
        scal = torch.stack([torch.full((B,), step / n_ctrl, device=dev),
                            torch.full((B,), args.episode_s / 10.0, device=dev)], -1)

        # Classifier-free guidance needs the SAME unconditional state training used: CLIP(''), and
        # for the policy no intent tokens at all (intent_flow.sample_actions). Passing the
        # conditional memory as both branches would make the guidance term identically zero.
        I_H = sample_latent(model.hip, B, mem_c, mv_c, mem_u, mv_u, num_steps=args.num_steps,
                            cfg_scale=args.cfg_scale, generator=gen, device=dev)
        hH_c = intent_hidden(model.hip, I_H, s_read, gen, mem=mem_c, mem_valid=mv_c)
        hH_u = intent_hidden(model.hip, I_H, s_read, gen, mem=mem_u, mem_valid=mv_u)
        I_I = sample_latent(model.iip, B, mem_c, mv_c, mem_u, mv_u, num_steps=args.num_steps,
                            cfg_scale=args.cfg_scale, generator=gen, prefix=lat_hist, scalars=scal,
                            extra=hH_c, extra_u=hH_u, device=dev)
        hI_c = intent_hidden(model.iip, I_I, s_read, gen, mem=mem_c, mem_valid=mv_c,
                             prefix_latent=lat_hist, scalars=scal, mem_extra=hH_c)
        toks, _ = model.intent_tokens(hH_c, hI_c, torch.ones(B, dtype=torch.bool, device=dev))
        x0 = sample_actions(model.policy, x_obs, obs_m, g,
                            (text, pooled, tlen), (text_u, pooled_u, tlen_u),
                            scal, toks, num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                            generator=gen)

        return x0

    if args.shadow:
        assert tracker is not None, "--shadow needs --warmup-tracker: the tracker has to drive"
        while step < n_ctrl:
            with torch.inference_mode():
                pred = plan()[:, H, PROPRIO_DIM:]
                a_tr = tracker(obs.detach())
                true_n = (a_tr - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:]
                shadow_se += float(((pred - true_n) ** 2).sum())
                shadow_var += float((true_n ** 2).sum())
                shadow_n += true_n.numel()
            for _ in range(hold):
                if step >= n_ctrl:
                    break
                obs, _, _, dones, _ = env.step(a_tr.detach())
                step += 1
            push_hist(a_tr.detach())
        nmse = shadow_se / max(shadow_var, 1e-9)
        print(f"\nshadow NMSE {nmse:.4f} over {shadow_n} action elements "
              f"({shadow_n // ACTION_DIM} planning steps x {ACTION_DIM} joints)")
        out.write_text(json.dumps(dict(mode="shadow", nmse=nmse, n_elements=shadow_n,
                                       policy=str(policy_path)), indent=2))
        print(f"wrote {out}")
        return


    while step < n_ctrl:

        with torch.inference_mode():
            x0 = plan()
            a_n = x0[:, H:H + K, PROPRIO_DIM:]
            actions = a_n * std[PROPRIO_DIM:] + mean[PROPRIO_DIM:]

        for k in range(K):
            if step >= n_ctrl:
                break
            a = actions[:, k]
            for _ in range(hold):
                if step >= n_ctrl:
                    break
                _, _, _, dones, _ = env.step(a)
                newly = dones.bool() & alive
                fall_step[newly] = step
                alive &= ~newly
                # legged_robot.py:2352 keeps this already shaped [num_envs, num_bodies, 3]; the raw
                # 13-wide state tensor is _rigid_body_state and is not reshaped per env.
                body_pos.append(env._rigid_body_pos.detach().clone().cpu())
                step += 1
            push_hist(actions[:, k])

    fall_rate = float((~alive).float().mean())
    duration = float((fall_step.float() / n_ctrl).mean())
    print(f"\nfall rate {fall_rate:.4f}   duration completion {duration:.4f}   "
          f"{time.time() - t0:.0f}s for {n_ctrl} control steps")

    bp = torch.stack(body_pos, 1).numpy()      # [B, T, n_bodies, 3]
    np.savez(out.with_suffix(".bodypos.npz"), body_pos=bp.astype(np.float16),
             fall_step=fall_step.cpu().numpy(), keys=np.array(keys))
    res = dict(policy=str(policy_path), warmup_tracker=bool(args.warmup_tracker), n_envs=B, episode_s=args.episode_s, gen_hz=gen_hz, H=H, F=F, K=K,
               num_steps=args.num_steps, cfg_scale=args.cfg_scale, seed=args.seed,
               fall_rate=fall_rate, duration_completion=duration, n_fell=int((~alive).sum()),
               note=("Single rollout, single computation (CLAUDE.md §4). Physical metrics only; the "
                     "semantic metrics are computed separately from the saved body positions through "
                     "hml_phys/g1_to_smpl.py -> the Guo evaluator."))
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out} and {out.with_suffix('.bodypos.npz').name}")


if __name__ == "__main__":
    main()
