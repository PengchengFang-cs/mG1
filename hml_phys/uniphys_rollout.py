"""Closed-loop text-to-motion rollouts with the official UniPhys checkpoint on HumanML3D test captions.

Adapted from UniPhys `DFHumanoid.evaluate_t2m_babel` (uniphys/algorithms/diffusion_forcing/df_humanoid.py):
same policy, same PULSE decoding, same environment; differences: one rollout per (item, caption) with a
per-item target length, per-episode fall bookkeeping, incremental saving.
Episode record: key, caption, rep, body_pos [L,24,3] float32 (Isaac/MuJoCo order, z-up, 30 fps; L = target
frames, or the frames up to and including the frame at which the fall was detected), target_frames, fell,
fall_step (1-based count of executed frames when `done` was first raised).
"""
import os, json, time
import numpy as np
import torch
import joblib
from tqdm import tqdm


def run_rollouts(experiment, items, out_path, save_every=1, seed=0):
    """experiment: UniPhys exp_isaac.IsaacExperiment after create_player(); items: list of dict(key, caption, target_frames)."""
    exp = experiment
    if not exp.algo:
        exp.algo = exp._build_algo()
    algo = exp.algo
    checkpoint = torch.load(exp.ckpt_path, map_location="cpu")
    print("restoring from", exp.ckpt_path)
    algo.load_state_dict(checkpoint["state_dict"], strict=False)
    stats = np.load(os.path.join(os.path.dirname(os.path.dirname(exp.ckpt_path)), "train_data_stats.npy"), allow_pickle=True)[()]
    algo.state_mean, algo.state_std = stats["Mean"], stats["Std"]
    algo.action_mean, algo.action_std = stats["ActionMean"], stats["ActionStd"]
    if "zMean" in stats:
        algo.z_mean, algo.z_std = stats["zMean"], stats["zStd"]
    algo.diffusion_model.cuda(); algo.diffusion_model.eval()
    algo.player = exp.player; algo.env = exp.player.env
    algo.env.task._termination_distances[:] = 1e6
    algo.env.task.termination_mode = "sampling"
    algo.save_dir = os.path.dirname(os.path.dirname(exp.ckpt_path))
    algo.resume_step = checkpoint["global_step"]

    from uniphys.utils.clip_utils import load_and_freeze_clip, encode_text
    torch.manual_seed(seed); np.random.seed(seed)
    num_envs = algo.env.task.num_envs
    algo.max_episode_length = algo.env.task.max_episode_length
    algo.guidance_fn = None
    algo.guidance_params = algo.cfg.guidance_params
    algo.n_samples = 1
    algo.clip_model = load_and_freeze_clip(clip_version="ViT-B/32", device="cuda")
    algo.text_embedding = None
    algo._clear_hist_buffer()
    if algo.norm_action:
        if algo.cfg.action_keys == ["action"]:
            algo.a_mean, algo.a_std = algo.action_mean, algo.action_std
        elif algo.cfg.action_keys == ["z"]:
            algo.a_mean, algo.a_std = algo.z_mean, algo.z_std
        else:
            raise ValueError("Unsupported action_keys")
    else:
        algo.a_mean, algo.a_std = np.zeros(algo.action_dim), np.ones(algo.action_dim)
    algo.env.task.completed_episode_lengths = []
    print(f"UniPhys rollout: {len(items)} items, {num_envs} envs, max_episode_length {algo.max_episode_length}, "
          f"exec_step {algo.exec_step}, guidance {algo.guidance_params}")

    episodes = []
    neutral_state = None  # first-frame state after reset (records the evaluation start pose)
    if os.path.exists(out_path):  # resume
        episodes = joblib.load(out_path)["episodes"]
        done_keys = {(e["key"], e["caption"], e.get("rep", 0)) for e in episodes}
        items = [it for it in items if (it["key"], it["caption"], it.get("rep", 0)) not in done_keys]
        print(f"resuming: {len(episodes)} episodes done, {len(items)} remaining")
    n_batches = (len(items) + num_envs - 1) // num_envs
    t_start = time.time()
    pbar = tqdm(total=len(items), desc="UniPhys HumanML3D rollouts")
    for b in range(n_batches):
        batch = items[b * num_envs:(b + 1) * num_envs]
        cur = len(batch)
        if cur < num_envs:
            batch = batch + [items[0]] * (num_envs - cur)
        captions = [it["caption"] for it in batch]
        targets = np.array([int(it["target_frames"]) for it in batch])
        # the 2 warm-up steps count towards progress_buf, so the env times out at executed step
        # max_episode_length-2; a target beyond that can never be reached and would look like a fall
        targets = np.minimum(targets, algo.max_episode_length - 2)
        algo.text_embedding = encode_text(algo.clip_model, captions)
        algo._clear_hist_buffer()
        algo.pred_pos, algo.pred_dof_pos, algo.root_state, algo.dof_state, algo.action = [], [], [], [], []
        episode_length = np.zeros(num_envs, dtype=int)
        is_done = np.zeros(num_envs, dtype=bool)
        fall_step = np.full(num_envs, -1, dtype=int)
        is_fallen = np.zeros(num_envs, dtype=bool)  # true falls (env terminate buffer), not timeouts
        obs_dict = algo.player.env_reset()
        if neutral_state is None:
            task = algo.env.task
            neutral_state = dict(root_state=task._root_states[task._humanoid_actor_ids][0].cpu().numpy().copy(),
                                 dof_state=task._dof_state.reshape(num_envs, -1, 2)[0].cpu().numpy().copy(),
                                 body_pos=task._rigid_body_pos[0].cpu().numpy().copy(),
                                 state_init=str(task._state_init),
                                 pd_action_offset=task._pd_action_offset.cpu().numpy().copy(),
                                 pd_action_scale=task._pd_action_scale.cpu().numpy().copy(),
                                 dof_names=list(getattr(task, "dof_names", [])) if hasattr(task, "dof_names") else None)
        done_indices = []
        t = 0
        with torch.no_grad():
            while t < algo.max_episode_length and not np.all(is_done | (episode_length >= targets)):
                while len(algo.root_state_buffer.buffer) < 2:  # 2 warm-up steps with the mean latent (official)
                    obs_dict = algo.player.env_reset(done_indices)
                    if algo.cfg.action_keys == ["z"]:
                        zero_z = torch.zeros((num_envs, algo.action_dim)).cuda()
                        zero_z = zero_z * torch.from_numpy(algo.a_std).cuda() + torch.from_numpy(algo.a_mean).cuda()
                        zero_action, _ = algo.player.dec_action(zero_z, obs_dict)
                    else:
                        zero_z = zero_action = torch.zeros((num_envs, algo.action_dim)).cuda()
                    obs_dict, r, done, info = algo.player.env_step(algo.env, zero_action)
                    algo.root_state_buffer.add(algo.env.task._root_states[algo.env.task._humanoid_actor_ids].clone())
                    algo.dof_state_buffer.add(algo.env.task._dof_state.reshape(num_envs, -1, 2).clone())
                    algo.joint_pos_buffer.add(algo.env.task._rigid_body_pos.clone())
                    algo.action_buffer.add(zero_z.clone())
                else:
                    action_pred, state_pred = algo.policy()
                    action_exec = action_pred[algo.H:].permute(1, 0, 2).float()
                    for i in range(algo.exec_step):
                        a = action_exec[:, i]
                        if algo.cfg.action_keys == ["z"]:
                            a_exec, _ = algo.player.dec_action(a, {"obs": obs_dict})
                        else:
                            a_exec = a
                        obs_dict, r, done, info = algo.player.env_step(algo.env, a_exec)
                        done = algo.post_step(info, done.clone(), save_dir="record")  # save_dir!=None -> record body_pos
                        algo.root_state_buffer.add(algo.env.task._root_states[algo.env.task._humanoid_actor_ids].clone())
                        algo.dof_state_buffer.add(algo.env.task._dof_state.reshape(num_envs, -1, 2).clone())
                        algo.joint_pos_buffer.add(algo.env.task._rigid_body_pos.clone())
                        algo.action_buffer.add(a.clone())
                        t += 1
                        episode_length[~is_done] += 1
                        d = done.cpu().numpy().astype(bool)
                        term = algo.env.task._terminate_buf.cpu().numpy().astype(bool)  # fall only (no timeout)
                        newly = d & ~is_done
                        fall_step[newly] = episode_length[newly]
                        is_fallen |= term & newly
                        is_done |= d
                        if np.all(is_done | (episode_length >= targets)):
                            break
        body_pos = np.stack([p.cpu().numpy() if torch.is_tensor(p) else p for p in algo.pred_pos])  # [T, B, 24, 3]
        for j in range(cur):
            it = batch[j]
            fell = bool(is_fallen[j]) and int(fall_step[j]) <= int(targets[j])
            L = int(min(targets[j], fall_step[j] if fell else targets[j], body_pos.shape[0]))
            episodes.append(dict(key=it["key"], caption=it["caption"], rep=int(it.get("rep", 0)), gt_length=int(it.get("gt_length", -1)),
                                 target_frames=int(targets[j]), fell=fell, fall_step=int(fall_step[j]) if fell else -1,
                                 body_pos=body_pos[:L, j].astype(np.float32)))
        pbar.update(cur)
        if (b + 1) % save_every == 0 or b == n_batches - 1:
            joblib.dump(dict(episodes=episodes, meta=dict(num_envs=num_envs, exec_step=int(algo.exec_step), neutral_state=neutral_state,
                        guidance=float(algo.guidance_params), max_episode_length=int(algo.max_episode_length),
                        elapsed_s=time.time() - t_start)), out_path)
    pbar.close()
    n_fell = sum(e["fell"] for e in episodes)
    print(f"done: {len(episodes)} episodes, fell before target: {n_fell} ({100*n_fell/max(1,len(episodes)):.1f}%), "
          f"{time.time()-t_start:.0f}s -> {out_path}")
