"""Closed-loop rollouts of the physics-policy DiT in the UniPhys Isaac Gym environment (docs/07 §5).

Every K executed frames: history = the dense last H raw physics frames plus a sparse selection reaching back
over L_max frames (SCRIPT eq. 6) -> window tokens (canonical frame = newest history frame) -> normalise ->
Euler sampling of F future frames with
x0-space CFG -> de-normalise -> execute the action channels of the first K future frames (raw 69-d action; the env
applies pd_target = offset + scale * a). Start: fixed standing pose (phc.env.stateInit=Start), history filled with
the start frame + hold action, 2 warm-up hold steps (like UniPhys). Bookkeeping / recording identical to
hml_phys.uniphys_rollout.run_rollouts.
"""
import os, time, json
import numpy as np, torch, joblib
from tqdm import tqdm
from hml_phys import tokens as tk, flow as fl
from hml_phys.mc_model import PhysPolicyDiT
from hml_phys.dataset import TokenStats, TextCache, norm_caption
from hml_phys.tokens import sample_sparse_history
from hml_phys.text_clip import ClipText


def load_policy(ckpt_path, device="cuda", weights="ema"):
    ck = torch.load(ckpt_path, map_location="cpu")
    a = ck["args"]
    rd = [int(x) for x in a["root_depth"].split(",")]; bd = [int(x) for x in a["body_depth"].split(",")]
    st = ck["stats"]
    model = PhysPolicyDiT(hidden_dim=a["hidden"], num_heads=a["heads"], root_depth_double=rd[0], root_depth_single=rd[1],
                          body_depth_double=bd[0], body_depth_single=bd[1], text_token_dim=768, text_pooled_dim=768, max_text_tokens=50,
                          local_root=bool(a.get("local_root", 0)),
                          root_stats=(st["root_mean"], st["root_std"]),
                          local_root_stats=(st.get("local_root_mean", np.zeros(4, np.float32)), st.get("local_root_std", np.ones(4, np.float32))))
    sd = ck["ema"] if weights == "ema" else ck["model"]
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in sd.items()}); model.to(device).eval()
    stats = TokenStats.__new__(TokenStats)
    stats.root_mean, stats.root_std, stats.body_mean, stats.body_std = st["root_mean"], st["root_std"], st["body_mean"], st["body_std"]
    stats.local_root_mean = st.get("local_root_mean", np.zeros(4, np.float32))
    stats.local_root_std = st.get("local_root_std", np.ones(4, np.float32))
    return model, stats, ck["env_constants"], a, ck["step"]


class HistoryBuffer:
    """Ring of the last `L` raw physics frames (L = L_max, the full span the sparse history may reach over).
    `n` counts how many of them are real; before that the buffer is filled with the reset frame."""
    def __init__(self, B, L):
        self.B, self.L = B, L; self.bp = None; self.n = 0
    def reset(self, bp, ds, rs, ac):
        self.bp = np.repeat(bp[:, None], self.L, 1); self.ds = np.repeat(ds[:, None], self.L, 1)
        self.rs = np.repeat(rs[:, None], self.L, 1); self.ac = np.repeat(ac[:, None], self.L, 1)
        self.n = 1
    def push(self, bp, ds, rs, ac):
        for name, v in (("bp", bp), ("ds", ds), ("rs", rs), ("ac", ac)):
            buf = getattr(self, name); buf[:, :-1] = buf[:, 1:]; buf[:, -1] = v
        self.n = min(self.n + 1, self.L)


def run_rollouts_mc(experiment, items, out_path, ckpt, save_every=1, seed=0, K=4, num_steps=32, cfg_scale=3.5,
                    weights="ema", h_sparse_override=None, alpha_override=None, l_max_override=None):
    exp = experiment; player = exp.player; env = player.env; task = env.task
    dev = torch.device("cuda")
    # the imitation env would otherwise terminate episodes that drift >0.25 m from its hidden reference motion
    task._termination_distances[:] = 1e6
    task.termination_mode = "sampling"
    model, stats, envc, margs, ckpt_step = load_policy(ckpt, weights=weights)
    assert np.allclose(np.asarray(envc["pd_offset"]), task._pd_action_offset.cpu().numpy(), atol=1e-5) and \
        np.allclose(np.asarray(envc["pd_scale"]), task._pd_action_scale.cpu().numpy(), atol=1e-5), "checkpoint PD offset/scale differ from the live env"
    if "Start" not in str(task._state_init):
        print(f"WARNING: env state init is {task._state_init}, protocol expects StateInit.Start (fixed standing pose)")
    H, F = int(margs["H"]), int(margs["F"]); T = H + F
    whole = bool(margs.get("whole_sequence", False))  # v2: future = remaining frames of the episode (capped at F)
    is_v3 = "H_sparse" in margs      # checkpoints trained before v3 used absolute positions and no sparse history
    H_sparse = int(margs.get("H_sparse", 0)); L_max = int(margs.get("L_max", H)); alpha = float(margs.get("alpha", 3.0))
    if h_sparse_override is not None:   # test-time history knob (docs/07 §15 改动 3)
        H_sparse = int(h_sparse_override)
    if alpha_override is not None:
        alpha = float(alpha_override)
    if l_max_override is not None:
        L_max = int(l_max_override)
    L_max = max(L_max, H + H_sparse)
    rng_np = np.random.RandomState(seed)
    clip_enc = ClipText()
    empty_tok, empty_pool, empty_len = clip_enc.encode([""])
    num_envs = task.num_envs; max_len = task.max_episode_length
    pd_off = torch.from_numpy(np.asarray(envc["pd_offset"])).to(dev).float(); pd_sc = torch.from_numpy(np.asarray(envc["pd_scale"])).to(dev).float()
    gen = torch.Generator(device=dev); gen.manual_seed(seed); np.random.seed(seed)
    print(f"MC policy rollout: ckpt step {ckpt_step} ({weights}), {len(items)} items, {num_envs} envs, "
          f"window [{H_sparse} sparse | {H} dense | {F} future] over L_max {L_max}, alpha {alpha}, K={K}, "
          f"Euler {num_steps}, cfg {cfg_scale}, local_root={model.local_root}, "
          f"positions={'signed (v3)' if is_v3 else 'absolute (v1/v2 compat)'}, stateInit={task._state_init}")
    episodes, neutral_state = [], None
    if os.path.exists(out_path):
        episodes = joblib.load(out_path)["episodes"]
        done_keys = {(e["key"], e["caption"], e.get("rep", 0)) for e in episodes}
        items = [it for it in items if (it["key"], it["caption"], it.get("rep", 0)) not in done_keys]
        print(f"resuming: {len(episodes)} done, {len(items)} remaining")
    n_batches = (len(items) + num_envs - 1) // num_envs; t_start = time.time()
    pbar = tqdm(total=len(items), desc="MC HumanML3D rollouts")

    def read_state():
        rs = task._root_states[task._humanoid_actor_ids].cpu().numpy().copy()
        ds = task._dof_state.reshape(num_envs, -1, 2).cpu().numpy().copy()
        bp = task._rigid_body_pos.cpu().numpy().copy()
        return bp, ds, rs

    for b in range(n_batches):
        batch = items[b * num_envs:(b + 1) * num_envs]; cur = len(batch)
        if cur < num_envs:
            batch = batch + [items[0]] * (num_envs - cur)
        captions = [norm_caption(it["caption"]) for it in batch]
        targets = np.minimum(np.array([int(it["target_frames"]) for it in batch]), max_len - 2)
        gt_len = np.array([float(it.get("gt_length", 0)) for it in batch]); total_len = np.where(gt_len > 0, gt_len / 20.0, targets / 30.0)
        tok, pool, tl = clip_enc.encode(captions)
        text = (torch.from_numpy(tok).to(dev), torch.from_numpy(pool).to(dev), torch.from_numpy(tl).to(dev))
        text_u = (torch.from_numpy(np.repeat(empty_tok, num_envs, 0)).to(dev), torch.from_numpy(np.repeat(empty_pool, num_envs, 0)).to(dev), torch.from_numpy(np.repeat(empty_len, num_envs, 0)).to(dev))
        obs = player.env_reset()
        bp, ds, rs = read_state()
        if neutral_state is None:
            neutral_state = dict(root_state=rs[0], dof_state=ds[0], body_pos=bp[0], state_init=str(task._state_init))
        hold = ((torch.from_numpy(ds[:, :, 0]).to(dev) - pd_off) / pd_sc).float()
        hist = HistoryBuffer(num_envs, L_max); hist.reset(bp, ds, rs, hold.cpu().numpy())
        rec_bp, episode_length = [], np.zeros(num_envs, dtype=int)
        is_done = np.zeros(num_envs, bool); is_fallen = np.zeros(num_envs, bool); fall_step = np.full(num_envs, -1)
        # 2 warm-up hold steps: fill the history with real physics frames; not recorded, not counted (UniPhys convention)
        for _ in range(2):
            obs, r, done, info = player.env_step(env, hold)
            bp, ds, rs = read_state(); hist.push(bp, ds, rs, hold.cpu().numpy())
        while not np.all(is_done | (episode_length >= targets)):
            # ---- plan. Window = [sparse distant history | dense recent history | future], exactly as in
            # training: tokens are computed on the contiguous buffer span and the chosen rows are gathered,
            # so every row keeps its true instantaneous velocities and its signed frame offset.
            if whole:
                fut = np.clip(targets - episode_length, K, F)
            else:
                fut = np.full(num_envs, F)
            Fb = int(fut.max())
            n_real = int(hist.n)                                   # real frames in the buffer
            l_distant = int(min(L_max - H, max(0, n_real - H)))    # frames available before the dense history
            dense0 = hist.L - H                                    # dense history starts here in the ring
            span0 = dense0 - l_distant
            raw_bp = np.concatenate([hist.bp[:, span0:], np.repeat(hist.bp[:, -1:], Fb, 1)], 1)
            raw_ds = np.concatenate([hist.ds[:, span0:], np.repeat(hist.ds[:, -1:], Fb, 1)], 1)
            raw_rs = np.concatenate([hist.rs[:, span0:], np.repeat(hist.rs[:, -1:], Fb, 1)], 1)
            raw_ac = np.concatenate([hist.ac[:, span0:], np.repeat(hist.ac[:, -1:], Fb, 1)], 1)
            sparse = sample_sparse_history(l_distant, H_sparse, alpha, rng_np)   # indices into [span0, dense0)
            rows = np.concatenate([sparse, np.arange(l_distant, l_distant + H + Fb)]).astype(np.int64)
            fidx = rows - (l_distant + H)                          # 0 = first generated frame
            n_hist = len(sparse) + H
            origin = int(rows[n_hist - 1])
            root_full, body_full = tk.window_tokens_batch(raw_bp, raw_ds, raw_rs, raw_ac, origin=origin)
            root, body = root_full[:, rows], body_full[:, rows]
            root, body = stats.norm(root, body)
            Tb = len(rows)
            xr = torch.from_numpy(root).to(dev); xb = torch.from_numpy(body).to(dev)
            xr[:, n_hist:] = 0; xb[:, n_hist:] = 0
            mask = torch.zeros(num_envs, Tb, device=dev); mask[:, :n_hist] = 1.0
            valid = (torch.arange(Tb, device=dev)[None] < (n_hist + torch.from_numpy(fut).to(dev))[:, None]).float()
            if is_v3:
                frame_index = torch.from_numpy(fidx).to(dev).long()[None].expand(num_envs, Tb).contiguous()
            else:   # v1/v2 were trained with pos = arange(T); keep their exact text<->motion offsets
                frame_index = torch.arange(Tb, device=dev).long()[None].expand(num_envs, Tb).contiguous()
            progress = np.clip(episode_length / np.maximum(1.0, total_len * 30.0), 0, 1)
            scal = torch.from_numpy(np.stack([progress, total_len / 10.0], -1)).float().to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x0r, x0b = fl.euler_sample(model, xr, xb, mask, text, text_u, scal, num_steps=num_steps,
                                           cfg_scale=cfg_scale, generator=gen, valid=valid, frame_index=frame_index)
            _, body_un = stats.denorm(x0r.float().cpu().numpy(), x0b.float().cpu().numpy())
            actions = body_un[:, n_hist:n_hist + K, 351:420]  # [B,K,69]
            # ---- execute K actions
            for k in range(K):
                a = torch.from_numpy(actions[:, k]).to(dev).float()
                obs, r, done, info = player.env_step(env, a)
                bp, ds, rs = read_state(); hist.push(bp, ds, rs, actions[:, k])
                rec_bp.append(info["body_pos"].cpu().numpy() if torch.is_tensor(info["body_pos"]) else info["body_pos"])
                d = done.cpu().numpy().astype(bool); term = task._terminate_buf.cpu().numpy().astype(bool)
                episode_length[~is_done] += 1
                newly = d & ~is_done; fall_step[newly] = episode_length[newly]; is_fallen |= term & newly; is_done |= d
                if np.all(is_done | (episode_length >= targets)):
                    break
        body_pos = np.stack(rec_bp)  # [T,B,24,3]
        for j in range(cur):
            it = batch[j]; fell = bool(is_fallen[j]) and int(fall_step[j]) <= int(targets[j])
            L = int(min(targets[j], fall_step[j] if fell else targets[j], body_pos.shape[0]))
            episodes.append(dict(key=it["key"], caption=it["caption"], rep=int(it.get("rep", 0)), gt_length=int(it.get("gt_length", -1)),
                                 target_frames=int(targets[j]), fell=fell, fall_step=int(fall_step[j]) if fell else -1, body_pos=body_pos[:L, j].astype(np.float32)))
        pbar.update(cur)
        if (b + 1) % save_every == 0 or b == n_batches - 1:
            joblib.dump(dict(episodes=episodes, meta=dict(num_envs=num_envs, K=K, num_steps=num_steps, cfg=cfg_scale, ckpt=ckpt, ckpt_step=ckpt_step,
                        weights=weights, neutral_state=neutral_state, max_episode_length=int(max_len), elapsed_s=time.time() - t_start)), out_path)
    pbar.close()
    n_fell = sum(e["fell"] for e in episodes)
    print(f"done: {len(episodes)} episodes, fell: {n_fell} ({100*n_fell/max(1,len(episodes)):.1f}%), {time.time()-t_start:.0f}s -> {out_path}")
