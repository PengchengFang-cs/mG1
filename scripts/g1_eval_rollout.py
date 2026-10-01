"""Closed-loop evaluation of a G1 text policy in the TextOp Isaac Lab environment (docs/04 line A).

Protocol is ADAPT's Table 1, kept identical to `scripts/g1_physical_protocol.py` so the physical numbers stay
comparable: N rollouts of 20 s (1000 steps @50 Hz) with the prompt switched every 5-10 s, fall = any body other
than the ankle/wrist links below `--contact_z` or the root tipped past 60 degrees, plus action smoothness,
transition smoothness and foot sliding.

What is added here is the SEMANTIC side.  ADAPT scores text alignment with TMR, which we do not have; instead
every episode's link positions are recorded, and `scripts/hml_phys/g1_eval_metrics.py` maps them onto the 22
SMPL joints (`hml_phys/g1_to_smpl.py`) and scores them with our Guo evaluator.  Because the prompt changes
during an episode, the recording is sliced into SEGMENTS -- one contiguous span per active prompt -- and each
segment is one item for R-precision and FID.  A 5-10 s segment is 100-200 frames at 20 fps, which is the range
HumanML3D itself covers; a whole 20 s episode would not be.

Fallen episodes are truncated at the fall, never dropped (project CLAUDE.md §2).

`--source` selects what drives the robot:
    policy   our G1 flow policy (with or without the intent modules; read from the checkpoint)
    tracker  the pretrained TextOp tracker following its reference motion -- the CEILING for the mapping and
             the evaluator, i.e. what score physically correct, genuinely on-label motion gets through this
             exact path.  Its "prompt" per segment is the BABEL label the reference motion carries.
    hold     a do-nothing control that holds the initial pose; the floor for the same path.

Run inside the Isaac Lab container from TextOpTracker/.  CLIP is not installed there, so the text features come
from `scripts/hml_phys/g1_build_text_dict.py`, run outside beforehand.
"""
import argparse, glob, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "TextOp", "TextOpTracker", "scripts", "rsl_rl"))
sys.path.insert(0, ROOT)
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-ProjGravObs-MNMLP-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rollouts", type=int, default=2048)
parser.add_argument("--steps", type=int, default=1000, help="20 s at 50 fps, ADAPT Table 1")
parser.add_argument("--source", default="policy", choices=["policy", "tracker", "hold"])
parser.add_argument("--ckpt", default="", help="our policy checkpoint (required for --source policy)")
parser.add_argument("--weights", default="ema", choices=["ema", "model"])
parser.add_argument("--text_npz", default=os.path.join(ROOT, "data/g1_rollouts/g1_eval_text_clip.npz"))
parser.add_argument("--prompt_file", default=os.path.join(ROOT, "data/g1_eval_prompts.txt"))
parser.add_argument("--stats", default=os.path.join(ROOT, "data/g1_rollouts/g1_token_stats.npz"))
parser.add_argument("--motion_glob", default="artifacts/val_all/*/motion.npz")
parser.add_argument("--motion_list", default="",
                    help="a whitelist of motion names (one per line, from g1_prompt_pool.py). ADAPT's pool "
                         "covers locomotion / exercises / upper-body gestures only, and a contact-based fall "
                         "criterion is meaningless on sitting or crawling, where torso contact is correct.")
parser.add_argument("--resume_path", default="logs/rsl_rl/Pretrained/checkpoints/model_75000.pt",
                    help="the pretrained TextOp tracker, for --source tracker and the warm-up")
parser.add_argument("--warmup_source", default="tracker", choices=["tracker", "hold"],
                    help="what fills the history buffer before the policy takes over")
parser.add_argument("--meta_pkl", default="", help="BABEL frame annotations, for --source tracker segments")
parser.add_argument("--num_steps", type=int, default=10, help="Euler steps for the action flow")
parser.add_argument("--cfg", type=float, default=2.5, help="classifier-free guidance scale")
parser.add_argument("--K", type=int, default=3, help="generated frames executed before replanning")
parser.add_argument("--s_read", type=float, default=-1.0, help="intent read level; <0 uses the checkpoint's cond_aug_test")
parser.add_argument("--act_clip", type=float, default=5.0, help="clip the generated action to +-this many training sigmas")
parser.add_argument("--switch_lo", type=float, default=5.0); parser.add_argument("--switch_hi", type=float, default=10.0)
parser.add_argument("--shadow_policy", default="",
                    help="harness check: ride a policy along a --source tracker rollout, predicting the action "
                         "at each state without executing it, and report how much of the tracker's action it "
                         "explains. The state never leaves the training distribution, so a bad number here is "
                         "a bug in the history/observation plumbing, not compounding error.")
parser.add_argument("--shadow_every", type=int, default=10)
parser.add_argument("--shadow_bug", default="none", choices=["none", "pairing", "rowh", "both"],
                    help="deliberately reintroduce one of the two harness bugs, for the SHADOW policy only, "
                         "to price what each one cost: 'pairing' pushes (state AFTER the step, action), "
                         "'rowh' hides the current state from the first generated row. Diagnostic only.")
parser.add_argument("--fall_mode", default="contact", choices=["contact", "height"],
                    help="ADAPT terminates on illegal torso contact (paper Appendix C), which is what "
                         "'contact' reproduces. 'height' is our earlier proxy, kept only to re-measure the gap.")
parser.add_argument("--fall_bodies", default="pelvis,waist_yaw_link,waist_roll_link,torso_link",
                    help="bodies whose ground contact counts as a fall, for --fall_mode contact")
parser.add_argument("--fall_force", type=float, default=1.0,
                    help="contact force (N) over the sensor's history that counts as contact")
parser.add_argument("--contact_z", type=float, default=0.06, help="only used by --fall_mode height")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym, torch, numpy as np, joblib
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import textop_tracker.tasks  # noqa

from hml_phys.g1_data import ACTION_DIM, L_INTENT, PROPRIO_DIM, TOKEN_DIM

FPS = 50


# --------------------------------------------------------------------------------------- text features
class TextBank:
    """The CLIP features precomputed outside the container; index 0 is always the empty caption."""

    def __init__(self, path, device):
        z = np.load(path, allow_pickle=True)
        self.texts = [str(t) for t in z["texts"]]
        self.index = {t: i for i, t in enumerate(self.texts)}
        assert self.texts[0] == "", "index 0 must be the empty caption (the unconditional branch)"
        self.tok = torch.from_numpy(z["tokens"].astype(np.float32)).to(device)
        self.pool = torch.from_numpy(z["pooled"].astype(np.float32)).to(device)
        self.len = torch.from_numpy(z["length"].astype(np.int64)).to(device)

    def gather(self, idx):
        return self.tok[idx], self.pool[idx], self.len[idx]


# --------------------------------------------------------------------------------------- our policy
class G1Planner:
    """Ring buffer of the last H (proprio, action) rows -> a plan of F action frames, K of them executed."""

    def __init__(self, ckpt, device, stats_path, num_steps, cfg, K, act_clip, s_read, weights="ema"):
        from hml_phys.g1_model import G1FlowPolicy, G1IntentPolicy
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = ck["args"]
        self.H, self.F, self.K = int(a["H"]), int(a["F"]), int(K)
        assert self.K <= self.F, f"cannot execute {self.K} of {self.F} generated frames"
        self.intent = bool(a["intent"])
        sd = ck[weights] if weights in ck else ck["model"]
        sd = {k: v.float() for k, v in sd.items()}
        if self.intent:
            self.model = G1IntentPolicy(ck["policy_kw"], os.path.join(ROOT, a["vae"]), a["latent_stats"],
                                        intent_dim=a["intent_dim"], intent_heads=a["intent_heads"],
                                        intent_depth=a["intent_depth"], intent_mlp=a["intent_mlp"], device=device)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            assert not unexpected, f"unexpected keys: {unexpected[:5]}"
            assert all(k.startswith("vae.") for k in missing), f"missing non-VAE keys: {[k for k in missing if not k.startswith('vae.')][:5]}"
            self.net = self.model.policy
        else:
            self.model = G1FlowPolicy(**ck["policy_kw"]).to(device)
            self.model.load_state_dict(sd, strict=True)
            self.net = self.model
        self.model.eval()
        self.s_read = float(a.get("cond_aug_test", 0.75)) if s_read < 0 else float(s_read)
        z = np.load(stats_path)
        self.mean = torch.from_numpy(z["mean"].astype(np.float32)).to(device)
        self.std = torch.from_numpy(z["std"].astype(np.float32)).to(device)
        self.obs_future_state = a.get("obs_future_state", "all")
        self.num_steps, self.cfg, self.act_clip = int(num_steps), float(cfg), float(act_clip)
        self.device, self.step = device, int(ck["step"])
        self.L = max(self.H, L_INTENT)                      # the intent history needs 28 rows, the policy 27
        print(f"[eval] policy step {self.step} intent={self.intent} H={self.H} F={self.F} K={self.K} "
              f"s_read={self.s_read} obs_future_state={self.obs_future_state}", flush=True)

    def reset(self, N, proprio0, action0):
        """Fill the buffer by replicating the first real frame; the warm-up steps then overwrite it."""
        self.buf = torch.zeros(N, self.L, TOKEN_DIM, device=self.device)
        self.buf[:, :, :PROPRIO_DIM] = proprio0[:, None]
        self.buf[:, :, PROPRIO_DIM:] = action0[:, None]
        self.hH = self.hH_u = None

    def push(self, proprio, action):
        self.buf = torch.roll(self.buf, -1, dims=1)
        self.buf[:, -1, :PROPRIO_DIM] = proprio
        self.buf[:, -1, PROPRIO_DIM:] = action

    def norm(self, x):
        return (x - self.mean) / self.std

    def _text(self, bank, idx, N):
        t_c = bank.gather(idx)
        t_u = bank.gather(torch.zeros(N, dtype=torch.long, device=self.device))
        return t_c, t_u

    @torch.no_grad()
    def new_prompt(self, bank, idx, sw=None):
        """HIP gives one holistic intent per prompt, from the text alone (MIND §4.3), so it is resampled
        exactly when the prompt changes -- the closed-loop analogue of 'once per episode'."""
        if not self.intent:
            return
        from hml_phys import intent_flow as ifl
        N = idx.shape[0]
        (t_c, t_u) = self._text(bank, idx, N)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mem_c, mv_c = self.model.adapter(t_c[0].float(), t_c[2])
            mem_u, mv_u = self.model.adapter(t_u[0].float(), t_u[2])
            I_H = ifl.sample_latent(self.model.hip, N, mem_c, mv_c, mem_u, mv_u, num_steps=self.num_steps,
                                    cfg_scale=self.cfg, device=self.device)
            # the guided branch of IIP needs BOTH read-outs of the same latent, exactly as on the SMPL side
            h_c = ifl.intent_hidden(self.model.hip, I_H, self.s_read, mem=mem_c, mem_valid=mv_c).float()
            h_u = ifl.intent_hidden(self.model.hip, I_H, self.s_read, mem=mem_u, mem_valid=mv_u).float()
        if self.hH is None or sw is None:
            self.hH, self.hH_u = h_c, h_u
        else:
            self.hH[sw] = h_c[sw]; self.hH_u[sw] = h_u[sw]

    @torch.no_grad()
    def plan(self, bank, idx, progress, total_len, prop_now):
        """-> [N, K, 29] raw actions to execute.

        Row alignment, which has to match the dataset exactly: a row is (state s_t, action a_t applied IN it).
        The H history rows are (s_{t-H}, a_{t-H}) .. (s_{t-1}, a_{t-1}); the first generated row is time t,
        whose STATE `prop_now` the simulator has just returned and whose action is what we are asking for.
        That is what `--obs_future_state first` means in training, so the row must carry the state here too --
        leaving it zero hands the model the dataset mean in the one place it learned to rely on."""
        from hml_phys import intent_flow as ifl
        N = idx.shape[0]
        H, F = self.H, self.F
        hist = self.norm(self.buf[:, self.L - H:])                        # [N,H,96]
        x = torch.cat([hist, torch.zeros(N, F, TOKEN_DIM, device=self.device)], 1)
        if self.obs_future_state == "first" and prop_now is not None:
            x[:, H, :PROPRIO_DIM] = (prop_now - self.mean[:PROPRIO_DIM]) / self.std[:PROPRIO_DIM]
        mask = torch.zeros(N, H + F, device=self.device); mask[:, :H] = 1.0
        scal = torch.stack([progress, total_len / 10.0], -1).float()
        (t_c, t_u) = self._text(bank, idx, N)
        kw = {}
        if self.intent:
            st = (self.buf[:, self.L - L_INTENT:, :PROPRIO_DIM] - self.mean[:PROPRIO_DIM]) / self.std[:PROPRIO_DIM]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                I_h = self.model.encode_latent(st)
                mem_c, mv_c = self.model.adapter(t_c[0].float(), t_c[2])
                mem_u, mv_u = self.model.adapter(t_u[0].float(), t_u[2])
                I_I = ifl.sample_latent(self.model.iip, N, mem_c, mv_c, mem_u, mv_u, num_steps=self.num_steps,
                                        cfg_scale=self.cfg, prefix=I_h, scalars=scal,
                                        extra=self.hH, extra_u=self.hH_u, device=self.device)
                hI = ifl.intent_hidden(self.model.iip, I_I, self.s_read, mem=mem_c, mem_valid=mv_c,
                                       prefix_latent=I_h, scalars=scal, mem_extra=self.hH)
            toks, _ = self.model.intent_tokens(self.hH, hI.float(),
                                               torch.ones(N, dtype=torch.bool, device=self.device))
            kw = dict(intent_tokens=toks)
        gen = torch.zeros(N, H + F, 1, device=self.device); gen[:, H:] = 1.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a_n = self._sample(x, mask, gen, t_c, t_u, scal, **kw)
        a_n = a_n[:, H:H + self.K].float().clamp(-self.act_clip, self.act_clip)
        return a_n * self.std[PROPRIO_DIM:] + self.mean[PROPRIO_DIM:]

    def _sample(self, x_obs, mask, gen, text, text_u, scal, intent_tokens=None):
        """Euler + x0-space CFG on the ACTION channels of the future rows, the state channels held at what the
        closed loop can actually observe -- the same masking `--obs_future_state` imposed in training."""
        N, T, _ = x_obs.shape
        H = self.H
        dev = self.device
        keep = H + (1 if self.obs_future_state == "first" else 0)
        if self.obs_future_state != "all":
            x_obs = x_obs.clone(); x_obs[:, keep:, :PROPRIO_DIM] = 0.0
        a_obs = x_obs[..., PROPRIO_DIM:]
        z = torch.randn(N, T, ACTION_DIM, device=dev) * gen + a_obs * (1 - gen)
        grid = torch.linspace(0, 1, self.num_steps + 1, device=dev)
        cat2 = lambda a: None if a is None else torch.cat([a, a])
        ones = None if intent_tokens is None else torch.ones(N, intent_tokens.shape[1], dtype=torch.bool, device=dev)
        for i in range(self.num_steps):
            t = grid[i].expand(N); dt = grid[i + 1] - grid[i]
            zz = x_obs.clone(); zz[..., PROPRIO_DIM:] = z
            if self.cfg != 1.0:
                kw = {} if intent_tokens is None else dict(extra_tokens=cat2(intent_tokens),
                                                           extra_valid=torch.cat([~ones, ones]))
                x = self.net(torch.cat([zz, zz]), cat2(mask), torch.cat([t, t]),
                             torch.cat([text_u[0], text[0]]), torch.cat([text_u[1], text[1]]),
                             torch.cat([text_u[2], text[2]]), cat2(scal), **kw)
                x = x[:N] + self.cfg * (x[N:] - x[:N])
            else:
                kw = {} if intent_tokens is None else dict(extra_tokens=intent_tokens, extra_valid=ones)
                x = self.net(zz, mask, t, text[0], text[1], text[2], scal, **kw)
            x = x.float()
            if i == self.num_steps - 1:
                return x * gen + a_obs * (1 - gen)
            z = z + dt * (x - z) / (1.0 - t.view(-1, 1, 1)).clamp_min(1e-4)
            z = z * gen + a_obs * (1 - gen)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    torch.manual_seed(args_cli.seed); np.random.seed(args_cli.seed)
    N = args_cli.num_envs
    env_cfg.scene.num_envs = N
    motion_files = sorted(glob.glob(args_cli.motion_glob))
    if args_cli.motion_list:
        keep = {l.strip() for l in open(args_cli.motion_list) if l.strip()}
        n0 = len(motion_files)
        motion_files = [f for f in motion_files if os.path.basename(os.path.dirname(f)) in keep]
        print(f"[eval] motion whitelist {args_cli.motion_list}: {len(motion_files)}/{n0} motions")
        assert motion_files, "the whitelist matched no motion under --motion_glob"
    env_cfg.commands.motion.motion_files = motion_files
    env_cfg.commands.motion.start_from_zero_step = True
    env_cfg.commands.motion.enable_adaptive_sampling = False
    env_cfg.commands.motion.pose_range = {}; env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 600.0
    env_cfg.terminations.anchor_pos = None; env_cfg.terminations.anchor_ori = None; env_cfg.terminations.ee_body_pos = None
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    uenv = env.unwrapped; robot = uenv.scene["robot"]; cmd = uenv.command_manager.get_term("motion")
    dev = uenv.device
    body_names = list(robot.body_names)
    act_term = uenv.action_manager.get_term("joint_pos")
    def per_env(v, fallback=None):
        """the action term stores scale/offset as a scalar, a [n_joints] vector or a [n_envs, n_joints] tensor."""
        t = v if torch.is_tensor(v) else (fallback if v is None else torch.full((robot.num_joints,), float(v)))
        t = t.to(dev).float()
        return t if t.dim() == 2 else t.reshape(1, -1).expand(N, -1)

    scale = per_env(getattr(act_term, "_scale", 1.0))
    offset = per_env(getattr(act_term, "_offset", None), fallback=robot.data.default_joint_pos)
    print(f"[eval] action scale {scale[0, :3].tolist()} offset {offset[0, :3].tolist()}")

    bad = torch.tensor([i for i, n in enumerate(body_names) if not ("ankle" in n or "wrist" in n)], device=dev)
    feet = torch.tensor([i for i, n in enumerate(body_names) if "ankle_roll" in n], device=dev)
    # ADAPT's criteria are contact-based (Appendix C: "illegal torso contact"; Eq. S11: peak contact force > 1 N
    # over a short history). The tracker scene already carries a ContactSensor over every robot body.
    csensor = uenv.scene.sensors.get("contact_forces") if hasattr(uenv.scene, "sensors") else None
    fall_names = [n.strip() for n in args_cli.fall_bodies.split(",") if n.strip()]
    fall_ids = torch.tensor([body_names.index(n) for n in fall_names if n in body_names], device=dev)
    if args_cli.fall_mode == "contact":
        assert csensor is not None, "the scene has no 'contact_forces' sensor; use --fall_mode height"
        assert len(fall_ids) == len(fall_names), f"unknown bodies in --fall_bodies: {fall_names}"
        # the sensor is created over Robot/.* in prim order, which need not equal robot.body_names order
        sensor_names = list(getattr(csensor, "body_names", body_names))
        s_fall = torch.tensor([sensor_names.index(n) for n in fall_names], device=dev)
        # keep the sensor's foot order identical to `feet` (robot body order), or the contact indicator of one
        # foot would be paired with the velocity of the other
        s_feet = torch.tensor([sensor_names.index(body_names[i]) for i in feet.tolist()], device=dev)
        assert len(s_feet) == 2, f"expected 2 ankle_roll links in the contact sensor, got {len(s_feet)}"
    else:
        s_fall = s_feet = None

    def contact_peak(ids):
        """peak |force| over the sensor's history window, per body -- the indicator of Eq. S11."""
        return csensor.data.net_forces_w_history[:, :, ids, :].norm(dim=-1).max(dim=1)[0]

    print(f"[eval] {len(body_names)} bodies; fall={args_cli.fall_mode} "
          f"({fall_names if args_cli.fall_mode == 'contact' else f'{len(bad)} bodies below {args_cli.contact_z} m'}); "
          f"source={args_cli.source}")

    prompts = [l.strip() for l in open(args_cli.prompt_file) if l.strip()]
    bank = TextBank(args_cli.text_npz, dev)
    missing = [p for p in prompts if p not in bank.index]
    assert not missing, f"{len(missing)} prompts absent from {args_cli.text_npz}: {missing[:5]} -- rerun g1_build_text_dict.py"
    pidx = torch.tensor([bank.index[p] for p in prompts], device=dev)

    shadow = None
    planner = None
    if args_cli.source == "policy":
        assert args_cli.ckpt, "--source policy needs --ckpt"
        planner = G1Planner(args_cli.ckpt, dev, args_cli.stats, args_cli.num_steps, args_cli.cfg,
                            args_cli.K, args_cli.act_clip, args_cli.s_read, args_cli.weights)
        warm, K = max(planner.L, L_INTENT), planner.K
    else:
        warm, K = L_INTENT, 1

    tracker, meta, names = None, {}, [os.path.basename(os.path.dirname(f)) for f in env_cfg.commands.motion.motion_files]
    if args_cli.source == "tracker" or args_cli.warmup_source == "tracker":
        # The warm-up exists to fill the history with REAL physics before the policy plans for the first time.
        # Holding the reset pose does not do that: the reset pose is a frame of a reference motion, not a
        # balanced stance, and the robot topples within ~0.3 s (measured: 16/16 fell in a hold-only run).  The
        # tracker is also the right choice on distribution grounds -- every history row the policy was trained
        # on is a tracker action.
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.resume_path)
        tracker = runner.get_inference_policy(device=dev)
        print(f"[eval] tracker {args_cli.resume_path} ({'source' if args_cli.source == 'tracker' else 'warm-up only'})")
    if args_cli.source == "tracker":
        assert args_cli.meta_pkl, "--source tracker needs --meta_pkl for the BABEL labels of the reference motions"
        meta = joblib.load(args_cli.meta_pkl)
        print(f"[eval] tracker ceiling over {len(names)} reference motions")

    if args_cli.shadow_policy:
        assert args_cli.source == "tracker", "--shadow_policy only means anything on a tracker-driven rollout"
        shadow = G1Planner(args_cli.shadow_policy, dev, args_cli.stats, args_cli.num_steps, args_cli.cfg,
                           1, args_cli.act_clip, args_cli.s_read, args_cli.weights)
        warm = max(warm, shadow.L, L_INTENT)

    fps = FPS
    shadow_tot = {"err": 0.0, "var": 0.0, "n": 0.0}
    episodes, tot = [], {"n": 0, "success": 0, "smooth_sum": 0.0, "smooth_n": 0, "trans_sum": 0.0,
                         "trans_n": 0, "slide_sum": 0.0, "slide_n": 0, "fall_steps": []}
    n_batches = int(np.ceil(args_cli.rollouts / N)); t0 = time.time(); infer = []
    total_len = torch.full((N,), args_cli.steps / fps, device=dev)

    for b in range(n_batches):
        # IsaacLab computes terminations before the command manager inside step(), so after an explicit reset the
        # command's relative body poses are stale and every env would terminate on step 1 (see record_tracker_rollouts.py)
        env.reset(); cmd.time_steps -= 1; cmd._update_command()
        obs, _ = env.get_observations()
        d = robot.data
        hold = ((d.joint_pos - offset) / scale).clone()
        prop = torch.cat([d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b, d.joint_pos, d.joint_vel], -1)
        for pl in (planner, shadow):
            if pl is not None:
                pl.reset(N, prop, hold)
        shadow_err = shadow_var = shadow_n = 0.0
        shadow_prev_a = None
        # ---- warm-up: fill the history with real physics; not recorded, not counted
        for _ in range(warm):
            d = robot.data
            prop = torch.cat([d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b, d.joint_pos, d.joint_vel], -1)
            hold = ((d.joint_pos - offset) / scale).clone()
            a = tracker(obs) if args_cli.warmup_source == "tracker" else hold
            for pl in (planner, shadow):
                if pl is not None:
                    pl.push(prop, a)                # (state BEFORE the step, action applied in it)
            obs, _, _, _ = env.step(a)

        ref0 = cmd.time_steps.clone().long()
        horizon = torch.full((N,), args_cli.steps, device=dev, dtype=torch.long)
        if args_cli.source == "tracker":
            horizon = torch.minimum(horizon, (cmd.motion_length.long() - ref0).clamp_min(0))
        def ref_label_idx(step):
            """text-bank index of the BABEL label covering the reference motion's current frame, per env."""
            out = np.zeros(N, dtype=np.int64)              # 0 = the empty caption
            for i in range(N):
                t = (int(ref0[i]) + step) / fps
                best, ov_best = "", 0.0
                for a, b, lab, *_ in meta.get(names[int(cmd.motion_idx[i])], {}).get("frame_ann", []):
                    if str(lab) == "transition":
                        continue
                    ov = min(float(b), t + 0.14) - max(float(a), t)
                    if ov > ov_best:
                        best, ov_best = str(lab), ov
                out[i] = bank.index.get(best, 0)
            return torch.from_numpy(out).to(dev)

        cur = torch.randint(0, len(prompts), (N,), device=dev)
        next_sw = (torch.rand(N, device=dev) * (args_cli.switch_hi - args_cli.switch_lo) + args_cli.switch_lo) * fps
        if planner is not None:
            planner.new_prompt(bank, pidx[cur])
        if shadow is not None:
            shadow.new_prompt(bank, ref_label_idx(0))
        seg_start = torch.zeros(N, dtype=torch.long, device=dev)
        segments = [[] for _ in range(N)]
        fallen = torch.zeros(N, dtype=torch.bool, device=dev); fall_step = torch.full((N,), -1, device=dev)
        prev_a = None; trans_win = torch.zeros(N, device=dev)
        sm_sum = torch.zeros(N, device=dev); sm_n = torch.zeros(N, device=dev)
        tr_sum = torch.zeros(N, device=dev); tr_n = torch.zeros(N, device=dev)
        sl_sum = torch.zeros(N, device=dev); sl_n = torch.zeros(N, device=dev)
        rec_bp, plan_actions, plan_i = [], None, 10 ** 9

        for step in range(args_cli.steps):
            sw = (step >= next_sw) & ~fallen if args_cli.source != "tracker" else torch.zeros(N, dtype=torch.bool, device=dev)
            if bool(sw.any()):
                for i in torch.where(sw)[0].tolist():
                    segments[i].append((int(seg_start[i]), step, int(cur[i])))
                seg_start[sw] = step
                cur[sw] = (cur[sw] + torch.randint(1, len(prompts), (int(sw.sum()),), device=dev)) % len(prompts)
                next_sw[sw] = step + (torch.rand(int(sw.sum()), device=dev) *
                                      (args_cli.switch_hi - args_cli.switch_lo) + args_cli.switch_lo) * fps
                trans_win[sw] = fps
                if planner is not None:
                    planner.new_prompt(bank, pidx[cur], sw)
                plan_i = 10 ** 9                                  # a new prompt always triggers a replan
            d = robot.data
            # the state the action is about to be applied in; `record_tracker_rollouts.py` snapshots the same
            # one before its own env.step, so this is the s_t of the dataset's (s_t, a_t) rows
            prop = torch.cat([d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b, d.joint_pos, d.joint_vel], -1)
            hold = ((d.joint_pos - offset) / scale).clone()
            if args_cli.source == "hold":
                a = hold
            elif args_cli.source == "tracker":
                a = tracker(obs)
            else:
                if plan_i >= K:
                    prog = torch.full((N,), step / max(1.0, args_cli.steps), device=dev)
                    ts = time.time()
                    plan_actions = planner.plan(bank, pidx[cur], prog, total_len, prop)
                    torch.cuda.synchronize(); infer.append((time.time() - ts) * 1000); plan_i = 0
                a = plan_actions[:, plan_i]; plan_i += 1
            if shadow is not None and step % args_cli.shadow_every == 0:
                pr = torch.full((N,), step / max(1.0, args_cli.steps), device=dev)
                # condition on the label of the motion the tracker is actually following, not a random prompt
                lab_idx = ref_label_idx(step)
                shadow.new_prompt(bank, lab_idx)          # the label moves with the reference; refresh HIP
                row_h = None if args_cli.shadow_bug in ("rowh", "both") else prop
                pred = shadow.plan(bank, lab_idx, pr, total_len, row_h)[:, 0]
                alive = ~fallen
                shadow_err += float((((pred - a) ** 2).sum(-1) * alive).sum())
                shadow_var += float((((a - a.mean(0)) ** 2).sum(-1) * alive).sum())
                shadow_n += float(alive.sum())
            if prev_a is not None:
                da = ((a - prev_a) ** 2).sum(-1); alive = ~fallen
                sm_sum += da * alive; sm_n += alive
                inwin = alive & (trans_win > 0); tr_sum += da * inwin; tr_n += inwin
            prev_a = a.clone(); trans_win = (trans_win - 1).clamp_min(0)
            if planner is not None:
                planner.push(prop, a)               # (s_t, a_t), the dataset's pairing
            if shadow is not None:
                if args_cli.shadow_bug in ("pairing", "both"):
                    # the old, wrong pairing: `prop` here is the state the PREVIOUS action produced
                    if shadow_prev_a is not None:
                        shadow.push(prop, shadow_prev_a)
                    shadow_prev_a = a.clone()
                else:
                    shadow.push(prop, a)
            obs, _, _, _ = env.step(a)
            bp = d.body_pos_w - uenv.scene.env_origins[:, None]
            rec_bp.append(bp.to(torch.float16).cpu().numpy())
            if args_cli.fall_mode == "contact":
                newly = (~fallen) & (contact_peak(s_fall) > args_cli.fall_force).any(-1)
                contact = (contact_peak(s_feet) > 1.0) & (~fallen)[:, None]     # Eq. S11: peak force > 1 N
            else:
                z = bp[:, :, 2]
                newly = (~fallen) & ((z[:, bad] < args_cli.contact_z).any(-1) | (d.projected_gravity_b[:, 2] > -0.5))
                contact = (bp[:, feet, 2] < args_cli.contact_z) & (~fallen)[:, None]
            fall_step[newly] = step; fallen |= newly
            sl_sum += (d.body_lin_vel_w[:, feet, :2].norm(dim=-1) * contact).sum(-1); sl_n += contact.sum(-1)
            if step % 250 == 0:
                print(f"[eval] batch {b+1}/{n_batches} step {step} fallen {int(fallen.sum())}/{N} "
                      f"plan {np.mean(infer[-100:]) if infer else 0:.0f} ms ({time.time()-t0:.0f}s)", flush=True)
        for i in range(N):
            segments[i].append((int(seg_start[i]), args_cli.steps, int(cur[i])))

        body_pos = np.stack(rec_bp)                                        # [T,N,B,3] float16
        take = min(N, args_cli.rollouts - tot["n"])
        for i in range(take):
            fell = bool(fallen[i]); L = min(int(horizon[i]), int(fall_step[i]) if fell else args_cli.steps)
            if args_cli.source == "tracker":
                # the ceiling's captions are the BABEL labels of the motion the tracker is following
                m, r0 = int(cmd.motion_idx[i]), int(ref0[i])
                ann = meta.get(names[m], {}).get("frame_ann", [])
                segs = []
                for a, bb, lab, *_ in sorted(ann):
                    lab = str(lab)
                    if lab == "transition":
                        continue
                    s0, s1 = int(round(float(a) * fps)) - r0, int(round(float(bb) * fps)) - r0
                    s0, s1 = max(0, s0), min(L, s1)
                    if s1 > s0:
                        segs.append((s0, s1, lab))
            else:
                segs = [(s, min(e, L), bank.texts[int(pidx[p])]) for s, e, p in segments[i] if s < L]
            episodes.append(dict(rollout=tot["n"] + i, fell=fell, fall_step=int(fall_step[i]),
                                 length=L, body_pos=body_pos[:L, i], segments=segs))
        tot["n"] += take; tot["success"] += int((~fallen[:take]).sum())
        tot["smooth_sum"] += float(sm_sum[:take].sum()); tot["smooth_n"] += float(sm_n[:take].sum())
        tot["trans_sum"] += float(tr_sum[:take].sum()); tot["trans_n"] += float(tr_n[:take].sum())
        tot["slide_sum"] += float(sl_sum[:take].sum()); tot["slide_n"] += float(sl_n[:take].sum())
        tot["fall_steps"] += fall_step[:take][fallen[:take]].tolist()
        shadow_tot["err"] += shadow_err; shadow_tot["var"] += shadow_var; shadow_tot["n"] += shadow_n
        print(f"[eval] batch {b+1} done: no-fall so far {tot['success']}/{tot['n']} ({time.time()-t0:.0f}s)", flush=True)

    res = {"source": args_cli.source, "ckpt": args_cli.ckpt, "rollouts": tot["n"],
           "success_rate": tot["success"] / max(tot["n"], 1),
           "action_smoothness": tot["smooth_sum"] / max(tot["smooth_n"], 1),
           "transition_smoothness": tot["trans_sum"] / max(tot["trans_n"], 1),
           "foot_sliding_mps": tot["slide_sum"] / max(tot["slide_n"], 1),
           "mean_fall_time_s": (float(np.mean(tot["fall_steps"])) / fps) if tot["fall_steps"] else None,
           "plan_ms": float(np.mean(infer)) if infer else None, "prompts": len(prompts),
           "steps": args_cli.steps, "num_steps": args_cli.num_steps, "cfg": args_cli.cfg, "K": args_cli.K,
           "fall_mode": args_cli.fall_mode, "fall_bodies": args_cli.fall_bodies,
           "contact_z": args_cli.contact_z, "seed": args_cli.seed}
    if shadow is not None:
        # 0 = the policy reproduces the tracker exactly, 1 = no better than always predicting the mean action
        res["shadow_policy"] = args_cli.shadow_policy
        res["shadow_bug"] = args_cli.shadow_bug
        res["shadow_action_nmse"] = shadow_tot["err"] / max(shadow_tot["var"], 1e-9)
        res["shadow_samples"] = shadow_tot["n"]
    print("[eval] PHYSICAL " + json.dumps(res), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    joblib.dump(dict(episodes=episodes, physical=res, body_names=body_names, fps=fps), args_cli.out, compress=3)
    json.dump(res, open(os.path.splitext(args_cli.out)[0] + "_physical.json", "w"), indent=1)
    print(f"[eval] wrote {args_cli.out} ({os.path.getsize(args_cli.out)/1e6:.0f} MB)")
    env.close()


if __name__ == "__main__":
    main(); app.close()
