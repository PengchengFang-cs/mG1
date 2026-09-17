"""ADAPT-style dataset built from tracker rollouts recorded by scripts/record_tracker_rollouts.py.

Rollout pkl format (see docs/04_g1_pipeline.md):
  proprio (T,67) = [v(3) w(3) g(3) q(29) qd(29)], prev_action (T,29), action (T,29), frame_ann [(start_s, end_s, label, [proc]),...]

Token_j = [a_{j-1} (29) | o_j (96)] with o_j = [v, 0.2 w, g, q, 0.05 qd, a_{j-1}] (ADAPT obs scaling).
A clip has n_frames tokens (default 20 = 5 history + 15 future), stride `stride`.
Text: one label sampled among BABEL segments overlapping the clip's time span (UniPhys rule).
"""
from __future__ import annotations
import glob
import hashlib
import os
import random
import re
from dataclasses import dataclass, field

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

OBS_DIM = 96
ACT_DIM = 29
TOKEN_DIM = ACT_DIM + OBS_DIM  # 125
FPS = 50


def build_obs(proprio: np.ndarray, prev_action: np.ndarray, zero_lin_vel: bool = True) -> np.ndarray:
    """proprio (T,67), prev_action (T,29) -> o (T,96) with ADAPT scaling."""
    v, w, g, q, qd = proprio[:, 0:3], proprio[:, 3:6], proprio[:, 6:9], proprio[:, 9:38], proprio[:, 38:67]
    if zero_lin_vel:
        v = np.zeros_like(v)
    return np.concatenate([v, 0.2 * w, g, q, 0.05 * qd, prev_action], axis=-1).astype(np.float32)


def clean_label(text: str) -> str:
    text = text.replace("transition to ", "")
    if "walk back " in text or text == "walk back":
        text = "walk"
    return text


def labels_overlapping(frame_ann, t_start: float, t_end: float, strict: float = 3.0 / FPS, drop=("transition",)):
    out = []
    for seg in frame_ann:
        if len(seg) < 3:
            continue
        s, e, label = float(seg[0]), float(seg[1]), str(seg[2])
        if label in drop:
            continue
        if not (s + strict > t_end or t_start > e - strict):
            out.append(clean_label(label))
    return out


@dataclass
class ClipDatasetCfg:
    rollout_globs: list[str] = field(default_factory=list)
    n_frames: int = 20
    n_hist: int = 5
    stride: int = 10
    only_success: bool = True
    zero_lin_vel: bool = True
    min_len: int = 25
    text_embedding_dict: str | None = None  # pkl {label: (512,) np.float32}
    holdout_frac: float = 0.0               # fraction of motions (by name hash) reserved for the "val" split
    use_expert_label: bool = True           # DAgger files: use expert_action (tracker) as the action label when present
    drop_tail_on_fail: int = 25             # when only_success=False: drop this many final frames of failed rollouts
    stats_path: str | None = None           # npz with token_mean/token_std; computed if None


class RolloutClipDataset(Dataset):
    def __init__(self, cfg: ClipDatasetCfg, split: str = "train"):
        self.cfg, self.split = cfg, split
        files = sorted(f for g in cfg.rollout_globs for f in glob.glob(g))
        assert files, f"no rollout files for {cfg.rollout_globs}"
        self.clips: list[dict] = []
        n_roll, n_used = 0, 0
        for f in files:
            d = joblib.load(f)
            for r in d["rollouts"]:
                n_roll += 1
                if cfg.only_success and not r["success"]:
                    continue
                if not r["success"] and cfg.drop_tail_on_fail > 0:
                    r = {k: (v[: -cfg.drop_tail_on_fail] if hasattr(v, "shape") and v.shape[0] == r["action"].shape[0] else v) for k, v in r.items()}
                if cfg.holdout_frac > 0:
                    h = int(hashlib.md5(str(r["motion"]).encode()).hexdigest()[:8], 16) % 1000
                    is_hold = h < int(cfg.holdout_frac * 1000)
                    if (split == "val") != is_hold:
                        continue
                T = r["action"].shape[0]
                if T < max(cfg.min_len, cfg.n_frames + 1):
                    continue
                n_used += 1
                obs = build_obs(r["proprio"], r["prev_action"], cfg.zero_lin_vel)      # (T,96)  (prev_action = executed a_{j-1})
                lab = r["expert_action"] if (cfg.use_expert_label and "expert_action" in r) else r["action"]   # label a_j
                # history tokens: [executed a_{j-1}, o_j] (what the policy sees at test time)
                # future tokens:  [expert   a_{j-1}, o_j] (prediction target). Identical for pure tracker data.
                tok_exec = np.concatenate([r["prev_action"].astype(np.float32), obs], -1)
                a_prev_lab = np.concatenate([np.zeros((1, ACT_DIM), np.float32), lab[:-1]], 0).astype(np.float32)
                tok_lab = np.concatenate([a_prev_lab, obs], -1)
                H = cfg.n_hist
                for j0 in range(0, T - cfg.n_frames + 1, cfg.stride):
                    j1 = j0 + cfg.n_frames
                    texts = labels_overlapping(r["frame_ann"], j0 / FPS, j1 / FPS)
                    if not texts:
                        continue
                    clip = np.concatenate([tok_exec[j0:j0 + H], tok_lab[j0 + H:j1]], 0)
                    self.clips.append({"tok": clip, "texts": sorted(set(texts)), "motion": r["motion"], "j0": j0})
        print(f"[RolloutClipDataset:{split}] rollouts {n_roll} used {n_used} clips {len(self.clips)}")
        self.text_emb = joblib.load(cfg.text_embedding_dict) if cfg.text_embedding_dict else None
        if cfg.stats_path and os.path.exists(cfg.stats_path):
            s = np.load(cfg.stats_path)
            self.mean, self.std = s["token_mean"], s["token_std"]
        else:
            allt = np.concatenate([c["tok"] for c in self.clips], 0)
            self.mean = allt.mean(0).astype(np.float32)
            self.std = (allt.std(0) + 1e-6).astype(np.float32)
            if cfg.stats_path:
                np.savez(cfg.stats_path, token_mean=self.mean, token_std=self.std)

    def vocab(self) -> list[str]:
        return sorted({t for c in self.clips for t in c["texts"]})

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        x = (c["tok"] - self.mean) / self.std
        text = random.choice(c["texts"]) if self.split == "train" else c["texts"][0]
        cond = {"text": text}
        if self.text_emb is not None:
            cond["text_embedding"] = torch.from_numpy(np.asarray(self.text_emb[text], dtype=np.float32))
        return torch.from_numpy(x.astype(np.float32)), cond
