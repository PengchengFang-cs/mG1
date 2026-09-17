"""Standalone HumanML3D text-to-motion evaluator (Guo et al. 2022 protocol) for physics rollouts.

* Evaluator: KV-Control checkpoints/t2m/text_mot_match (finest.tar) + Comp_v6_KLD005 opt/meta stats.
* Inputs are RAW (un-normalised) 263-d HumanML3D features; normalisation with the evaluator's own
  meta mean/std happens here (that is what the official Text2MotionDatasetEval does).
* Metrics: R-Precision top-1/2/3, FID, MM-Dist, Diversity (300), MultiModality (optional),
  batch 32 with drop_last, repeated `replications` times, mean and 95% CI.
* GT loader reproduces Text2MotionDatasetEval exactly (min len 40, < 200, sub-clip captions with
  time tags, random caption choice, unit-length quantisation, random crop, zero-pad to 196).

Item format (both GT and generated): dict(key=str, motion=np.ndarray [L,263] raw float32,
length=int, texts=[dict(caption=str, tokens=[ 'word/POS', ... ])]).
"""
import os
import codecs as cs
from math import sqrt
import numpy as np
import torch

from hml_phys.t2m.get_opt import get_opt
from hml_phys.t2m.t2m_eval_wrapper import EvaluatorModelWrapper
from hml_phys.t2m.word_vectorizer import WordVectorizer
from hml_phys.t2m.metrics import (calculate_R_precision, euclidean_distance_matrix,
                                  calculate_activation_statistics, calculate_diversity,
                                  calculate_multimodality, calculate_frechet_distance)
from hml_phys.sim2hml import hml263_to_joints
from hml_phys import phys_metrics

KV_ROOT = "/iridisfs/scratch/pf2m24/projects/Umdd/KV-Control"
CKPT_DIR = os.path.join(KV_ROOT, "checkpoints")
EVAL_OPT = os.path.join(CKPT_DIR, "t2m", "Comp_v6_KLD005", "opt.txt")
GLOVE_DIR = os.path.join(KV_ROOT, "glove")
HML_ROOT = "/iridisfs/scratch/pf2m24/data/HumanML3D/HumanML3D"

MAX_MOTION_LEN = 196
MAX_TEXT_LEN = 20
UNIT_LEN = 4
MIN_MOTION_LEN = 40
BATCH = 32  # R-precision is defined over batches of 32 – do not change


# ----------------------------------------------------------------------------- GT loading
def read_split(split, root=HML_ROOT):
    with cs.open(os.path.join(root, f"{split}.txt"), "r") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def read_texts(name, root=HML_ROOT):
    """Return list of dict(caption, tokens, f_tag, to_tag) from texts/<name>.txt."""
    out = []
    with cs.open(os.path.join(root, "texts", name + ".txt")) as f:
        for line in f.readlines():
            parts = line.strip().split("#")
            if len(parts) < 4:
                continue
            f_tag = float(parts[2]); to_tag = float(parts[3])
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
            out.append(dict(caption=parts[0], tokens=parts[1].split(" "), f_tag=f_tag, to_tag=to_tag))
    return out


def build_gt_items(split="test", root=HML_ROOT, motion_dir=None, id_list=None, rng=None):
    """Official Text2MotionDatasetEval item construction from raw 263-d files.

    motion_dir: directory of raw 263-d <name>.npy (default HumanML3D new_joint_vecs).
    Returns list of items; sub-clip captions become separate items (key '<letter>_<name>') like the
    official code. Also returns the number of split ids that were dropped by the length filter.
    """
    rng = rng or np.random.RandomState(0)
    motion_dir = motion_dir or os.path.join(root, "new_joint_vecs")
    ids = id_list if id_list is not None else read_split(split, root)
    items, dropped, used_keys = [], 0, set()
    for name in ids:
        p = os.path.join(motion_dir, name + ".npy")
        if not os.path.exists(p):
            dropped += 1; continue
        motion = np.load(p).astype(np.float32)
        if len(motion) < MIN_MOTION_LEN or len(motion) >= 200:
            dropped += 1; continue
        whole = []
        for t in read_texts(name, root):
            if t["f_tag"] == 0.0 and t["to_tag"] == 0.0:
                whole.append(dict(caption=t["caption"], tokens=t["tokens"]))
            else:
                sub = motion[int(t["f_tag"] * 20): int(t["to_tag"] * 20)]
                if len(sub) < MIN_MOTION_LEN or len(sub) >= 200:
                    continue
                key = "ABCDEFGHIJKLMNOPQRSTUVW"[rng.randint(23)] + "_" + name
                while key in used_keys:  # official code loops until the key is unique
                    key = "ABCDEFGHIJKLMNOPQRSTUVW"[rng.randint(23)] + "_" + name
                used_keys.add(key)
                items.append(dict(key=key, base=name, motion=sub, length=len(sub),
                                  texts=[dict(caption=t["caption"], tokens=t["tokens"])]))
        if whole:
            items.append(dict(key=name, base=name, motion=motion, length=len(motion), texts=whole))
    return items, dropped


# ----------------------------------------------------------------------------- evaluator
class HMLEvaluator:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        opt = get_opt(EVAL_OPT, self.device, checkpoints_dir=CKPT_DIR)
        opt.meta_dir = os.path.join(CKPT_DIR, "t2m", "Comp_v6_KLD005", "meta")  # get_opt derives it from the relative path in opt.txt
        self.opt = opt
        self.wrapper = EvaluatorModelWrapper(opt)
        self.mean = np.load(os.path.join(opt.meta_dir, "mean.npy")).astype(np.float32)
        self.std = np.load(os.path.join(opt.meta_dir, "std.npy")).astype(np.float32)
        self.w_vectorizer = WordVectorizer(GLOVE_DIR, "our_vab")

    # -- text
    def encode_tokens(self, tokens):
        if len(tokens) < MAX_TEXT_LEN:
            tokens = ["sos/OTHER"] + list(tokens) + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (MAX_TEXT_LEN + 2 - sent_len)
        else:
            tokens = ["sos/OTHER"] + list(tokens[:MAX_TEXT_LEN]) + ["eos/OTHER"]
            sent_len = len(tokens)
        pos, emb = [], []
        for tok in tokens:
            we, po = self.w_vectorizer[tok]
            pos.append(po[None]); emb.append(we[None])
        return np.concatenate(emb, 0).astype(np.float32), np.concatenate(pos, 0).astype(np.float32), sent_len

    # -- motion
    def prepare_motion(self, motion, m_length):
        """raw [L,263] -> normalised, zero-padded [196,263]; m_length is used as-is."""
        m = (motion[:m_length] - self.mean) / self.std
        if m_length < MAX_MOTION_LEN:
            m = np.concatenate([m, np.zeros((MAX_MOTION_LEN - m_length, m.shape[1]), np.float32)], 0)
        return m.astype(np.float32)

    @staticmethod
    def official_crop(length, rng):
        """Text2MotionDatasetEval length quantisation + random crop start."""
        coin2 = rng.choice(["single", "single", "double"])
        if coin2 == "double":
            m_len = (length // UNIT_LEN - 1) * UNIT_LEN
        else:
            m_len = (length // UNIT_LEN) * UNIT_LEN
        max_aligned = (MAX_MOTION_LEN // UNIT_LEN) * UNIT_LEN
        m_len = min(m_len, max_aligned)
        start = rng.randint(0, length - m_len + 1)
        return start, m_len

    def batch_embeddings(self, samples):
        """samples: list of dict(word_emb, pos, sent_len, motion196, m_length) (any order)."""
        order = np.argsort([-s["sent_len"] for s in samples], kind="stable")  # packed GRU needs desc
        samples = [samples[i] for i in order]
        we = torch.from_numpy(np.stack([s["word_emb"] for s in samples]))
        po = torch.from_numpy(np.stack([s["pos"] for s in samples]))
        sl = torch.tensor([s["sent_len"] for s in samples])
        mo = torch.from_numpy(np.stack([s["motion196"] for s in samples]))
        ml = torch.tensor([s["m_length"] for s in samples])
        et, em = self.wrapper.get_co_embeddings(we, po, sl, mo, ml)
        return et.cpu().numpy(), em.cpu().numpy()

    def make_sample(self, item, rng, official_crop=True, text_idx=None):
        texts = item["texts"]
        t = texts[rng.randint(len(texts))] if text_idx is None else texts[text_idx]
        we, po, sl = self.encode_tokens(t["tokens"])
        L = int(item["length"])
        if official_crop:
            start, m_len = self.official_crop(L, rng)
        else:
            start, m_len = 0, min(L, MAX_MOTION_LEN)
        motion = item["motion"][start:start + m_len]
        return dict(word_emb=we, pos=po, sent_len=sl, motion196=self.prepare_motion(motion, m_len),
                    m_length=m_len, caption=t["caption"])

    # -- one replication
    def _run_set(self, items, rng, official_crop):
        """Returns (top-k hit counts[3], matching score sum, n, motion embeddings [n,512])."""
        assert len(items) >= BATCH, f"need at least {BATCH} items for one R-precision batch, got {len(items)}"
        idx = rng.permutation(len(items))
        r_prec = np.zeros(3); match = 0.0; n = 0; ems = []
        for b in range(0, len(idx) - BATCH + 1, BATCH):
            samples = [self.make_sample(items[i], rng, official_crop) for i in idx[b:b + BATCH]]
            et, em = self.batch_embeddings(samples)
            r_prec += calculate_R_precision(et, em, top_k=3, sum_all=True)
            match += euclidean_distance_matrix(et, em).trace()
            n += BATCH; ems.append(em)
        return r_prec, match, n, np.concatenate(ems, 0)

    def evaluate(self, gt_items, gen_items=None, replications=20, seed=0, diversity_times=300,
                 gen_official_crop=False, unique_per_key=True, mm_items=None, mm_times=10,
                 physics=True, log=print):
        """gen_items=None evaluates GT against itself (sanity: R@1≈0.511, FID≈0.002 for HumanML3D test)."""
        rng = np.random.RandomState(seed)
        np.random.seed(seed)  # calculate_diversity / multimodality use np.random
        per_rep = []
        for r in range(replications):
            gt_r, gt_m, gt_n, gt_em = self._run_set(gt_items, rng, official_crop=True)
            mu_gt, cov_gt = calculate_activation_statistics(gt_em)
            res = dict(rep=r, real_top1=float(gt_r[0] / gt_n), real_top2=float(gt_r[1] / gt_n), real_top3=float(gt_r[2] / gt_n),
                       real_mm_dist=float(gt_m / gt_n), real_diversity=float(calculate_diversity(gt_em, min(diversity_times, gt_n - 1))))
            if gen_items is not None:
                items = gen_items
                if unique_per_key:
                    keys = {}
                    for i in rng.permutation(len(gen_items)):
                        keys.setdefault(gen_items[i]["key"], i)
                    items = [gen_items[i] for i in sorted(keys.values())]
                g_r, g_m, g_n, g_em = self._run_set(items, rng, official_crop=gen_official_crop)
                mu, cov = calculate_activation_statistics(g_em)
                res.update(top1=float(g_r[0] / g_n), top2=float(g_r[1] / g_n), top3=float(g_r[2] / g_n),
                           mm_dist=float(g_m / g_n), fid=float(calculate_frechet_distance(mu_gt, cov_gt, mu, cov)),
                           diversity=float(calculate_diversity(g_em, min(diversity_times, g_n - 1))), n_gen=int(g_n))
            if mm_items:
                res["multimodality"] = self._multimodality(mm_items, rng, mm_times)
            per_rep.append(res)
            log(f"[rep {r}] " + ", ".join(f"{k}={v:.4f}" for k, v in res.items() if k != "rep"))
        summary = {}
        for k in per_rep[0]:
            if k == "rep":
                continue
            vals = np.array([p[k] for p in per_rep], dtype=np.float64)
            summary[k] = dict(mean=float(vals.mean()), ci95=float(vals.std() * 1.96 / sqrt(len(vals))))
        if physics:
            summary["physics_gt"] = self.physics(gt_items)
            if gen_items is not None:
                summary["physics_gen"] = self.physics(gen_items)
        return summary, per_rep

    def _multimodality(self, mm_items, rng, mm_times):
        """mm_items: list of dict(texts=[one text], motions=[list of raw [L,263]], lengths=[...])."""
        acts = []
        for it in mm_items:
            we, po, sl = self.encode_tokens(it["texts"][0]["tokens"])
            samples = [dict(word_emb=we, pos=po, sent_len=sl, motion196=self.prepare_motion(m, int(l)), m_length=int(l))
                       for m, l in zip(it["motions"], it["lengths"])]
            ems = []
            for b in range(0, len(samples), BATCH):
                chunk = samples[b:b + BATCH]
                mo = torch.from_numpy(np.stack([s["motion196"] for s in chunk]))
                ml = torch.tensor([s["m_length"] for s in chunk])
                # get_motion_embeddings returns embeddings sorted by length; order irrelevant for MModality
                ems.append(self.wrapper.get_motion_embeddings(mo, ml).cpu().numpy())
            acts.append(np.concatenate(ems, 0))
        acts = np.stack(acts, 0)
        return float(calculate_multimodality(acts, mm_times))

    @staticmethod
    def physics(items):
        """Physics metrics on joints recovered from the 263-d features (HumanML3D representation: each
        clip is re-floored by process_file, so Penetration is 0 by construction and Floating is relative
        to the clip's lowest point). For simulated data prefer physics_raw()."""
        joints = [hml263_to_joints(it["motion"][:int(it["length"])]) for it in items]
        return {k: float(v) for k, v in phys_metrics.all_metrics(joints).items()}

    @staticmethod
    def physics_raw(joints_list):
        """Physics metrics on raw y-up 20 fps joints with the true simulator ground plane (y = 0)."""
        return {k: float(v) for k, v in phys_metrics.all_metrics(joints_list).items()}


def format_summary(summary, prefix=""):
    lines = []
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v and "ci95" in v:
            lines.append(f"{prefix}{k:16s} {v['mean']:.4f} ± {v['ci95']:.4f}")
        elif isinstance(v, dict) and all(isinstance(b, (int, float)) for b in v.values()):
            lines.append(f"{prefix}{k}: " + ", ".join(f"{a}={b:.4f}" for a, b in v.items()))
        elif isinstance(v, dict):
            lines.append(f"{prefix}{k}:"); lines.append(format_summary(v, prefix + "    "))
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)
