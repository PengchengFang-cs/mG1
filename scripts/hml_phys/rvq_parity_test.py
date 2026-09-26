#!/usr/bin/env python
"""Parity tests for `hml_phys/rvq.py` (PartRVQVAE) against the MoMask reference RVQVAE.

Reference: vendor_momask/ (clone of EricGuo5513/momask-codes), read in docs/08_rvq_spec.md.
This file NEVER edits hml_phys/rvq.py -- it only reports mismatches.

Contract under test (from the lead):
    PartRVQVAE(input_dim, part_channels, width, down_t, depth, dilation,
               code_dim, nb_code, n_quant, shared_codebook, dropout_prob)
      .encode(x [B,T,D]) -> dict(codes [B,T/down,n_parts,n_quant] long,
                                 z_q   [B,T/down,n_parts,code_dim])
      .forward(x)        -> (x_rec [B,T,D], losses dict with 'commit' + per-part usage stats)
      .decode(codes or z_q) -> x_rec

Tests
  1. weight-transfer parity: one part over all channels, MoMask weights copied in,
     reconstruction / codes / commit must agree to `--tol` (default 1e-5) in eval mode.
  2. residual property: reconstruction error must not increase as quantiser layers are added.
  3. code validity + round-trip determinism.
  4. quantise-dropout: off at eval, MoMask's uniform kept-depth distribution at train.

Run on a compute node (never the login node):
  srun --jobid=<job> --overlap -n1 bash -lc 'source scripts/activate_uniphys.sh >/dev/null 2>&1; \
    cd /iridisfs/scratch/pf2m24/projects/motion_rebot; export CUDA_VISIBLE_DEVICES=1; \
    python scripts/hml_phys/rvq_parity_test.py'

Env overrides (for harness self-validation only):
  RVQ_MODULE=<dotted.module.path>   import PartRVQVAE from somewhere other than hml_phys.rvq
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
import random
import re
import sys
import traceback
from collections import Counter, OrderedDict
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENDOR_MOMASK = os.path.join(REPO, "vendor_momask")

# quantiser layers / codebook size / code dim / input dim used by every test below.
# Deliberately small: parity at 1e-5 in float32 wants modest activations, and the whole
# suite has to stay inside a few seconds of one GPU's CPU budget.
CFG = dict(input_dim=48, width=64, down_t=2, depth=2, dilation=3,
           code_dim=32, nb_code=64, n_quant=3, dropout_prob=0.2)
BATCH, FRAMES = 8, 64          # T must be a multiple of 2**down_t
FIT_STEPS = 700                # optimiser steps; an untrained autoencoder makes tests 2 and 4 vacuous
COMMIT_W = 0.02                # vq_option.py:23


# --------------------------------------------------------------------------------------- reporting
class Report:
    def __init__(self):
        self.rows = []          # (name, status, detail)

    def add(self, name, status, detail=""):
        self.rows.append((name, status, detail))
        tag = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP", "INCONCLUSIVE": "INCO"}[status]
        print(f"[{tag}] {name}" + (f"\n       {detail}" if detail else ""), flush=True)

    def summary(self):
        print("\n" + "=" * 78)
        for name, status, _ in self.rows:
            print(f"  {status:<12s} {name}")
        print("=" * 78)
        n_fail = sum(1 for _, s, _ in self.rows if s == "FAIL")
        n_inc = sum(1 for _, s, _ in self.rows if s == "INCONCLUSIVE")
        print(f"  {len(self.rows)} checks: "
              f"{sum(1 for _, s, _ in self.rows if s == 'PASS')} pass, {n_fail} fail, "
              f"{n_inc} inconclusive, {sum(1 for _, s, _ in self.rows if s == 'SKIP')} skip")
        return n_fail


# --------------------------------------------------------------------------------------- imports
def import_reference():
    """MoMask RVQVAE. Needs vendor_momask on sys.path (its modules import `models.vq.*`)."""
    if not os.path.isdir(VENDOR_MOMASK):
        raise RuntimeError(f"vendor_momask/ not found at {VENDOR_MOMASK}; clone it on the login node:\n"
                           f"  git clone --depth 1 https://github.com/EricGuo5513/momask-codes {VENDOR_MOMASK}")
    if VENDOR_MOMASK not in sys.path:
        sys.path.insert(0, VENDOR_MOMASK)
    from models.vq.model import RVQVAE  # noqa: E402
    return RVQVAE


def import_ours():
    """PartRVQVAE, or None if the lead has not pushed hml_phys/rvq.py yet."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    mod_name = os.environ.get("RVQ_MODULE", "hml_phys.rvq")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        return None, f"cannot import {mod_name}: {exc}"
    cls = getattr(mod, "PartRVQVAE", None)
    if cls is None:
        return None, f"{mod_name} has no attribute PartRVQVAE"
    return cls, mod_name


def build_reference(RVQVAE, cfg):
    args = Namespace(num_quantizers=cfg["n_quant"], shared_codebook=False,
                     quantize_dropout_prob=cfg["dropout_prob"], mu=0.99)
    return RVQVAE(args,
                  cfg["input_dim"],          # input_width
                  cfg["nb_code"],
                  cfg["code_dim"],
                  cfg["code_dim"],           # output_emb_width == code_dim (model.py:23)
                  cfg["down_t"],
                  2,                         # stride_t: decoder hard-codes x2 (encdec.py:57)
                  cfg["width"],
                  cfg["depth"],
                  cfg["dilation"],
                  "relu",
                  None)


def build_ours(cls, cfg, part_channels, dropout_prob=None):
    """Single part covering every channel, so the model is structurally MoMask's."""
    kw = dict(input_dim=cfg["input_dim"], part_channels=part_channels, width=cfg["width"],
              down_t=cfg["down_t"], depth=cfg["depth"], dilation=cfg["dilation"],
              code_dim=cfg["code_dim"], nb_code=cfg["nb_code"], n_quant=cfg["n_quant"],
              shared_codebook=False,
              dropout_prob=cfg["dropout_prob"] if dropout_prob is None else dropout_prob)
    try:
        return cls(**kw)
    except TypeError:
        # positional fallback, in the contract's declared order
        return cls(kw["input_dim"], kw["part_channels"], kw["width"], kw["down_t"], kw["depth"],
                   kw["dilation"], kw["code_dim"], kw["nb_code"], kw["n_quant"],
                   kw["shared_codebook"], kw["dropout_prob"])


# --------------------------------------------------------------------------------------- weight copy
_GRP_RE = re.compile(r"(encoder|decoder)s?\.")
_INT_RE = re.compile(r"(\d+)")


def _norm_key(key):
    """('encoder'|'decoder', structural-suffix) or ('codebook', None) or None.

    Strips whatever prefix the attribute happens to be called and any leading part/layer
    index, so `part_encoders.0.model.3.weight` and `encoder.model.3.weight` collide.
    """
    if "codebook" in key.lower():
        return ("codebook", None)
    m = _GRP_RE.search(key.lower())
    if m is None:
        return None
    rest = re.sub(r"^(\d+\.)+", "", key[m.end():])
    return (m.group(1), rest)


def _last_int(key):
    ints = _INT_RE.findall(key)
    return int(ints[-1]) if ints else -1


# MoMask keeps these as plain Python attributes (quantizer.py:44-46, 63-65) so they never reach
# state_dict(); registering them as buffers instead is an improvement (the EMA state survives a
# checkpoint), so the copy helper fills them rather than treating them as unmappable.
EMA_EXTRA_SUM = ("code_sum", "embed_avg", "code_avg")
EMA_EXTRA_COUNT = ("code_count", "cluster_size")
EMA_EXTRA_FLAG = ("inited", "init", "initialized", "initialised")
EMA_EXTRA = EMA_EXTRA_SUM + EMA_EXTRA_COUNT + EMA_EXTRA_FLAG


def copy_reference_weights(ref, ours, cfg):
    """Copy MoMask's encoder/decoder/codebooks into `ours`. -> (ok, detail)."""
    ref_sd, our_sd = ref.state_dict(), ours.state_dict()

    ref_index = {}
    for k, v in ref_sd.items():
        n = _norm_key(k)
        if n is None or n[0] == "codebook":
            continue
        ref_index.setdefault(n, []).append(k)

    mapping, unmapped_ours, ema_extra = OrderedDict(), [], []
    for k, v in our_sd.items():
        if k.rsplit(".", 1)[-1] in EMA_EXTRA:
            ema_extra.append(k)
            continue
        n = _norm_key(k)
        if n is None:
            unmapped_ours.append(f"{k} {tuple(v.shape)} (no encoder/decoder/codebook in name)")
            continue
        if n[0] == "codebook":
            continue
        cands = [c for c in ref_index.get(n, []) if tuple(ref_sd[c].shape) == tuple(v.shape)]
        if len(cands) == 1:
            mapping[k] = cands[0]
        else:
            unmapped_ours.append(f"{k} {tuple(v.shape)} -> {len(cands)} reference candidates "
                                 f"for structural key {n}")

    # --- codebooks -------------------------------------------------------------------
    ref_cbs = sorted([k for k in ref_sd if "codebook" in k.lower()], key=_last_int)
    our_cbs = sorted([k for k in our_sd if "codebook" in k.lower()], key=_last_int)
    cb_assign = {}
    if len(our_cbs) == len(ref_cbs) and len(ref_cbs) == cfg["n_quant"]:
        for ok, rk in zip(our_cbs, ref_cbs):
            if tuple(our_sd[ok].shape) != tuple(ref_sd[rk].shape):
                unmapped_ours.append(f"{ok} {tuple(our_sd[ok].shape)} vs reference "
                                     f"{rk} {tuple(ref_sd[rk].shape)}")
            else:
                cb_assign[ok] = ref_sd[rk].clone()
    elif len(our_cbs) == 1:
        ok = our_cbs[0]
        t = our_sd[ok]
        stacked = torch.stack([ref_sd[k] for k in ref_cbs], 0)       # [Q, nb_code, code_dim]
        if tuple(t.shape) == tuple(stacked.shape):
            cb_assign[ok] = stacked
        elif tuple(t.shape) == (1,) + tuple(stacked.shape):          # [P=1, Q, nb_code, code_dim]
            cb_assign[ok] = stacked.unsqueeze(0)
        else:
            unmapped_ours.append(f"single codebook tensor {ok} {tuple(t.shape)}; expected "
                                 f"{tuple(stacked.shape)} or {(1,) + tuple(stacked.shape)}")
    else:
        unmapped_ours.append(f"{len(our_cbs)} codebook tensors {our_cbs} vs {len(ref_cbs)} in reference")

    # --- EMA bookkeeping buffers, primed to be consistent with the copied codebook ----
    ema_assign = {}
    for k in ema_extra:
        tail = k.rsplit(".", 1)[-1]
        cb_key = k[: -len(tail)] + "codebook"
        cb = cb_assign.get(cb_key, our_sd.get(cb_key))
        if cb is None:
            unmapped_ours.append(f"{k}: no sibling codebook at {cb_key}")
            continue
        if tail in EMA_EXTRA_SUM:
            ema_assign[k] = cb.clone().reshape(our_sd[k].shape)
        else:                                   # counts and the "already initialised" flag
            ema_assign[k] = torch.ones_like(our_sd[k])

    if unmapped_ours:
        return False, ("could not map these of your parameters onto the MoMask reference:\n         "
                       + "\n         ".join(unmapped_ours[:20]))

    new_sd = OrderedDict(our_sd)
    for ok, rk in mapping.items():
        new_sd[ok] = ref_sd[rk].clone()
    new_sd.update(cb_assign)
    new_sd.update(ema_assign)
    ours.load_state_dict(new_sd, strict=True)

    n_set = set_ema_state(ours, cfg)
    detail = (f"copied {len(mapping)} enc/dec tensors + {len(cb_assign)} codebook tensor(s), "
              f"primed {len(ema_assign)} EMA bookkeeping buffer(s) and "
              f"{n_set} attribute-style quantiser state(s)")
    return True, detail


def set_ema_state(model, cfg):
    """MoMask keeps `init`/`code_sum`/`code_count` as plain attributes -> absent from state_dict.

    Without them an eval-mode model quantises against a zero codebook, and a train-mode model
    re-initialises the codebook from the first batch. Prime them from whatever codebook is loaded.
    """
    n = 0
    for m in model.modules():
        cb = getattr(m, "codebook", None)
        if not torch.is_tensor(cb) or not hasattr(m, "init"):
            continue
        m.init = True
        flat = cb.reshape(-1, cb.shape[-1])
        if getattr(m, "code_sum", None) is None:
            m.code_sum = cb.clone()
        if getattr(m, "code_count", None) is None:
            m.code_count = torch.ones(flat.shape[0], device=cb.device).reshape(cb.shape[:-1])
        n += 1
    return n


# --------------------------------------------------------------------------------------- helpers
def smooth_motion(batch, frames, dim, device, generator):
    """Low-frequency random signals: an untrained encoder on white noise gives a degenerate
    latent distribution, which makes the residual/monotonicity test meaningless."""
    t = torch.linspace(0, 1, frames, device=device).view(1, frames, 1)
    out = torch.zeros(batch, frames, dim, device=device)
    for k in range(1, 6):
        a = torch.randn(batch, 1, dim, device=device, generator=generator) / k
        p = torch.rand(batch, 1, dim, device=device, generator=generator) * 2 * math.pi
        out = out + a * torch.sin(2 * math.pi * k * t + p)
    return out / out.std()


def set_quantize_dropout_prob(model, p):
    """Set every quantise-dropout probability we can find. -> number of attributes set."""
    names = ("quantize_dropout_prob", "dropout_prob", "quantise_dropout_prob", "qdp")
    n = 0
    for m in model.modules():
        for nm in names:
            if hasattr(m, nm) and isinstance(getattr(m, nm), (int, float)):
                setattr(m, nm, float(p))
                n += 1
    return n


def zero_nn_dropout(model):
    """MoMask puts a hard-coded Dropout(0.2) in every residual block (resnet.py:13, 46).

    It is train-mode-only noise that has nothing to do with quantise-dropout; silence it when
    the point of the measurement is the quantiser."""
    saved = []
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            saved.append((m, m.p))
            m.p = 0.0
    return saved


def restore_nn_dropout(saved):
    for m, p in saved:
        m.p = p


def snapshot_codebooks(model):
    return [(m, m.codebook.clone()) for m in model.modules()
            if torch.is_tensor(getattr(m, "codebook", None))]


def restore_codebooks(snap):
    for m, cb in snap:
        m.codebook = cb.clone()


def fit(model, data, steps, is_reference, cfg, lr=1e-3, init_steps=15):
    """Briefly fit the tiny autoencoder with MoMask's objective.

    Not optional: with random weights every truncated reconstruction has the same error as the
    zero prediction, so the residual and dropout tests degenerate into noise. Quantise-dropout
    is off for the first `init_steps` so that every codebook gets its lazy EMA initialisation.
    """
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    crit = nn.SmoothL1Loss()                       # vq_option.py:25, vq_trainer.py:34
    set_quantize_dropout_prob(model, 0.0)
    model.train()
    last = float("nan")
    for i in range(steps):
        if i == init_steps:
            set_quantize_dropout_prob(model, cfg["dropout_prob"])
        x = data[i % len(data)]
        out = as_tuple(model(x))
        rec = out[0]
        if is_reference:
            commit = out[1]
        else:
            commit = out[1]["commit"] if (len(out) > 1 and isinstance(out[1], dict)
                                          and "commit" in out[1]) else rec.new_zeros(())
        loss = crit(rec, x) + COMMIT_W * commit    # vq_trainer.py:50, minus the RIC-slice term
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss)
    set_quantize_dropout_prob(model, cfg["dropout_prob"])
    model.eval()
    return last


def as_tuple(out):
    return out if isinstance(out, (tuple, list)) else (out,)


DECODE_STYLE = {"codes": None, "z_q": None}     # 'positional' | 'keyword', remembered after the first hit


def call_decode(model, value, kind):
    """`decode(codes or z_q)`: accept either the positional or the keyword spelling."""
    styles = ([DECODE_STYLE[kind]] if DECODE_STYLE[kind] else []) + ["positional", "keyword"]
    err = None
    for style in styles:
        try:
            out = model.decode(value) if style == "positional" else model.decode(**{kind: value})
            DECODE_STYLE[kind] = style
            return out
        except Exception as exc:                 # noqa: BLE001 - we want to try the other spelling
            err = exc
    raise err


def mse(a, b):
    return float(torch.mean((a - b) ** 2))


def maxabs(a, b):
    return float(torch.max(torch.abs(a - b)))


def conv_dilation_profile(model):
    """{'encoder'|'decoder': {structural-suffix: dilation}}.

    MoMask's Resnet1D defaults to reverse_dilation=True (resnet.py:73) and the ENCODER relies on
    that default (encdec.py:27), so its encoder dilations run 3**(depth-1) ... 3, 1 -- the same
    order as the decoder. Building the encoder blocks in the forward order is the single easiest
    way to get shape-compatible but numerically different weights.
    """
    prof = {"encoder": {}, "decoder": {}}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv1d):
            continue
        n = _norm_key(name)
        if n is None or n[0] == "codebook":
            continue
        prof[n[0]].setdefault(n[1], m.dilation[0])
    return prof


def dilation_mismatch(ref, ours):
    rp, op = conv_dilation_profile(ref), conv_dilation_profile(ours)
    out = []
    for grp in ("encoder", "decoder"):
        rk = sorted(k for k, v in rp[grp].items() if v != 1)
        ok = sorted(k for k, v in op[grp].items() if v != 1)
        if sorted(rp[grp].items()) != sorted(op[grp].items()):
            out.append(f"{grp}: reference dilations "
                       f"{[(k, rp[grp][k]) for k in sorted(set(rk) | set(ok))]} vs yours "
                       f"{[(k, op[grp].get(k)) for k in sorted(set(rk) | set(ok))]}")
    return out


def train_forward_jitter(model, x, cfg):
    """max|diff| between two identical train-mode forwards, with nn.Dropout, quantise-dropout and
    EMA drift all neutralised -- i.e. purely the code-assignment stochasticity."""
    saved = zero_nn_dropout(model)
    snap = snapshot_codebooks(model)
    set_quantize_dropout_prob(model, 0.0)
    model.train()
    with torch.no_grad():
        a = as_tuple(model(x))[0].clone()
        restore_codebooks(snap)
        b = as_tuple(model(x))[0].clone()
        restore_codebooks(snap)
    model.eval()
    restore_nn_dropout(saved)
    set_quantize_dropout_prob(model, cfg["dropout_prob"])
    return maxabs(a, b)


def ref_encode_sum(ref, x):
    """MoMask encode -> (codes [B,T,Q], summed latent [B,T,D]).

    `RVQVAE.encode` returns all_codes in NCT ([Q,B,D,T], residual_vq.py:188/191) while
    `get_codes_from_indices` returns NTC -- see docs/08_rvq_spec.md section 2.5.
    """
    codes, all_codes = ref.encode(x)
    z = all_codes.sum(0).permute(0, 2, 1)          # [Q,B,D,T] -> [B,T,D]
    return codes, z


# --------------------------------------------------------------------------------------- test 1
def test_parity(rep, ref, ours, x, tol):
    ref.eval()
    ours.eval()
    with torch.no_grad():
        ref_rec, ref_commit, ref_perp = ref(x)
        ref_codes, ref_z = ref_encode_sum(ref, x)
        our_out = as_tuple(ours(x))
        our_rec = our_out[0]
        our_losses = our_out[1] if len(our_out) > 1 else {}
        enc = ours.encode(x)

    problems = []

    for msg in dilation_mismatch(ref, ours):
        problems.append("residual-block DILATION ORDER differs -> " + msg + "\n"
                        "         Weight shapes are identical either way, so this copies cleanly and "
                        "then produces different numbers.\n"
                        "         MoMask: Resnet1D(reverse_dilation=True) by default (resnet.py:73) "
                        "and the encoder uses that default (encdec.py:27).")

    if tuple(our_rec.shape) != tuple(ref_rec.shape):
        problems.append(f"x_rec shape {tuple(our_rec.shape)} != reference {tuple(ref_rec.shape)}")
    else:
        d = maxabs(our_rec, ref_rec)
        scale = float(ref_rec.abs().max())
        if d > tol:
            problems.append(f"x_rec max|diff| = {d:.3e} > tol {tol:.0e} "
                            f"(relative {d / max(scale, 1e-12):.3e})")

    if not isinstance(enc, dict):
        problems.append(f"encode() returned {type(enc).__name__}, contract says dict(codes=..., z_q=...)")
    else:
        codes = enc.get("codes")
        z_q = enc.get("z_q")
        if codes is None:
            problems.append("encode() result has no 'codes'")
        else:
            want = (x.shape[0], x.shape[1] // (2 ** CFG["down_t"]), 1, CFG["n_quant"])
            if tuple(codes.shape) != want:
                problems.append(f"codes shape {tuple(codes.shape)} != contract {want}")
            else:
                mism = int((codes[:, :, 0, :] != ref_codes).sum())
                if mism:
                    problems.append(f"{mism}/{ref_codes.numel()} code indices differ from the reference")
        if z_q is None:
            problems.append("encode() result has no 'z_q'")
        else:
            want = (x.shape[0], x.shape[1] // (2 ** CFG["down_t"]), 1, CFG["code_dim"])
            if tuple(z_q.shape) != want:
                problems.append(f"z_q shape {tuple(z_q.shape)} != contract {want}")
            else:
                d = maxabs(z_q[:, :, 0, :], ref_z)
                if d > tol:
                    problems.append(f"z_q max|diff| vs summed reference latent = {d:.3e} > tol "
                                    f"(is your z_q summed over quantisers, NTC?)")

    if isinstance(our_losses, dict) and "commit" in our_losses:
        c = float(our_losses["commit"])
        if abs(c - float(ref_commit)) > max(tol, 1e-6 * abs(float(ref_commit))):
            problems.append(f"losses['commit'] = {c:.6e} vs reference {float(ref_commit):.6e} "
                            f"(reference = mean over quantiser layers of "
                            f"mse(z_e, z_q.detach()), unweighted -- quantizer.py:147, residual_vq.py:157)")
    else:
        problems.append("forward() losses dict has no 'commit' key")

    if problems:
        rep.add("1. weight-transfer parity vs MoMask RVQVAE", "FAIL", "\n       ".join(problems))
    else:
        rep.add("1. weight-transfer parity vs MoMask RVQVAE", "PASS",
                f"x_rec max|diff| {maxabs(our_rec, ref_rec):.2e}, codes identical, "
                f"commit {float(ref_commit):.6f}")
    return not problems


# --------------------------------------------------------------------------------------- test 2
def truncated_errors(model, x, n_quant, is_reference):
    """[err(x, decode(codes[..., :k])) for k = 1..n_quant], or None if truncation is unsupported."""
    model.eval()
    errs = []
    with torch.no_grad():
        if is_reference:
            codes, _ = model.encode(x)                       # [B, T, Q]
            for k in range(1, n_quant + 1):
                errs.append(mse(model.forward_decoder(codes[..., :k].contiguous()), x))
        else:
            codes = model.encode(x)["codes"]                 # [B, T, P, Q]
            for k in range(1, n_quant + 1):
                errs.append(mse(call_decode(model, codes[..., :k].contiguous(), "codes"), x))
    return errs


def test_residual(rep, ref, ours, x, n_quant):
    try:
        ref_errs = truncated_errors(ref, x, n_quant, True)
    except Exception as exc:
        rep.add("2. residual property (more layers -> lower error)", "INCONCLUSIVE",
                f"reference control failed: {exc}")
        return False

    ref_mono = all(ref_errs[i] <= ref_errs[i - 1] * 1.001 + 1e-9 for i in range(1, n_quant))
    if not ref_mono:
        rep.add("2. residual property (more layers -> lower error)", "INCONCLUSIVE",
                f"the MoMask control itself is not monotone under this protocol "
                f"({['%.4e' % e for e in ref_errs]}); tighten the warm-up before trusting ours")
        return False

    if ours is None:
        rep.add("2. residual property (more layers -> lower error)", "SKIP", "no module")
        return False
    try:
        our_errs = truncated_errors(ours, x, n_quant, False)
    except Exception as exc:
        rep.add("2. residual property (more layers -> lower error)", "FAIL",
                f"decode(codes[..., :k]) with k < n_quant raised {type(exc).__name__}: {exc}\n"
                f"       MoMask supports this: get_codes_from_indices right-pads the missing "
                f"quantiser columns with -1 and zeroes those code vectors "
                f"(residual_vq.py:71-72, 81-89). Please accept truncated code stacks.")
        return False

    bad = [i for i in range(1, n_quant) if our_errs[i] > our_errs[i - 1] * 1.001 + 1e-9]
    fmt = ", ".join(f"k={i + 1}: {e:.4e}" for i, e in enumerate(our_errs))
    trivial = mse(torch.zeros_like(x), x)
    if our_errs[0] > 0.9 * trivial:
        rep.add("2. residual property (more layers -> lower error)", "INCONCLUSIVE",
                f"the fitted model does not reconstruct at all (k=1 error {our_errs[0]:.4e} vs "
                f"{trivial:.4e} for predicting zero), so the comparison is vacuous: {fmt}")
        return False
    if bad:
        rep.add("2. residual property (more layers -> lower error)", "FAIL",
                f"error increases when adding layer(s) {[b + 1 for b in bad]}: {fmt}\n"
                f"       reference: " + ", ".join(f"k={i + 1}: {e:.4e}" for i, e in enumerate(ref_errs)))
        return False
    if not our_errs[-1] < our_errs[0]:
        rep.add("2. residual property (more layers -> lower error)", "FAIL",
                f"{n_quant} layers is not better than 1: {fmt}")
        return False
    rep.add("2. residual property (more layers -> lower error)", "PASS", fmt)
    return True


# --------------------------------------------------------------------------------------- test 3
def test_codes_and_roundtrip(rep, ours, x, cfg, tol):
    problems = []
    ours.eval()
    with torch.no_grad():
        e1 = ours.encode(x)
        e2 = ours.encode(x)
        c1, c2 = e1["codes"], e2["codes"]

        if c1.dtype not in (torch.int64, torch.int32, torch.long):
            problems.append(f"codes dtype {c1.dtype}, contract says long")
        lo, hi = int(c1.min()), int(c1.max())
        if lo < 0 or hi >= cfg["nb_code"]:
            problems.append(f"code range [{lo}, {hi}] outside [0, {cfg['nb_code']})")
        if not torch.equal(c1, c2):
            problems.append(f"encode() is not deterministic in eval mode: "
                            f"{int((c1 != c2).sum())} indices differ. MoMask's tokenisation path "
                            f"uses temperature 0 -> plain argmax (residual_vq.py:178, quantizer.py:78).")

        r1 = call_decode(ours, c1, "codes")
        r2 = call_decode(ours, c1, "codes")
        if not torch.equal(r1, r2):
            problems.append(f"decode() is not deterministic: max|diff| {maxabs(r1, r2):.3e}")
        if tuple(r1.shape) != tuple(x.shape):
            problems.append(f"decode(codes) shape {tuple(r1.shape)} != input {tuple(x.shape)}")

        fwd = as_tuple(ours(x))[0]
        if tuple(fwd.shape) == tuple(r1.shape):
            d = maxabs(fwd, r1)
            if d > tol:
                problems.append(f"decode(encode(x)) differs from forward(x) by {d:.3e} in eval mode "
                                f"(both should be the deterministic argmax path)")

        # decode() must also take z_q, per the contract
        try:
            rz = call_decode(ours, e1["z_q"], "z_q")
            if tuple(rz.shape) != tuple(x.shape):
                problems.append(f"decode(z_q) shape {tuple(rz.shape)} != input {tuple(x.shape)}")
            elif maxabs(rz, r1) > tol:
                problems.append(f"decode(z_q) differs from decode(codes) by {maxabs(rz, r1):.3e}")
        except Exception as exc:
            problems.append(f"decode(z_q) raised {type(exc).__name__}: {exc} "
                            f"(contract says decode takes codes or z_q)")

    if problems:
        rep.add("3. code validity + round-trip determinism", "FAIL", "\n       ".join(problems))
        return False
    rep.add("3. code validity + round-trip determinism", "PASS",
            f"codes {tuple(c1.shape)} {c1.dtype} in [{lo}, {hi}] < {cfg['nb_code']}; "
            f"decode(encode(x)) == forward(x) to {maxabs(fwd, r1):.1e}")
    return True


# --------------------------------------------------------------------------------------- test 4
DEPTH_KEYS = ("n_quant_active", "active_layers", "n_active", "kept_layers", "kept_depth",
              "quant_depth", "dropout_depth", "start_drop_quantize_index")


def _per_part_depths(losses, n_quant):
    """Kept depth per part, read out of the losses dict; None if the model does not expose it.

    `perplexity_per_layer` works because a live layer's perplexity is exp(entropy) >= 1, while a
    dropped one is recorded as exactly 0.
    """
    if not isinstance(losses, dict):
        return None
    ppl = losses.get("perplexity_per_layer")
    if torch.is_tensor(ppl) and ppl.shape[-1] == n_quant:
        m = ppl.reshape(-1, n_quant)
        return [int((row > 0).sum()) for row in m]
    for k in DEPTH_KEYS:
        if k in losses:
            v = losses[k]
            v = v.reshape(-1).tolist() if torch.is_tensor(v) else [v]
            return [int(u) + (1 if k == "start_drop_quantize_index" else 0) for u in v]
    if "codes" in losses and torch.is_tensor(losses["codes"]):
        c = losses["codes"]
        return [int((c.reshape(-1, c.shape[-1]) >= 0).all(0).sum())]
    return None


def _depth_from_losses(losses, n_quant):
    d = _per_part_depths(losses, n_quant)
    return None if d is None else d[0]


def _depth_by_error(model, x, ref_errs, n_quant, trials, cfg, p):
    """Classify each train-mode forward by which truncated-reconstruction error it lands on.

    nn.Dropout is silenced and the codebooks restored after every call, so the only thing that
    moves between trials is the quantise-dropout draw (plus MoMask's Gumbel sampling at
    temperature 0.5, model.py:72, which perturbs but does not reorder the depth modes).
    """
    saved = zero_nn_dropout(model)
    snap = snapshot_codebooks(model)
    set_quantize_dropout_prob(model, p)
    model.train()
    counts = Counter()
    exposed = Counter()
    have_exposed = True
    with torch.no_grad():
        for _ in range(trials):
            out = as_tuple(model(x))
            losses = out[1] if len(out) > 1 else None
            d = _depth_from_losses(losses, n_quant)
            if d is None:
                have_exposed = False
            else:
                exposed[d] += 1
            e = mse(out[0], x)
            counts[1 + int(np.argmin([abs(e - r) for r in ref_errs]))] += 1
            restore_codebooks(snap)
    model.eval()
    restore_nn_dropout(saved)
    set_quantize_dropout_prob(model, cfg["dropout_prob"])
    return counts, (exposed if have_exposed else None)


def _ref_depth_distribution(ref, x, trials, cfg, p):
    """Ground truth for the reference: -1 columns in all_indices (residual_vq.py:119, 134)."""
    saved = zero_nn_dropout(ref)
    snap = snapshot_codebooks(ref)
    ref.quantizer.quantize_dropout_prob = p
    ref.train()
    counts = Counter()
    errs_by_depth = {}
    with torch.no_grad():
        for _ in range(trials):
            z = ref.encoder(x.permute(0, 2, 1).float())
            q, idx, _, _ = ref.quantizer(z, sample_codebook_temp=0.5)
            depth = int((idx.reshape(-1, idx.shape[-1]) >= 0).all(0).sum())
            counts[depth] += 1
            errs_by_depth.setdefault(depth, []).append(mse(ref.decoder(q), x))
            restore_codebooks(snap)
    ref.eval()
    restore_nn_dropout(saved)
    ref.quantizer.quantize_dropout_prob = cfg["dropout_prob"]
    return counts, errs_by_depth


def test_train_time_noise(rep, ref, ours, x, cfg):
    """Two MoMask train-time behaviours that eval-mode parity cannot see."""
    problems = []

    ref_jit = train_forward_jitter(ref, x, cfg)
    our_jit = train_forward_jitter(ours, x, cfg)
    if ref_jit > 0 and our_jit == 0:
        problems.append(
            f"train-mode code assignment is DETERMINISTIC in yours (two identical forwards differ "
            f"by {our_jit:.1e}) but STOCHASTIC in MoMask ({ref_jit:.3e}). MoMask hard-codes "
            f"sample_codebook_temp=0.5 in RVQVAE.forward (model.py:72) and draws the code with "
            f"Gumbel noise while training (quantizer.py:78); tokenisation stays argmax "
            f"(residual_vq.py:178). Deliberate simplification, or an oversight?")
    elif ref_jit > 0 and our_jit > 0:
        pass
    elif ref_jit == 0:
        problems.append(f"reference control is deterministic too ({ref_jit:.1e}); cannot judge")

    ref_do = [m.p for m in ref.modules() if isinstance(m, nn.Dropout) and m.p > 0]
    our_do = [m.p for m in ours.modules() if isinstance(m, nn.Dropout) and m.p > 0]
    if ref_do and not our_do:
        problems.append(f"no nn.Dropout anywhere in your model; MoMask puts Dropout(p=0.2) in every "
                        f"residual block ({len(ref_do)} of them here, resnet.py:13/46/67). Also a "
                        f"train-time-only difference, and it is the only regulariser MoMask's "
                        f"tokenizer has (weight_decay is 0.0, vq_option.py:22).")

    if problems:
        rep.add("5. train-time behaviours invisible to eval parity", "FAIL", "\n       ".join(problems))
        return False
    rep.add("5. train-time behaviours invisible to eval parity", "PASS",
            f"stochastic assignment jitter {our_jit:.3e} (reference {ref_jit:.3e}); "
            f"{len(our_do)} active nn.Dropout modules (reference {len(ref_do)})")
    return True


def test_dropout_granularity(rep, cls, cfg, device, data, trials=200):
    """MoMask draws the dropout depth ONCE per forward (residual_vq.py:112-117).

    With a part-structured model there are n_parts residual stacks; if each draws its own depth
    the training regime is no longer MoMask's.
    """
    half = cfg["input_dim"] // 2
    chans = [np.arange(0, half, dtype=np.int64), np.arange(half, cfg["input_dim"], dtype=np.int64)]
    try:
        m = build_ours(cls, cfg, chans).to(device)
    except Exception as exc:
        rep.add("4b. dropout draw granularity (MoMask: one draw per forward)", "INCONCLUSIVE",
                f"could not build a 2-part model: {exc}")
        return
    set_quantize_dropout_prob(m, 0.0)
    m.train()
    with torch.no_grad():
        for i in range(5):
            m(data[i % len(data)])
    set_quantize_dropout_prob(m, 1.0)
    same = diff = 0
    with torch.no_grad():
        for i in range(trials):
            out = as_tuple(m(data[i % len(data)]))
            d = _per_part_depths(out[1] if len(out) > 1 else None, cfg["n_quant"])
            if d is None or len(d) < 2:
                rep.add("4b. dropout draw granularity (MoMask: one draw per forward)", "INCONCLUSIVE",
                        "forward()'s losses dict does not report a per-part kept depth "
                        "(expected e.g. 'perplexity_per_layer' shaped [n_parts, n_quant] with 0 "
                        "for dropped layers)")
                return
            same += int(len(set(d)) == 1)
            diff += int(len(set(d)) > 1)
    if diff == 0:
        rep.add("4b. dropout draw granularity (MoMask: one draw per forward)", "PASS",
                f"all {trials} forwards dropped the same depth in both parts -> one shared draw")
    else:
        rep.add("4b. dropout draw granularity (MoMask: one draw per forward)", "FAIL",
                f"{diff}/{trials} forwards gave the two parts DIFFERENT kept depths, so each part's "
                f"ResidualVQ draws its own dropout depth. MoMask draws once per forward and applies "
                f"it to the whole (single) stack (residual_vq.py:112-117). Flag this back to me if "
                f"independent per-part draws are the intended design -- it is a defensible choice, "
                f"but it is not MoMask's, and it changes the effective dropout rate seen by the "
                f"downstream model from p to 1-(1-p)^n_parts.")


def test_dropout(rep, ref, ours, x, cfg, trials=600):
    n_quant = cfg["n_quant"]

    # (a) eval mode must not drop anything.
    ours.eval()
    with torch.no_grad():
        codes = ours.encode(x)["codes"]
        eval_rec = as_tuple(ours(x))[0]
    if int((codes < 0).sum()):
        rep.add("4. quantise-dropout", "FAIL",
                "encode() produced negative (dropped) indices in eval mode; MoMask's tokenisation "
                "path has no dropout at all (residual_vq.py:171-194)")
        return False

    # train-mode encode must also be dropout-free (MoMask: ResidualVQ.quantize ignores dropout)
    saved = zero_nn_dropout(ours)
    snap = snapshot_codebooks(ours)
    set_quantize_dropout_prob(ours, 1.0)
    ours.train()
    with torch.no_grad():
        tr_codes = ours.encode(x)["codes"]
    restore_codebooks(snap)
    ours.eval()
    restore_nn_dropout(saved)
    set_quantize_dropout_prob(ours, cfg["dropout_prob"])
    if int((tr_codes < 0).sum()):
        rep.add("4. quantise-dropout", "FAIL",
                "encode() drops layers in train mode. MoMask keeps tokenisation dropout-free: "
                "ResidualVQ.quantize() has no dropout branch (residual_vq.py:171-194); only "
                "ResidualVQ.forward() drops (residual_vq.py:112-136).")
        return False

    # (b) calibrate the error-mode classifier on the reference.
    try:
        ref_errs = truncated_errors(ref, x, n_quant, True)
        ref_counts, ref_errs_by_depth = _ref_depth_distribution(ref, x, trials, cfg, 1.0)
    except Exception as exc:
        rep.add("4. quantise-dropout", "INCONCLUSIVE", f"reference control failed: {exc}")
        return False
    mis = 0
    tot = 0
    for depth, es in ref_errs_by_depth.items():
        for e in es:
            tot += 1
            mis += int(1 + int(np.argmin([abs(e - r) for r in ref_errs])) != depth)
    classifier_ok = tot and (1 - mis / tot) >= 0.95

    our_errs = truncated_errors(ours, x, n_quant, False)

    # (c) prob = 0 -> never drop;  prob = 1 -> kept depth uniform on {1..n_quant}.
    c0, e0 = _depth_by_error(ours, x, our_errs, n_quant, max(60, trials // 6), cfg, 0.0)
    c1, e1 = _depth_by_error(ours, x, our_errs, n_quant, trials, cfg, 1.0)
    used, obs0, obs1 = ("exposed", e0, e1) if (e0 is not None and e1 is not None) \
        else ("error-mode", c0, c1)

    if used == "error-mode" and not classifier_ok:
        rep.add("4. quantise-dropout", "INCONCLUSIVE",
                f"forward() does not report the kept quantiser depth (looked for {DEPTH_KEYS[:4]} "
                f"in the losses dict) and the error-mode fallback only classifies the reference "
                f"correctly {100 * (1 - mis / max(tot, 1)):.0f}% of the time.\n"
                f"       Please expose the kept depth in the losses dict so this can be checked.")
        return False

    problems = []
    n0 = sum(v for k, v in obs0.items() if k < n_quant)
    if n0 > 0.02 * sum(obs0.values()):
        problems.append(f"dropout_prob=0 still drops layers in {n0}/{sum(obs0.values())} forwards "
                        f"(depths seen: {dict(sorted(obs0.items()))})")

    total = sum(obs1.values())
    expected = total / n_quant
    missing = [k for k in range(1, n_quant + 1) if obs1.get(k, 0) == 0]
    if missing:
        problems.append(f"dropout_prob=1: kept depths {missing} never occur. MoMask draws "
                        f"start = randrange(0, n_quant) and keeps start+1 layers, so every depth in "
                        f"1..{n_quant} must appear (residual_vq.py:117, 133).")
    else:
        chi2 = sum((obs1.get(k, 0) - expected) ** 2 / expected for k in range(1, n_quant + 1))
        # 99.9th percentile of chi2 with n_quant-1 dof; generous, we only want gross deviations
        crit = {1: 10.8, 2: 13.8, 3: 16.3, 4: 18.5, 5: 20.5, 6: 22.5}.get(n_quant - 1, 25.0)
        if chi2 > crit:
            problems.append(f"dropout_prob=1: kept-depth histogram {dict(sorted(obs1.items()))} is "
                            f"not uniform on 1..{n_quant} (chi2 {chi2:.1f} > {crit}); MoMask's is "
                            f"uniform (measured, docs/08_rvq_spec.md section 2.4)")

    detail = (f"via {used}; prob=0 -> {dict(sorted(obs0.items()))}; "
              f"prob=1 -> {dict(sorted(obs1.items()))} "
              f"(reference prob=1 -> {dict(sorted(ref_counts.items()))})")
    if problems:
        rep.add("4. quantise-dropout", "FAIL", "\n       ".join(problems) + "\n       " + detail)
        return False
    rep.add("4. quantise-dropout", "PASS", detail)
    return True


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default; deterministic, and the model here is tiny) or cuda")
    ap.add_argument("--tol", type=float, default=1e-5, help="parity tolerance on max|diff|")
    ap.add_argument("--seed", type=int, default=3407, help="MoMask's seed (vq_option.py:66)")
    ap.add_argument("--trials", type=int, default=600, help="forwards used for the dropout statistics")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(False)
    device = torch.device(args.device)

    rep = Report()
    print(f"repo      : {REPO}")
    print(f"reference : {VENDOR_MOMASK}")
    print(f"device    : {device}   torch {torch.__version__}")
    print(f"config    : {CFG}, batch {BATCH}, frames {FRAMES}\n")

    cls, where = import_ours()
    if cls is None:
        rep.add("0. import hml_phys.rvq.PartRVQVAE", "SKIP", where)
        print("\nNothing to test yet -- the model file is not in the tree. "
              "Re-run once hml_phys/rvq.py lands.")
        rep.summary()
        return 0
    rep.add("0. import hml_phys.rvq.PartRVQVAE", "PASS", f"from {where}")

    RVQVAE = import_reference()
    gen = torch.Generator(device=device).manual_seed(args.seed)
    data = [smooth_motion(BATCH, FRAMES, CFG["input_dim"], device, gen) for _ in range(8)]
    x = data[0]

    # ---- reference, briefly fitted so its EMA codebooks are live and it actually reconstructs
    ref = build_reference(RVQVAE, CFG).to(device)
    ref_loss = fit(ref, data, FIT_STEPS, True, CFG)
    with torch.no_grad():
        print(f"reference fitted: loss {ref_loss:.4f}, eval recon MSE "
              f"{mse(ref(x)[0], x):.4e} (predicting zero: {mse(torch.zeros_like(x), x):.4e})\n")

    # ---- ours
    try:
        ours = build_ours(cls, CFG, [np.arange(CFG["input_dim"], dtype=np.int64)]).to(device)
    except Exception:
        try:
            ours = build_ours(cls, CFG, [list(range(CFG["input_dim"]))]).to(device)
        except Exception:
            rep.add("0b. construct PartRVQVAE", "FAIL",
                    "constructor raised with both a numpy and a list part_channels:\n"
                    + traceback.format_exc())
            rep.summary()
            return 1
    rep.add("0b. construct PartRVQVAE", "PASS",
            f"{sum(p.numel() for p in ours.parameters()) / 1e3:.1f}k params vs reference "
            f"{sum(p.numel() for p in ref.parameters()) / 1e3:.1f}k")

    ok, detail = copy_reference_weights(ref, ours, CFG)
    if ok:
        rep.add("0c. weight-copy helper (MoMask -> PartRVQVAE)", "PASS", detail)
        test_parity(rep, ref, ours, x, args.tol)
    else:
        rep.add("0c. weight-copy helper (MoMask -> PartRVQVAE)", "FAIL", detail)
        rep.add("1. weight-transfer parity vs MoMask RVQVAE", "SKIP",
                "cannot run without a weight mapping; see 0c")
        fit(ours, data, FIT_STEPS, False, CFG)   # so tests 2-4 have a model that reconstructs

    test_residual(rep, ref, ours, x, CFG["n_quant"])
    test_codes_and_roundtrip(rep, ours, x, CFG, args.tol)
    try:
        test_dropout(rep, ref, ours, x, CFG, trials=args.trials)
    except Exception:
        rep.add("4. quantise-dropout", "INCONCLUSIVE", traceback.format_exc())
    try:
        test_dropout_granularity(rep, cls, CFG, device, data)
    except Exception:
        rep.add("4b. dropout draw granularity (MoMask: one draw per forward)", "INCONCLUSIVE",
                traceback.format_exc())
    try:
        test_train_time_noise(rep, ref, ours, x, CFG)
    except Exception:
        rep.add("5. train-time behaviours invisible to eval parity", "INCONCLUSIVE",
                traceback.format_exc())

    return 1 if rep.summary() else 0


if __name__ == "__main__":
    sys.exit(main())
