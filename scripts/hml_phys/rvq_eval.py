"""Evaluate a trained residual VQ-VAE tokeniser on the HumanML3D-physics TEST split (docs/08).

ONE pass over the split, ONE metric computation -- no repetitions, no seeds, no confidence intervals
(project CLAUDE.md §4).  The val split is never touched (§1).

What is reported, for the channel variant the checkpoint was trained on (hml_phys/rvq_data.py):

1. Reconstruction error per CHANNEL GROUP, in physical units (de-normalised with token_stats_v3.npz):
       root_trans        mm      mean L2 of the 3-vector error
       root_rot_6d       deg     geodesic angle after Gram-Schmidt (+ raw RMS of the 6 entries)
       root_trans_vel    m/s     mean L2  (the token stores the recorded root velocity, already m/s)
       root_rot_vel      rad/s   mean |.|
       local_positions   mm      mean L2 per joint over 24 joints -- the MPJPE-style number
                                 scripts/hml_phys/train_intent_vae.py prints
       local_vel         m/s     mean L2 per joint; the token stores m/frame at 30 fps, so x30
       dof_pose_6d       deg     geodesic angle per joint (+ raw RMS)
       dof_vel           rad/s   mean |.|
       action            RMS in raw PD-action units, plus the same error expressed as a fraction of the
                                 action scale under BOTH readings of "scale":
                                   .pd_target_rad/.pd_target_deg = pd_action_scale * error, i.e. the error of
                                     the PD target the action encodes (pd_tar = pd_offset + pd_scale * a;
                                     our pd_offset is 0 and pd_scale is 5.0 or pi per joint axis)
                                   .frac_of_data_std = error / per-channel std of the action channels in
                                     token_stats_v3, i.e. the error relative to the scale the actions actually
                                     occupy in the data (1.0 would mean "as bad as predicting the mean")
   plus the normalised-space MSE overall and per part.

2. Codebook statistics per CODE GROUP (one group under --structure whole) and per RESIDUAL LAYER: usage (fraction of the nb_code entries ever selected),
   perplexity exp(-sum p log p) over the empirical code histogram, and perplexity / nb_code.

3. The same physical-unit table when only the first k of the n_quant residual layers are decoded, k = 1..n_quant
   (PartRVQVAE.encode(x, n_layers=k) -> PartRVQVAE.decode(codes=...)).  k = n_quant is cross-checked against
   PartRVQVAE.forward, and the max abs difference is reported.

Checkpoint: dict(model=<state_dict>, args=<vars(args) of train_rvq.py>, parts=[per-part channel arrays], ...).
`--self_test` runs the whole metric path against a stub with the same API, so the script can be exercised
without a trained checkpoint.
"""
import argparse, importlib, json, math, os, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import ROOT, TokenStats, load_env_constants
from hml_phys.rvq_data import DEFAULT_STATS, DEFAULT_WINDOW, DOWNSAMPLE, FEATURE_UNITS, RVQWindowDataset, token_mean_std

REC_KEYS = ("rec", "recon", "reconstruction", "x_rec", "x_hat", "pred")


def _as_rec(out):
    """model output -> the reconstruction tensor (PartRVQVAE.forward returns (rec, stats))."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for k in REC_KEYS:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
    if isinstance(out, (list, tuple)):
        for o in out:
            if torch.is_tensor(o) and o.dim() >= 3:
                return o
    raise RuntimeError(f"cannot find the reconstruction tensor in a model output of type {type(out)}")


def _as_codes(out):
    """PartRVQVAE.encode output -> code tensor [B, T', n_parts, n_quant]."""
    if isinstance(out, dict):
        return out["codes"]
    if isinstance(out, (list, tuple)):
        return out[0] if torch.is_tensor(out[0]) else torch.stack(list(out), 2)
    return out


# --------------------------------------------------------------------------------------- metrics
def rot6d_to_matrix(v):
    """[..., 6] = [M00, M01, M10, M11, M20, M21] (the first two COLUMNS) -> [..., 3, 3] via Gram-Schmidt."""
    a = v.reshape(*v.shape[:-1], 3, 2)
    b1 = a[..., 0]
    b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    a2 = a[..., 1]
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def geodesic_deg(v_gt, v_pr):
    A, B = rot6d_to_matrix(v_gt), rot6d_to_matrix(v_pr)
    cos = (((A * B).sum((-2, -1)) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.arccos(cos) * (180.0 / math.pi)


class Acc:
    """weighted-mean accumulator; keys ending in '^2' are reported as sqrt(mean) (an RMS)."""

    def __init__(self):
        self.s, self.n = {}, 0.0

    def add(self, vals, w):
        for k, v in vals.items():
            self.s[k] = self.s.get(k, 0.0) + float(v) * w
        self.n += w

    def result(self):
        out = {}
        for k, v in self.s.items():
            m = v / max(self.n, 1e-9)
            out[k[:-2] if k.endswith("^2") else k] = math.sqrt(max(m, 0.0)) if k.endswith("^2") else m
        return out


def group_metrics(xs, rs, fgroups, pd_scale_act, act_std):
    """xs, rs: PHYSICAL-unit [B, T, C].  -> per-batch means; keys ending '^2' are means of squares."""
    out = {}
    d = xs - rs
    for name, loc in fgroups.items():
        unit, w = FEATURE_UNITS[name]
        dd = d[..., loc]
        e = dd.reshape(*dd.shape[:-1], -1, w)
        out[f"{name}.rms^2"] = (dd ** 2).mean()
        if unit == "m":
            out[f"{name}.mm"] = e.norm(dim=-1).mean() * 1000.0
        elif unit == "m/s":
            out[f"{name}.m_s"] = e.norm(dim=-1).mean()
        elif unit == "m/frame":
            out[f"{name}.m_s"] = e.norm(dim=-1).mean() * 30.0
        elif unit == "rad/s":
            out[f"{name}.rad_s"] = dd.abs().mean()
        elif unit == "rot6d":
            out[f"{name}.deg"] = geodesic_deg(xs[..., loc].reshape(*dd.shape[:-1], -1, 6),
                                              rs[..., loc].reshape(*dd.shape[:-1], -1, 6)).mean()
        elif unit == "pd":
            out[f"{name}.mean_abs"] = dd.abs().mean()
            out[f"{name}.pd_target_rad^2"] = ((dd * pd_scale_act) ** 2).mean()
            out[f"{name}.frac_of_data_std^2"] = ((dd / act_std) ** 2).mean()
    return out


def part_metrics(xn, rn, names, groups):
    """normalised-space MSE overall and per part."""
    d = (xn - rn) ** 2
    out = {"all.mse": d.mean(), "all.l1": (xn - rn).abs().mean()}
    for n, g in zip(names, groups):
        out[f"{n}.mse"] = d[..., g].mean()
    return out


def finalise(acc, pd_scale_mean):
    r = acc.result()
    if "action.pd_target_rad" in r:
        r["action.pd_target_deg"] = r["action.pd_target_rad"] * 180.0 / math.pi
        r["action.pd_scale_mean"] = pd_scale_mean
    return r


ORDER = ["root_trans.mm", "root_rot_6d.deg", "root_rot_6d.rms", "root_trans_vel.m_s", "root_rot_vel.rad_s",
         "local_positions.mm", "local_vel.m_s", "dof_pose_6d.deg", "dof_pose_6d.rms", "dof_vel.rad_s",
         "action.rms", "action.mean_abs", "action.frac_of_data_std", "action.pd_target_rad", "action.pd_target_deg"]


def fmt_group(r):
    return "  ".join(f"{k}={r[k]:.4g}" for k in ORDER if k in r)


# --------------------------------------------------------------------------------------- stub for --self_test
class _StubRVQ(torch.nn.Module):
    """Stand-in with PartRVQVAE's API, to exercise the metric path without a trained checkpoint."""

    def __init__(self, dim, groups, n_quant=6, nb_code=512, down=DOWNSAMPLE, seed=0):
        super().__init__()
        self.n_parts, self.n_quant, self.nb_code, self.down = len(groups), n_quant, nb_code, down
        self.groups = [torch.as_tensor(np.asarray(g)) for g in groups]
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("noise", torch.randn(dim, generator=g) * 0.05)

    def _rec(self, x, n_layers=None):
        k = self.n_quant if n_layers is None else int(n_layers)
        return x + self.noise * (self.n_quant / max(k, 1))

    def forward(self, x):
        return self._rec(x), dict(commit=x.new_zeros(()), perplexity=x.new_zeros(()))

    @torch.no_grad()
    def encode(self, x, n_layers=None):
        B, T, _ = x.shape
        t, k = T // self.down, self.n_quant if n_layers is None else int(n_layers)
        cols = []
        for g in self.groups:
            h = x[:, : t * self.down, g.to(x.device)].reshape(B, t, self.down, -1).mean((2, 3)) * 97.0
            cols.append(torch.stack([(h.long() + 13 * q).abs() % self.nb_code for q in range(k)], -1))
        return dict(codes=torch.stack(cols, 2), z_q=None)   # [B, t, n_parts, k]

    @torch.no_grad()
    def decode(self, codes=None, z_q=None):
        k = codes.shape[-1]
        B, t = codes.shape[0], codes.shape[1]
        x = torch.zeros(B, t * self.down, self.noise.shape[0], device=codes.device)
        return self._rec(x, k)                              # content is irrelevant; only the plumbing is tested


# --------------------------------------------------------------------------------------- model construction
def build_model(ck, cargs, ds, dev):
    rvq = importlib.import_module("hml_phys.rvq")
    cls = getattr(rvq, "PartRVQVAE")
    parts = ck.get("parts")
    parts = [np.asarray(p, dtype=np.int64) for p in parts] if parts is not None else list(ds.part_groups)
    kw = dict(width=int(cargs.get("width", 512)), down_t=int(cargs.get("down_t", 2)),
              depth=int(cargs.get("depth", 3)), dilation=int(cargs.get("dilation", 3)),
              code_dim=int(cargs.get("code_dim", 512)), nb_code=int(cargs.get("nb_code", 512)),
              n_quant=int(cargs.get("n_quant", 6)), shared_codebook=bool(cargs.get("shared_codebook", 0)),
              dropout_prob=float(cargs.get("dropout_prob", 0.2)), mu=float(cargs.get("mu", 0.99)))
    model = cls(int(cargs.get("n_channels", ds.n_channels)), parts, **kw).to(dev)
    model.load_state_dict(ck["model"])      # strict: on a silent mismatch the codebooks would stay random and every
    return model                            # number below would still look plausible


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="checkpoint saved by scripts/hml_phys/train_rvq.py")
    ap.add_argument("--split", default="test", help="test only; val is banned (CLAUDE.md §1)")
    ap.add_argument("--variant", default="", help="override ckpt['args']['variant']")
    ap.add_argument("--window", type=int, default=0, help="override ckpt['args']['window']")
    ap.add_argument("--stride", type=int, default=0, help="window stride; 0 = the window length, a tiling that "
                                                          "tiles each clip from the start; the tail frames of a clip that do not fill a whole "
                                                          "window, and clips shorter than the window, are skipped")
    ap.add_argument("--stats", default="", help="override ckpt['args']['stats']")
    ap.add_argument("--env_constants", default=os.path.join(ROOT, "env_constants.npz"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_items", type=int, default=0, help="subset the pass for a smoke test (0 = whole split)")
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--no_layers", action="store_true", help="skip the first-k-layers table")
    ap.add_argument("--self_test", action="store_true", help="run against a stub model (no checkpoint needed)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="", help="name for the report file (default: the checkpoint's basename)")
    ap.add_argument("--out", default="", help="JSON report; default <ckpt dir>/rvq_eval_<split>_<tag>.json")
    args = ap.parse_args()
    assert args.split != "val", "the val split is banned in this project (CLAUDE.md §1)"
    assert args.ckpt or args.self_test, "--ckpt is required (or --self_test)"

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ck, cargs = None, {}
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        cargs = dict(ck.get("args", {}) or {})
    variant = args.variant or cargs.get("variant", "action")
    window = args.window or int(cargs.get("window", DEFAULT_WINDOW))
    stats_path = args.stats or cargs.get("stats", cargs.get("stats_path", DEFAULT_STATS))
    stride = args.stride if args.stride > 0 else window
    down = 2 ** int(cargs.get("down_t", 2))

    ds = RVQWindowDataset(args.split, variant=variant, window=window, stride=stride, downsample=down,
                          stats_path=stats_path, max_clips=args.max_clips, max_windows=args.max_items)
    print(ds.describe(), flush=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.self_test:
        model = _StubRVQ(ds.n_channels, ds.part_groups, down=down).to(dev)
        print("[self_test] stub model, no checkpoint loaded", flush=True)
    else:
        try:
            model = build_model(ck, cargs, ds, dev)
        except ModuleNotFoundError as e:
            print(f"hml_phys/rvq.py is not importable yet ({e}) -- nothing to evaluate. "
                  f"Use --self_test to exercise the metric code path.", flush=True)
            return 0
    model.eval()
    n_q = int(getattr(model, "n_quant", cargs.get("n_quant", 6)))
    nb_code = int(getattr(model, "nb_code", cargs.get("nb_code", 2048)))
    # The codebook axis belongs to the MODEL, not to the dataset's body-part split: under --structure whole there is
    # exactly one codebook stack over every channel.  Reading it off ds.part_names printed the one real codebook under
    # the label "root" and five all-zero ghost rows.
    n_parts = int(getattr(model, "n_parts", len(ds.part_names)))
    group_names = ["all"] if n_parts == 1 else list(ds.part_names[:n_parts])
    group_dims = [int(d) for d in getattr(model, "dims", [len(g) for g in ds.part_groups])]
    print(f"model: {n_parts} code group(s) {group_names}, {n_q} residual layers, {nb_code} codes, "
          f"{window}->{window // down} frames"
          + (f", {sum(p.numel() for p in model.parameters())/1e6:.1f}M params" if not args.self_test else ""), flush=True)

    # ---- physical-unit constants
    _, tstd = token_mean_std(TokenStats(stats_path))
    mean = torch.from_numpy(ds.mean.copy()).to(dev)
    std = torch.from_numpy(ds.std.copy()).to(dev)
    act_loc = ds.feature_groups.get("action")
    if act_loc is not None:
        dofs = ds.dof_index()
        pd_scale = torch.from_numpy(load_env_constants(args.env_constants)["pd_scale"][dofs].astype(np.float32)).to(dev)
        act_std = torch.from_numpy(tstd[ds.channels[act_loc]].copy()).to(dev)
        pd_scale_mean = float(pd_scale.mean())
    else:
        pd_scale, act_std, pd_scale_mean = None, None, float("nan")

    full, parts_acc = Acc(), Acc()
    layer_accs = {} if args.no_layers else {k: Acc() for k in range(1, n_q + 1)}
    counts = torch.zeros(n_parts, n_q, nb_code, dtype=torch.float64)
    enc_ok, layers_ok, n_codes = True, bool(layer_accs), 0
    fwd_vs_full = 0.0
    t0, n_items, n_frames = time.time(), 0, 0

    with torch.no_grad():
        for x in loader:
            x = (x if torch.is_tensor(x) else x["x"]).to(dev, non_blocking=True)
            B = x.shape[0]
            xs = x * std + mean
            r = _as_rec(model(x))
            full.add(group_metrics(xs, r * std + mean, ds.feature_groups, pd_scale, act_std), B)
            parts_acc.add(part_metrics(x, r, ds.part_names, ds.part_groups), B)

            if layers_ok:
                try:
                    for k in range(1, n_q + 1):
                        rk = _as_rec(model.decode(codes=_as_codes(model.encode(x, n_layers=k))))
                        layer_accs[k].add({**group_metrics(xs, rk * std + mean, ds.feature_groups, pd_scale, act_std),
                                           **part_metrics(x, rk, ds.part_names, ds.part_groups)}, B)
                        if k == n_q:
                            fwd_vs_full = max(fwd_vs_full, float((rk - r).abs().max()))
                except Exception as e:                              # noqa: BLE001
                    print(f"[warn] the first-k-layers table is unavailable ({type(e).__name__}: {e})", flush=True)
                    layers_ok, layer_accs = False, {}

            if enc_ok:
                try:
                    c = _as_codes(model.encode(x)).permute(0, 2, 3, 1).cpu()    # [B, T', P, Q] -> [B, P, Q, T']
                    for p in range(min(n_parts, c.shape[1])):
                        for q in range(min(n_q, c.shape[2])):
                            v = c[:, p, q].reshape(-1)
                            counts[p, q] += torch.bincount(v[v >= 0].clamp(0, nb_code - 1).long(),
                                                           minlength=nb_code).double()
                    n_codes += c.shape[0] * c.shape[3]
                except Exception as e:                              # noqa: BLE001
                    print(f"[warn] codebook statistics unavailable ({type(e).__name__}: {e})", flush=True)
                    enc_ok = False
            n_items += B; n_frames += B * window

    dt = time.time() - t0
    rep = dict(ckpt=args.ckpt, split=args.split, variant=variant, window=window, stride=stride, n_quant=n_q,
               nb_code=nb_code, code_groups=group_names, code_group_dims=group_dims,
               body_parts=ds.part_names, body_part_dims=[len(g) for g in ds.part_groups],
               n_windows=n_items, n_frames=n_frames, stats=stats_path, seconds=dt,
               protocol="single pass over the split, single metric computation (CLAUDE.md §4)",
               canonical_frame="window frame 0 (origin=0); self-contained window",
               forward_vs_encode_decode_maxabs=fwd_vs_full,
               reconstruction=finalise(full, pd_scale_mean), parts_mse=parts_acc.result())

    print(f"\n=== RVQ reconstruction ({args.split}, variant {variant}, window {window}, stride {stride}) ===")
    print(f"{n_items} windows / {n_frames} frames in {dt:.1f}s ({1000 * dt / max(n_items, 1):.2f} ms/window)")
    print("physical units:", fmt_group(rep["reconstruction"]))
    print("normalised    :", "  ".join(f"{k}={v:.4g}" for k, v in rep["parts_mse"].items()))

    if layer_accs:
        rep["layers"] = {}
        print(f"\n--- first k of {n_q} residual layers (encode(n_layers=k) -> decode) ---")
        for k in sorted(layer_accs):
            r = finalise(layer_accs[k], pd_scale_mean)
            rep["layers"][k] = r
            print(f"k={k}: mse={r['all.mse']:.5g}  " + fmt_group(r))
        print(f"consistency: max|forward - decode(encode(k={n_q}))| = {fwd_vs_full:.3g}"
              " (TF32 convolution noise on GPU; the same comparison in fp32 on CPU is ~5e-7)")

    if enc_ok and float(counts.sum()) > 0:
        rep["codebook"] = {}
        print(f"\n--- codebook usage / perplexity (nb_code={nb_code}, {n_codes} codes per group-layer) ---")
        for p, name in enumerate(group_names):
            row = []
            for q in range(n_q):
                c = counts[p, q]
                tot = float(c.sum())
                pr = (c / max(tot, 1.0)).numpy()
                nz = pr[pr > 0]
                ppl = float(np.exp(-(nz * np.log(nz)).sum())) if len(nz) else 0.0
                use = float((c > 0).double().mean())
                rep["codebook"][f"{name}.layer{q + 1}"] = dict(usage=use, perplexity=ppl,
                                                               perplexity_norm=ppl / nb_code, n_codes=int(tot))
                row.append(f"L{q + 1} use={use:.3f} ppl={ppl:.1f}")
            print(f"  {name:10s} " + "  ".join(row))
    else:
        print("\n[warn] no codebook statistics (model.encode unavailable or produced nothing)")

    out = args.out
    if not out and args.ckpt:
        tag = args.tag or os.path.splitext(os.path.basename(args.ckpt))[0]
        out = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), f"rvq_eval_{args.split}_{tag}.json")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(rep, open(out, "w"), indent=1, default=float)
        print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
