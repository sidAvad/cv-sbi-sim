"""
Train a pure-sim NPE flow for identifiability analysis.

Observation types:
  pas_hr    : Pas waveform (201 pts, z-scored) + HR scalar → 202-dim
  cath_lab  : 4 pressure waveforms (Prv/Pra/Pvp/Pap) + 5 scalars → 809-dim
  all_waves : all 24 continuous sim waveforms (z-scored) → 4824-dim (theoretical ceiling)

Encoder and flow are trained jointly from NLL loss on sims only.
Optional --flow-warmup freezes encoder for initial epochs.

Usage:
  python train_flow_sim.py \\
      --run exp-cath-lab-v1 \\
      --obs-type cath_lab \\
      --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314

  python train_flow_sim.py \\
      --run exp-pas-hr-v1 \\
      --obs-type pas_hr \\
      --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.multiprocessing.set_sharing_strategy("file_system")

from sbi.neural_nets import posterior_nn

from dataset import (
    load_stats, load_manifest,
    CVDataset, CVDatasetHR, ReducedCVDataset, PasHRDataset,
    PARAM_KEYS_INFER,
)
from models import AutoencoderEncoder, ReducedAutoencoderEncoder, PasHREncoder

STATS_PATH     = Path("norm_stats.json")
N_PARAMS_INFER = len(PARAM_KEYS_INFER)  # 24
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 512
VAL_FRAC       = 0.05


# ── Helpers ───────────────────────────────────────────────────────────────────

def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class Tee:
    def __init__(self, fh):
        self._fh, self._stdout = fh, sys.stdout

    def write(self, msg):
        self._fh.write(msg); self._stdout.write(msg)

    def flush(self):
        try:
            self._fh.flush()
        except ValueError:
            pass
        self._stdout.flush()


def parse_run(name: str):
    if name.startswith("dry-"):
        return "dry", Path("dry-runs") / name
    elif name.startswith("exp-"):
        return "exp", Path("outputs") / name
    else:
        raise ValueError("--run must start with 'exp-' or 'dry-'")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sim_data(obs_type: str, data_dir: Path, manifest: dict,
                  stats: dict, n: int, log) -> tuple[torch.Tensor, torch.Tensor]:
    index = manifest["index"][:n]
    if obs_type == "cath_lab":
        ds = ReducedCVDataset(str(data_dir), index, stats)
    elif obs_type == "all_waves":
        ds = CVDataset(str(data_dir), index, stats)
    else:
        ds = PasHRDataset(str(data_dir), index, stats)

    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=4,
                        pin_memory=False)
    thetas, xs, loaded = [], [], 0
    for theta_b, x_b in loader:
        thetas.append(theta_b); xs.append(x_b)
        loaded += len(theta_b)
        print(f"\r  loading {loaded}/{n}", end="", flush=True)
    print()
    ds.close()
    theta_all = torch.cat(thetas)
    x_all     = torch.cat(xs)
    log(f"Loaded {loaded} sims  θ={tuple(theta_all.shape)}  x={tuple(x_all.shape)}")
    return theta_all, x_all


# ── Flow construction ─────────────────────────────────────────────────────────

def build_flow(latent_dim: int, theta_sample: torch.Tensor,
               hidden_features: int, num_transforms: int) -> nn.Module:
    """MAF flow conditioned on z (latent). Uses identity embedding — encoder is separate."""
    build_fn = posterior_nn(
        model="maf",
        embedding_net=nn.Identity(),
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        z_score_theta="independent",
        z_score_x="none",
    )
    z_dummy = torch.zeros(len(theta_sample), latent_dim)
    return build_fn(theta_sample.cpu(), z_dummy)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run",            required=True)
    p.add_argument("--obs-type",       required=True, choices=["pas_hr", "cath_lab", "all_waves"])
    p.add_argument("--sim-data-root",  required=True)
    p.add_argument("--n-sims",         type=int,   default=None,
                   help="Number of sims to use (default: all in manifest)")
    p.add_argument("--latent-dim",     type=int,   default=128)
    p.add_argument("--hidden-features",type=int,   default=128)
    p.add_argument("--num-transforms", type=int,   default=5)
    p.add_argument("--max-epochs",     type=int,   default=200)
    p.add_argument("--flow-warmup",    type=int,   default=5,
                   help="Epochs to train flow only before unfreezing encoder")
    p.add_argument("--patience",       type=int,   default=20)
    p.add_argument("--lr-encoder",     type=float, default=5e-4)
    p.add_argument("--lr-flow",        type=float, default=5e-4)
    p.add_argument("--device",         default=DEVICE)
    args = p.parse_args()

    run_type, run_dir = parse_run(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / f"train_{args.run}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_fh   = open(log_path, "w")
    sys.stdout = Tee(log_fh)

    def log(msg=""):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    log(f"Run: {args.run}  obs_type={args.obs_type}  device={args.device}")
    log(f"git: {git_hash()}")
    log(f"cmd: {' '.join(sys.argv)}")

    device = args.device
    stats    = load_stats(STATS_PATH)
    sim_root = Path(args.sim_data_root)
    manifest = load_manifest(sim_root / "manifest_train.json")
    n_sims   = args.n_sims or len(manifest["index"])
    if run_type == "dry":
        n_sims = min(n_sims, 512)
        log("DRY RUN — capped at 512 sims")

    # ── Load data ─────────────────────────────────────────────────────────────
    log(f"Loading {n_sims} sims ({args.obs_type})...")
    theta_all, x_all = load_sim_data(
        args.obs_type, sim_root / "train", manifest, stats, n_sims, log
    )

    n_val   = max(64, min(int(len(theta_all) * VAL_FRAC), 1024))
    n_train = len(theta_all) - n_val
    idx     = torch.randperm(len(theta_all))
    tr_idx, val_idx = idx[:n_train], idx[n_train:]

    theta_tr, x_tr   = theta_all[tr_idx], x_all[tr_idx]
    theta_val, x_val = theta_all[val_idx], x_all[val_idx]
    log(f"Train: {n_train}  Val: {n_val}")

    train_ds = TensorDataset(theta_tr, x_tr)
    val_ds   = TensorDataset(theta_val, x_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Models ────────────────────────────────────────────────────────────────
    if args.obs_type == "cath_lab":
        encoder = ReducedAutoencoderEncoder(latent_dim=args.latent_dim).to(device)
    elif args.obs_type == "all_waves":
        encoder = AutoencoderEncoder(latent_dim=args.latent_dim).to(device)
    else:
        encoder = PasHREncoder(latent_dim=args.latent_dim).to(device)

    flow_net = build_flow(
        args.latent_dim,
        theta_all[:1024],
        args.hidden_features,
        args.num_transforms,
    ).to(device)

    log(f"Encoder: {encoder.describe()}")
    log(f"Flow: {type(flow_net).__name__}  transforms={args.num_transforms}  hidden={args.hidden_features}")

    opt_enc  = torch.optim.Adam(encoder.parameters(),  lr=args.lr_encoder)
    opt_flow = torch.optim.Adam(flow_net.parameters(), lr=args.lr_flow)

    # ── Run info (written upfront so config is captured even if training crashes) ─
    run_info = dict(
        run=args.run,
        type=run_type,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        command=" ".join(sys.argv),
        git_hash=git_hash(),
        obs_type=args.obs_type,
        device=args.device,
        encoder=encoder.describe(),
        flow=dict(
            model="maf",
            hidden_features=args.hidden_features,
            num_transforms=args.num_transforms,
            latent_dim=args.latent_dim,
        ),
        data=dict(
            sim_data_root=args.sim_data_root,
            n_sims=n_sims,
            n_train=n_train,
            n_val=n_val,
        ),
        training=dict(
            lr_encoder=args.lr_encoder,
            lr_flow=args.lr_flow,
            batch_size=BATCH_SIZE,
            max_epochs=args.max_epochs,
            flow_warmup=args.flow_warmup,
            patience=args.patience,
        ),
        best_val_nll=None,
    )
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    # ── Training ──────────────────────────────────────────────────────────────
    csv_path = run_dir / f"train_log_{args.run}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    csv_fh   = open(csv_path, "w", newline="")
    csv_w    = csv.writer(csv_fh)
    csv_w.writerow(["epoch", "phase", "train_nll", "val_nll"])

    best_val  = float("inf")
    wait      = 0
    t0        = __import__("time").time()

    for epoch in range(1, args.max_epochs + 1):
        phase = 0 if epoch <= args.flow_warmup else 1  # 0=flow only, 1=joint

        encoder.train()
        flow_net.train()
        if phase == 0:
            for p in encoder.parameters():
                p.requires_grad_(False)
        else:
            for p in encoder.parameters():
                p.requires_grad_(True)

        train_nll = 0.0
        for theta_b, x_b in train_loader:
            theta_b = theta_b.to(device)
            x_b     = x_b.to(device)

            with torch.set_grad_enabled(True):
                z    = encoder(x_b)
                loss = -flow_net.log_prob(theta_b, condition=z).mean()

            opt_flow.zero_grad()
            if phase == 1:
                opt_enc.zero_grad()
            loss.backward()
            opt_flow.step()
            if phase == 1:
                opt_enc.step()

            train_nll += loss.item() * len(theta_b)

        train_nll /= n_train

        encoder.eval()
        flow_net.eval()
        val_nll = 0.0
        with torch.no_grad():
            for theta_b, x_b in val_loader:
                theta_b = theta_b.to(device)
                x_b     = x_b.to(device)
                z       = encoder(x_b)
                val_nll += -flow_net.log_prob(theta_b, condition=z).mean().item() * len(theta_b)
        val_nll /= n_val

        phase_str = "flow-warmup" if phase == 0 else "joint"
        elapsed   = __import__("time").time() - t0
        log(f"ep {epoch:4d}/{args.max_epochs}  [{phase_str}]  "
            f"train={train_nll:.4f}  val={val_nll:.4f}  "
            f"({elapsed:.0f}s)")
        csv_w.writerow([epoch, phase, f"{train_nll:.6f}", f"{val_nll:.6f}"])
        csv_fh.flush()

        if val_nll < best_val:
            best_val = val_nll
            wait     = 0
            torch.save(encoder.state_dict(), run_dir / "encoder.pt")
            torch.save(flow_net,             run_dir / "flow_net.pt")
        else:
            if phase == 1:
                wait += 1
                if wait >= args.patience:
                    log(f"Early stop at epoch {epoch}  best_val={best_val:.4f}")
                    break

    log(f"Done. Best val NLL: {best_val:.4f}")
    log(f"Saved encoder.pt  flow_net.pt  to {run_dir}")

    # ── Update run_info with final result ─────────────────────────────────────
    run_info["best_val_nll"] = best_val
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    log(f"run_info.json written  git={run_info['git_hash']}")

    csv_fh.close()
    log_fh.close()


if __name__ == "__main__":
    main()
