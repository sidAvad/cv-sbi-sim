"""
Train the surrogate decoder: θ (25-dim) → 24 continuous waveforms × 201 pts.

Purely supervised regression on sim pairs. Loss is per-channel MSE over
z-scored waveforms. θ is z-scored using parameter stats from norm_stats.json.
Saves decoder.pt and theta_norm.json (mean/std vectors) so the decoder can
be used standalone at inference time.

Usage:
  python train_surrogate.py \\
      --run exp-v1_surrogate_sims \\
      --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314

  python train_surrogate.py \\
      --run dry-v1_surrogate_sims \\
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

from dataset import (
    load_stats, load_manifest,
    SurrogateDataset,
    PARAM_KEYS, WAVE_KEYS_CONT,
    N_PARAMS, N_CONT, T,
)
from models import SurrogateDecoder

STATS_PATH = Path("norm_stats.json")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
VAL_FRAC   = 0.05
LOG_CH_EVERY = 10  # log per-channel val MSE every N epochs


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

def load_sim_data(data_dir: Path, manifest: dict, stats: dict,
                  n: int, log) -> tuple[torch.Tensor, torch.Tensor]:
    index = manifest["index"][:n]
    ds    = SurrogateDataset(str(data_dir), index, stats)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=4,
                        pin_memory=False)
    thetas, waves, loaded = [], [], 0
    for theta_b, wave_b in loader:
        thetas.append(theta_b); waves.append(wave_b)
        loaded += len(theta_b)
        print(f"\r  loading {loaded}/{n}", end="", flush=True)
    print()
    ds.close()
    theta_all = torch.cat(thetas)  # (N, 25)
    wave_all  = torch.cat(waves)   # (N, N_CONT*T)
    log(f"Loaded {loaded} sims  θ={tuple(theta_all.shape)}  waves={tuple(wave_all.shape)}")
    return theta_all, wave_all


def build_theta_norm(theta_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = theta_all.mean(dim=0)
    std  = theta_all.std(dim=0).clamp(min=1e-8)
    return mean, std


# ── Training ──────────────────────────────────────────────────────────────────

def per_channel_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Returns MSE per channel, shape (N_CONT,)."""
    p = pred.view(-1, N_CONT, T)
    t = target.view(-1, N_CONT, T)
    return ((p - t) ** 2).mean(dim=(0, 2))


def per_channel_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Returns R² per channel, shape (N_CONT,)."""
    p = pred.view(-1, N_CONT, T)
    t = target.view(-1, N_CONT, T)
    ss_res = ((p - t) ** 2).sum(dim=(0, 2))
    ss_tot = ((t - t.mean(dim=(0, 2), keepdim=True)) ** 2).sum(dim=(0, 2))
    return 1.0 - ss_res / (ss_tot + 1e-8)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run",           required=True)
    p.add_argument("--sim-data-root", required=True)
    p.add_argument("--n-sims",        type=int,   default=None)
    p.add_argument("--hidden",        type=int,   default=512)
    p.add_argument("--n-layers",      type=int,   default=4)
    p.add_argument("--max-epochs",    type=int,   default=300)
    p.add_argument("--patience",      type=int,   default=20)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--device",        default=DEVICE)
    args = p.parse_args()

    run_type, run_dir = parse_run(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / f"train_{args.run}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_fh   = open(log_path, "w")
    sys.stdout = Tee(log_fh)

    def log(msg=""):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    log(f"Run: {args.run}  device={args.device}")
    log(f"git: {git_hash()}")

    device   = args.device
    stats    = load_stats(STATS_PATH)
    sim_root = Path(args.sim_data_root)
    manifest = load_manifest(sim_root / "manifest_train.json")
    n_sims   = args.n_sims or len(manifest["index"])
    if run_type == "dry":
        n_sims = min(n_sims, 512)
        log("DRY RUN — capped at 512 sims")

    # ── Load data ─────────────────────────────────────────────────────────────
    log(f"Loading {n_sims} sims...")
    theta_all, wave_all = load_sim_data(
        sim_root / "train", manifest, stats, n_sims, log
    )

    # Z-score theta from data statistics
    theta_mean, theta_std = build_theta_norm(theta_all)
    theta_all_z = (theta_all - theta_mean) / theta_std

    n_val   = max(64, min(int(len(theta_all) * VAL_FRAC), 1024))
    n_train = len(theta_all) - n_val
    idx     = torch.randperm(len(theta_all))
    tr_idx, val_idx = idx[:n_train], idx[n_train:]

    theta_tr,  wave_tr  = theta_all_z[tr_idx],  wave_all[tr_idx]
    theta_val, wave_val = theta_all_z[val_idx],  wave_all[val_idx]
    log(f"Train: {n_train}  Val: {n_val}")

    train_ds = TensorDataset(theta_tr,  wave_tr)
    val_ds   = TensorDataset(theta_val, wave_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Move val data to device once
    theta_val_d = theta_val.to(device)
    wave_val_d  = wave_val.to(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SurrogateDecoder(hidden=args.hidden, n_layers=args.n_layers).to(device)
    log(f"Model: {model.describe()}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    csv_path = run_dir / f"train_log_{args.run}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    csv_fh   = open(csv_path, "w", newline="")
    csv_w    = csv.writer(csv_fh)
    csv_w.writerow(["epoch", "train_mse", "val_mse", "lr"])

    best_val = float("inf")
    wait     = 0
    t0       = __import__("time").time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_mse = 0.0
        for theta_b, wave_b in train_loader:
            theta_b = theta_b.to(device)
            wave_b  = wave_b.to(device)
            pred    = model(theta_b)
            loss    = per_channel_mse(pred, wave_b).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_mse += loss.item() * len(theta_b)
        train_mse /= n_train

        model.eval()
        with torch.no_grad():
            pred_val = model(theta_val_d)
            ch_mse   = per_channel_mse(pred_val, wave_val_d)  # (N_CONT,)
            val_mse  = ch_mse.mean().item()

        sched.step(val_mse)
        current_lr = opt.param_groups[0]["lr"]
        elapsed    = __import__("time").time() - t0

        log(f"ep {epoch:4d}/{args.max_epochs}  "
            f"train={train_mse:.6f}  val={val_mse:.6f}  "
            f"lr={current_lr:.2e}  ({elapsed:.0f}s)")

        if epoch % LOG_CH_EVERY == 0:
            ch_names = "  ".join(f"{k}={v:.4f}" for k, v in zip(WAVE_KEYS_CONT, ch_mse.tolist()))
            log(f"  per-ch val MSE: {ch_names}")

        csv_w.writerow([epoch, f"{train_mse:.8f}", f"{val_mse:.8f}", f"{current_lr:.2e}"])
        csv_fh.flush()

        if val_mse < best_val:
            best_val = val_mse
            wait     = 0
            torch.save(model.state_dict(), run_dir / "decoder.pt")
        else:
            wait += 1
            if wait >= args.patience:
                log(f"Early stop at epoch {epoch}  best_val={best_val:.6f}")
                break

    # ── Final per-channel R² on val set ───────────────────────────────────────
    model.load_state_dict(torch.load(run_dir / "decoder.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        pred_val = model(theta_val_d)
        r2 = per_channel_r2(pred_val, wave_val_d)  # (N_CONT,)

    log("Final val R² per channel:")
    for k, v in zip(WAVE_KEYS_CONT, r2.tolist()):
        log(f"  {k:6s}: {v:.4f}")
    log(f"Mean R²: {r2.mean().item():.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    theta_norm = {
        "keys":  PARAM_KEYS,
        "mean":  theta_mean.tolist(),
        "std":   theta_std.tolist(),
    }
    with open(run_dir / "theta_norm.json", "w") as f:
        json.dump(theta_norm, f, indent=2)

    run_info = {
        "run":        args.run,
        "git_hash":   git_hash(),
        "n_sims":     n_sims,
        "n_train":    n_train,
        "n_val":      n_val,
        "hidden":     args.hidden,
        "n_layers":   args.n_layers,
        "max_epochs": args.max_epochs,
        "patience":   args.patience,
        "lr":         args.lr,
        "best_val_mse": best_val,
        "final_r2":   dict(zip(WAVE_KEYS_CONT, r2.tolist())),
        "model":      model.describe(),
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    log(f"Done. Best val MSE: {best_val:.6f}")
    log(f"Saved decoder.pt  theta_norm.json  run_info.json  to {run_dir}")

    csv_fh.close()
    log_fh.close()


if __name__ == "__main__":
    main()
