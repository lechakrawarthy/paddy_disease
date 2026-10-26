"""Shared config, data access, model and training helpers (PRD project 29)."""
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("PADDY_DATA", ROOT / "data"))
ART = ROOT / "artifacts"
RUNS = ROOT / "runs"
FIGS = ROOT / "reports" / "figs"
for _d in (ART, RUNS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def load_split():
    """Returns dict with uint8 NCHW tensors X_*, int64 labels y_*, and class names."""
    z = np.load(ART / "dataset_128.npz", allow_pickle=True)
    s = np.load(ART / "split.npz")
    out = {"classes": list(z["classes"])}
    for name in ("train", "val", "test"):
        idx = s[name]
        out[f"X_{name}"] = torch.from_numpy(z["X"][idx]).permute(0, 3, 1, 2).contiguous()
        out[f"y_{name}"] = torch.from_numpy(z["y"][idx]).long()
    return out


def to_float(xb):
    return xb.to(DEVICE).float().div_(255.0)


def augment(x):
    """Train-only: h-flip, rotation within +-15 deg, brightness jitter. x in [0,1], NCHW on device."""
    n = x.shape[0]
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    ang = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(15)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False),
                      padding_mode="reflection", align_corners=False)
    bright = 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * 0.2
    return (x * bright).clamp_(0, 1)


# ---------------------------------------------------------------- model
class Encoder(nn.Module):
    """3 stride-2 'same' convs: 128x128x3 -> 16x16x128 -> dense latent."""

    def __init__(self, latent=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Linear(16 * 16 * 128, latent)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.fc = nn.Linear(latent, 16 * 16 * 128)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 128, 16, 16))


class CAE(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.enc, self.dec = Encoder(latent), Decoder(latent)
        for m in self.modules():  # He normal init
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.dec(self.enc(x))


def make_head(latent, n_classes=10):
    h = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, n_classes))
    for m in h:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    return h


@torch.no_grad()
def encode(enc, X, bs=256):
    enc.eval()
    return torch.cat([enc(to_float(X[i:i + bs])).cpu() for i in range(0, len(X), bs)])


# ---------------------------------------------------------------- metrics / logging
def metrics(y_true, y_pred, proba=None):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(p),
           "macro_recall": float(r), "macro_f1": float(f)}
    if proba is not None:
        out["macro_auc_ovr"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    return out


def log_run(row):
    path = RUNS / "experiment_log.csv"
    cols = ["run", "stage", "latent", "batch", "lr", "momentum", "epochs_run",
            "best_val_loss", "val_macro_f1", "sec_per_epoch", "note"]
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t
