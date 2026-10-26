"""Stage 1: convolutional autoencoder, plain mini-batch GD on MSE (no labels)."""
import argparse

import torch
import torch.nn.functional as F

from common import (DEVICE, RUNS, CAE, Timer, augment, load_split, log_run, save_json,
                    set_seed, to_float)


@torch.no_grad()
def eval_mse(model, X, bs=256):
    model.eval()
    tot = 0.0
    for i in range(0, len(X), bs):
        xb = to_float(X[i:i + bs])
        tot += F.mse_loss(model(xb), xb, reduction="sum").item()
    return tot / X.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    set_seed()
    name = f"cae_z{a.latent}_b{a.batch}_lr{a.lr}_m{a.momentum}"
    d = load_split()
    Xtr, Xva = d["X_train"], d["X_val"]
    model = CAE(a.latent).to(DEVICE)
    # Plain SGD: theta -= lr * mean-batch-gradient (momentum 0 unless ablation)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum)

    hist = {"train": [], "val": [], "sec": []}
    best, best_state, bad = float("inf"), None, 0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        run, n = 0.0, 0
        with Timer() as t:
            for i in range(0, len(perm), a.batch):
                xb = to_float(Xtr[perm[i:i + a.batch]])
                xb = augment(xb)
                loss = F.mse_loss(model(xb), xb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run += loss.item() * len(xb)
                n += len(xb)
        v = eval_mse(model, Xva)
        hist["train"].append(run / n); hist["val"].append(v); hist["sec"].append(t.dt)
        print(f"ep {ep + 1:02d} train {run / n:.5f} val {v:.5f} {t.dt:.1f}s")
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break

    model.load_state_dict(best_state)
    torch.save({"state": best_state, "latent": a.latent}, RUNS / f"{name}.pt")
    save_json({**vars(a), "hist": hist, "best_val_mse": best}, RUNS / f"{name}.json")
    log_run({"run": name, "stage": "ae", "latent": a.latent, "batch": a.batch, "lr": a.lr,
             "momentum": a.momentum, "epochs_run": len(hist["train"]), "best_val_loss": best,
             "sec_per_epoch": sum(hist["sec"]) / len(hist["sec"]), "note": a.note})
    print("saved", name)


if __name__ == "__main__":
    main()
