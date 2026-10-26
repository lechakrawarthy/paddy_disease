"""Stage 2: dense head (latent -> 64 -> 10) on frozen-encoder latents, mini-batch GD,
class-weighted cross-entropy. --finetune unfreezes the encoder (E4, low LR).
Test is evaluated only with --test (final run)."""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (DEVICE, RUNS, CAE, Timer, augment, encode, load_split, log_run, make_head,
                    metrics, save_json, set_seed, to_float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", required=True, help="AE run name, e.g. cae_z128_b64_lr0.01_m0.0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--enc_lr", type=float, default=1e-3)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    set_seed()
    ck = torch.load(RUNS / f"{a.ae}.pt")
    cae = CAE(ck["latent"]); cae.load_state_dict(ck["state"])
    enc = cae.enc.to(DEVICE)
    head = make_head(ck["latent"]).to(DEVICE)
    name = f"{a.ae}__head{'_ft' if a.finetune else ''}_b{a.batch}_lr{a.lr}"
    d = load_split()
    ytr, yva = d["y_train"].to(DEVICE), d["y_val"].to(DEVICE)

    cnt = torch.bincount(d["y_train"], minlength=10).float()
    cw = (cnt.sum() / (10 * cnt)).to(DEVICE)  # inverse-frequency class weights

    if a.finetune:
        opt = torch.optim.SGD([{"params": head.parameters(), "lr": a.lr},
                               {"params": enc.parameters(), "lr": a.enc_lr}], momentum=a.momentum)
        feats = lambda X, idx, train: enc(augment(to_float(X[idx])) if train else to_float(X[idx]))
    else:
        for p in enc.parameters():
            p.requires_grad_(False)
        ztr, zva = encode(enc, d["X_train"]), encode(enc, d["X_val"])
        mu, sd = ztr.mean(0), ztr.std(0) + 1e-6  # standardise with train stats only
        ztr, zva = ((ztr - mu) / sd).to(DEVICE), ((zva - mu) / sd).to(DEVICE)
        opt = torch.optim.SGD(head.parameters(), lr=a.lr, momentum=a.momentum)

    @torch.no_grad()
    def evaluate(split):
        enc.eval(); head.eval()
        X, y = d[f"X_{split}"], d[f"y_{split}"].to(DEVICE)
        if a.finetune:
            z = torch.cat([enc(to_float(X[i:i + 256])) for i in range(0, len(X), 256)])
        else:
            z = ((encode(enc, X) - mu) / sd).to(DEVICE) if split != "val" else zva
        logits = head(z)
        return (F.cross_entropy(logits, y, weight=cw).item(), logits.argmax(1).cpu().numpy(), z,
                logits.softmax(1).cpu().numpy())

    best, best_state, bad, secs = float("inf"), None, 0, []
    hist = {"train": [], "val": [], "train_acc": [], "val_acc": [], "val_f1": []}
    for ep in range(a.epochs):
        head.train(); enc.train(a.finetune)
        perm = torch.randperm(len(ytr))
        run, correct = 0.0, 0
        with Timer() as t:
            for i in range(0, len(perm), a.batch):
                idx = perm[i:i + a.batch]
                z = feats(d["X_train"], idx, True) if a.finetune else ztr[idx.to(DEVICE)]
                logits = head(z)
                loss = F.cross_entropy(logits, ytr[idx.to(DEVICE)], weight=cw)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                run += loss.item() * len(idx)
                correct += (logits.argmax(1) == ytr[idx.to(DEVICE)]).sum().item()
        vl, vp, _, _ = evaluate("val")
        vm = metrics(d["y_val"].numpy(), vp)
        hist["train"].append(run / len(perm)); hist["val"].append(vl); secs.append(t.dt)
        hist["train_acc"].append(correct / len(perm))
        hist["val_acc"].append(vm["accuracy"]); hist["val_f1"].append(vm["macro_f1"])
        print(f"ep {ep + 1:02d} train {run / len(perm):.4f} val {vl:.4f} valF1 {vm['macro_f1']:.4f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = ({k: v.clone() for k, v in head.state_dict().items()},
                          {k: v.clone() for k, v in enc.state_dict().items()})
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break

    head.load_state_dict(best_state[0]); enc.load_state_dict(best_state[1])
    vl, vp, _, _ = evaluate("val")
    res = {**vars(a), "hist": hist, "best_val_loss": best, "val": metrics(d["y_val"].numpy(), vp)}
    if a.test:
        _, tp, tz, tprob = evaluate("test")
        res["test"] = metrics(d["y_test"].numpy(), tp, tprob)
        np.save(RUNS / f"{name}_test_pred.npy", tp)
        np.save(RUNS / f"{name}_test_latent.npy", tz.cpu().numpy())
    save_json(res, RUNS / f"{name}.json")
    log_run({"run": name, "stage": "head_ft" if a.finetune else "head", "latent": ck["latent"],
             "batch": a.batch, "lr": a.lr, "momentum": a.momentum, "epochs_run": len(secs),
             "best_val_loss": best, "val_macro_f1": res["val"]["macro_f1"],
             "sec_per_epoch": sum(secs) / len(secs), "note": a.note})
    print(res["val"], res.get("test", ""))


if __name__ == "__main__":
    main()
