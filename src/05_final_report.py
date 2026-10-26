"""Final plots/tables. Run AFTER choosing configs on validation and running
02_baseline.py --test and 04_train_head.py --test for the chosen model."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from common import DEVICE, FIGS, RUNS, CAE, load_json, load_split, to_float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", required=True)
    ap.add_argument("--head", required=True, help="head run name (must have been run with --test)")
    a = ap.parse_args()
    d = load_split()
    classes, yte = d["classes"], d["y_test"].numpy()
    ae, hd, b1 = load_json(RUNS / f"{a.ae}.json"), load_json(RUNS / f"{a.head}.json"), load_json(RUNS / "B1.json")

    # loss curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ae["hist"]["train"], label="train"); ax[0].plot(ae["hist"]["val"], label="val")
    ax[0].set(title="CAE reconstruction MSE", xlabel="epoch"); ax[0].legend()
    ax[1].plot(hd["hist"]["train"], label="train"); ax[1].plot(hd["hist"]["val"], label="val")
    ax[1].set(title="Head weighted CE", xlabel="epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(FIGS / "loss_curves.png", dpi=150); plt.close()

    # head accuracy per epoch (the autoencoder itself has no accuracy)
    h = hd["hist"]
    ep = np.arange(1, len(h["train_acc"]) + 1)
    best = int(np.argmin(h["val"])) + 1
    plt.figure(figsize=(7, 3.6))
    plt.plot(ep, h["train_acc"], label="train accuracy")
    plt.plot(ep, h["val_acc"], label="validation accuracy")
    plt.plot(ep, h["val_f1"], "--", label="validation macro-F1")
    plt.axvline(best, color="grey", lw=0.8, ls=":")
    plt.text(best + 0.2, 0.25, f"best val loss (epoch {best})", fontsize=8, color="grey")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.title("Classifier head: accuracy per epoch")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIGS / "accuracy_curves.png", dpi=150); plt.close()

    # reconstructions + test MSE
    ck = torch.load(RUNS / f"{a.ae}.pt")
    cae = CAE(ck["latent"]).to(DEVICE); cae.load_state_dict(ck["state"]); cae.eval()
    with torch.no_grad():
        tot = sum(F.mse_loss(cae(to_float(d["X_test"][i:i + 256])), to_float(d["X_test"][i:i + 256]),
                             reduction="sum").item() for i in range(0, len(yte), 256))
        test_mse = tot / d["X_test"].numel()
        pick = [int(np.nonzero(yte == c)[0][0]) for c in range(len(classes))]
        x = to_float(d["X_test"][pick]); r = cae(x)
    # 4 x 5 grid: rows 1/3 originals, rows 2/4 reconstructions
    fig, ax = plt.subplots(4, 5, figsize=(10, 8.8))
    for c in range(len(classes)):
        row, col = 2 * (c // 5), c % 5
        ax[row, col].imshow(x[c].permute(1, 2, 0).cpu())
        ax[row + 1, col].imshow(r[c].permute(1, 2, 0).cpu())
        ax[row, col].set_title(classes[c], fontsize=11)
        ax[row, col].axis("off"); ax[row + 1, col].axis("off")
    plt.tight_layout(); plt.savefig(FIGS / "reconstructions.png", dpi=150); plt.close()

    # t-SNE of test latents
    z = np.load(RUNS / f"{a.head}_test_latent.npy")
    emb = TSNE(2, random_state=42, init="pca", perplexity=30).fit_transform(z)
    plt.figure(figsize=(8, 6))
    for c in range(len(classes)):
        m = yte == c
        plt.scatter(emb[m, 0], emb[m, 1], s=6, label=classes[c])
    plt.legend(markerscale=2, fontsize=7); plt.title("t-SNE of test latents")
    plt.tight_layout(); plt.savefig(FIGS / "tsne.png", dpi=150); plt.close()

    # confusion matrix + per-class report
    pred = np.load(RUNS / f"{a.head}_test_pred.npy")
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=classes, xticks_rotation=60,
                                            ax=ax, colorbar=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout(); plt.savefig(FIGS / "confusion_matrix.png", dpi=150); plt.close()
    rep = classification_report(yte, pred, target_names=classes, digits=3)

    lines = [f"Test MSE (CAE): {test_mse:.5f}",
             f"E1 test: {hd['test']}", f"B1 test: {b1['test']}",
             f"E1 beats B1 on macro-F1: {hd['test']['macro_f1'] > b1['test']['macro_f1']}",
             f"Target >= 0.60 met: {hd['test']['macro_f1'] >= 0.60}", "", rep]
    (FIGS.parent / "final_results.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
