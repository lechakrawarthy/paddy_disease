"""B1: PCA(128) on flattened 64x64 pixels -> logistic regression. PCA fit on train only.
Test is evaluated only with --test (final run)."""
import argparse

import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import RUNS, Timer, load_split, log_run, metrics, save_json, set_seed


def flat64(X):
    return F.interpolate(X.float() / 255, size=64, mode="area").flatten(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    set_seed()
    d = load_split()
    Xtr, Xva = flat64(d["X_train"]), flat64(d["X_val"])
    with Timer() as t:
        pca = PCA(128, random_state=42).fit(Xtr)
        sc = StandardScaler().fit(pca.transform(Xtr))
        f = lambda X: sc.transform(pca.transform(X))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(f(Xtr), d["y_train"].numpy())
    res = {"val": metrics(d["y_val"].numpy(), clf.predict(f(Xva))), "fit_seconds": t.dt,
           "explained_var": float(pca.explained_variance_ratio_.sum())}
    if args.test:
        zte = f(flat64(d["X_test"]))
        pred = clf.predict(zte)
        res["test"] = metrics(d["y_test"].numpy(), pred, clf.predict_proba(zte))
        np.save(RUNS / "B1_test_pred.npy", pred)
    save_json(res, RUNS / "B1.json")
    log_run({"run": "B1", "stage": "baseline", "latent": 128, "val_macro_f1": res["val"]["macro_f1"],
             "note": "PCA128 + LogReg"})
    print(res)


if __name__ == "__main__":
    main()
