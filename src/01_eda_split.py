"""EDA, near-duplicate check, grouped stratified 70/15/15 split, 128x128 cache.

Expects Kaggle layout: data/train.csv (image_id,label,variety,age) and
data/train_images/<label>/<image_id>.

Near-duplicates: 64-bit dHash, Hamming distance <= --hamming. Clusters are
union-find components; every cluster stays inside ONE split (StratifiedGroupKFold,
20 folds: 3 test + 3 val + 14 train = 15/15/70).
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from common import ART, DATA, FIGS, IMG, SEED, save_json


def dhash(img):
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def clusters_from_hashes(bits, thr):
    n = len(bits)
    packed = np.packbits(bits, axis=1)
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n - 1):
        d = pop[packed[i] ^ packed[i + 1:]].sum(1)
        for j in np.nonzero(d <= thr)[0]:
            ra, rb = find(i), find(i + 1 + j)
            if ra != rb:
                parent[rb] = ra
    return np.array([find(i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hamming", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(DATA / "train.csv")
    classes = sorted(df["label"].unique())
    y = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    counts = df["label"].value_counts()
    print("Class distribution:\n", counts.to_string())
    print(f"max/min class ratio: {counts.max() / counts.min():.2f}")

    X = np.zeros((len(df), IMG, IMG, 3), np.uint8)
    bits = np.zeros((len(df), 64), bool)
    sizes = set()
    for k, r in enumerate(tqdm(df.itertuples(), total=len(df), desc="load")):
        im = Image.open(DATA / "train_images" / r.label / r.image_id).convert("RGB")
        sizes.add(im.size)
        bits[k] = dhash(im)
        X[k] = np.asarray(im.resize((IMG, IMG), Image.BILINEAR))
    print("image sizes seen:", sorted(sizes)[:5], f"({len(sizes)} distinct)")

    groups = clusters_from_hashes(bits, args.hamming)
    _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    dup_imgs = int((cnt[inv] > 1).sum())
    mixed = sum(len(set(y[inv == g])) > 1 for g in np.nonzero(cnt > 1)[0])
    print(f"clusters: {len(cnt)}; images in multi-image clusters: {dup_imgs}; "
          f"clusters with mixed labels: {mixed}")

    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    folds = [te for _, te in sgkf.split(X, y, inv)]
    test = np.concatenate(folds[:3])
    val = np.concatenate(folds[3:6])
    train = np.concatenate(folds[6:])
    for n, s in (("train", train), ("val", val), ("test", test)):
        print(n, len(s), np.bincount(y[s], minlength=len(classes)))

    np.savez_compressed(ART / "split.npz", train=train, val=val, test=test)
    np.savez(ART / "dataset_128.npz", X=X, y=y, classes=np.array(classes))
    save_json({"counts": counts.to_dict(), "n_images": len(df), "dup_images": dup_imgs,
               "n_clusters": int(len(cnt)), "mixed_label_clusters": int(mixed),
               "hamming": args.hamming, "split_sizes": [len(train), len(val), len(test)]},
              ART / "eda.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = counts.sort_values().plot.barh(figsize=(7, 4))
    ax.set_title("Images per class")
    plt.tight_layout()
    plt.savefig(FIGS / "class_distribution.png", dpi=150)


if __name__ == "__main__":
    main()
