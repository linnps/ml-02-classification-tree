"""
Synthetic 2-D classification data with a deliberately non-linear,
known decision boundary.

Class 0 lives at the (-,-) and (+,+) quadrants of the input space;
class 1 lives at (-,+) and (+,-) — i.e. an XOR-shape. A linear classifier
*cannot* solve this: it's the canonical demonstration that you need
non-linear models for non-linear structure.

Two extra noise features (`x3`, `x4`) are added so feature-importance
plots have something interesting to say: the trees should ignore them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    n_samples: int = 1500
    cluster_std: float = 0.55
    n_noise_features: int = 2
    seed: int = 42


def generate(cfg: DataConfig) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(cfg.seed)

    # Four Gaussian blobs in the four quadrants. Diagonal pair = class 0,
    # anti-diagonal pair = class 1. This is XOR in continuous space.
    centers = np.array([
        [-1.5, -1.5],   # class 0
        [ 1.5,  1.5],   # class 0
        [-1.5,  1.5],   # class 1
        [ 1.5, -1.5],   # class 1
    ])
    labels = np.array([0, 0, 1, 1])

    per_blob = cfg.n_samples // 4
    points = []
    ys = []
    for c, lab in zip(centers, labels):
        pts = rng.normal(loc=c, scale=cfg.cluster_std, size=(per_blob, 2))
        points.append(pts)
        ys.append(np.full(per_blob, lab))

    X_informative = np.vstack(points)
    y = np.concatenate(ys)

    # Add pure-noise features that should be ignored by good models.
    if cfg.n_noise_features > 0:
        noise = rng.normal(0.0, 1.0, size=(len(y), cfg.n_noise_features))
        X = np.hstack([X_informative, noise])
    else:
        X = X_informative

    # Shuffle so the class order isn't trivially predictable.
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    feature_names = ["x1", "x2"] + [f"x{i+3}_noise" for i in range(cfg.n_noise_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="label")
    return X_df, y_series


def save(out_dir: Path, X: pd.DataFrame, y: pd.Series) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_dir / "X.csv", index=False)
    y.to_csv(out_dir / "y.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic XOR-shape classification data.")
    p.add_argument("--n-samples", type=int, default=1500)
    p.add_argument("--cluster-std", type=float, default=0.55)
    p.add_argument("--n-noise-features", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    cfg = DataConfig(
        n_samples=args.n_samples,
        cluster_std=args.cluster_std,
        n_noise_features=args.n_noise_features,
        seed=args.seed,
    )
    X, y = generate(cfg)
    save(args.out_dir, X, y)

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    print(f"Generated {len(X)} samples, {X.shape[1]} features "
          f"(2 informative + {cfg.n_noise_features} noise).")
    print(f"  class 0: {n0}   class 1: {n1}")
    print(f"  saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
