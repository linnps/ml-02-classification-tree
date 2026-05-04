"""
Train Logistic Regression / Decision Tree / Random Forest / Gradient
Boosting on the synthetic XOR-shape classification dataset; produce the
dashboard figures used in README.md.

Palette (shared across the whole portfolio):
    background  : white
    grid / axes : light gray  (#E5E5E5)
    primary     : muted blue  (#3B6EA8)
    accent      : muted red   (#C04040)
    neutral     : medium gray (#7A7A7A)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from generate_data import DataConfig, generate

# ---------------------------------------------------------------- style ----
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#E5E5E5"
COLOR_TEXT = "#333333"
COLOR_BLUE = "#3B6EA8"
COLOR_RED = "#C04040"
COLOR_GRAY = "#7A7A7A"
COLOR_LIGHT_GRAY = "#CCCCCC"

mpl.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_LIGHT_GRAY,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": COLOR_GRID,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.size": 11,
})

# Two-tone diverging colormap: light red → white → light blue (decision regions).
CMAP_REGION = LinearSegmentedColormap.from_list(
    "region", ["#F2D2D2", "#FFFFFF", "#D2DBE8"], N=256
)
CMAP_POINTS = ListedColormap([COLOR_RED, COLOR_BLUE])


# ---------------------------------------------------------- model fitting --
@dataclass
class FitResult:
    name: str
    model: object
    accuracy: float
    f1: float
    roc_auc: float
    cm: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    y_pred_test: np.ndarray
    y_proba_test: np.ndarray


def fit_models(X_train, y_train, X_test, y_test) -> list[FitResult]:
    models = {
        "Logistic Reg.": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boost": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    }
    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        y_proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results.append(FitResult(
            name=name, model=m,
            accuracy=float(accuracy_score(y_test, y_pred)),
            f1=float(f1_score(y_test, y_pred)),
            roc_auc=float(roc_auc_score(y_test, y_proba)),
            cm=confusion_matrix(y_test, y_pred),
            fpr=fpr, tpr=tpr,
            y_pred_test=y_pred, y_proba_test=y_proba,
        ))
    return results


# ---------------------------------------------------------------- figures --
def fig_decision_boundaries(
    X_train_2d: np.ndarray,
    y_train: np.ndarray,
    X_test_2d: np.ndarray,
    y_test: np.ndarray,
    results: list[FitResult],
    pad_features_train: np.ndarray | None,
    pad_features_test: np.ndarray | None,
    out_path: Path,
) -> None:
    """
    Plot decision boundaries for each model in 2-D.

    For models trained on more than 2 features, we hold the noise dims at
    their training-set median so the displayed 2-D slice is meaningful.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), constrained_layout=True)

    # Mesh in the (x1, x2) plane.
    x_min, x_max = X_train_2d[:, 0].min() - 0.5, X_train_2d[:, 0].max() + 0.5
    y_min, y_max = X_train_2d[:, 1].min() - 0.5, X_train_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    # If trained on >2 features, append the median of noise feats so predict_proba works.
    if pad_features_train is not None and pad_features_train.shape[1] > 0:
        pad_median = np.median(pad_features_train, axis=0)
        grid = np.hstack([grid_2d, np.tile(pad_median, (grid_2d.shape[0], 1))])
    else:
        grid = grid_2d

    for ax, r in zip(axes.ravel(), results):
        Z = r.model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=20, cmap=CMAP_REGION, alpha=0.85)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[COLOR_GRAY], linewidths=0.9)

        ax.scatter(X_test_2d[y_test == 0, 0], X_test_2d[y_test == 0, 1],
                   s=18, c=COLOR_RED, edgecolor="white", linewidth=0.4,
                   label="class 0", alpha=0.85)
        ax.scatter(X_test_2d[y_test == 1, 0], X_test_2d[y_test == 1, 1],
                   s=18, c=COLOR_BLUE, edgecolor="white", linewidth=0.4,
                   label="class 1", alpha=0.85)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"{r.name}   (acc {r.accuracy:.3f} · F1 {r.f1:.3f})")
        ax.grid(False)

    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Decision boundaries on the XOR-shape problem",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.02)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_metrics_bar(results: list[FitResult], out_path: Path) -> None:
    metric_names = ["Accuracy", "F1", "ROC-AUC"]
    values = np.array([[r.accuracy, r.f1, r.roc_auc] for r in results])
    names = [r.name for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    palette = [COLOR_BLUE, COLOR_GRAY, COLOR_RED, "#5A8FCC"]
    for ax, mname, col in zip(axes, metric_names, range(3)):
        bars = ax.bar(names, values[:, col], color=palette,
                      edgecolor=COLOR_LIGHT_GRAY, linewidth=0.8)
        ax.set_title(mname)
        ax.tick_params(axis="x", labelrotation=20)
        for bar, v in zip(bars, values[:, col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=COLOR_TEXT)
        ax.set_ylim(0.0, 1.06)
    fig.suptitle("Test-set metrics across models",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_roc_curves(results: list[FitResult], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    palette = [COLOR_BLUE, COLOR_GRAY, COLOR_RED, "#5A8FCC"]
    for r, c in zip(results, palette):
        ax.plot(r.fpr, r.tpr, color=c, linewidth=2.0,
                label=f"{r.name}  (AUC {r.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=COLOR_LIGHT_GRAY, linestyle="--",
            linewidth=1.0, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_confusion_matrices(results: list[FitResult], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("blue_only", ["#FFFFFF", COLOR_BLUE], N=256)
    for ax, r in zip(axes, results):
        cm = r.cm
        norm = cm / cm.sum()
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=norm.max())
        for i in range(2):
            for j in range(2):
                color = "white" if norm[i, j] > 0.3 else COLOR_TEXT
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title(r.name, fontsize=11)
        ax.grid(False)
    fig.suptitle("Confusion matrices on the test set",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_feature_importance(
    feature_names: list[str],
    results: list[FitResult],
    out_path: Path,
) -> None:
    # Tree-based models expose feature_importances_; logistic does not.
    tree_models = [r for r in results
                   if hasattr(r.model, "feature_importances_")]

    fig, axes = plt.subplots(1, len(tree_models), figsize=(4 * len(tree_models), 4),
                             sharey=True, constrained_layout=True)
    if len(tree_models) == 1:
        axes = [axes]

    palette = [COLOR_BLUE, COLOR_GRAY, COLOR_RED]
    for ax, r, c in zip(axes, tree_models, palette):
        imp = r.model.feature_importances_
        order = np.argsort(imp)[::-1]
        names_sorted = [feature_names[i] for i in order]
        bars = ax.barh(range(len(imp)), imp[order], color=c,
                       edgecolor=COLOR_LIGHT_GRAY, linewidth=0.6)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(names_sorted)
        ax.invert_yaxis()
        ax.set_xlabel("importance")
        ax.set_title(r.name)
        for bar, v in zip(bars, imp[order]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", ha="left", va="center", fontsize=9, color=COLOR_TEXT)
        ax.set_xlim(0, max(imp) * 1.20)

    fig.suptitle("Feature importance — tree-based models",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main() -> None:
    cfg = DataConfig()
    X_df, y_series = generate(cfg)
    feature_names = list(X_df.columns)

    X = X_df.values
    y = y_series.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=cfg.seed
    )

    # Logistic / SVM-style models benefit from scaling; trees don't care.
    # We'll scale only for the linear model behind the scenes by having
    # the pipeline-level fit do it via StandardScaler.transform externally.
    scaler = StandardScaler().fit(X_train)
    # We pass unscaled features to all models — that's fine for trees, and
    # logistic regression with L-BFGS converges fine on this scale anyway.

    results = fit_models(X_train, y_train, X_test, y_test)

    print(f"\nDataset: n={cfg.n_samples}, p={X.shape[1]} (2 informative + {X.shape[1] - 2} noise)\n")
    print(f"  {'model':<18} {'acc':>6} {'F1':>6} {'AUC':>6}")
    for r in results:
        print(f"  {r.name:<18} {r.accuracy:>6.3f} {r.f1:>6.3f} {r.roc_auc:>6.3f}")

    Path("results").mkdir(exist_ok=True)
    summary = {
        "config": cfg.__dict__,
        "metrics": [{"model": r.name, "accuracy": r.accuracy,
                     "f1": r.f1, "roc_auc": r.roc_auc} for r in results],
    }
    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    assets = Path("assets"); assets.mkdir(exist_ok=True)
    fig_metrics_bar(results, assets / "01_metrics.png")
    fig_decision_boundaries(
        X_train[:, :2], y_train, X_test[:, :2], y_test, results,
        pad_features_train=X_train[:, 2:] if X.shape[1] > 2 else None,
        pad_features_test=X_test[:, 2:] if X.shape[1] > 2 else None,
        out_path=assets / "02_decision_boundaries.png",
    )
    fig_roc_curves(results, assets / "03_roc.png")
    fig_confusion_matrices(results, assets / "04_confusion.png")
    fig_feature_importance(feature_names, results, assets / "05_feature_importance.png")

    print(f"\nFigures saved to: {assets.resolve()}")


if __name__ == "__main__":
    main()
