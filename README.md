<div align="center">

# Tree-Based Classification — Why Ensembles Beat a Single Tree

**Logistic · Decision Tree · Random Forest · Gradient Boosting · on a synthetic XOR-shape dataset**

![status](https://img.shields.io/badge/status-complete-3B6EA8?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-3B6EA8?style=flat-square)
![data](https://img.shields.io/badge/data-self--generated-7A7A7A?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-7A7A7A?style=flat-square)

</div>

---

## At a glance

> Train four classifiers on a synthetic dataset whose **decision boundary is XOR-shaped** (linear models cannot solve it by construction) and watch the model-family ladder reveal itself: linear fails → single tree mostly works → ensembles nail it.

<table>
<tr>
<td align="center" width="25%">
<sub>Logistic Regression</sub><br>
<b style="font-size:1.5em; color:#C04040;">46.1%</b><br>
<sub>accuracy — fails by construction</sub>
</td>
<td align="center" width="25%">
<sub>Decision Tree (depth 6)</sub><br>
<b style="font-size:1.5em; color:#7A7A7A;">86.9%</b><br>
<sub>works, but jagged</sub>
</td>
<td align="center" width="25%">
<sub>Random Forest</sub><br>
<b style="font-size:1.5em; color:#3B6EA8;">98.1%</b><br>
<sub>clean XOR recovery</sub>
</td>
<td align="center" width="25%">
<sub>Gradient Boosting</sub><br>
<b style="font-size:1.5em; color:#3B6EA8;">97.9%</b><br>
<sub>matches RF</sub>
</td>
</tr>
</table>

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.461 | 0.468 | 0.452 |
| Decision Tree (depth 6) | 0.869 | 0.855 | 0.966 |
| **Random Forest (200 trees)** | **0.981** | **0.981** | **0.997** |
| **Gradient Boosting (200 stages)** | **0.979** | **0.979** | **0.995** |

<sub>**Headline finding:** Logistic regression is *worse than chance* — that's not bad luck, it's a mathematical guarantee. XOR is the canonical example of a problem no linear classifier can solve, and a synthetic dataset lets us demonstrate that cleanly. The ensembles' 98% is the gap a single decision tree leaves on the table.</sub>

---

## Dashboard

### 1. Test-set metrics across models

![metrics](assets/01_metrics.png)

The bar chart maps the model-complexity ladder almost perfectly: a linear model floors at ~50% (random), a single tree climbs to ~87%, and either ensemble of trees lands at ~98%. The next four figures show *why*.

### 2. Decision boundaries — the visual proof

![decision boundaries](assets/02_decision_boundaries.png)

This is the figure that tells the whole story.

- **Logistic Regression** can only draw one straight line. There is no straight line that separates the four blobs of an XOR pattern, so it picks an arbitrary one and gets ~half the points wrong.
- **Decision Tree** draws axis-aligned rectangles. Visually you can see the four quadrants forming, but the boundary is jagged and a few clusters at corners get chopped up.
- **Random Forest** averages 200 axis-aligned trees with bagging + feature subsampling. The result is a smooth, almost-circular soft boundary around each class — much closer to what the data actually looks like.
- **Gradient Boosting** arrives at a similar boundary by a completely different algorithm (sequential weak learners that correct each other's residuals). The shape converges because the data has a true structure both algorithms can find.

### 3. ROC curves on the test set

![roc](assets/03_roc.png)

Logistic's ROC sits on the diagonal — the empirical proof that its score function carries no information about the label. Random Forest's curve almost touches the top-left corner (AUC = 0.997), and Gradient Boosting follows a near-identical trajectory.

### 4. Confusion matrices

![confusion](assets/04_confusion.png)

Logistic's matrix is balanced-but-wrong: it predicts roughly equal numbers of each class but they're uncorrelated with the truth. The other three matrices have the off-diagonals drained to nearly zero — exactly what a working classifier looks like.

### 5. Feature importance — the noise rejection check

![feature importance](assets/05_feature_importance.png)

Synthetic data ships with a built-in test: we added two pure-noise features (`x3_noise`, `x4_noise`) on top of the 2 informative ones. A working tree-based model **must** assign them near-zero importance. All three pass:

- Decision Tree: `x1`/`x2` get >97% of the importance combined; the noise features score below 2%.
- Random Forest: same shape, with the noise features at ~5–6% (a slight bias due to bagged feature subsampling).
- Gradient Boosting: most ruthless of the three — noise features under 1%.

This is exactly why people say "tree-based models are robust to irrelevant features."

---

## What's actually happening

### The XOR problem in continuous space

Class 0 lives at the (−,−) and (+,+) quadrants of (x1, x2). Class 1 lives at (−,+) and (+,−). Knowing only `x1` is *uninformative*: each side has 50% of each class. Same for `x2`. You need *both* dimensions, and a non-linear interaction, to separate them.

### Logistic regression — necessarily fails

The decision rule is `sign(w₁·x₁ + w₂·x₂ + b)`. That's a single straight line in the (x1, x2) plane. There is no choice of (w₁, w₂, b) that separates all four blobs. Logistic isn't unlucky here — it is *mathematically incapable* of solving this.

### Decision tree — works, but in a brittle way

A tree carves the input space into axis-aligned rectangles. Four rectangles is enough to solve XOR, so even a depth-2 tree works *if it picks the right splits* — but greedy CART tries to maximize info gain *one split at a time*, and the optimal first split for XOR has zero info gain at the root. Trees usually find the structure anyway by descending two more levels — provided their depth budget isn't exhausted by spurious splits on noise features. Hence depth 6 here, not depth 4.

### Random forest — bagging + feature subsampling

Train many trees on bootstrap samples of the data, with each split considering a random subset of features. Each tree is a weak, slightly-different classifier; averaging them smooths the rectangular bias into something resembling the true smooth decision boundary. Feature subsampling also dilutes the impact of noise features.

### Gradient boosting — sequential weak learners

Fit a shallow tree to the data; compute its residuals; fit *another* shallow tree to the residuals; add it in with a small learning rate. Repeat. The result is a strong learner built out of many tiny trees that each correct the predecessor's mistakes. Often slightly more accurate than RF but more sensitive to hyperparameter tuning.

### Mental model

| Model family | Where it fits | What kills it |
|---|---|---|
| Linear (Logistic, SVM-linear) | Boundaries that look like single straight lines / hyperplanes | Anything non-linear |
| Single tree | Axis-aligned rectangular regions | Noise features eating depth budget; high variance |
| Random Forest | Smooth-ish boundaries via bagging + feature randomization | Cannot extrapolate beyond training-data range |
| Gradient Boosting | Same regions as RF, often a bit sharper | Sensitive to learning rate and tree depth |

---

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py    # synthetic XOR-shape dataset (deterministic)
python train.py            # fit, evaluate, render dashboard figures
```

Outputs land in `assets/` (the five PNGs embedded above) and `results/metrics.json`.

### Tweak the difficulty

`DataConfig` in [`generate_data.py`](generate_data.py):

```python
DataConfig(
    n_samples=1500,
    cluster_std=0.55,         # blob spread — higher = more class overlap
    n_noise_features=2,       # how many distractor features to add
    seed=42,
)
```

Try `cluster_std=0.9` to see all models degrade gracefully, or `n_noise_features=10` to watch the single decision tree start to choke even at depth 6.

---

## Project layout

```
02-classification-tree/
├── README.md              ← this dashboard
├── requirements.txt
├── generate_data.py       ← synthetic XOR-shape dataset (deterministic)
├── train.py               ← four-model pipeline + dashboard figures
├── assets/                ← rendered dashboard figures (5 PNGs)
└── results/metrics.json
```

---

## What I learned

- **The "linear models can't solve XOR" claim is something you should *see*, not just memorize.** A boundary plot beside a 46% accuracy bar is twenty times more memorable than the textbook sentence.
- **A single decision tree's accuracy is a misleading proxy for "trees work."** With noise features in the mix, a depth-4 tree on this exact dataset bottoms out near random — *not because trees are bad*, but because greedy splits get distracted before they reach the XOR structure. Bumping depth to 6 fixes it; that's the kind of failure that Random Forest's feature subsampling avoids by construction.
- **Synthetic data lets you bake in a noise-rejection test.** Adding two known-irrelevant features and then plotting feature importance is a one-line audit of any tree-based model. With real data, you only ever check that the *important* features make sense; you can rarely check that the *unimportant* ones really are noise.
- **Ensembles win here, but the win is qualitatively different from the linear-vs-tree gap.** Linear → tree is a "wrong model family" gap; tree → ensemble is a "right family, but variance reduction matters" gap. Both gaps are real; only the first is unbridgeable without changing the model family.

---

<div align="center">
<sub>Part of a hands-on machine-learning portfolio. Data is fully synthetic and self-generated.</sub>
</div>
