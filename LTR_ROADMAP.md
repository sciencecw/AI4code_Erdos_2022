# Roadmap: From Pairwise Classifier to Measurable Ranking

This document describes a minimal but non-trivial path to turn the existing
pairwise classification pipeline into a system that actually produces cell
orderings and measures them with Kendall Tau -- the competition metric.

## The Gap

The project currently trains binary classifiers on pairs of cells (does cell A
come before cell B?) and reports pairwise accuracy (~66%). But it never:

1. Aggregates those pairwise predictions into a full ordering of cells within
   a notebook
2. Evaluates that ordering against ground truth using Kendall Tau
3. Compares the result to baselines (random ordering, original-file-order)

Without step 1, the project has no ranking prediction. Without steps 2-3, there
is no way to know if the approach is working at all as a learning-to-rank
system.

## Proposed Steps

### Step 1: Build the aggregation function (~30 lines)

Given a trained pairwise classifier and a notebook's cells, predict the
ordering:

```python
from itertools import combinations

def predict_ordering(model, vectorize_pair_fn, cell_ids, cell_features):
    """
    Predict the ordering of cells in a single notebook.

    For each pair (i, j), ask the classifier P(i before j).
    Sum up "wins" for each cell. Rank by descending win count.
    This is a simple Copeland-style aggregation.
    """
    n = len(cell_ids)
    wins = {cid: 0.0 for cid in cell_ids}

    for (idx_a, id_a), (idx_b, id_b) in combinations(enumerate(cell_ids), 2):
        # Build the same feature vector that P4 creates
        pair_features = vectorize_pair_fn(cell_features[idx_a],
                                          cell_features[idx_b])
        prob_a_first = model.predict_proba(pair_features)[0, 1]
        wins[id_a] += prob_a_first
        wins[id_b] += (1.0 - prob_a_first)

    # Sort by win count (highest = earliest in notebook)
    ranked = sorted(cell_ids, key=lambda c: -wins[c])
    return ranked
```

Using `predict_proba` rather than hard `predict` gives a smoother ranking
signal. The Copeland aggregation (sum of win probabilities) is simple and
well-understood in social choice theory. More sophisticated methods
(e.g., Bradley-Terry, or solving a linear program) can come later.

### Step 2: Evaluate on validation notebooks (~40 lines)

The validation set already exists (`valid_P3_code/`, `valid_P3_md/`). Each
file is one notebook with ground-truth ordering. The key is to iterate over
these files, predict an ordering, and compute Kendall Tau:

```python
def evaluate_ranking(model, vectorize_pair_fn, val_dir, cell_type="code"):
    """
    For each notebook in val_dir:
      1. Load cells and ground-truth order
      2. Predict ordering via pairwise aggregation
      3. Compute Kendall Tau
    Return mean Kendall Tau across all notebooks.
    """
    ground_truths = []
    predictions = []

    for nb_file in os.listdir(val_dir):
        df = pd.read_json(os.path.join(val_dir, nb_file))
        if len(df) < 2:
            continue

        order_col = f"true_order_{cell_type}"
        gt_order = df.sort_values(order_col).index.tolist()

        cell_ids = df.index.tolist()
        cell_features = ...  # extract per-cell feature vectors

        pred_order = predict_ordering(model, vectorize_pair_fn,
                                      cell_ids, cell_features)

        ground_truths.append(gt_order)
        predictions.append(pred_order)

    tau = kendall_tau(ground_truths, predictions)
    return tau
```

### Step 3: Establish baselines for comparison

To know whether the model is useful, compare against trivial baselines:

| Baseline | Description | Expected Kendall Tau |
|----------|-------------|---------------------|
| **Random** | Shuffle cells randomly | ~0.0 (by definition) |
| **File order** | Use the order cells appear in the JSON file | Weak positive (file order has some signal) |
| **Always-True** | Predict every pair as True (A before B) | Depends on data; likely near 0 |

A model Kendall Tau meaningfully above 0 validates the approach. The
competition leaderboard median was around 0.85 (using transformer-based
methods), so a traditional ML approach in the 0.2-0.5 range would already
demonstrate that the features carry signal.

### Step 4: Quick wins to improve the score

Once end-to-end evaluation works, these are the highest-leverage improvements
in rough priority order:

1. **Combine code + markdown predictions.** Currently the two cell types are
   ranked independently. The final notebook ordering interleaves them. Even a
   simple merge (rank code cells among themselves, rank markdown cells among
   themselves, then interleave by normalized rank) should improve Kendall Tau.

2. **Use more training data.** The pipeline currently subsamples 1-in-30.
   Increasing to 1-in-10 or 1-in-5 triples/sextuples the training set. The
   MLP and Random Forest should benefit most.

3. **Use LightGBM or XGBoost instead of scikit-learn classifiers.** Gradient
   boosted trees are the standard workhorse for tabular learning-to-rank.
   LightGBM even has a built-in `lambdarank` objective that directly optimizes
   ranking metrics. This replaces the Copeland aggregation with a principled
   LTR loss.

4. **Add positional features.** The current feature set doesn't encode where a
   cell appears relative to the notebook (e.g., fraction through the notebook,
   total cell count). These are cheap to add in P4 and strong for ranking.

## Verifiable Endpoint

**Definition of done:** a script or notebook cell that prints:

```
Random baseline Kendall Tau:  0.003
File-order baseline:          0.12
MLP pairwise model:           0.XX
```

where `0.XX` is a real number computed on the held-out validation notebooks.
This is a single measurable metric (Kendall Tau on validation set) that proves
the system can go from raw notebooks to ranked cell predictions -- the core
capability the project set out to build.

## Estimated Effort

| Step | New code | Dependencies |
|------|----------|--------------|
| Step 1: Aggregation | ~30 lines | Trained model from N5 |
| Step 2: Evaluation loop | ~40 lines | Validation data from P3, Kendall Tau from NX |
| Step 3: Baselines | ~15 lines | None |
| Step 4: Improvements | Variable | Steps 1-3 working |

Steps 1-3 are the minimal path. They require no new data processing, no new
dependencies, and no retraining -- just connecting the pieces that already
exist.
