#!/usr/bin/env python3
"""
Alternative LTR approaches for AI4Code cell ordering.

Compares three simpler approaches against the original pairwise pipeline:

1. Pointwise Regression: Predict each cell's normalized position directly.
   No pairwise feature generation, no aggregation — just argsort predictions.

2. LightGBM LambdaRank: Use LightGBM's built-in lambdarank objective which
   handles pairwise comparisons internally and optimizes ranking metrics.

3. XGBoost rank:pairwise: Similar idea, XGBoost's pairwise ranking objective.

All three use the same underlying features (TF-IDF for markdown, CountVec +
AST ancestry for code) but differ in how they frame the learning problem.

Usage:
    python3 ltr_alternatives.py
"""

import os
import re
import ast
import random
import warnings
from bisect import bisect
from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix, hstack as sp_hstack

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

DATA_DIR = "data"

# ---------------------------------------------------------------------------
# Feature extraction (shared with original pipeline)
# ---------------------------------------------------------------------------

EXTRA_STOPWORDS = ["li", "br", "http", "https", "www", "com", "class", "alert"]
RE_HTML = re.compile(r'<[^>]+>')
RE_CHAR = re.compile(r"[^a-zA-Z0-9]")
RE_NOMAGIC = re.compile(r"^[!%].*\n?", flags=re.MULTILINE)
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english")) | set(EXTRA_STOPWORDS)


def preprocessing_unit(sourcetext: str) -> list:
    words = sourcetext.lower()
    words = RE_HTML.sub('', words)
    words = RE_CHAR.sub(' ', words)
    words = nltk.word_tokenize(words)
    words = [w for w in words if w not in STOP_WORDS]
    words = [w for w in words if not w.isdigit()]
    words = [w for w in words if len(w) < 20]
    words = [STEMMER.stem(w) for w in words]
    return words


def ast_unit(sourcetext: str):
    sourcetext = RE_NOMAGIC.sub(" ", sourcetext.lower())
    try:
        root = ast.parse(sourcetext)
    except SyntaxError:
        return ([], [])

    all_vars = {node.id for node in ast.walk(root) if isinstance(node, ast.Name)}
    assignment = {n.id for node in ast.walk(root)
                  if isinstance(node, ast.Assign)
                  for n in node.targets
                  if isinstance(n, ast.Name)}
    all_imports = {n.asname or n.name for node in ast.walk(root)
                   if isinstance(node, (ast.Import, ast.ImportFrom))
                   for n in node.names}

    right_values = all_vars - assignment - all_imports
    left_values = assignment | all_imports
    return list(left_values), list(right_values)


# ---------------------------------------------------------------------------
# Kendall Tau metric
# ---------------------------------------------------------------------------

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    if total_2max == 0:
        return 0.0
    return 1 - 4 * total_inversions / total_2max


# ---------------------------------------------------------------------------
# Load notebooks (same as ltr_evaluate.py)
# ---------------------------------------------------------------------------

def load_notebooks(orders_df, indices):
    notebooks = []
    for idx in indices:
        row = orders_df.iloc[idx]
        nb_id = row["id"]
        cell_order = row["cell_order"].split() if isinstance(row["cell_order"], str) else row["cell_order"]
        true_order = {h: i for i, h in enumerate(cell_order)}

        path = os.path.join(DATA_DIR, "train", f"{nb_id}.json")
        if not os.path.exists(path):
            continue
        df = pd.read_json(path)
        df["true_order"] = df.index.map(true_order)

        df_code = df[df.cell_type == "code"].copy()
        df_md = df[df.cell_type == "markdown"].copy()
        df_code["true_order_code"] = df_code["true_order"].rank(method="min").astype(int)
        df_md["true_order_md"] = df_md["true_order"].rank(method="min").astype(int)

        notebooks.append({
            "id": nb_id,
            "code": df_code,
            "md": df_md,
            "all": df,
            "cell_order": cell_order,
        })
    return notebooks


def fit_vectorizers(notebooks):
    md_sentences = []
    code_sentences = []

    for nb in notebooks:
        for _, row in nb["md"].iterrows():
            tokens = preprocessing_unit(row["source"])
            md_sentences.append(" ".join(tokens))

        for _, row in nb["code"].iterrows():
            lv, rv = ast_unit(row["source"])
            code_sentences.append(" ".join(lv + rv))

    tfidf = TfidfVectorizer(min_df=0.01)
    tfidf.fit(md_sentences)

    countvec = CountVectorizer(max_features=250)
    countvec.fit(code_sentences)

    print(f"  TF-IDF features: {len(tfidf.get_feature_names_out())}")
    print(f"  Code features:   {len(countvec.get_feature_names_out())}")
    return tfidf, countvec


# ---------------------------------------------------------------------------
# Build pointwise (per-cell) training data
# ---------------------------------------------------------------------------

def build_pointwise_data(notebooks, tfidf, countvec):
    """
    Build per-cell feature matrices and targets for ALL cell types combined.

    Each cell gets:
      - Its text features (TF-IDF for markdown, CountVec for code)
      - A cell_type indicator (0=code, 1=markdown)
      - Normalized position target in [0, 1]

    Returns: X (n_cells, n_features), y (n_cells,), groups (list of group sizes)
    """
    n_tfidf = len(tfidf.get_feature_names_out())
    n_code = len(countvec.get_feature_names_out())
    # Unified feature width: max(n_tfidf, n_code) + 2 ancestry + 1 cell_type
    # + 2 structural features (n_cells_in_notebook, fraction_of_type)
    n_feat = max(n_tfidf, n_code) + 2 + 1 + 2
    X_rows = []
    y_rows = []
    groups = []  # number of cells per notebook (for LTR group queries)
    cell_ids_per_nb = []

    for nb in notebooks:
        all_df = nb["all"]
        n_total = len(all_df)
        if n_total < 2:
            continue

        n_code_cells = len(nb["code"])
        n_md_cells = len(nb["md"])
        nb_cell_ids = []
        nb_X = []
        nb_y = []

        # Process code cells
        for cell_id, row in nb["code"].iterrows():
            lv, rv = ast_unit(row["source"])
            vec = countvec.transform([" ".join(lv + rv)]).toarray()[0]
            feat = np.zeros(n_feat)
            feat[:n_code] = vec
            feat[max(n_tfidf, n_code)] = len(lv)   # ancestry: num L-values
            feat[max(n_tfidf, n_code) + 1] = len(rv)  # ancestry: num R-values
            feat[max(n_tfidf, n_code) + 2] = 0  # cell_type = code
            feat[max(n_tfidf, n_code) + 3] = n_total  # notebook size
            feat[max(n_tfidf, n_code) + 4] = n_code_cells / max(n_total, 1)

            # Target: normalized position in [0, 1]
            target = row["true_order"] / max(n_total - 1, 1)
            nb_X.append(feat)
            nb_y.append(target)
            nb_cell_ids.append(cell_id)

        # Process markdown cells
        for cell_id, row in nb["md"].iterrows():
            tokens = preprocessing_unit(row["source"])
            vec = tfidf.transform([" ".join(tokens)]).toarray()[0]
            feat = np.zeros(n_feat)
            feat[:n_tfidf] = vec
            feat[max(n_tfidf, n_code) + 2] = 1  # cell_type = markdown
            feat[max(n_tfidf, n_code) + 3] = n_total
            feat[max(n_tfidf, n_code) + 4] = n_md_cells / max(n_total, 1)

            target = row["true_order"] / max(n_total - 1, 1)
            nb_X.append(feat)
            nb_y.append(target)
            nb_cell_ids.append(cell_id)

        X_rows.extend(nb_X)
        y_rows.extend(nb_y)
        groups.append(len(nb_X))
        cell_ids_per_nb.append(nb_cell_ids)

    X = np.array(X_rows)
    y = np.array(y_rows)
    return X, y, groups, cell_ids_per_nb


# ---------------------------------------------------------------------------
# Approach 1: Pointwise Regression
# ---------------------------------------------------------------------------

def evaluate_pointwise_regression(train_nbs, val_nbs, tfidf, countvec):
    """
    Simplest possible approach: predict each cell's position directly
    with Ridge regression, then argsort to get the ordering.
    """
    print("\n" + "-" * 60)
    print("APPROACH 1: Pointwise Regression (Ridge)")
    print("-" * 60)

    print("  Building pointwise training data...")
    t1 = perf_counter()
    X_train, y_train, _, _ = build_pointwise_data(train_nbs, tfidf, countvec)
    print(f"  Training data: {X_train.shape} in {perf_counter()-t1:.1f}s")

    print("  Building pointwise validation data...")
    X_val, y_val, val_groups, val_cell_ids = build_pointwise_data(val_nbs, tfidf, countvec)
    print(f"  Validation data: {X_val.shape}")

    print("  Training Ridge regression...")
    t1 = perf_counter()
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    print(f"  Done in {perf_counter()-t1:.1f}s")

    # Predict and evaluate
    y_pred = model.predict(X_val)

    gt_all, pred_all = [], []
    offset = 0
    for g, nb, cell_ids in zip(val_groups, val_nbs, val_cell_ids):
        if g < 2:
            offset += g
            continue
        # Ground truth ordering
        all_df = nb["all"]
        gt = [x[0] for x in sorted(
            [(cid, all_df.loc[cid, "true_order"]) for cid in cell_ids],
            key=lambda x: x[1]
        )]
        # Predicted ordering: argsort predicted positions
        preds_for_nb = y_pred[offset:offset + g]
        ranked_indices = np.argsort(preds_for_nb)
        pred = [cell_ids[i] for i in ranked_indices]

        gt_all.append(gt)
        pred_all.append(pred)
        offset += g

    tau = kendall_tau(gt_all, pred_all)
    print(f"\n  >>> Pointwise Regression Kendall Tau: {tau:.4f} ({len(gt_all)} notebooks)")
    return tau


# ---------------------------------------------------------------------------
# Approach 2: LightGBM LambdaRank
# ---------------------------------------------------------------------------

def evaluate_lgbm_lambdarank(train_nbs, val_nbs, tfidf, countvec):
    """
    Use LightGBM's built-in lambdarank objective.
    The library handles pairwise comparisons internally.
    We provide: features, relevance labels, and group sizes.
    """
    print("\n" + "-" * 60)
    print("APPROACH 2: LightGBM LambdaRank")
    print("-" * 60)

    print("  Building pointwise training data...")
    t1 = perf_counter()
    X_train, y_train_raw, train_groups, _ = build_pointwise_data(
        train_nbs, tfidf, countvec)

    # LambdaRank needs integer relevance labels (higher = more relevant)
    # Convert normalized position to relevance: cells near the start get
    # higher relevance. We use 10 bins.
    y_train = np.round((1 - y_train_raw) * 10).astype(int)
    print(f"  Training data: {X_train.shape} in {perf_counter()-t1:.1f}s")

    print("  Building pointwise validation data...")
    X_val, y_val_raw, val_groups, val_cell_ids = build_pointwise_data(
        val_nbs, tfidf, countvec)
    y_val = np.round((1 - y_val_raw) * 10).astype(int)
    print(f"  Validation data: {X_val.shape}")

    print("  Training LightGBM LambdaRank...")
    t1 = perf_counter()

    train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
    val_data = lgb.Dataset(X_val, label=y_val, group=val_groups,
                           reference=train_data)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "num_leaves": 63,
        "learning_rate": 0.1,
        "min_child_samples": 20,
        "n_estimators": 200,
        "verbosity": -1,
        "seed": 137,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(period=50)],
    )
    print(f"  Done in {perf_counter()-t1:.1f}s")

    # Predict: LambdaRank outputs relevance scores (higher = earlier)
    y_pred = model.predict(X_val)

    gt_all, pred_all = [], []
    offset = 0
    for g, nb, cell_ids in zip(val_groups, val_nbs, val_cell_ids):
        if g < 2:
            offset += g
            continue
        all_df = nb["all"]
        gt = [x[0] for x in sorted(
            [(cid, all_df.loc[cid, "true_order"]) for cid in cell_ids],
            key=lambda x: x[1]
        )]
        # Higher score = should come earlier (lower position)
        preds_for_nb = y_pred[offset:offset + g]
        ranked_indices = np.argsort(-preds_for_nb)  # descending
        pred = [cell_ids[i] for i in ranked_indices]

        gt_all.append(gt)
        pred_all.append(pred)
        offset += g

    tau = kendall_tau(gt_all, pred_all)
    print(f"\n  >>> LightGBM LambdaRank Kendall Tau: {tau:.4f} ({len(gt_all)} notebooks)")
    return tau


# ---------------------------------------------------------------------------
# Approach 3: XGBoost rank:pairwise
# ---------------------------------------------------------------------------

def evaluate_xgb_pairwise(train_nbs, val_nbs, tfidf, countvec):
    """
    Use XGBoost's rank:pairwise objective.
    Similar to LightGBM LambdaRank but uses XGBoost's implementation.
    """
    print("\n" + "-" * 60)
    print("APPROACH 3: XGBoost rank:pairwise")
    print("-" * 60)

    print("  Building pointwise training data...")
    t1 = perf_counter()
    X_train, y_train_raw, train_groups, _ = build_pointwise_data(
        train_nbs, tfidf, countvec)

    # XGBoost rank:pairwise uses relevance labels too
    y_train = np.round((1 - y_train_raw) * 10).astype(int)
    print(f"  Training data: {X_train.shape} in {perf_counter()-t1:.1f}s")

    print("  Building pointwise validation data...")
    X_val, y_val_raw, val_groups, val_cell_ids = build_pointwise_data(
        val_nbs, tfidf, countvec)
    y_val = np.round((1 - y_val_raw) * 10).astype(int)
    print(f"  Validation data: {X_val.shape}")

    print("  Training XGBoost rank:pairwise...")
    t1 = perf_counter()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(train_groups)
    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(val_groups)

    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "max_depth": 6,
        "eta": 0.1,
        "min_child_weight": 20,
        "seed": 137,
        "verbosity": 0,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dval, "val")],
        verbose_eval=50,
    )
    print(f"  Done in {perf_counter()-t1:.1f}s")

    y_pred = model.predict(dval)

    gt_all, pred_all = [], []
    offset = 0
    for g, nb, cell_ids in zip(val_groups, val_nbs, val_cell_ids):
        if g < 2:
            offset += g
            continue
        all_df = nb["all"]
        gt = [x[0] for x in sorted(
            [(cid, all_df.loc[cid, "true_order"]) for cid in cell_ids],
            key=lambda x: x[1]
        )]
        preds_for_nb = y_pred[offset:offset + g]
        ranked_indices = np.argsort(-preds_for_nb)  # descending
        pred = [cell_ids[i] for i in ranked_indices]

        gt_all.append(gt)
        pred_all.append(pred)
        offset += g

    tau = kendall_tau(gt_all, pred_all)
    print(f"\n  >>> XGBoost rank:pairwise Kendall Tau: {tau:.4f} ({len(gt_all)} notebooks)")
    return tau


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = perf_counter()

    print("=" * 60)
    print("AI4Code: Alternative LTR Approaches")
    print("=" * 60)

    # Load notebook metadata
    orders = pd.read_csv(os.path.join(DATA_DIR, "train_orders.csv"))

    # Splits (matching original pipeline)
    val_indices = [i for i in range(len(orders)) if i % 1000 == 137]
    train_indices = [i for i in range(len(orders)) if i % 30 == 0 and i % 1000 != 137]

    print(f"\nLoading {len(val_indices)} validation notebooks...")
    val_notebooks = load_notebooks(orders, val_indices)
    print(f"  Loaded {len(val_notebooks)} validation notebooks")

    # Use more training data than the original (1-in-30 -> still 1-in-30 but
    # more notebooks for vectorizer fitting)
    n_train = min(len(train_indices), 1500)
    print(f"\nLoading {n_train} training notebooks...")
    train_notebooks = load_notebooks(orders, train_indices[:n_train])
    print(f"  Loaded {len(train_notebooks)} training notebooks")

    print("\nFitting vectorizers...")
    tfidf, countvec = fit_vectorizers(train_notebooks)

    # Run all three approaches
    tau_ridge = evaluate_pointwise_regression(
        train_notebooks, val_notebooks, tfidf, countvec)

    tau_lgbm = evaluate_lgbm_lambdarank(
        train_notebooks, val_notebooks, tfidf, countvec)

    tau_xgb = evaluate_xgb_pairwise(
        train_notebooks, val_notebooks, tfidf, countvec)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: All Approaches")
    print("=" * 60)
    print(f"  Original pairwise MLP (from ltr_evaluate.py):  0.258")
    print(f"  File-order baseline:                           0.408")
    print(f"  ---")
    print(f"  Pointwise Ridge Regression:                    {tau_ridge:.4f}")
    print(f"  LightGBM LambdaRank:                           {tau_lgbm:.4f}")
    print(f"  XGBoost rank:pairwise:                         {tau_xgb:.4f}")
    print(f"\n  Total time: {perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
