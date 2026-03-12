#!/usr/bin/env python3
"""
End-to-end learning-to-rank evaluation for AI4Code.

Trains pairwise classifiers on pre-built P4 feature matrices,
then evaluates notebook-level Kendall Tau on held-out validation notebooks
by rebuilding features from raw JSONs.

Usage:
    python3 ltr_evaluate.py
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
from scipy.sparse import load_npz, csr_matrix

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

warnings.filterwarnings("ignore")

# Ensure NLTK data is available
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

DATA_DIR = "data"

# ---------------------------------------------------------------------------
# NLP + AST feature extraction (mirrors N4 pipeline)
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
# Kendall Tau metric (from NX)
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
# Build per-notebook data from raw JSONs
# ---------------------------------------------------------------------------

def load_notebooks(orders_df, indices):
    """Load raw notebooks and split into code/markdown with ground truth order."""
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
    """Fit TF-IDF (markdown) and CountVectorizer (code) on training notebooks."""
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


def vectorize_notebook_cells(nb, tfidf, countvec):
    """Vectorize all cells in a notebook, returning per-cell feature dicts."""
    code_cells = []
    for cell_id, row in nb["code"].iterrows():
        tokens = preprocessing_unit(row["source"])
        lv, rv = ast_unit(row["source"])
        vec = countvec.transform([" ".join(lv + rv)]).toarray()[0]
        code_cells.append({
            "cell_id": cell_id,
            "vec": vec,
            "lv": lv,
            "rv": rv,
            "true_order": row["true_order"],
            "true_order_within": row["true_order_code"],
        })

    md_cells = []
    for cell_id, row in nb["md"].iterrows():
        tokens = preprocessing_unit(row["source"])
        vec = tfidf.transform([" ".join(tokens)]).toarray()[0]
        md_cells.append({
            "cell_id": cell_id,
            "vec": vec,
            "true_order": row["true_order"],
            "true_order_within": row["true_order_md"],
        })

    return code_cells, md_cells


def build_pairwise_features_code(cells):
    """Build pairwise feature matrix for code cells (matching P4 format).
    Feature vector: [vec_A (250) | vec_B (250) | ancestry_A_first | ancestry_B_first]
    Total: 502 features.
    """
    if len(cells) < 2:
        return None, None, None
    X_rows = []
    y_rows = []
    pairs = []
    for (i, a), (j, b) in combinations(enumerate(cells), 2):
        feat = np.concatenate([a["vec"], b["vec"]])
        # Ancestry features
        a_first = len(set(b["rv"]) & set(a["lv"]))
        b_first = len(set(a["rv"]) & set(b["lv"]))
        feat = np.append(feat, [a_first, b_first])
        X_rows.append(feat)
        y_rows.append(a["true_order_within"] < b["true_order_within"])
        pairs.append((i, j))
    return np.array(X_rows), np.array(y_rows), pairs


def build_pairwise_features_md(cells):
    """Build pairwise feature matrix for markdown cells (matching P4 format).
    Feature vector: [vec_A (229) | vec_B (229)]
    Total: 458 features.
    """
    if len(cells) < 2:
        return None, None, None
    X_rows = []
    y_rows = []
    pairs = []
    for (i, a), (j, b) in combinations(enumerate(cells), 2):
        feat = np.concatenate([a["vec"], b["vec"]])
        X_rows.append(feat)
        y_rows.append(a["true_order_within"] < b["true_order_within"])
        pairs.append((i, j))
    return np.array(X_rows), np.array(y_rows), pairs


# ---------------------------------------------------------------------------
# Ranking aggregation: pairwise predictions -> full ordering
# ---------------------------------------------------------------------------

def predict_ordering(model, cells, build_fn):
    """Predict cell ordering using Copeland aggregation of pairwise probabilities."""
    if len(cells) <= 1:
        return list(range(len(cells)))

    X, _, pairs = build_fn(cells)
    if X is None:
        return list(range(len(cells)))

    probs = model.predict_proba(X)[:, 1]  # P(A before B)

    wins = np.zeros(len(cells))
    for (i, j), p in zip(pairs, probs):
        wins[i] += p
        wins[j] += (1.0 - p)

    # Rank by descending win count -> earliest cell first
    ranking = sorted(range(len(cells)), key=lambda c: -wins[c])
    return ranking


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = perf_counter()

    print("=" * 60)
    print("AI4Code: End-to-End Learning-to-Rank Evaluation")
    print("=" * 60)

    # Load notebook metadata
    orders = pd.read_csv(os.path.join(DATA_DIR, "train_orders.csv"))

    # Identify train/val splits (matching original pipeline)
    val_indices = [i for i in range(len(orders)) if i % 1000 == 137]
    # Use a subset of training notebooks for fitting vectorizers (speed)
    train_indices = [i for i in range(len(orders)) if i % 30 == 0 and i % 1000 != 137]

    print(f"\nLoading {len(val_indices)} validation notebooks...")
    val_notebooks = load_notebooks(orders, val_indices)
    print(f"  Loaded {len(val_notebooks)} validation notebooks")

    # Use a manageable subset of training notebooks to fit vectorizers
    # (we need the same vocabulary as the P4 data was built with)
    n_train_for_vocab = min(len(train_indices), 500)
    print(f"\nLoading {n_train_for_vocab} training notebooks for vectorizer fitting...")
    train_notebooks = load_notebooks(orders, train_indices[:n_train_for_vocab])
    print(f"  Loaded {len(train_notebooks)} training notebooks")

    print("\nFitting vectorizers...")
    tfidf, countvec = fit_vectorizers(train_notebooks)

    # -----------------------------------------------------------------------
    # Train models on P4 data
    # -----------------------------------------------------------------------
    print("\nLoading P4 training data...")
    Xc = load_npz(os.path.join(DATA_DIR, "train_P4_code_X.npz"))
    yc = np.load(os.path.join(DATA_DIR, "train_P4_code.npy"))
    Xm = load_npz(os.path.join(DATA_DIR, "train_P4_md_X.npz"))
    ym = np.load(os.path.join(DATA_DIR, "train_P4_md.npy"))
    print(f"  Code pairs: {Xc.shape}, Markdown pairs: {Xm.shape}")

    # SGDClassifier with log loss = logistic regression, very fast on large data
    print("\nTraining SGD (logistic) on code pairs...")
    t1 = perf_counter()
    sgd_code = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=50,
                             random_state=137, n_jobs=-1)
    sgd_code.fit(Xc, yc)
    print(f"  Done in {perf_counter()-t1:.1f}s, "
          f"train acc: {sgd_code.score(Xc, yc):.4f}")

    print("\nTraining SGD (logistic) on markdown pairs...")
    t1 = perf_counter()
    sgd_md = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=50,
                           random_state=137, n_jobs=-1)
    sgd_md.fit(Xm, ym)
    print(f"  Done in {perf_counter()-t1:.1f}s, "
          f"train acc: {sgd_md.score(Xm, ym):.4f}")

    # Also train MLP (smaller, faster version)
    print("\nTraining MLP on code pairs...")
    t1 = perf_counter()
    mlp_code = MLPClassifier(solver='sgd', alpha=1e-3, activation="logistic",
                             learning_rate_init=0.01, hidden_layer_sizes=(100,),
                             random_state=137, learning_rate="invscaling",
                             max_iter=30)
    mlp_code.fit(Xc, yc)
    print(f"  Done in {perf_counter()-t1:.1f}s, "
          f"train acc: {mlp_code.score(Xc, yc):.4f}")

    print("\nTraining MLP on markdown pairs...")
    t1 = perf_counter()
    mlp_md = MLPClassifier(solver='sgd', alpha=1e-3, activation="logistic",
                           learning_rate_init=0.01, hidden_layer_sizes=(100,),
                           random_state=137, learning_rate="invscaling",
                           max_iter=30)
    mlp_md.fit(Xm, ym)
    print(f"  Done in {perf_counter()-t1:.1f}s, "
          f"train acc: {mlp_md.score(Xm, ym):.4f}")

    # -----------------------------------------------------------------------
    # Evaluate: predict per-notebook ordering and compute Kendall Tau
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluating on validation notebooks...")
    print("=" * 60)

    models = {
        "SGD (logistic)": (sgd_code, sgd_md),
        "MLP (100)": (mlp_code, mlp_md),
    }
    results = {}  # model_name -> (tau_code, tau_md, tau_combined)

    for model_name, (m_code, m_md) in models.items():
        gt_code_all, pred_code_all = [], []
        gt_md_all, pred_md_all = [], []
        gt_combined_all, pred_combined_all = [], []

        for nb in val_notebooks:
            code_cells, md_cells = vectorize_notebook_cells(nb, tfidf, countvec)

            # --- Code cells ---
            if len(code_cells) >= 2:
                gt_order = [c["cell_id"] for c in sorted(code_cells,
                            key=lambda c: c["true_order_within"])]
                ranking = predict_ordering(m_code, code_cells,
                                           build_pairwise_features_code)
                pred_order = [code_cells[r]["cell_id"] for r in ranking]
                gt_code_all.append(gt_order)
                pred_code_all.append(pred_order)

            # --- Markdown cells ---
            if len(md_cells) >= 2:
                gt_order = [c["cell_id"] for c in sorted(md_cells,
                            key=lambda c: c["true_order_within"])]
                ranking = predict_ordering(m_md, md_cells,
                                           build_pairwise_features_md)
                pred_order = [md_cells[r]["cell_id"] for r in ranking]
                gt_md_all.append(gt_order)
                pred_md_all.append(pred_order)

            # --- Combined (all cells, interleaved) ---
            all_cells_gt = sorted(
                [(c["cell_id"], c["true_order"]) for c in code_cells] +
                [(c["cell_id"], c["true_order"]) for c in md_cells],
                key=lambda x: x[1])
            gt_combined = [x[0] for x in all_cells_gt]

            # Predict within-type orderings and merge by normalized rank
            pred_combined = []
            if len(code_cells) >= 2:
                code_ranking = predict_ordering(m_code, code_cells,
                                                build_pairwise_features_code)
            elif len(code_cells) == 1:
                code_ranking = [0]
            else:
                code_ranking = []

            if len(md_cells) >= 2:
                md_ranking = predict_ordering(m_md, md_cells,
                                              build_pairwise_features_md)
            elif len(md_cells) == 1:
                md_ranking = [0]
            else:
                md_ranking = []

            # Assign normalized positions [0, 1] within each type, then merge
            merged = []
            n_code = len(code_ranking)
            n_md = len(md_ranking)
            for rank_pos, orig_idx in enumerate(code_ranking):
                norm_pos = rank_pos / max(n_code, 1)
                merged.append((code_cells[orig_idx]["cell_id"], norm_pos, "code"))
            for rank_pos, orig_idx in enumerate(md_ranking):
                norm_pos = rank_pos / max(n_md, 1)
                merged.append((md_cells[orig_idx]["cell_id"], norm_pos, "md"))

            # Sort by normalized position (ties broken by type - code first)
            merged.sort(key=lambda x: (x[1], x[2]))
            pred_combined_order = [x[0] for x in merged]

            if len(gt_combined) >= 2:
                gt_combined_all.append(gt_combined)
                pred_combined_all.append(pred_combined_order)

        # Compute Kendall Tau
        tau_code = kendall_tau(gt_code_all, pred_code_all) if gt_code_all else 0
        tau_md = kendall_tau(gt_md_all, pred_md_all) if gt_md_all else 0
        tau_combined = kendall_tau(gt_combined_all, pred_combined_all) if gt_combined_all else 0

        results[model_name] = (tau_code, tau_md, tau_combined)
        print(f"\n  {model_name}:")
        print(f"    Code cells Kendall Tau:     {tau_code:.4f}  ({len(gt_code_all)} notebooks)")
        print(f"    Markdown cells Kendall Tau: {tau_md:.4f}  ({len(gt_md_all)} notebooks)")
        print(f"    Combined Kendall Tau:       {tau_combined:.4f}  ({len(gt_combined_all)} notebooks)")

    # -----------------------------------------------------------------------
    # Baselines
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Baselines")
    print("=" * 60)

    # Random baseline
    random.seed(42)
    gt_rand, pred_rand = [], []
    for nb in val_notebooks:
        code_cells, md_cells = vectorize_notebook_cells(nb, tfidf, countvec)
        all_cells = (
            [(c["cell_id"], c["true_order"]) for c in code_cells] +
            [(c["cell_id"], c["true_order"]) for c in md_cells]
        )
        if len(all_cells) < 2:
            continue
        gt = [x[0] for x in sorted(all_cells, key=lambda x: x[1])]
        pred = gt.copy()
        random.shuffle(pred)
        gt_rand.append(gt)
        pred_rand.append(pred)

    tau_random = kendall_tau(gt_rand, pred_rand)
    print(f"\n  Random shuffle Kendall Tau:    {tau_random:.4f}")

    # File-order baseline (order cells appear in JSON)
    gt_file, pred_file = [], []
    for nb in val_notebooks:
        all_df = nb["all"]
        if len(all_df) < 2:
            continue
        gt = all_df.sort_values("true_order").index.tolist()
        pred = all_df.index.tolist()  # JSON file order
        gt_file.append(gt)
        pred_file.append(pred)

    tau_file = kendall_tau(gt_file, pred_file)
    print(f"  File-order baseline Kendall Tau: {tau_file:.4f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Random baseline:     {tau_random:.4f}")
    print(f"  File-order baseline: {tau_file:.4f}")
    for model_name, (tc, tm, tb) in results.items():
        print(f"  {model_name} (combined): {tb:.4f}  (code: {tc:.4f}, md: {tm:.4f})")
    elapsed = perf_counter() - t0
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
