# AI4Code

#### A Coding Challenge to match Markdown comments to code in Jupyter notebooks

> "The goal of this competition is to understand the relationship between code
> and comments in Python notebooks. You are challenged to reconstruct the order
> of markdown cells in a given notebook based on the order of the code cells,
> demonstrating comprehension of which natural language references which code."
>
> -- [Kaggle AI4Code Competition](https://www.kaggle.com/c/AI4Code)

## Project Status

This project was developed as an exercise during the Erdos Institute 2022
program. It implements a **pairwise learning-to-rank** approach: for each
notebook, every pair of cells is compared and a binary classifier predicts
which cell comes first. The full cell ordering is then reconstructable from
these pairwise predictions.

**What works today:**

- Full preprocessing pipeline (P1-P4) that turns raw notebook JSON into
  trainable pairwise feature matrices
- Feature engineering combining TF-IDF text features with AST-derived code
  ancestry features (L-values / R-values)
- Binary classifiers (Naive Bayes, MLP, Random Forest) trained on pairwise
  data, reaching ~66% pairwise accuracy on code cells
- Kendall Tau metric implementation matching the competition specification

**End-to-end evaluation** (`ltr_evaluate.py`) closes the loop by:

- Aggregating pairwise predictions into full cell orderings via Copeland
  ranking (sum of win probabilities)
- Evaluating with the competition's Kendall Tau metric on 140 held-out
  validation notebooks
- Comparing against random and file-order baselines

See [LTR_ROADMAP.md](LTR_ROADMAP.md) for further improvement ideas.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| **N1** `N1_reconstruct_nb.ipynb` | Reconstruct Jupyter notebooks from competition JSON format |
| **N2** `N2_test_ast.ipynb` | Test AST parsing to extract variable assignments (L-values) and usage (R-values) |
| **N3** `N3_nlpclean.ipynb` | Test NLP cleaning: tokenization, stopword removal, stemming |
| **N4** `N4_preprocess_pipeline.ipynb` | Main preprocessing pipeline (P1 through P4) -- does the heavy lifting |
| **N5** `N5_Training.ipynb` | Train classifiers and evaluate pairwise accuracy |
| **NX** `NX_kendall_tau.ipynb` | Demonstrate and test the Kendall Tau competition metric |
| **E2E** `ltr_evaluate.py` | End-to-end: train, predict notebook orderings, evaluate Kendall Tau |

## Pipeline Architecture

```
Raw JSON notebooks (139K files, 2 GB)
  │
  ├─ P1: Subsample (1-in-30), extract cell order, split code/markdown
  │
  ├─ P2: NLP preprocessing (tokenize, stem) + AST feature extraction
  │       (left-values: assignments/imports, right-values: variable usage)
  │
  ├─ P3: Vectorize
  │       • Markdown cells → TF-IDF (229 features)
  │       • Code cells → CountVectorizer (250 features)
  │
  └─ P4: Create pairwise combinations (n*(n-1)/2 per notebook)
          • Concatenate feature vectors of cell A and cell B
          • Add ancestry features (L-value/R-value overlap)
          • Label: True if A comes before B in ground truth
```

**Output:** Sparse matrices (`*_P4_*_X.npz`) and label arrays (`*_P4_*.npy`)
for training and validation, with code and markdown handled separately.

## Results

### Pairwise Classification Accuracy (Code Cells)

| Model | Accuracy | Precision | MSE |
|-------|----------|-----------|-----|
| Gaussian Naive Bayes | 57.9% | 65.0% | 0.421 |
| MLP (2x100, SGD) | 65.6% | 65.3% | 0.344 |

### End-to-End Kendall Tau (140 Validation Notebooks)

| Method | Code | Markdown | Combined |
|--------|------|----------|----------|
| Random baseline | — | — | -0.006 |
| File-order baseline | — | — | 0.408 |
| SGD (logistic) | 0.212 | 0.140 | 0.183 |
| MLP (100 hidden) | 0.319 | 0.164 | 0.258 |

The MLP model achieves a combined Kendall Tau of **0.258**, well above random
(~0) but below the file-order baseline (0.408). The code-cell ordering
(0.319) is notably stronger than markdown (0.164), likely because AST
ancestry features provide useful ordering signal. The file-order baseline
is surprisingly strong because the competition JSON files partially preserve
the original notebook structure.

## Data

Data is not included in this repo (2 GB). Download instructions:
https://www.kaggle.com/c/AI4Code

## Tech Stack

Python 3.9+ · scikit-learn · pandas · NumPy · SciPy · NLTK · Keras/TensorFlow · matplotlib · seaborn
