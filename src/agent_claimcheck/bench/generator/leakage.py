"""The leakage audit: can `final_claim.text` alone predict the label?

TF-IDF (1,2-gram, sublinear tf) + logistic regression, 5-fold stratified
cross-validated out-of-fold AUROC. Positive class is failure (the same
convention as `metrics.py`), so a text-only classifier that could guess the
label from wording alone would score high; the benchmark is meant to make
that hard.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def rank_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC by the Mann-Whitney rank method, ties averaged (metrics.py's
    method), positive = label 1.
    """
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def leakage_auroc_train_test(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
) -> float:
    """The same TF-IDF + logistic-regression pipeline as `leakage_auroc`, but
    fitted once on the full train split and scored on the test split,
    instead of cross-validated. Computed a single time, after the generator
    is frozen: never used to tune the generator.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(x_train, np.array(train_labels))
    scores = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    return rank_auroc(scores, np.array(test_labels))


def leakage_auroc(texts: list[str], labels: list[int]) -> float:
    """Out-of-fold AUROC of TF-IDF + logistic regression on `texts`.

    `labels` is 1 for failure (positive class), 0 for success, matching
    `metrics.py`'s convention. 5-fold `StratifiedKFold(shuffle=True,
    random_state=0)`.
    """
    y = np.array(labels)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(texts))
    for train_idx, test_idx in splitter.split(texts, y):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_test = vectorizer.transform([texts[i] for i in test_idx])
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(x_train, y[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    return rank_auroc(oof, y)
