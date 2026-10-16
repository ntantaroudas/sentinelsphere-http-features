"""Shared helpers for Paper B: models, metrics and confidence intervals."""
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def make_model(name, seed):
    """Models used throughout. 'mlp' mirrors the SentinelSphere 64-64 network
    (ReLU, Adam, learning rate 1e-3, L2 1e-3, early stopping with patience 5)."""
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 64), activation="relu", solver="adam", alpha=1e-3,
            learning_rate_init=1e-3, batch_size=128, max_iter=50, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=5, random_state=seed))
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    raise ValueError(name)


def fit(model, X, y, balanced=False):
    """Fit, optionally with inverse class frequency sample weights (weighted loss)."""
    if not balanced:
        return model.fit(X, y)
    w = np.where(y == 1, 0.5 / max(y.mean(), 1e-12), 0.5 / max(1 - y.mean(), 1e-12))
    last = model.steps[-1][0] if hasattr(model, "steps") else None
    if last is not None:
        return model.fit(X, y, **{f"{last}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def scores(model, X):
    return model.predict_proba(X)[:, 1]


def metrics(y, p, threshold=0.5):
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": prec, "recall": rec, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(y),
        "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan"),
        "pr_auc": average_precision_score(y, p) if y.sum() else float("nan"),
    }


def mean_ci(values):
    v = np.asarray([x for x in values if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    m = v.mean()
    if len(v) == 1:
        return m, m, m
    half = T_975.get(len(v), 1.96) * v.std(ddof=1) / math.sqrt(len(v))
    return m, m - half, m + half


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h
