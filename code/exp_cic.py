#!/usr/bin/env python3
"""Paper B, experiments on CIC-IDS2017 Thursday morning web attack flows.

RQ1  protocol sensitivity of the SentinelSphere style detector (constant HTTP columns,
     duplicates, threshold 0.4 versus 0.5, weighted loss, feature sets, model families)
RQ2  what the corrected labels of Engelen et al. (2021) change
RQ3  single feature shortcuts
Writes JSON and CSV files to ../results.  Run:  python exp_cic.py  (about one hour on 2 vCPUs)
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.tree import DecisionTreeClassifier

from cic_data import (HTTP10, SENTINEL21, corrected_name, load_corrected, load_original,
                      original_features)
from common import fit, make_model, mean_ci, metrics, scores, wilson

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SEEDS = [0, 1, 2, 3, 4]
SHORTCUTS = ["Destination Port", "Init_Win_bytes_forward", "Init_Win_bytes_backward"]
LOG = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def repeated_split(X, y, model_name, balanced=False, seeds=SEEDS, thresholds=(0.4, 0.5)):
    runs = []
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=s, stratify=y)
        m = make_model(model_name, s)
        fit(m, Xtr, ytr, balanced)
        p = scores(m, Xte)
        runs.append({str(t): metrics(yte, p, t) for t in thresholds})
    return runs


def summarise(runs, threshold="0.5"):
    keys = ["precision", "recall", "f1", "fpr", "accuracy", "roc_auc", "pr_auc", "fp", "fn"]
    return {k: mean_ci([r[threshold][k] for r in runs]) for k in keys}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    df = load_original()
    feats = original_features(df)
    y = df["y"].to_numpy()
    res["dataset"] = {
        "rows": int(len(df)), "attacks": int(y.sum()), "label_counts": df["Label"].value_counts().to_dict(),
        "duplicate_rows": int(df.duplicated(subset=feats + ["Label"]).sum()),
        "duplicate_attack_rows": int(df[df.y == 1].duplicated(subset=feats + ["Label"]).sum()),
        "n_features_all": len(feats),
    }
    log(f"original: {res['dataset']['rows']} rows, {res['dataset']['attacks']} attacks, "
        f"{res['dataset']['duplicate_rows']} duplicate rows")

    # RQ1 ---------------------------------------------------------------------------
    X21 = df[SENTINEL21].to_numpy()
    X31 = np.hstack([X21, np.zeros((len(df), len(HTTP10)))])      # HTTP columns are constant on flows
    configs = {
        "mlp_s21_plus_http_zero": (X31, y, "mlp", False),
        "mlp_s21": (X21, y, "mlp", False),
        "mlp_s21_weighted": (X21, y, "mlp", True),
        "mlp_all": (df[feats].to_numpy(), y, "mlp", False),
        "rf_s21": (X21, y, "rf", False),
        "rf_all": (df[feats].to_numpy(), y, "rf", False),
        "hgb_all": (df[feats].to_numpy(), y, "hgb", False),
        "lr_all": (df[feats].to_numpy(), y, "lr", False),
        "rf_all_minus_shortcuts": (df[[c for c in feats if c not in SHORTCUTS]].to_numpy(), y, "rf", False),
    }
    dedup = df.drop_duplicates(subset=feats + ["Label"])
    configs["mlp_s21_dedup"] = (dedup[SENTINEL21].to_numpy(), dedup["y"].to_numpy(), "mlp", False)
    configs["rf_all_dedup"] = (dedup[feats].to_numpy(), dedup["y"].to_numpy(), "rf", False)
    res["rq1"] = {}
    for name, (X, yy, model, bal) in configs.items():
        t0 = time.time()
        runs = repeated_split(X, yy, model, bal)
        res["rq1"][name] = {"runs": runs, "summary_t05": summarise(runs, "0.5"), "summary_t04": summarise(runs, "0.4")}
        s = res["rq1"][name]["summary_t05"]
        log(f"RQ1 {name:26s} F1 {s['f1'][0]:.4f}  FPR {s['fpr'][0]:.5f}  FP {s['fp'][0]:.1f}  "
            f"FN {s['fn'][0]:.1f}  ({time.time() - t0:.0f}s)")
        json.dump(res, open(os.path.join(OUT, "cic_results.json"), "w"), indent=1, default=float)

    # per class detection, out of fold over all flows (5 fold, stratified by attack type)
    res["per_class_oof"] = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, X, model, thr in (("mlp_s21", X21, "mlp", 0.4), ("rf_all", df[feats].to_numpy(), "rf", 0.5)):
        t0 = time.time()
        p = np.zeros(len(df))
        for tr, te in skf.split(X, df["Label"]):
            m = make_model(model, 0)
            fit(m, X[tr], y[tr])
            p[te] = scores(m, X[te])
        out = {}
        for lab in sorted(df["Label"].unique()):
            idx = (df["Label"] == lab).to_numpy()
            k = int((p[idx] >= thr).sum()) if lab != "BENIGN" else int((p[idx] < thr).sum())
            lo, hi = wilson(k, int(idx.sum()))
            out[lab] = {"correct": k, "n": int(idx.sum()), "rate": k / idx.sum(), "ci": [lo, hi]}
        res["per_class_oof"][name] = out
        log(f"OOF {name}: " + ", ".join(f"{k} {v['correct']}/{v['n']}" for k, v in out.items()) +
            f" ({time.time() - t0:.0f}s)")
    json.dump(res, open(os.path.join(OUT, "cic_results.json"), "w"), indent=1, default=float)

    # RQ3 single feature shortcuts ---------------------------------------------------------
    aucs = []
    for c in feats:
        x = df[c].to_numpy()
        if np.all(x == x[0]):
            continue
        a = roc_auc_score(y, x)
        aucs.append((c, max(a, 1 - a)))
    aucs.sort(key=lambda t: -t[1])
    res["rq3_single_feature_auc"] = aucs[:15]
    stumps = {}
    for c, _ in aucs[:5]:
        X1 = df[[c]].to_numpy()
        p = cross_val_predict(DecisionTreeClassifier(max_depth=3, random_state=0), X1, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        stumps[c] = metrics(y, p, 0.5)
    res["rq3_depth3_tree_single_feature"] = stumps
    log("RQ3 top single features: " + ", ".join(f"{c} {a:.3f}" for c, a in aucs[:6]))
    json.dump(res, open(os.path.join(OUT, "cic_results.json"), "w"), indent=1, default=float)

    # RQ2 corrected labels ----------------------------------------------------------------------
    cf = load_corrected()
    c21 = [corrected_name(c) for c in SENTINEL21]
    common = [corrected_name(c) for c in feats if corrected_name(c) in cf.columns]
    res["rq2_dataset"] = {"rows": int(len(cf)), "label_counts": cf["Label"].value_counts().to_dict(),
                          "web_flows": int(cf["web"].sum()), "attempted": int((cf["web"] & cf["attempted"]).sum()),
                          "real_attacks": int((cf["web"] & (1 - cf["attempted"])).sum()), "common_features": len(common)}
    log(f"corrected: {res['rq2_dataset']}")
    labelings = {
        "attempted_as_attack": (cf, cf["web"].to_numpy()),
        "attempted_as_benign": (cf, (cf["web"] & (1 - cf["attempted"])).to_numpy()),
        "attempted_removed": (cf[~((cf["web"] == 1) & (cf["attempted"] == 1))],
                              None),
    }
    sub = labelings["attempted_removed"][0]
    labelings["attempted_removed"] = (sub, sub["web"].to_numpy())
    res["rq2"] = {}
    for lname, (frame, yy) in labelings.items():
        for model, cols in (("rf", common), ("mlp", c21)):
            t0 = time.time()
            runs = repeated_split(frame[cols].to_numpy(), yy, model, False, thresholds=(0.5,))
            res["rq2"][f"{model}_{lname}"] = {"runs": runs, "summary": summarise(runs, "0.5"),
                                               "positives": int(yy.sum())}
            s = res["rq2"][f"{model}_{lname}"]["summary"]
            log(f"RQ2 {model}_{lname:22s} pos {int(yy.sum())}  F1 {s['f1'][0]:.4f}  P {s['precision'][0]:.3f}  "
                f"R {s['recall'][0]:.3f}  ({time.time() - t0:.0f}s)")
            json.dump(res, open(os.path.join(OUT, "cic_results.json"), "w"), indent=1, default=float)

    # cross release, train on the original labels, score corrected web flows
    m = make_model("rf", 0)
    fit(m, df[SENTINEL21].to_numpy(), y)
    p = scores(m, cf[c21].to_numpy())
    cross = {}
    for lab in sorted(cf["Label"].unique()):
        idx = (cf["Label"] == lab).to_numpy()
        k = int((p[idx] >= 0.5).sum())
        cross[lab] = {"flagged": k, "n": int(idx.sum()), "rate": k / idx.sum(), "ci": list(wilson(k, int(idx.sum())))}
    res["rq2_cross_release_rf_s21"] = cross
    log("cross release: " + ", ".join(f"{k} {v['flagged']}/{v['n']}" for k, v in cross.items()))
    res["log"] = LOG
    json.dump(res, open(os.path.join(OUT, "cic_results.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
