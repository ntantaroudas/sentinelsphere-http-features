#!/usr/bin/env python3
"""Paper B, experiments on CSIC 2010 HTTP requests.

RQ4  how much do the ten SentinelSphere HTTP features detect when request text exists,
     with and without percent decoding, with and without the POST body, against a richer
     handcrafted set and a character n-gram baseline
RQ5  which of the ten features carry the signal (drop one ablation)
Run:  python exp_csic.py   (about 30 minutes on 2 vCPUs)
"""
import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import fit, make_model, mean_ci, metrics, scores
from http_features import csic_frame, extended, sentinel10, with_path

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PAPER_B_CSIC", os.path.join(HERE, "..", "data", "csic"))
OUT = os.path.join(HERE, "..", "results")
SEEDS = [0, 1, 2, 3, 4]
LOG = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def load():
    files = [(os.path.join(DATA, "normalTrafficTraining.txt"), 0),
             (os.path.join(DATA, "normalTrafficTest.txt"), 0),
             (os.path.join(DATA, "anomalousTrafficTest.txt"), 1)]
    df = csic_frame(files)
    key = ["method", "url", "body"]
    conflicts = df.groupby(key)["label"].nunique()
    n_conflict = int((conflicts > 1).sum())
    df = df.drop_duplicates(subset=key + ["label"])
    df = df[~df.set_index(key).index.isin(conflicts[conflicts > 1].index)].reset_index(drop=True)
    return df, n_conflict


def main():
    os.makedirs(OUT, exist_ok=True)
    raw, n_conflict = load()
    y = raw["label"].to_numpy()
    res = {"dataset": {"requests_after_dedup": int(len(raw)), "anomalous": int(y.sum()),
                       "conflicting_duplicates_removed": n_conflict,
                       "post_share_anomalous": float((raw[raw.label == 1]["method"] == "POST").mean()),
                       "post_share_normal": float((raw[raw.label == 0]["method"] == "POST").mean()),
                       "distinct_user_agents": int(raw["user_agent"].nunique())}}
    log(f"CSIC after dedup {res['dataset']}")

    url_only = with_path(raw, include_body=False)
    url_body = with_path(raw, include_body=True)
    reps = {
        "s10_url_raw": sentinel10(url_only, decode=False),
        "s10_url_decoded": sentinel10(url_only, decode=True),
        "s10_urlbody_decoded": sentinel10(url_body, decode=True),
        "ext24_urlbody": extended(url_body),
    }
    res["constant_features"] = {k: [c for c in v.columns if v[c].nunique() <= 1] for k, v in reps.items()}
    log(f"constant features {res['constant_features']}")

    res["rq4"] = {}
    for rname, F in reps.items():
        X = F.to_numpy(dtype=float)
        for model in ("rf", "mlp"):
            t0 = time.time()
            runs = []
            for s in SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=s, stratify=y)
                m = make_model(model, s)
                fit(m, Xtr, ytr)
                runs.append(metrics(yte, scores(m, Xte), 0.5))
            summ = {k: mean_ci([r[k] for r in runs]) for k in ("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc")}
            res["rq4"][f"{model}_{rname}"] = {"runs": runs, "summary": summ}
            log(f"RQ4 {model}_{rname:22s} F1 {summ['f1'][0]:.4f}  P {summ['precision'][0]:.3f}  "
                f"R {summ['recall'][0]:.3f}  FPR {summ['fpr'][0]:.4f}  AUC {summ['roc_auc'][0]:.4f} ({time.time() - t0:.0f}s)")
            json.dump(res, open(os.path.join(OUT, "csic_results.json"), "w"), indent=1, default=float)

    # character n-gram baseline on decoded URL and body
    text = url_body["path"].str.lower().to_numpy()
    t0 = time.time()
    runs = []
    for s in SEEDS:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=s, stratify=y)
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=30000, sublinear_tf=True)
        Xtr, Xte = vec.fit_transform(text[tr]), vec.transform(text[te])
        m = LogisticRegression(max_iter=2000, C=4.0)
        m.fit(Xtr, y[tr])
        runs.append(metrics(y[te], m.predict_proba(Xte)[:, 1], 0.5))
    summ = {k: mean_ci([r[k] for r in runs]) for k in ("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc")}
    res["rq4"]["lr_char_ngram_urlbody"] = {"runs": runs, "summary": summ}
    log(f"RQ4 lr_char_ngram_urlbody F1 {summ['f1'][0]:.4f}  FPR {summ['fpr'][0]:.4f}  AUC {summ['roc_auc'][0]:.4f} ({time.time() - t0:.0f}s)")

    # RQ5 drop one ablation of the ten features (decoded URL and body, random forest)
    F = reps["s10_urlbody_decoded"]
    res["rq5_ablation"] = {}

    def rf_auc_f1(X, seeds=(0, 1, 2)):
        out = []
        for s in seeds:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=s, stratify=y)
            m = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=s).fit(Xtr, ytr)
            mm = metrics(yte, m.predict_proba(Xte)[:, 1], 0.5)
            out.append((mm["f1"], mm["roc_auc"]))
        return np.mean([o[0] for o in out]), np.mean([o[1] for o in out])

    base_f1, base_auc = rf_auc_f1(F.to_numpy(dtype=float))
    res["rq5_ablation"]["_all"] = {"f1": base_f1, "auc": base_auc}
    for c in F.columns:
        f1, auc = rf_auc_f1(F.drop(columns=[c]).to_numpy(dtype=float))
        res["rq5_ablation"][c] = {"f1": f1, "auc": auc, "delta_f1": f1 - base_f1, "delta_auc": auc - base_auc}
        log(f"RQ5 drop {c:20s} dF1 {f1 - base_f1:+.4f}  dAUC {auc - base_auc:+.4f}")
    res["log"] = LOG
    json.dump(res, open(os.path.join(OUT, "csic_results.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
