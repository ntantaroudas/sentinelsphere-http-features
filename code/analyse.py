#!/usr/bin/env python3
"""Turn the Paper B result JSON files into the manuscript tables and figures."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # figstyle.py sits next to this script
import figstyle                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")


def ci(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def table_rq1(cic):
    lines = ["config,F1,FPR,FP,FN (threshold 0.5)"]
    for name, d in cic["rq1"].items():
        s = d["summary_t05"]
        lines.append(f"{name},{ci(s['f1'])},{ci(s['fpr'])},{s['fp'][0]:.1f},{s['fn'][0]:.1f}")
    open(os.path.join(RES, "tableB1_rq1.csv"), "w").write("\n".join(lines))


def clean_label(k):
    """CIC-IDS2017 web attack labels carry a mis-encoded separator in the released CSV, so map them
    to plain names by keyword."""
    low = k.lower()
    if "brute" in low:
        return "Brute force"
    if "sql" in low:
        return "SQL injection"
    if "xss" in low:
        return "Cross site scripting"
    return k


def fig_per_class(cic):
    import matplotlib.pyplot as plt
    oof = cic["per_class_oof"]["mlp_s21"]
    labels = [k for k in oof if k != "BENIGN"]
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH_IN, 3.5))
    xs = range(len(labels))
    rates = [oof[k]["rate"] for k in labels]
    err = [[oof[k]["rate"] - oof[k]["ci"][0] for k in labels], [oof[k]["ci"][1] - oof[k]["rate"] for k in labels]]
    ax.bar(xs, rates, color=figstyle.OKABE["blue"], yerr=err, capsize=4)
    for i, k in enumerate(labels):
        ax.text(i, oof[k]["ci"][1] + 0.03, f"{oof[k]['correct']} of {oof[k]['n']}", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([clean_label(k) for k in labels])
    ax.set_ylabel("detection rate per attack type")
    ax.set_ylim(0, 1.0)
    ax.set_title("Detection by attack type, with Wilson 95% intervals")
    figstyle.save(fig, os.path.join(FIG, "figB1_per_class"))


def fig_csic(csic):
    import matplotlib.pyplot as plt
    order = ["rf_s10_url_raw", "rf_s10_url_decoded", "rf_s10_urlbody_decoded", "rf_ext24_urlbody", "lr_char_ngram_urlbody"]
    names = {"rf_s10_url_raw": "10 features, request line, not decoded", "rf_s10_url_decoded": "10 features, request line, decoded",
             "rf_s10_urlbody_decoded": "10 features, line and body, decoded", "rf_ext24_urlbody": "24 features, line and body",
             "lr_char_ngram_urlbody": "character ngram model"}
    present = [k for k in order if k in csic["rq4"]]
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH_IN - 0.45, 3.9))
    xs = range(len(present))
    f1 = [csic["rq4"][k]["summary"]["f1"][0] for k in present]
    err = [[csic["rq4"][k]["summary"]["f1"][0] - csic["rq4"][k]["summary"]["f1"][1] for k in present],
           [csic["rq4"][k]["summary"]["f1"][2] - csic["rq4"][k]["summary"]["f1"][0] for k in present]]
    ax.bar(xs, f1, color=figstyle.SEQ[:len(present)], yerr=err, capsize=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([names[k] for k in present], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("F1 on CSIC 2010 (test)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Detecting anomalous web requests in CSIC 2010")
    figstyle.save(fig, os.path.join(FIG, "figB2_csic_f1"))


def main():
    os.makedirs(FIG, exist_ok=True)
    figstyle.apply()
    cic = json.load(open(os.path.join(RES, "cic_results.json")))
    table_rq1(cic)
    fig_per_class(cic)
    if os.path.exists(os.path.join(RES, "csic_results.json")):
        csic = json.load(open(os.path.join(RES, "csic_results.json")))
        fig_csic(csic)
    print("Paper B figures and tables written")


if __name__ == "__main__":
    main()
