"""HTTP request features for Paper B.

sentinel10()   the ten SentinelSphere HTTP features exactly as in the public
               http_feature_extractor.py, with one repair, the XSS pattern list is
               regex escaped so that the extractor runs at all ('alert(' crashed it).
               No URL decoding, like the original.
sentinel10(decode=True)
               the same ten features computed on the percent decoded request, which
               is what the pattern features need to see encoded payloads.
extended()     a richer set of 24 handcrafted request features used as a stronger baseline.
parse_csic()   parser for the raw CSIC 2010 request files.
"""
import math
import re
from collections import Counter
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd

LOGIN = ["login", "signin", "auth", "admin", "wp-login"]
SQL = ["union", "select", "insert", "update", "delete", "drop"]
XSS = ["<script", "javascript:", "onerror", "onload", "alert("]
BOTS = ["bot", "crawler", "spider", "scan", "nikto", "sqlmap"]
RE_LOGIN = "|".join(map(re.escape, LOGIN))
RE_SQL = "|".join(map(re.escape, SQL))
RE_XSS = "|".join(map(re.escape, XSS))          # the repair, the released code left 'alert(' unescaped
RE_BOTS = "|".join(map(re.escape, BOTS))


def entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def sentinel10(df, decode=False):
    """df needs columns path, status, size, user_agent (status and size may be constant)."""
    path = df["path"].fillna("")
    if decode:
        path = path.map(lambda s: unquote_plus(s, encoding="latin-1"))
    status = df["status"].astype(float)
    size = df["size"].astype(float)
    ua = df["user_agent"].fillna("")
    f = pd.DataFrame(index=df.index)
    f["request_complexity"] = (path.str.len() * path.str.count(r"[^a-zA-Z0-9/]") / (status + 1)).fillna(0)
    f["response_pattern"] = (size / (status + 1)).fillna(0)
    f["path_entropy"] = path.map(entropy)
    f["encoding_indicator"] = df["path"].fillna("").str.contains(r"%[0-9a-fA-F]{2}", regex=True).astype(int)
    f["scan_pattern"] = ((status == 404) & (path.str.len() < 50)).astype(int)
    f["brute_force_pattern"] = path.str.contains(RE_LOGIN, case=False, regex=True).astype(int)
    f["injection_pattern"] = (path.str.contains(RE_SQL, case=False, regex=True)
                              | path.str.contains("'", regex=False) | path.str.contains("--", regex=False)).astype(int)
    f["xss_pattern"] = path.str.contains(RE_XSS, case=False, regex=True).astype(int)
    f["automated_pattern"] = ua.str.contains(RE_BOTS, case=False, regex=True).astype(int)
    f["suspicious_timing"] = ((status >= 500) | (status == 403)).astype(int)
    return f


SPECIAL = {"quote": "'", "dquote": '"', "lt": "<", "gt": ">", "semicolon": ";", "paren": "(",
           "dash2": "--", "slashstar": "/*", "pipe": "|", "backtick": "`", "dotdot": "..", "pct": "%"}
KEYWORDS = ["select", "union", "insert", "drop", "or 1=1", "script", "alert", "onerror",
            "etc/passwd", "cmd", "exec", "sleep(", "benchmark("]


def extended(df):
    """24 handcrafted features on the decoded request text (URL and body)."""
    raw = df["path"].fillna("")
    dec = raw.map(lambda s: unquote_plus(s, encoding="latin-1").lower())
    f = pd.DataFrame(index=df.index)
    f["len"] = dec.str.len()
    f["n_params"] = dec.str.count("&") + dec.str.contains(r"\?|=", regex=True).astype(int)
    f["digits_ratio"] = dec.str.count(r"\d") / f["len"].clip(lower=1)
    f["upper_raw_ratio"] = raw.str.count(r"[A-Z]") / raw.str.len().clip(lower=1)
    f["nonalnum_ratio"] = dec.str.count(r"[^a-z0-9/]") / f["len"].clip(lower=1)
    f["pct_encodings"] = raw.str.count(r"%[0-9a-fA-F]{2}")
    f["entropy"] = dec.map(entropy)
    f["max_token_len"] = dec.map(lambda s: max((len(t) for t in re.split(r"[/?&=+]", s)), default=0))
    for name, ch in SPECIAL.items():
        f[f"c_{name}"] = dec.str.count(re.escape(ch))
    for kw in KEYWORDS[:4]:
        f[f"k_{re.sub('[^a-z]', '', kw)}"] = dec.str.contains(re.escape(kw), regex=True).astype(int)
    return f


REQ_START = re.compile(r"^(GET|POST|PUT) (\S+) HTTP/1\.[01]$")


def parse_csic(path, label):
    """Parse a raw CSIC 2010 file into method, url (path and query), body, user_agent."""
    rows, cur = [], None
    with open(path, encoding="latin-1") as fh:
        lines = fh.read().split("\n")
    i = 0
    while i < len(lines):
        m = REQ_START.match(lines[i].strip())
        if m:
            if cur:
                rows.append(cur)
            url = m.group(2).replace("http://localhost:8080", "")
            cur = {"method": m.group(1), "url": url, "body": "", "user_agent": "", "label": label}
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                if lines[i].lower().startswith("user-agent:"):
                    cur["user_agent"] = lines[i].split(":", 1)[1].strip()
                i += 1
            if cur["method"] in ("POST", "PUT"):
                i += 1
                if i < len(lines) and lines[i].strip() and not REQ_START.match(lines[i].strip()):
                    cur["body"] = lines[i].strip()
            continue
        i += 1
    if cur:
        rows.append(cur)
    return pd.DataFrame(rows)


def csic_frame(files):
    df = pd.concat([parse_csic(p, lab) for p, lab in files], ignore_index=True)
    df["status"] = 200          # requests only, no responses in CSIC 2010
    df["size"] = 0
    return df


def with_path(df, include_body):
    out = df.copy()
    out["path"] = (out["url"] + np.where(out["body"] != "", "?" + out["body"], "")) if include_body else out["url"]
    return out
