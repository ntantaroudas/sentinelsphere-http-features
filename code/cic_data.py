"""Loading the CIC-IDS2017 Thursday morning web attack flows, original release and
the corrected release of Engelen, Rimmer and Joosen (2021)."""
import os

import numpy as np
import pandas as pd

DATA = os.environ.get("PAPER_B_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
ORIGINAL = os.path.join(DATA, "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
CORRECTED = os.path.join(DATA, "engelen_thursday.csv")

# The 21 network features named in the SentinelSphere paper (flow duration, packet counts,
# packet length statistics, rates, inter arrival statistics, flag counts, active and idle time)
SENTINEL21 = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Packet Length Mean", "Max Packet Length", "Min Packet Length", "Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "SYN Flag Count", "FIN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "Active Mean", "Idle Mean",
]
HTTP10 = ["request_complexity", "response_pattern", "path_entropy", "encoding_indicator", "scan_pattern",
          "brute_force_pattern", "injection_pattern", "xss_pattern", "automated_pattern", "suspicious_timing"]

# names used by the corrected release for the same quantities
CORRECTED_NAMES = {
    "Total Fwd Packets": "Total Fwd Packet", "Total Backward Packets": "Total Bwd packets",
    "Max Packet Length": "Packet Length Max", "Min Packet Length": "Packet Length Min",
    "Destination Port": "Dst Port", "Init_Win_bytes_forward": "FWD Init Win Bytes",
    "Init_Win_bytes_backward": "Bwd Init Win Bytes",
}


def _clean(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(0)
    return df


def load_original():
    df = pd.read_csv(ORIGINAL, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df["Label"] = (df["Label"].str.replace(r"Web Attack \W+ ", "Web Attack - ", regex=True)
                   .str.replace("Sql Injection", "SQL Injection"))
    df = _clean(df)
    df["y"] = (df["Label"] != "BENIGN").astype(int)
    return df


def original_features(df):
    return [c for c in df.columns if c not in ("Label", "y") and np.issubdtype(df[c].dtype, np.number)]


def load_corrected():
    df = pd.read_csv(CORRECTED)
    df = df[~df["Label"].str.startswith("Infiltration")].copy()   # keep BENIGN and web attack flows
    df = _clean(df)
    df["attempted"] = df["Label"].str.endswith("Attempted").astype(int)
    df["web"] = df["Label"].str.startswith("Web Attack").astype(int)
    return df


def corrected_name(col):
    return CORRECTED_NAMES.get(col, col)
