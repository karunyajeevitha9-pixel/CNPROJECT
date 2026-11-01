"""
sender.py - replays network flows into the live detection server (live_server.py).

Usage:
    python sender.py                      # 20 flows/s, 15% attack mix, drift after 600 flows
    python sender.py --rate 50 --count 3000 --drift-after 1000
    python sender.py --no-drift
    python sender.py --source live_flows.csv     # send your own captured flows instead

The default source is the CIC-IDS2017 flow file (all 78 model features), so the
model can score every flow. Flows captured by flow_extractor.py have only ~26
columns and are rejected by the server until the extractor produces all 78.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import requests

URL = "http://127.0.0.1:5000"
DATASET = "dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"


def safe_read(path, **kw):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def clean(row):
    out = {}
    for k, v in row.items():
        try:
            v = float(v)
            if np.isnan(v) or np.isinf(v):
                v = 0.0
        except (TypeError, ValueError):
            pass
        out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=DATASET)
    p.add_argument("--rate", type=float, default=20, help="flows per second")
    p.add_argument("--count", type=int, default=3000)
    p.add_argument("--attack-share", type=float, default=0.15)
    p.add_argument("--drift-after", type=int, default=600, help="flow number where drift starts")
    p.add_argument("--no-drift", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    try:
        requests.get(URL + "/health", timeout=3).raise_for_status()
    except Exception:
        raise SystemExit("Server not reachable. Start it first:  python live_server.py")

    df = safe_read(a.source)
    if df.empty:
        raise SystemExit(f"No data in {a.source}")
    df.columns = df.columns.str.strip()

    rng = np.random.default_rng(a.seed)
    labels = df["Label"].astype(str).str.strip() if "Label" in df.columns else None
    feat_cols = [c for c in df.columns if c != "Label"]

    if labels is not None:
        ben = np.where(labels == "BENIGN")[0]
        att = np.where(labels != "BENIGN")[0]
        n_att = int(a.count * a.attack_share)
        idx = np.concatenate([rng.choice(ben, a.count - n_att, replace=False),
                              rng.choice(att, n_att, replace=False)])
        rng.shuffle(idx)
    else:
        idx = rng.choice(len(df), min(a.count, len(df)), replace=False)

    num = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    drift_cols = rng.choice(feat_cols, size=len(feat_cols) // 3, replace=False)

    print(f"Sending {len(idx)} flows at {a.rate}/s to {URL}/flow"
          + ("" if a.no_drift else f" | simulated traffic drift starts at flow {a.drift_after}"))
    sent = fail = 0
    for n, i in enumerate(idx, 1):
        row = num.iloc[i].copy()
        if not a.no_drift and n > a.drift_after:
            ramp = min(1.0, (n - a.drift_after) / 500)       # gradual drift
            row[drift_cols] = row[drift_cols] * (1 + 0.8 * ramp)
        try:
            r = requests.post(URL + "/flow", json=clean(row.to_dict()), timeout=5)
            if r.ok:
                sent += 1
                ev = r.json()
                if ev.get("drift_detected"):
                    print(f"  >>> ADWIN DRIFT DETECTED at flow {ev['flow_number']}")
            else:
                fail += 1
                if fail <= 3:
                    print("  server error:", r.text[:200])
        except requests.RequestException as exc:
            fail += 1
            print("  request failed:", exc)
        if n % 100 == 0:
            print(f"  sent {n}/{len(idx)}")
        time.sleep(1.0 / a.rate)

    print(f"\nDONE. Sent: {sent}  Failed: {fail}")


if __name__ == "__main__":
    main()