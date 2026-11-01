
import os
import sys
import pickle
import numpy as np
import pandas as pd

# ==========================================
# 1. LOAD TRAINED FILES
# ==========================================

required_files = [
    "saved_model.pkl",
    "scaler.pkl",
    "features.pkl",
    "live_flows.csv"
]

for file in required_files:
    if not os.path.exists(file):
        print("ERROR: File not found:", file)
        sys.exit(1)

try:
    with open("saved_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("features.pkl", "rb") as f:
        features = list(pickle.load(f))

except Exception as e:
    print("ERROR loading model files:", e)
    sys.exit(1)

print("\nMODEL LOADED SUCCESSFULLY")
print("Model:", type(model).__name__)
print("Scaler:", type(scaler).__name__)
print("Expected features:", len(features))

# ==========================================
# 2. LOAD LIVE TRAFFIC
# ==========================================

try:
    df = pd.read_csv("live_flows.csv")
    df.columns = df.columns.str.strip()
except Exception as e:
    print("ERROR reading CSV:", e)
    sys.exit(1)

print("\nLIVE TRAFFIC INFORMATION")
print("========================")
print("Total captured flows:", len(df))

if df.empty:
    print("No traffic available.")
    sys.exit(0)

# ==========================================
# 3. FEATURE MAPPING
# ==========================================

mapping = {
    "Destination Port": "dst_port",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "totlen_fwd_pkts",
    "Total Length of Bwd Packets": "totlen_bwd_pkts",

    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",

    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",

    "Flow Bytes/s": "flow_byts_s",
    "Flow Packets/s": "flow_pkts_s",

    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",

    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",

    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",

    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",

    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",

    "Fwd Packets/s": "fwd_pkts_s",
    "Bwd Packets/s": "bwd_pkts_s",

    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",

    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "CWE Flag Count": "cwr_flag_count",
    "ECE Flag Count": "ece_flag_cnt",

    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "pkt_size_avg",

    "Avg Fwd Segment Size": "fwd_seg_size_avg",
    "Avg Bwd Segment Size": "bwd_seg_size_avg",

    "Fwd Header Length.1": "fwd_header_len",

    "Fwd Avg Bytes/Bulk": "fwd_byts_b_avg",
    "Fwd Avg Packets/Bulk": "fwd_pkts_b_avg",
    "Fwd Avg Bulk Rate": "fwd_blk_rate_avg",

    "Bwd Avg Bytes/Bulk": "bwd_byts_b_avg",
    "Bwd Avg Packets/Bulk": "bwd_pkts_b_avg",
    "Bwd Avg Bulk Rate": "bwd_blk_rate_avg",

    "Subflow Fwd Packets": "subflow_fwd_pkts",
    "Subflow Fwd Bytes": "subflow_fwd_byts",
    "Subflow Bwd Packets": "subflow_bwd_pkts",
    "Subflow Bwd Bytes": "subflow_bwd_byts",

    "Init_Win_bytes_forward": "init_fwd_win_byts",
    "Init_Win_bytes_backward": "init_bwd_win_byts",

    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",

    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",

    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min"
}

# ==========================================
# 4. PREPARE MODEL INPUT
# ==========================================

X = pd.DataFrame(index=df.index)
missing = []

for feature in features:

    source = mapping.get(feature, feature)

    if source not in df.columns:
        missing.append((feature, source))
        continue

    X[feature] = pd.to_numeric(
        df[source],
        errors="coerce"
    )

if missing:
    print("\nMISSING FEATURES:")
    for feature, source in missing:
        print(feature, "<-", source)

    print("\nPREDICTION STOPPED.")
    print("Fix feature extraction before continuing.")
    sys.exit(1)

# Enforce exact training feature order
X = X[features]

print("\nFEATURE VALIDATION")
print("==================")
print("Expected features:", len(features))
print("Available features:", len(X.columns))

# ==========================================
# 5. HANDLE INVALID VALUES
# ==========================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

valid_mask = X.notna().all(axis=1)

X_valid = X.loc[valid_mask].copy()

invalid_count = len(df) - len(X_valid)

if X_valid.empty:
    print("\nNo valid flows available.")
    sys.exit(0)

# ==========================================
# 6. APPLY TRAINED SCALER
# ==========================================

try:
    X_scaled = scaler.transform(X_valid)
except Exception as e:
    print("\nSCALING ERROR:", e)
    sys.exit(1)

# ==========================================
# 7. ISOLATION FOREST PREDICTION
# ==========================================

try:
    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)
except Exception as e:
    print("\nPREDICTION ERROR:", e)
    sys.exit(1)

# ==========================================
# 8. GENERATE RESULTS
# ==========================================

results = pd.DataFrame(index=X_valid.index)

results["Source IP"] = (
    df.loc[X_valid.index, "src_ip"].values
    if "src_ip" in df.columns else "UNKNOWN"
)

results["Destination IP"] = (
    df.loc[X_valid.index, "dst_ip"].values
    if "dst_ip" in df.columns else "UNKNOWN"
)

results["Destination Port"] = (
    df.loc[X_valid.index, "dst_port"].values
    if "dst_port" in df.columns else 0
)

results["Anomaly Score"] = scores

results["Prediction"] = np.where(
    predictions == -1,
    "ANOMALY DETECTED",
    "NORMAL"
)

# ==========================================
# 9. DISPLAY FLOW RESULTS
# ==========================================

normal = int(np.sum(predictions == 1))
anomalies = int(np.sum(predictions == -1))

print("\nFLOW DETECTION RESULTS")
print("======================")

print(
    results[
        ["Source IP", "Destination IP",
         "Destination Port", "Anomaly Score", "Prediction"]
    ].tail(20).to_string(index=False)
)

# ==========================================
# 10. SAVE PREDICTIONS
# ==========================================

results.to_csv(
    "prediction_results.csv",
    index=False
)

# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\nFINAL SUMMARY")
print("=============")

print("Total captured flows:", len(df))
print("Valid flows:", len(X_valid))
print("Normal flows:", normal)
print("Anomalies:", anomalies)
print("Invalid flows:", invalid_count)

print("\nANOMALY SCORE STATISTICS")
print("========================")
print("Mean:", round(float(np.mean(scores)), 6))
print("Standard deviation:", round(float(np.std(scores)), 6))
print("Minimum:", round(float(np.min(scores)), 6))
print("Maximum:", round(float(np.max(scores)), 6))

print("\nResults saved: prediction_results.csv")
print("\nML DETECTION COMPLETED")
