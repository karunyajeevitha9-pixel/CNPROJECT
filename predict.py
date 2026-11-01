
import pickle
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. LOAD TRAINED FILES
# ==========================================

with open("saved_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = list(pickle.load(f))

print("Model loaded:", type(model).__name__)
print("Expected features:", len(feature_names))


# ==========================================
# 2. FEATURE NAME MAPPING
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
    "Flow IAT Total": "flow_iat_total",

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
# 3. NORMALIZE FEATURE NAMES
# ==========================================

def normalize(name):
    import re
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================

def predict_traffic(data):

    if isinstance(data, dict):
        data = pd.DataFrame([data])

    data = data.copy()

    data.columns = data.columns.astype(str).str.strip()

    if "Label" in data.columns:
        data = data.drop(columns=["Label"])

    available = {
        normalize(col): col for col in data.columns
    }

    X = pd.DataFrame(index=data.index)

    missing = []

    for feature in feature_names:

        source = mapping.get(feature, feature)

        found = None

        # Try exact match
        if source in data.columns:
            found = source

        elif feature in data.columns:
            found = feature

        # Try normalized match
        else:
            for candidate in [source, feature]:
                key = normalize(candidate)

                if key in available:
                    found = available[key]
                    break

        if found is None:
            missing.append(feature)
        else:
            X[feature] = pd.to_numeric(
                data[found], errors="coerce"
            )

    # Check missing features
    if missing:
        print("MISSING MODEL FEATURES:")
        for feature in missing:
            print(feature)

        raise ValueError(
            f"Input data is missing {len(missing)} required features"
        )

    # Clean invalid values
    X = X.replace(
        [np.inf, -np.inf], np.nan
    )

    X = X.fillna(0)

    # Ensure correct training order
    X = X[feature_names]

    # Apply trained scaler
    scaled = scaler.transform(X)

    scaled = pd.DataFrame(
        scaled,
        columns=feature_names,
        index=X.index
    )

    # Isolation Forest prediction
    scores = model.decision_function(scaled)

    predictions = model.predict(scaled)

    # Prepare results
    results = []

    for score, prediction in zip(scores, predictions):

        if prediction == -1:
            status = "ANOMALY"
            alert = "ALERT"
        else:
            status = "NORMAL"
            alert = "NO ALERT"

        results.append({
            "Anomaly_Score": float(score),
            "Status": status,
            "Alert": alert
        })

    return pd.DataFrame(results)


# ==========================================
# 5. TEST PREDICTION
# ==========================================

if __name__ == "__main__":

    print("\nLIVE TRAFFIC PREDICTION TEST")
    print("============================")

    if not os.path.exists("live_flows.csv"):
        print("ERROR: live_flows.csv not found")
        raise SystemExit(1)

    df = pd.read_csv("live_flows.csv")

    df.columns = df.columns.str.strip()

    print("Captured flows:", len(df))

    result = predict_traffic(df)

    print("\nPREDICTION RESULTS")
    print("==================")

    print(result.to_string(index=False))

    print("\nSUMMARY")
    print("=======")

    print(
        "Normal:",
        (result["Status"] == "NORMAL").sum()
    )

    print(
        "Anomalies:",
        (result["Status"] == "ANOMALY").sum()
    )

    print("\nPREDICTION COMPLETED SUCCESSFULLY")
