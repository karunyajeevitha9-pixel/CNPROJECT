from flask import Flask, request, jsonify
from river.drift import ADWIN
from predict import predict_traffic
from collections import deque
from datetime import datetime
import threading
import json
import math
import os

app = Flask(__name__)

adwin = ADWIN(delta=0.002)
lock = threading.Lock()

total_flows = 0
total_anomalies = 0
total_normal = 0
drift_events = []
recent_flows = deque(maxlen=200)

# ==========================================
# Column map: model feature name -> CSV column name
# (created by the compatibility script as column_map.json)
# ==========================================
COLUMN_MAP = {}
if os.path.exists("column_map.json"):
    with open("column_map.json") as f:
        COLUMN_MAP = json.load(f)
    print(f"Loaded column_map.json ({len(COLUMN_MAP)} features)")
else:
    print("WARNING: column_map.json not found; incoming keys are used as-is")


def to_model_features(data):
    """Return a dict keyed by model feature names.
    Accepts keys already in model naming, or CSV column naming."""
    out = {}
    for feature, csv_col in COLUMN_MAP.items():
        if feature in data:
            value = data[feature]
        elif csv_col in data:
            value = data[csv_col]
        else:
            continue  # predict_traffic will report it as missing

        # JSON/model can't handle NaN or inf
        try:
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                value = 0.0
        except (TypeError, ValueError):
            pass

        out[feature] = value

    # if no map was loaded, fall back to the raw data
    return out if COLUMN_MAP else data


@app.route("/health", methods=["GET"])
def health():
    with lock:
        return jsonify({
            "status": "running",
            "service": "Live Network Anomaly Detection",
            "total_flows": total_flows
        })


@app.route("/flow", methods=["POST"])
def receive_flow():
    global total_flows, total_anomalies, total_normal

    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data:
        return jsonify({"error": "Invalid flow data"}), 400

    try:
        data = to_model_features(data)

        result = predict_traffic(data)
        row = result.iloc[0]

        score = float(row["Anomaly_Score"])
        status = str(row["Status"])
        alert = str(row["Alert"])

        with lock:
            total_flows += 1

            if status == "ANOMALY":
                total_anomalies += 1
                adwin_input = 1.0
            else:
                total_normal += 1
                adwin_input = 0.0

            adwin.update(adwin_input)

            drift_detected = bool(adwin.drift_detected)

            event = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "flow_number": total_flows,
                "anomaly_score": score,
                "status": status,
                "alert": alert,
                "adwin_input": adwin_input,
                "drift_detected": drift_detected,
                "adwin_window_size": int(adwin.width)
            }

            recent_flows.append(event)

            if drift_detected:
                drift_events.append(event.copy())

        return jsonify(event)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(exc),
            "type": type(exc).__name__
        }), 400


@app.route("/events", methods=["GET"])
def get_events():
    with lock:
        return jsonify({
            "total_flows": total_flows,
            "anomalies": total_anomalies,
            "normal": total_normal,
            "recent_flows": list(recent_flows)
        })


@app.route("/drift", methods=["GET"])
def get_drift():
    with lock:
        return jsonify({
            "detector": "ADWIN",
            "delta": 0.002,
            "total_flows": total_flows,
            "drift_event_count": len(drift_events),
            "window_size": int(adwin.width),
            "recent_drift_events": drift_events[-20:]
        })


if __name__ == "__main__":
    print("LIVE NETWORK ANOMALY SERVER")
    print("ADWIN DRIFT DETECTION ENABLED")
    app.run(host="0.0.0.0", port=5000, debug=False)