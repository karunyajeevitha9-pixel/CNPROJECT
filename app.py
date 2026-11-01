
import streamlit as st
import pandas as pd
import requests
from datetime import datetime

API = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="Drift-Adaptive Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #e2e8f0 !important;
}
p, label {
    color: #cbd5e1;
}
div[data-testid="stMetric"] {
    background: #172238;
    border: 1px solid #334155;
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetric"] label {
    color: #93c5fd;
}
div[data-testid="stMetricValue"] {
    color: white;
}
hr {
    border-color: #263449;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.title("🛡️ Drift-Adaptive Unsupervised Anomaly Detection")
st.subheader(
    "for Encrypted Enterprise Network Traffic in Real Time"
)
st.caption(
    "Information Security Research Project | "
    "Isolation Forest + ADWIN Concept Drift Detection"
)

st.divider()

# =====================================================
# AUTO REFRESH
# =====================================================

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=2000, key="live_refresh")
except ImportError:
    st.warning(
        "Automatic refresh unavailable. Install streamlit-autorefresh."
    )

# =====================================================
# FLASK API
# =====================================================

try:
    health_response = requests.get(API + "/health", timeout=3)
    events_response = requests.get(API + "/events", timeout=3)
    drift_response = requests.get(API + "/drift", timeout=3)

    health_response.raise_for_status()
    events_response.raise_for_status()
    drift_response.raise_for_status()

    health = health_response.json()
    events = events_response.json()
    drift = drift_response.json()

except requests.RequestException:
    st.error("Flask server is not reachable. Run python live_server.py first.")
    st.stop()

total = events.get("total_flows", 0)
normal = events.get("normal", 0)
anomalies = events.get("anomalies", 0)
drift_count = drift.get("drift_event_count", 0)
recent = events.get("recent_flows", [])

# =====================================================
# SYSTEM STATUS
# =====================================================

st.header("⚙️ System Status")

s1, s2 = st.columns(2)

with s1:
    if health.get("status") == "running":
        st.success("Flask Detection Server: ONLINE")
    else:
        st.error("Flask Detection Server: OFFLINE")

with s2:
    st.success("ADWIN Drift Detector: ACTIVE")

st.divider()

# =====================================================
# LIVE NETWORK OVERVIEW
# =====================================================

st.header("📊 Live Network Security Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Flows Analysed", total)
c2.metric("Normal Traffic", normal)
c3.metric("Anomalies Detected", anomalies)
c4.metric("Drift Events", drift_count)

st.divider()

# =====================================================
# ZERO-DAY / ANOMALY MONITOR
# =====================================================

st.header("🚨 Potential Unknown Threat Monitor")

if anomalies > 0:
    st.error(
        f"{anomalies} anomalous network flows detected. "
        "These may indicate suspicious or previously unseen traffic."
    )
else:
    st.success("No anomalous flows detected so far.")

if drift_count > 0:
    st.warning(
        f"ADWIN has detected {drift_count} concept drift event(s). "
        "The incoming traffic distribution may have changed."
    )
else:
    st.info("No ADWIN drift event has been reported yet.")

st.divider()

# =====================================================
# LIVE CLASSIFICATION AND SCORE TREND
# =====================================================

left, right = st.columns(2)

with left:
    st.subheader("Traffic Classification")

    if total > 0:
        chart_data = pd.DataFrame({
            "Traffic Type": ["Normal", "Anomaly"],
            "Number of Flows": [normal, anomalies]
        })

        st.bar_chart(
            chart_data.set_index("Traffic Type")
        )
    else:
        st.info("Waiting for incoming network flows...")

with right:
    st.subheader("Anomaly Score Trend")

    if recent:
        score_df = pd.DataFrame(recent)

        if "anomaly_score" in score_df.columns:
            score_df["flow_number"] = pd.to_numeric(
                score_df["flow_number"], errors="coerce"
            )
            score_df["anomaly_score"] = pd.to_numeric(
                score_df["anomaly_score"], errors="coerce"
            )
            score_df = score_df.dropna(
                subset=["flow_number", "anomaly_score"]
            )

            if not score_df.empty:
                st.line_chart(
                    score_df.set_index("flow_number")["anomaly_score"]
                )
    else:
        st.info("No anomaly scores received yet.")

st.divider()

# =====================================================
# RECENT DETECTION RESULTS
# =====================================================

st.header("🔍 Recent Network Detection Results")

if recent:
    recent_df = pd.DataFrame(recent)

    columns = [
        col for col in [
            "timestamp",
            "flow_number",
            "anomaly_score",
            "status",
            "alert",
            "drift_detected"
        ] if col in recent_df.columns
    ]

    st.dataframe(
        recent_df[columns].iloc[::-1],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No network traffic received yet.")

st.divider()

# =====================================================
# ADWIN DRIFT MONITORING
# =====================================================

st.header("📈 Concept Drift Monitoring – ADWIN")

d1, d2, d3 = st.columns(3)

d1.metric("Detector", drift.get("detector", "ADWIN"))
d2.metric("Current Window Size", drift.get("window_size", 0))
d3.metric("Total Drift Events", drift_count)

drift_events = drift.get("recent_drift_events", [])

if drift_events:
    drift_df = pd.DataFrame(drift_events)

    st.subheader("Detected Drift Events")

    st.dataframe(
        drift_df.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )
else:
    st.success("No drift events detected yet.")

st.divider()

# =====================================================
# BENCHMARK COMPARISON
# =====================================================

st.header("📊 Quantitative Performance & Benchmark Comparison")

st.caption(
    "Comparison of static, non-adaptive and proposed "
    "drift-adaptive anomaly detection approaches."
)

st.warning(
    "Important: The values below are reference benchmark figures, "
    "not experimentally measured results from the current live system. "
    "Do not report them as your own experimental results without validation."
)

benchmark = pd.DataFrame({
    "Evaluation Metric": [
        "Zero-Day Detection Rate (Recall)",
        "False Positive Rate (FPR)",
        "Average Latency per Flow",
        "Throughput Capacity",
        "Adaptation Recovery Time"
    ],
    "Static Unsupervised Baseline (Isolation Forest)": [
        "71.4%",
        "14.2%",
        "4.8 ms",
        "2.1 Gbps",
        "N/A"
    ],
    "Standard Autoencoder (Non-Adaptive)": [
        "79.2%",
        "11.6%",
        "8.1 ms",
        "4.5 Gbps",
        "N/A"
    ],
    "Proposed Drift-Adaptive Approach": [
        "94.8%",
        "1.1%",
        "0.88 ms",
        "10.2 Gbps",
        "< 1 Second"
    ]
})

st.subheader("Benchmark Comparison Table")

st.dataframe(
    benchmark,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Reference comparison only. The autoencoder and proposed approach "
    "figures require independent experimental validation."
)

st.divider()

# =====================================================
# 1. ZERO-DAY DETECTION RATE
# =====================================================

st.subheader("1. Zero-Day Detection Rate Comparison")

detection_data = pd.DataFrame({
    "Model": [
        "Isolation Forest",
        "Standard Autoencoder",
        "Proposed Drift-Adaptive"
    ],
    "Detection Rate (%)": [
        71.4,
        79.2,
        94.8
    ]
})

st.bar_chart(
    detection_data.set_index("Model")
)

st.info(
    "Detection rate indicates the proportion of attack flows "
    "correctly identified. A genuine zero-day evaluation requires "
    "testing on attack types excluded from training."
)

st.divider()

# =====================================================
# 2. FALSE POSITIVE RATE
# =====================================================

st.subheader("2. False Positive Rate Comparison")

fpr_data = pd.DataFrame({
    "Model": [
        "Isolation Forest",
        "Standard Autoencoder",
        "Proposed Drift-Adaptive"
    ],
    "False Positive Rate (%)": [
        14.2,
        11.6,
        1.1
    ]
})

st.bar_chart(
    fpr_data.set_index("Model")
)

st.info(
    "A lower false positive rate means fewer normal network flows "
    "are incorrectly classified as anomalies."
)

st.divider()

# =====================================================
# 3. AVERAGE LATENCY
# =====================================================

st.subheader("3. Average Latency per Flow")

latency_data = pd.DataFrame({
    "Model": [
        "Isolation Forest",
        "Standard Autoencoder",
        "Proposed Drift-Adaptive"
    ],
    "Latency (ms)": [
        4.8,
        8.1,
        0.88
    ]
})

st.bar_chart(
    latency_data.set_index("Model")
)

st.info(
    "Latency measures the time taken to process an individual "
    "network flow."
)

st.divider()

# =====================================================
# 4. THROUGHPUT
# =====================================================

st.subheader("4. Throughput Capacity Comparison")

throughput_data = pd.DataFrame({
    "Model": [
        "Isolation Forest",
        "Standard Autoencoder",
        "Proposed Drift-Adaptive"
    ],
    "Throughput (Gbps)": [
        2.1,
        4.5,
        10.2
    ]
})

st.bar_chart(
    throughput_data.set_index("Model")
)

st.info(
    "Throughput represents the amount of network traffic "
    "processed per second."
)

st.divider()

# =====================================================
# 5. ADAPTATION RECOVERY TIME
# =====================================================

st.subheader("5. Adaptation Recovery Time after Concept Drift")

r1, r2, r3 = st.columns(3)

r1.metric("Isolation Forest", "N/A")
r2.metric("Standard Autoencoder", "N/A")
r3.metric("Proposed Drift-Adaptive", "< 1 Second")

st.info(
    "Reference target only. The current implementation uses ADWIN "
    "to detect changes in the incoming anomaly signal. Detection "
    "does not by itself mean that the Isolation Forest has been retrained."
)

st.divider()

# =====================================================
# FOOTER
# =====================================================

st.caption(
    "Drift-Adaptive Unsupervised Anomaly Detection for Encrypted "
    "Enterprise Network Traffic in Real Time"
)

st.caption(
    "Live dashboard refresh: " + datetime.now().strftime("%d-%m-%Y %H:%M:%S")
)

st.caption(
    "Isolation Forest | ADWIN Concept Drift Detection | "
    "Flask REST API | Streamlit"
)
