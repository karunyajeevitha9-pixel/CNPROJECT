
import pandas as pd
from river.drift import ADWIN

INPUT_FILE = "anomaly_results.csv"
OUTPUT_FILE = "drift_results.csv"

df = pd.read_csv(INPUT_FILE)

if "Anomaly_Score" not in df.columns:
    raise ValueError("Anomaly_Score column not found in anomaly_results.csv")

adwin = ADWIN(delta=0.002)
events = []

for i, score in enumerate(df["Anomaly_Score"]):
    if pd.isna(score):
        continue

    adwin.update(float(score))

    if adwin.drift_detected:
        events.append({
            "Record_Index": i,
            "Anomaly_Score": float(score),
            "Drift": True,
            "ADWIN_Window_Size": adwin.width
        })

results = pd.DataFrame(
    events,
    columns=[
        "Record_Index",
        "Anomaly_Score",
        "Drift",
        "ADWIN_Window_Size"
    ]
)

results.to_csv(OUTPUT_FILE, index=False)

print("===================================")
print("ADWIN DRIFT DETECTION")
print("===================================")
print("Records monitored:", len(df))
print("Drift events:", len(results))
print("Result saved to:", OUTPUT_FILE)

if len(results) > 0:
    print("DRIFT DETECTED")
    print(results.to_string(index=False))
else:
    print("NO DRIFT DETECTED")
