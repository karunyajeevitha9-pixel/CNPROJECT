import pandas as pd
import numpy as np
import pickle

DATA_FILE = "dataset/processed_data.csv"
MODEL_FILE = "saved_model.pkl"
DRIFT_FILE = "drift_results.csv"

df = pd.read_csv(DATA_FILE)
X = df.drop(columns=["Label"])

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

drift_results = pd.read_csv(DRIFT_FILE)

drift_count = drift_results["Drift"].sum()

scores = model.decision_function(X)

window_size = 5000
recent_scores = scores[window_size:window_size * 2]

old_threshold = 0.0

if drift_count > 0:
    new_threshold = np.percentile(recent_scores, 50)
    drift_status = "DRIFT DETECTED"
    action = "THRESHOLD UPDATED"
else:
    new_threshold = old_threshold
    drift_status = "NO DRIFT"
    action = "THRESHOLD UNCHANGED"

predictions = np.where(scores < new_threshold, 1, 0)

alerts = np.where(
    predictions == 1,
    "ALERT",
    "NO ALERT"
)

results = pd.DataFrame({
    "Anomaly_Score": scores,
    "Prediction": predictions,
    "Alert": alerts
})

print("========================================")
print("ADAPTIVE RESPONSE")
print("========================================")

print("\nDrift Status:", drift_status)
print("Old Threshold:", round(old_threshold, 6))
print("New Threshold:", round(new_threshold, 6))
print("Action:", action)

print("\nTotal Alerts:", (predictions == 1).sum())
print("Normal Records:", (predictions == 0).sum())

results.to_csv("adaptive_results.csv", index=False)

with open("adaptive_threshold.pkl", "wb") as f:
    pickle.dump(new_threshold, f)

print("\nFiles created:")
print("adaptive_results.csv")
print("adaptive_threshold.pkl")