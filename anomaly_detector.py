import pandas as pd
import numpy as np
import pickle

DATA_FILE = "dataset/processed_data.csv"
MODEL_FILE = "saved_model.pkl"

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Label"])
y = df["Label"]

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

scores = model.decision_function(X)

predictions = np.where(scores < 0, 1, 0)

results = pd.DataFrame()

results["Anomaly_Score"] = scores
results["Actual"] = y
results["Prediction"] = predictions

results["Status"] = np.where(
    predictions == 1,
    "ANOMALY",
    "NORMAL"
)

print("========================================")
print("ANOMALY DETECTION")
print("========================================")

print("\nSample results:")

print(
    results[
        ["Anomaly_Score", "Actual", "Prediction", "Status"]
    ].head(10).to_string(index=False)
)

print("\nAnomaly count:")
print((predictions == 1).sum())

print("\nNormal count:")
print((predictions == 0).sum())

results.to_csv(
    "anomaly_results.csv",
    index=False
)

print("\nFile created:")
print("anomaly_results.csv")