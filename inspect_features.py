
import pickle
import pandas as pd

with open("features.pkl", "rb") as f:
    expected = pickle.load(f)

if hasattr(expected, "tolist"):
    expected = expected.tolist()

actual = list(pd.read_csv("live_flows.csv", nrows=0).columns)

print("\n========== MODEL FEATURES (78) ==========")
for i, feature in enumerate(expected):
    print(i + 1, ":", feature)

print("\n========== CICFLOWMETER FEATURES (82) ==========")
for i, feature in enumerate(actual):
    print(i + 1, ":", feature)
