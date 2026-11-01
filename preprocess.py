import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

file = "dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(file)
df.columns = df.columns.str.strip()

print("Original dataset:")
print("Rows:", len(df))
print("Columns:", len(df.columns))

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("\nAfter cleaning:")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Removed:", 225745 - len(df))

print("\nMissing values:", df.isnull().sum().sum())

print("\nLabels:")
print(df["Label"].value_counts())

y = df["Label"].str.strip()

y = y.map({
    "BENIGN": 0,
    "DDoS": 1
})

valid = y.notna()

df = df[valid]
y = y[valid].astype(int)

X = df.drop(columns=["Label"])

X = X.select_dtypes(include=[np.number])

print("\nFeature preparation:")
print("Number of features:", X.shape[1])

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

processed = X_scaled.copy()
processed["Label"] = y.values

processed.to_csv("dataset/processed_data.csv", index=False)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("\nPreprocessing completed successfully")
print("Processed data:", processed.shape)
print("Features:", X.shape[1])

print("\nFinal class distribution:")
print(y.value_counts().rename({
    0: "BENIGN",
    1: "DDoS"
}))

print("\nFiles created:")
print("dataset/processed_data.csv")
print("scaler.pkl")
print("features.pkl")