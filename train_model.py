import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

file = "dataset/processed_data.csv"

df = pd.read_csv(file)

X = df.drop(columns=["Label"])
y = df["Label"]

print("Features used:", X.shape[1])
print("Records:", X.shape[0])

contamination_values = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

results = []

best_model = None
best_score = -1
best_contamination = None

for contamination in contamination_values:

    print("\nTesting contamination:", contamination)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    pred = model.predict(X)

    pred = np.where(pred == -1, 1, 0)

    accuracy = accuracy_score(y, pred)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)

    results.append([
        contamination,
        accuracy,
        precision,
        recall,
        f1
    ])

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))

    if f1 > best_score:
        best_score = f1
        best_model = model
        best_contamination = contamination

results_df = pd.DataFrame(
    results,
    columns=[
        "Contamination",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(results_df.to_string(index=False))

print("\n========================================")
print("BEST MODEL")
print("========================================")

print("Best contamination:", best_contamination)
print("Best F1-score:", round(best_score, 4))

best_pred = best_model.predict(X)
best_pred = np.where(best_pred == -1, 1, 0)

print("\nConfusion Matrix:")
print(confusion_matrix(y, best_pred))

print("\nBest Model Performance:")
print("Accuracy :", round(accuracy_score(y, best_pred), 4))
print("Precision:", round(precision_score(y, best_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y, best_pred, zero_division=0), 4))
print("F1-score :", round(f1_score(y, best_pred, zero_division=0), 4))

with open("saved_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

results_df.to_csv("model_comparison.csv", index=False)

print("\nFiles created:")
print("saved_model.pkl")
print("model_comparison.csv")