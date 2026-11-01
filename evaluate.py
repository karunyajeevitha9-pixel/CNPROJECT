import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

file = "dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(file)
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna()

X = df.drop(columns=["Label"])
y = df["Label"]
X = X.select_dtypes(include=np.number)

model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
model.fit(X)

pred = np.where(model.predict(X) == -1, "DDoS", "BENIGN")

cm = confusion_matrix(y, pred, labels=["BENIGN", "DDoS"])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["BENIGN", "DDoS"]
)

disp.plot()
plt.title("Isolation Forest Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

actual = y.value_counts()
predicted = pd.Series(pred).value_counts()

plt.figure()
plt.bar(["BENIGN", "DDoS"], [
    actual.get("BENIGN", 0),
    actual.get("DDoS", 0)
])
plt.title("Actual Traffic Distribution")
plt.ylabel("Number of Records")
plt.savefig("actual_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.bar(["BENIGN", "DDoS"], [
    predicted.get("BENIGN", 0),
    predicted.get("DDoS", 0)
])
plt.title("Isolation Forest Predicted Distribution")
plt.ylabel("Number of Records")
plt.savefig("predicted_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

print("Graphs generated successfully!")
