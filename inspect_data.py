import pandas as pd

file = "dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(file)
df.columns = df.columns.str.strip()

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for i, col in enumerate(df.columns):
    print(i, col)

print("\nLabels:")
print(df["Label"].value_counts())
