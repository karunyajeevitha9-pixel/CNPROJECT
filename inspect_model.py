
import pickle
import os

files = [
    "saved_model.pkl",
    "scaler.pkl",
    "features.pkl",
    "live_model.pkl",
    "live_scaler.pkl",
    "live_features.pkl"
]

for filename in files:
    if not os.path.exists(filename):
        print(f"\n{filename}: NOT FOUND")
        continue

    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)

        print(f"\n{'='*45}")
        print("FILE:", filename)
        print("TYPE:", type(obj).__name__)

        if isinstance(obj, (list, tuple)):
            print("FEATURE COUNT:", len(obj))
            print("FEATURES:", obj)

        elif hasattr(obj, "n_features_in_"):
            print("EXPECTED FEATURES:", obj.n_features_in_)
            if hasattr(obj, "feature_names_in_"):
                print("FEATURE NAMES:", list(obj.feature_names_in_))

        else:
            print("OBJECT:", obj)

    except Exception as e:
        print("ERROR:", e)
