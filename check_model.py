
import pickle

for filename in ["saved_model.pkl", "scaler.pkl", "features.pkl"]:
    print("\n" + "=" * 50)
    print("FILE:", filename)

    with open(filename, "rb") as f:
        obj = pickle.load(f)

    print("TYPE:", type(obj).__name__)

    if hasattr(obj, "n_features_in_"):
        print("EXPECTED INPUT FEATURES:", obj.n_features_in_)

    if hasattr(obj, "feature_names_in_"):
        print("FEATURE NAMES:")
        print(list(obj.feature_names_in_))

    if filename == "features.pkl":
        print("FEATURE COUNT:", len(obj))
        print("FIRST 10 FEATURES:", obj[:10])
