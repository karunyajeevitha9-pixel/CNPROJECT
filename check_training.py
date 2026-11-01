
import pickle
import numpy as np
import os

print("=" * 65)
print("       MODEL AND TRAINING COMPATIBILITY CHECK")
print("=" * 65)

files = {
    "Model": "saved_model.pkl",
    "Scaler": "scaler.pkl",
    "Features": "features.pkl"
}

for name, path in files.items():
    if not os.path.exists(path):
        print(f"\nERROR: {path} not found!")
        raise SystemExit(1)

with open("saved_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = list(pickle.load(f))

print("\n1. LOADED OBJECTS")
print("-" * 65)
print("Model type       :", type(model).__name__)
print("Scaler type      :", type(scaler).__name__)
print("Feature count    :", len(features))

model_count = getattr(model, "n_features_in_", None)
scaler_count = getattr(scaler, "n_features_in_", None)

print("\n2. FEATURE DIMENSIONS")
print("-" * 65)
print("Model expects    :", model_count)
print("Scaler expects   :", scaler_count)
print("Features file    :", len(features))

print("\n3. MODEL FEATURE NAMES")
print("-" * 65)

model_names = getattr(model, "feature_names_in_", None)
scaler_names = getattr(scaler, "feature_names_in_", None)

if model_names is not None:
    print("Model has feature names:", len(model_names))
    print("First 10:", list(model_names[:10]))
else:
    print("Model feature names: Not available")

print("\n4. SCALER FEATURE NAMES")
print("-" * 65)

if scaler_names is not None:
    print("Scaler has feature names:", len(scaler_names))
    print("First 10:", list(scaler_names[:10]))
else:
    print("Scaler feature names: Not available")

print("\n5. SCALER STATISTICS")
print("-" * 65)

if hasattr(scaler, "mean_"):
    print("Scaler mean (first 10):")
    print(scaler.mean_[:10])

if hasattr(scaler, "scale_"):
    print("\nScaler scale (first 10):")
    print(scaler.scale_[:10])

print("\n6. FEATURE ORDER CHECK")
print("-" * 65)

if model_names is not None:
    model_order_ok = list(model_names) == features
    print("Model order matches features.pkl:", model_order_ok)
else:
    model_order_ok = None
    print("Model order: Cannot verify")

if scaler_names is not None:
    scaler_order_ok = list(scaler_names) == features
    print("Scaler order matches features.pkl:", scaler_order_ok)
else:
    scaler_order_ok = None
    print("Scaler order: Cannot verify")

print("\n7. PREPROCESSING COMPATIBILITY")
print("-" * 65)

if model_count == scaler_count == len(features):
    print("Feature dimensions are compatible.")
else:
    print("WARNING: Feature dimension mismatch!")

if model_names is not None and scaler_names is not None:
    if list(model_names) == list(scaler_names):
        print("Model and scaler feature names match.")
    else:
        print("WARNING: Model and scaler feature names differ.")

print("\nIMPORTANT:")
print("This diagnostic checks saved-object compatibility.")
print("It cannot prove how the model was originally trained.")
print("Check the original training script to verify whether")
print("scaling was applied before model.fit().")

print("\n" + "=" * 65)
print("              CHECK COMPLETED")
print("=" * 65)
