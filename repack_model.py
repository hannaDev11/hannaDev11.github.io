import joblib

# Load the original model (replace with actual filename if needed)
model = joblib.load("model/cancer_treatment_protocol_model.pkl")

# Re-save in compatible format
joblib.dump(model, "model/repacked_model.pkl")

print("✅ Model re-saved successfully")
