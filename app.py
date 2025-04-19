from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import csv 
import requests
import os

app = Flask(__name__, template_folder='.')

# Util: download file from Google Drive
def download_file_from_drive(file_id, destination):
    if os.path.exists(destination):
        return
    print(f"📥 Downloading {destination}...")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        f.write(response.content)
    print(f"✅ Downloaded {destination}")

# Google Drive file IDs
drive_files = {
    "model/cancer_treatment_protocol_model.pkl": "1E6rdfIvjT3WyXGLZsslXXr0lVb3RvQLd",
    "model/scaler.pkl": "1ghjucsm2dlEoomt53PCWe7rcbgPHr6ft",
    "model/label_encoder.pkl":"1I-qCObWfWfEJr4l6Mgl_VIAYrzdVZ-FE",
    "model/feature_names.pkl": "1jgJY2qzXXfLCljILlpBfwswm6OA9guDv"
}

# Download all model components
for path, file_id in drive_files.items():
    download_file_from_drive(file_id, path)

# ✅ Load them into memory
model = joblib.load("model/cancer_treatment_protocol_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
feature_names = joblib.load("model/feature_names.pkl")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract raw input data for saving
        raw_data = {
            "age": data["age"],
            "tumor_grade": data["tumor_grade"],
            "family_cancer_history": data["family_cancer_history"],
            "tumor_size": data["tumor_size"],
            "survival_time": data["survival_time"],
            "cancer_stage": data["cancer_stage"],
            "metastasis": data["metastasis"]
        }

        # Preprocessing input data for model prediction
        age = float(data["age"])
        tumor_grade = int(data["tumor_grade"])
        family_cancer_history = int(data["family_cancer_history"])
        tumor_size = float(data["tumor_size"])
        survival_time = float(data["survival_time"]) / 365
        cancer_stage = data["cancer_stage"]
        metastasis = data["metastasis"]

        input_data = pd.DataFrame([{
            "age_at_diagnosis": age,
            "tumor_grade": tumor_grade,
            "family_cancer_history": family_cancer_history,
            "survival_time": survival_time,
            "tumor_largest_dimension_diameter": tumor_size,
            "ajcc_clinical_stage": cancer_stage,
            "metastasis_at_diagnosis": metastasis
        }])

        input_data = pd.get_dummies(input_data, drop_first=False)

        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[feature_names]

        numeric_features = ["age_at_diagnosis", "tumor_grade", "family_cancer_history", "survival_time", "tumor_largest_dimension_diameter"]
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])

        # Get the prediction and the recommended protocol
        prediction = model.predict(input_data)
        protocol = label_encoder.inverse_transform(prediction)[0]
        raw_data["recommended_protocol"] = protocol

        # Save raw data and recommended protocol to CSV
        with open('new_data.csv', 'a', newline='') as csvfile:
            fieldnames = list(raw_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:  # Write header if file is empty
                writer.writeheader()
            writer.writerow(raw_data)

        return jsonify({"prediction": protocol})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
