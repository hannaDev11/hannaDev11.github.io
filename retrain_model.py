@app.route("/retrain", methods=["POST"])
def retrain():
    global model
    # Call the retrain script or logic
    exec(open("retrain_model.py").read())
    model = joblib.load("model/cancer_treatment_protocol_model.pkl")
    return jsonify({"status": "Model retrained successfully"})
