from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import random
import os

app = Flask(__name__, static_folder="dist")
CORS(app)  # Enable CORS for all routes

# Load the saved models
model_lr = joblib.load("../models/logistic_regression_model.joblib")
model_knn = joblib.load("../models/knn_model.joblib")
model_dtc = joblib.load("../models/dtc_model.joblib")
model_rfc = joblib.load("../models/rfc_model.joblib")

# Load the dataset
df = pd.read_csv("../datasets/creditcard_test.csv")

# Separate features and target
x = df.drop("Class", axis=1)
y = df["Class"]


# Function to make predictions
def make_predictions(models, features):
    feature_names = ["V4", "V10", "V11", "V12", "V14", "V17"]
    features_df = pd.DataFrame([features], columns=feature_names)
    predictions = {
        model_name: "Fraud" if model.predict(features_df)[0] == 1 else "Valid"
        for model_name, model in models.items()
    }
    return predictions


# API endpoint to load a random transaction
@app.route("/random", methods=["GET"])
def random_transaction():
    random_index = random.randint(0, len(df) - 1)
    transaction = df.iloc[random_index]
    response = {
        "features": transaction[["V4", "V10", "V11", "V12", "V14", "V17"]].to_dict(),
        "class": "Fraud" if transaction["Class"] == 1 else "Valid",
    }
    return jsonify(response)


# API endpoint to load a random fraud transaction
@app.route("/fraud", methods=["GET"])
def random_fraud_transaction():
    fraud_df = df[df["Class"] == 1]
    random_index = random.randint(0, len(fraud_df) - 1)
    transaction = fraud_df.iloc[random_index]
    response = {
        "features": transaction[["V4", "V10", "V11", "V12", "V14", "V17"]].to_dict(),
        "class": "Fraud",
    }
    return jsonify(response)


# API endpoint to load a random non-fraud (valid) transaction
@app.route("/valid", methods=["GET"])
def random_non_fraud_transaction():
    non_fraud_df = df[df["Class"] == 0]
    random_index = random.randint(0, len(non_fraud_df) - 1)
    transaction = non_fraud_df.iloc[random_index]
    response = {
        "features": transaction[["V4", "V10", "V11", "V12", "V14", "V17"]].to_dict(),
        "class": "Valid",
    }
    return jsonify(response)


# API endpoint to predict if a transaction is fraud or not using all models
@app.route("/predict", methods=["POST"])
def predict():
    try:
        feature_names = ["V4", "V10", "V11", "V12", "V14", "V17"]
        features = {name: float(request.form[name]) for name in feature_names}
        models = {
            "Logistic Regression": model_lr,
            "K-Nearest Neighbors": model_knn,
            "Decision Tree Classifier": model_dtc,
            "Random Forest Classifier": model_rfc,
        }
        predictions = make_predictions(models, features)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Serve the Vue.js frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    print(f"Serving static files from: {os.path.abspath(app.static_folder)}")
    app.run(host="0.0.0.0", port=5000)
