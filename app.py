# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import numpy as np

# app = Flask(__name__)
# CORS(app, origins="*")  # Enable CORS globally

# # 🔹 Load models + scalers
# heart_model = joblib.load("models/logistic_model.pkl")
# heart_scaler = joblib.load("models/scaler.pkl")

# parkinson_model = joblib.load("models/svm_model.pkl")
# parkinson_scaler = joblib.load("models/scaler1.pkl")

# diabetes_model = joblib.load("models/svm_classifier.pkl")
# diabetes_scaler = joblib.load("models/scaler2.pkl")

# # 🔹 Heart Prediction
# @app.route("/predict-heart", methods=["POST"])
# def predict_heart():
#     data = request.json
#     features = [
#         data["age"], data["sex"], data["cp"], data["trestbps"], data["chol"],
#         data["fbs"], data["restecg"], data["thalach"], data["exang"],
#         data["oldpeak"], data["slope"], data["ca"], data["thal"]
#     ]
#     features_scaled = heart_scaler.transform([features])
#     prediction = int(heart_model.predict(features_scaled)[0])
#     return jsonify({"prediction": prediction})

# # 🔹 Parkinson Prediction (UPDATED to use all 22 features)
# @app.route("/predict-parkinson", methods=["POST"])
# def predict_parkinson():
#     data = request.json

#     features = [
#         data["MDVP_Fo"], data["MDVP_Fhi"], data["MDVP_Flo"], data["MDVP_Jitter"],
#         data["MDVP_Shimmer"], data["NHR"], data["HNR"], data["RPDE"], data["DFA"],
#         data["spread1"], data["spread2"], data["D2"], data["PPE"],

#         # 9 hidden fields (ensure these keys are exactly sent from frontend)
#         data["age"], data["sex"], data["test_time"],
#         data["other_feature1"], data["other_feature2"], data["other_feature3"],
#         data["other_feature4"], data["other_feature5"], data["other_feature6"]
#     ]

#     features_scaled = parkinson_scaler.transform([features])
#     prediction = int(parkinson_model.predict(features_scaled)[0])
#     return jsonify({"prediction": prediction})

# # 🔹 Diabetes Prediction
# @app.route("/predict-diabetes", methods=["POST"])
# def predict_diabetes():
#     data = request.json
#     features = [
#         data["Pregnancies"], data["Glucose"], data["BloodPressure"], data["SkinThickness"],
#         data["Insulin"], data["BMI"], data["DiabetesPedigreeFunction"], data["Age"]
#     ]
#     features_scaled = diabetes_scaler.transform([features])
#     prediction = int(diabetes_model.predict(features_scaled)[0])
#     return jsonify({"prediction": prediction})

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app, origins="*")  # Enable CORS globally

# 🔹 Load models + scalers
heart_model = joblib.load("models/logistic_model.pkl")
heart_scaler = joblib.load("models/scaler.pkl")

parkinson_model = joblib.load("models/svm_model.pkl")
parkinson_scaler = joblib.load("models/scaler1.pkl")

diabetes_model = joblib.load("models/svm_classifier.pkl")
diabetes_scaler = joblib.load("models/scaler2.pkl")

# 🔹 Root route
@app.route("/")
def home():
    return "✅ Multi-Disease Prediction API is running!"

# 🔹 Heart Prediction
@app.route("/predict-heart", methods=["POST"])
def predict_heart():
    data = request.json
    features = [
        data["age"], data["sex"], data["cp"], data["trestbps"], data["chol"],
        data["fbs"], data["restecg"], data["thalach"], data["exang"],
        data["oldpeak"], data["slope"], data["ca"], data["thal"]
    ]
    features_scaled = heart_scaler.transform([features])
    prediction = int(heart_model.predict(features_scaled)[0])
    return jsonify({"prediction": prediction})

# 🔹 Parkinson Prediction
@app.route("/predict-parkinson", methods=["POST"])
def predict_parkinson():
    data = request.json
    features = [
        data["MDVP_Fo"], data["MDVP_Fhi"], data["MDVP_Flo"], data["MDVP_Jitter"],
        data["MDVP_Shimmer"], data["NHR"], data["HNR"], data["RPDE"], data["DFA"],
        data["spread1"], data["spread2"], data["D2"], data["PPE"],
        data["age"], data["sex"], data["test_time"],
        data["other_feature1"], data["other_feature2"], data["other_feature3"],
        data["other_feature4"], data["other_feature5"], data["other_feature6"]
    ]
    features_scaled = parkinson_scaler.transform([features])
    prediction = int(parkinson_model.predict(features_scaled)[0])
    return jsonify({"prediction": prediction})

# 🔹 Diabetes Prediction
@app.route("/predict-diabetes", methods=["POST"])
def predict_diabetes():
    data = request.json
    features = [
        data["Pregnancies"], data["Glucose"], data["BloodPressure"], data["SkinThickness"],
        data["Insulin"], data["BMI"], data["DiabetesPedigreeFunction"], data["Age"]
    ]
    features_scaled = diabetes_scaler.transform([features])
    prediction = int(diabetes_model.predict(features_scaled)[0])
    return jsonify({"prediction": prediction})

# 🔹 Run app (Render-compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
