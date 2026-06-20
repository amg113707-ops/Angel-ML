from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models

ann_model = load_model("Obesity_ann.keras")
lr_model = joblib.load("modl.pkl")
x_scaler = joblib.load("x_scaler.pkl")

# Encoders

Gender_encoder = joblib.load("Gender.pkl")
CALC_encoder = joblib.load("CALC.pkl")
FAVC_encoder = joblib.load("FAVC.pkl")
SCC_encoder = joblib.load("SCC.pkl")
SMOKE_encoder = joblib.load("SMOKE.pkl")
FHWO_encoder = joblib.load("family_history_with_overweight.pkl")
CAEC_encoder = joblib.load("CAEC.pkl")
MTRANS_encoder = joblib.load("MTRANS.pkl")
NObeyesdad_encoder = joblib.load("NObeyesdad.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
 data = request.json
 Age = int(data["Age"])
 Gender = Gender_encoder.transform([data["Gender"]])[0]
 CALC = CALC_encoder.transform([data["CALC"]])[0]
 FAVC = FAVC_encoder.transform([data["FAVC"]])[0]
 SCC = SCC_encoder.transform([data["SCC"]])[0]
 SMOKE = SMOKE_encoder.transform([data["SMOKE"]])[0]
 family_history = FHWO_encoder.transform([data["family_history_with_overweight"]])[0]
 CAEC = CAEC_encoder.transform([data["CAEC"]])[0]
 MTRANS = MTRANS_encoder.transform([data["MTRANS"]])[0]
 Height = float(data["Height"])
 Weight = float(data["Weight"])
 FCVC = float(data["FCVC"])
 NCP = float(data["NCP"])
 CH2O = float(data["CH2O"])
 FAF = float(data["FAF"])
 TUE = float(data["TUE"])
 features = np.array([[
    Age,
    Gender,
    Height,
    Weight,
    CALC,
    FAVC,
    FCVC,
    NCP,
    SCC,
    SMOKE,
    CH2O,
    family_history,
    FAF,
    TUE,
    CAEC,
    MTRANS]])
 features_scaled = x_scaler.transform(features)
 lr_prediction = lr_model.predict(features_scaled)
 lr_result = NObeyesdad_encoder.inverse_transform(lr_prediction)[0]
 ann_prediction = ann_model.predict(features_scaled)
 ann_class = np.argmax(ann_prediction)
 ann_result = NObeyesdad_encoder.inverse_transform([ann_class])[0]
 return jsonify({
    "success": True,
    "logistic_prediction": lr_result,
    "ann_prediction": ann_result
})


if __name__=="__main__":
   app.run(debug=True)
   



