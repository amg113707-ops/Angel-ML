
import joblib
import numpy as np
from tensorflow.keras.models import load_model
ann_model=load_model("heart_ann.keras")
modl=joblib.load('modl.pkl')
x_scaler=joblib.load('x_scaler.pkl')
thal_encoder=joblib.load('thal_encoded.pkl')
age=int(input("Enter age of patient:"))

sex=int(input("Enter Sex(0 for female and 1 for male): "))

cp=float(input("Enter CP value:"))
trestbps=float(input("Enter trestbps:"))
chol=float(input("Enter cholesterol count:"))
fbs=float(input("Enter fbs count:"))
restecg=float(input("Enter rest ecg count:"))
thalach=float(input("Enter thalach:"))
exang=float(input("Enter exang count:"))
oldpeak=float(input("Enter oldpeak count:"))
slope=float(input("Enter slope:"))
ca=float(input("Enter ca:"))
thal = input("Enter thal: ").strip()
thal_encoded=thal_encoder.transform([thal])[0]
features = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal_encoded]])
features_scaled = x_scaler.transform(features)
prediction = ann_model.predict(features_scaled)
if prediction[0][0] >= 0.5:
    print("Heart Patient")
else:
    print("Healthy heart")
      


