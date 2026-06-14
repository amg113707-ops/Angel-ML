import joblib
import numpy as np
modl=joblib.load('modl.pkl')
x_scaler=joblib.load('x_scaler.pkl')
gender_encoder=joblib.load('gender_encoded.pkl')
age=int(input("Enter age of patient:"))

gender= input("Enter Sex: ").strip().lower()
gender_encoded=gender_encoder.fit_transform([gender])[0]
tot_bilirubin=float(input("Enter total bilirubin count:"))
direct_bilirubin=float(input("Enter direct bilirubin count:"))
tot_proteins=float(input("Enter total proteins count:"))
albumin=float(input("Enter total albumin count:"))
ag_ratio=float(input("Enter ag_ratio:"))
sgpt=float(input("Enter sgpt count:"))
sgot=float(input("Enter sgot count:"))
alkphos=float(input("Enter alkphos count:"))

features = np.array([[age,tot_bilirubin,direct_bilirubin,tot_proteins,albumin,ag_ratio,sgpt,sgot,alkphos,gender_encoded]])
features_scaled = x_scaler.transform(features)
prediction = modl.predict(features_scaled)
if prediction[0]==1:
    print("Liver Patient")
else:
    print("Healthy liver")
      


