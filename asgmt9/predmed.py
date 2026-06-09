import joblib
import numpy as np
model=joblib.load('model.pkl')
x_scaler=joblib.load('x_scaler.pkl')
y_scaler=joblib.load('y_scaler.pkl')
sex_encoder = joblib.load('sex_encoder.pkl')
smoker_encoder = joblib.load('smoker_encoder.pkl')
region_encoder = joblib.load('region_encoder.pkl')
age=int(input("Enter age of patient:"))
sex = input("Enter Sex: ").strip().lower()
bmi = float(input("Enter BMI : "))
children = int(input("Enter Number of Children: "))
smoker = input("Smoker? (yes/no): ").strip().lower()
region = input("Enter Region: ").strip().lower()
sex_encoded = sex_encoder.transform([sex])[0]
smoker_encoded = smoker_encoder.transform([smoker])[0]
region_encoded = region_encoder.transform([region])[0]
features = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
features_scaled = x_scaler.transform(features)
prediction_scaled = model.predict(features_scaled)
prediction_original = y_scaler.inverse_transform(prediction_scaled)
print(f"Estimated Medical Insurance Cost: Rs.{prediction_original[0][0]}")