import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
label_encoder=LabelEncoder()

data=pd.read_csv("insurance.csv")
sex_encoder = LabelEncoder()
smoker_encoder = LabelEncoder()
region_encoder = LabelEncoder()

data["sex"] = sex_encoder.fit_transform(data["sex"])
data["smoker"] = smoker_encoder.fit_transform(data["smoker"])
data["region"] = region_encoder.fit_transform(data["region"])
#print(data)
x=data.drop("charges", axis=1)
y=data[["charges"]]
x_scaler=StandardScaler()
y_scaler=StandardScaler()
x_scaled=x_scaler.fit_transform(x)
y_scaled=y_scaler.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_scaled,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
error=mean_squared_error(y_test,y_pred)
rms=np.sqrt(error)
print(rms)
joblib.dump(model, 'model.pkl')
joblib.dump(x_scaler,'x_scaler.pkl')
joblib.dump(y_scaler,'y_scaler.pkl')
joblib.dump(sex_encoder, 'sex_encoder.pkl')
joblib.dump(smoker_encoder, 'smoker_encoder.pkl')
joblib.dump(region_encoder, 'region_encoder.pkl')

print("Model and scalers are saved successfully")



