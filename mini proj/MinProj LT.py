import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
label_encoder=LabelEncoder()
data=pd.read_csv("Indian Liver Patient Dataset.csv")
data=data.drop_duplicates()
data["alkphos"] = data["alkphos"].fillna(data["alkphos"].mode()[0])
data["gender_encoded"]=label_encoder.fit_transform(data["gender"])
x=data.drop(columns=["gender","is_patient"])
y=data[["is_patient"]]
x_scaler=StandardScaler()

x_scaled=x_scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
joblib.dump(model, 'modl.pkl')
joblib.dump(x_scaler,'x_scaler.pkl')
joblib.dump(label_encoder,'gender_encoded.pkl')

print(data.duplicated().sum())
print(data.isnull().sum())











