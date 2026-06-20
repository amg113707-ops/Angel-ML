import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
label_encoder=LabelEncoder()
data=pd.read_csv("Obdataset.csv")
#print(data)
data=data.drop_duplicates()
gender_encoder = LabelEncoder()
calc_encoder = LabelEncoder()
favc_encoder = LabelEncoder()
scc_encoder = LabelEncoder()
smoke_encoder = LabelEncoder()
fhwo_encoder = LabelEncoder()
caec_encoder = LabelEncoder()
mtrans_encoder = LabelEncoder()
obesity_encoder = LabelEncoder()

data["Gender"]=gender_encoder.fit_transform(data["Gender"])
data["CALC"]=calc_encoder.fit_transform(data["CALC"])
data["FAVC"]=favc_encoder.fit_transform(data["FAVC"])
data["SCC"]=scc_encoder.fit_transform(data["SCC"])
data["SMOKE"]=smoke_encoder.fit_transform(data["SMOKE"])
data["family_history_with_overweight"]=fhwo_encoder.fit_transform(data["family_history_with_overweight"])
data["CAEC"]=caec_encoder.fit_transform(data["CAEC"])
data["MTRANS"]=mtrans_encoder.fit_transform(data["MTRANS"])
data["NObeyesdad"]=obesity_encoder.fit_transform(data["NObeyesdad"])

x=data.drop(columns=["NObeyesdad"])
#print(x)

y=data["NObeyesdad"]
#print(y)

x_scaler=StandardScaler()

x_scaled=x_scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
joblib.dump(model, 'modl.pkl')
joblib.dump(x_scaler,'x_scaler.pkl')
joblib.dump(gender_encoder,'Gender.pkl')
joblib.dump(calc_encoder,'CALC.pkl')
joblib.dump(favc_encoder,'FAVC.pkl')
joblib.dump(scc_encoder,'SCC.pkl')
joblib.dump(smoke_encoder,'SMOKE.pkl')
joblib.dump(fhwo_encoder,'family_history_with_overweight.pkl')
joblib.dump(caec_encoder,'CAEC.pkl')
joblib.dump(mtrans_encoder,'MTRANS.pkl')
joblib.dump(obesity_encoder,'NObeyesdad.pkl')


print(data.duplicated().sum())
print(data.isnull().sum())

model = Sequential([
    Dense(64, input_shape=(16,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)
model.save("Obesity_ann.keras")



