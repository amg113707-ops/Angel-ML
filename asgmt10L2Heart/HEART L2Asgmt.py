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
data=pd.read_csv("heart.csv")
##print(data)
data=data.drop_duplicates()

data["thal_encoded"]=label_encoder.fit_transform(data["thal"])
x=data.drop(columns=["thal","target"])
y=data[["target"]]
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
joblib.dump(label_encoder,'thal_encoded.pkl')

print(data.duplicated().sum())
print(data.isnull().sum())

model=Sequential(
    [
        Dense(13, input_shape=(13,),activation='relu'),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(1, activation='sigmoid'),
    ]
)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)
model.save("heart_ann.keras")

'''
new_data=pd.DataFrame([[34,0,3.4,5,6,4.5,6.0,6.7,1.2,9.0,4.5,0.6,2.4]],columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal_encoded'])
new_data_scaled = x_scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
prediction=model.predict(new_data)
print(prediction[0][0])
if prediction==1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")'''


