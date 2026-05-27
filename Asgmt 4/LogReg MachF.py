import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("machine_failure.csv")
label_encoder=LabelEncoder()
data["Type- encoded"]=label_encoder.fit_transform(data["Type"])
#print(data)

x=data[["Type- encoded","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","TWF","HDF","PWF","OSF","RNF"]]
#print(x)
y=data[["Machine failure"]]
#print(y)
model=LogisticRegression(max_iter=10001)
y=y.values.ravel()
model.fit(x,y)
new_data=pd.DataFrame([[2,301,311,1450,52,28,1,0,0,0,0]],
                      columns=x.columns)
print(model.predict(new_data))










