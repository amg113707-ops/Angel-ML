import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data=pd.read_csv("iris.csv")
print(data)
x=data.drop(columns=["Id","Species"])
y=data[["Species"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

user_input=list(map(float,input("EnterSepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm separated by space").split()))

prediction=model.predict([user_input])[0]
print("Input Measurements:")
print(f"Sepal Length = {user_input[0]} cm")
print(f"Sepal Width  = {user_input[1]} cm")
print(f"Petal Length = {user_input[2]} cm")
print(f"Petal Width  = {user_input[3]} cm")
print("\nPredicted Species:", prediction.upper())