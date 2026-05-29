import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("Decision_Tree.csv")
print(data)

label_encoder=LabelEncoder()

data["Temperature"]=label_encoder.fit_transform(data["Temperature"])
data["Vibration"]=label_encoder.fit_transform(data["Vibration"])
data["Failure"]=label_encoder.fit_transform(data["Failure"])

x=data[["Temperature","Vibration"]]
#print(x)
y=data[["Failure"]]
#print(y)
model=DecisionTreeClassifier(criterion="entropy")
model.fit(x,y)
sample=[[0,2]]
prediction=model.predict(sample)
if prediction==1:
    print("machine will fail")
else:
    print("machine will not fail")

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
tree.plot_tree(
    model,
    feature_names=["Temperature","Vibration"],
    class_names=["Yes","No"]
)
plt.title("Decision Tree-Machine Failure")
plt.show()