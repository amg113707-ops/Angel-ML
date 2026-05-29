import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("dataDT.csv")
print(data)

x=data[["tempMode","AQ","USS","CS","VOC","RP","IP","Temperature"]]
#print(x)
y=data[["fail"]]
#print(y)
model=DecisionTreeClassifier(criterion="entropy")
model.fit(x,y)
sample=[[4,5,3,6,1,45,5,1]]
prediction=model.predict(sample)
if prediction==1:
    print("machine will fail")
else:
    print("machine will not fail")

import matplotlib.pyplot as plt
plt.figure(figsize=(50,50))
tree.plot_tree(
    model,
    feature_names=["tempMode","AQ","USS","CS","VOC","RP","IP","Temperature"],
    class_names=["Yes","No"]
)
plt.title("Decision Tree-Machine Failure2")
plt.show()
#print(data.shape)