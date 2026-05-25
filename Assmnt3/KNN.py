import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
data=pd.read_csv("KNN_Dataset.csv")
print(data)

x=data[["Temperature"]]

y=data[["Fuel_Consumption"]]

model=KNeighborsRegressor(n_neighbors=3)

model.fit(x,y)

print(model.predict([[58]]))

