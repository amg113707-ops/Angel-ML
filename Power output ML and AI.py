import pandas as pd
from sklearn.linear_model import LinearRegression
data=pd.read_csv("updated_power_data.csv")
print(data)

x=data[["Wind_Speed","Blade_Angle","Rotor_Speed"]]
print(x)
y=data[["Power_Output"]]
print(y)
model=LinearRegression()
model.fit(x,y)
new_Power=model.predict([[9,11,100]])
print(new_Power)