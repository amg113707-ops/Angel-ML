import pandas as pd

from sklearn.linear_model import LinearRegression

data=pd.read_csv("machine_failure.csv")

print(data)
'''x=data[["Load (N)"]]
print(x)
y=data[["Extension (mm)"]]
print(y)
plt.scatter(x,y)
plt.show()
model=LinearRegression()
model.fit(x,y)
new_extension=model.predict([[55]])
print(new_extension)'''











