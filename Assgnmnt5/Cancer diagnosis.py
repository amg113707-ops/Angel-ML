import pandas as pd
from sklearn.linear_model import LogisticRegression
#rom sklearn.preprocessing import LabelEncoder

user_input=[]

data=pd.read_csv("data.csv")
#abel_encoder=LabelEncoder()
#ata["diagnosis- encoded"]=label_encoder.fit_transform(data["diagnosis"])
#print(data)

x=data[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]]
y=data[["diagnosis"]]
#print(y)
model=LogisticRegression(max_iter=10000)
y=y.values.ravel()
model.fit(x,y)
for col in x.columns:
    val=float(input(f"enter values for {col}:"))
    user_input.append(val)
new_data=pd.DataFrame([user_input],
                      columns=x.columns)
print(model.predict(new_data))
