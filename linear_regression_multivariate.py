import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('homeprices.csv')
df

df.bedrooms.median()

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
df

reg = linear_model.LinearRegression()
reg.fit(df.drop('price',axis='columns'),df.price)


reg.coef_
reg.intercept_

reg.predict([[3000, 3, 40]])

reg.predict([[2500, 4, 5]])