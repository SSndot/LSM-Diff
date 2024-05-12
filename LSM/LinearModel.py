# -*- coding = utf-8 -*-
# @Time: 2024/3/17 13:04
# @File: LinearModel.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from SeqInput import data

X = np.array(data[:2000])
X = X.reshape(-1,1)
y = data[100:2100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)