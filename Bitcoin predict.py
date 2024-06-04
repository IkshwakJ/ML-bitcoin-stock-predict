import numpy as np 
import pandas as pd 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("BTC-USD.csv")
df.info()
df.describe()
df = df.dropna()
df = df.drop(["Date"], axis=1)
forecast_col = "Close"
test_size = 0.2
y = df[[forecast_col]]
predictors = list(df.columns)
predictors.remove(forecast_col)
x = df[predictors]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1234) 
model = LinearRegression().fit(x_train, y_train)
print("Intercept: ",model.intercept_)
print("Coefficents",model.coef_)
print("Score:", model.score(x_test,y_test))
y_pred = model.predict(x_test)
print("MAE",mean_absolute_error(y_test, y_pred))

from prophet import Prophet
df = pd.read_csv('BTC-USD.csv')
df = df[["Date", "Close"]]
df.columns = ["ds", "y"]
print(df)
prophet = Prophet()
prophet.fit(df)
future = prophet.make_future_dataframe(periods=365)
print(future)
forecast = prophet.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(200)
from prophet.plot import plot
prophet.plot(forecast, figsize=(20, 10)).show()
input("Press Any  Key to Terminate")
exit