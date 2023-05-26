import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('./data/Copy of market_data_df.csv')
data = df.copy()
data['DATETIME'] = pd.to_datetime(data["DATETIME"])

data.set_index("DATETIME", inplace=True)

#define numerical and categorical features
numeric_features = [feature for feature in data.columns if data[feature].dtype != 'O']
categorical_features = [feature for feature in data.columns if data[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))

for features in numeric_features:

  median=data[features].median()
  

  data[features].fillna(value=median, inplace=True)

    # checking for correaltion
new_df = pd.DataFrame(columns=numeric_features)
for features in numeric_features:
    new_df[features] = data[features]


# h0 null hypothesis = data is no stationary
# h1 data is stationary
from statsmodels.tsa.stattools import adfuller
# adfuller gives 5 different values
'''ArithmeticError
1. ADF test statistic
2. P=Vaule
3. #lags used
4. Number of observations used'''

from statsmodels.tsa.stattools import adfuller
def adfuller_test(marginal_price):

    result = adfuller(marginal_price)
    labels = ["ADF Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"]

    for value, label in zip(result, labels):
        print(label+':'+str(value))
        if result[1]<=0.05:
            print("Strong evidence against the null hypothesis, reject it")
        else:
            print("weak evidence against null hypothesis, time series has a unit root")

# define the target feature
test_result = adfuller(data['target'])
print(test_result)


#if data is no stationary, we need to make it so

# we just shift 1 position of 
# data["target"].shift(1)

# if data is seasonal, instead of shifting by 1, we shift by 12

# and subtract it from the actualsales/forecaseted data

# and make a NEW feature called SEASONAL FIRST DIFFERENCE 

# than do the adfuller again after removing nan

# autoregressive model

# how many previous data needs to be considered is told by autto correlation graph
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(data["target"])
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig = plot_acf(df["target"], lags=50)
fig = plot_pacf(df["target"], lags=40)

from statsmodels.tsa.arima.model import ARIMA
dt = data.copy()
Y= dt["target"]
X= dt.drop("target")
model=ARIMA(dt["target)"],dataorder=(0,0,0))
model_fit=model.fit()


model_fit.summary()

# plot prediction

d1= '2020-10-9 00:00:00'
d2= '2020-10-9 23:00:00'
df1= pd.to_datetime(d1)
df2= pd.to_datetime(d2)
#X['forecast']=
print(model_fit.predict(start=df1, end=df2,dynamic=True))
M = model_fit.predict(start=df1, end=df2,dynamic=True)

# calculate error

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(M.values, Y.loc[d1:d2].values)
print(mse)
pred=M.values
actual=Y.loc[d1:d2].values
plt.plot(pred,"r")
plt.plot(actual,'g')