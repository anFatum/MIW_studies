from pandas import Series
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

dataset_train = pd.read_csv('./lab05/ABCDATA.mst')
training_set = dataset_train.iloc[:1258, 2:3].values
train = training_set
test = dataset_train.iloc[1258:, 2:3].values

# plot_acf(test, lags=30)
# plt.show()
#
# plot_pacf(training_set, lags=30)
# plt.show()

model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params

# walk forward over time steps in test
history = train[len(train) - window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length - window, length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d + 1] * lag[window - d - 1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# Visualising the results
plt.plot(test, color='blue', label='Real Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# plot
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()

# dataset_train = pd.read_csv('./lab05/ABCDATA.mst')
# training_set = dataset_train.iloc[:1258, 2:3].values
# train = training_set
# test = dataset_train.iloc[1258:, 2:3].values
# history = [x for x in train]
# predictions = list()
# for t in range(len(test)):
#     model = ARIMA(history, order=(5, 0, 0))
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t]
#     history.append(obs)
#     print('predicted=%f, expected=%f' % (yhat, obs))
#
# error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)
# # plot results
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()
