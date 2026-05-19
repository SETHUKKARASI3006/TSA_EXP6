# Ex.No: 6               HOLT WINTERS METHOD
### Date: 

### AIM:
To create a model based on holt winters method.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('/content/education_vs_income_econometrics_samples.csv.csv')

data.head()

df_for_resample = data.copy()
start_year = 2000
df_for_resample['year_col'] = start_year + (df_for_resample['obs'] - 1)
df_for_resample['date_index'] = pd.to_datetime(df_for_resample['year_col'], format='%Y')
df_for_resample = df_for_resample.set_index('date_index')
df_for_resample = df_for_resample.drop(columns=['obs', 'year_col', 'sample'])
data_yearly = df_for_resample.resample('YS').sum()
data_yearly.head()
data_yearly.plot()

scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_yearly.values.reshape(-1, 1)).flatten())
scaled_data.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
series_to_decompose = data_yearly['y_income_thousands_usd']
decomposition = seasonal_decompose(series_to_decompose, model="additive")
decomposition.plot()
plt.show()

scaled_data=scaled_data+1
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=2).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

np.sqrt(mean_squared_error(test_data, test_predictions_add))

np.sqrt(scaled_data.var()),scaled_data.mean()

final_model = ExponentialSmoothing(data_yearly['y_income_thousands_usd'], trend='add', seasonal='mul', seasonal_periods=2).fit()

final_predictions = final_model.forecast(steps=int(len(data_yearly)/4))

ax=data_yearly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Income')
ax.set_ylabel('Years')
ax.set_title('Prediction')
```

### OUTPUT:


TEST PREDICTION

<img width="547" height="435" alt="image" src="https://github.com/user-attachments/assets/ebbedb3f-e60d-4ca3-b1a2-f9c48d190f23" />


FINAL PREDICTION

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/592c5113-f273-4630-b1e9-7cc564b88f4f" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
