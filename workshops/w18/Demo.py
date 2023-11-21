import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


df = pd.DataFrame({
    'ds': pd.date_range(start='2022-01-01', end='2022-01-28', freq='D'),
    'y': [8,8,9,8,7,1,3,9,6,8,10,9,2,3,6,7,10,9,6,1,2,8,7,10,7,9,3,2]  # Weekly dip in the data on weekends
})


#model = Prophet(weekly_seasonality=False) #Predictions with and without weekly seasonality

model = Prophet(weekly_seasonality=True) #Forecasts future days with accounting for the weekly dip on weekends

model.fit(df)
future = model.make_future_dataframe(periods=28, freq='D')  # Forecasting for the next 4 Weeks
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Prophet Forecast with Weekly Seasonality (Weekend Dip)')
plt.show()
model.add_seasonality(name='Weekly', period=7, fourier_order=7)

