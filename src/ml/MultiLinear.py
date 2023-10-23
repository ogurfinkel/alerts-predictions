import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load your dataset into a Pandas DataFrame
# Path of the file to read
# Path of the file to read
relative_path = "../../data/alertsHistory.json"
absolute_path = os.path.dirname(__file__)
full_path = os.path.join(absolute_path, relative_path)

data = pd.read_json(full_path)

# Assuming 'alertDate' is the name of your timestamp column
timestamps = data['alertDate']

# Convert the timestamps to datetime objects if they are not already in that format
timestamps = pd.to_datetime(timestamps)
#timestamps.plot()

# Sort the timestamps in ascending order
timestamps.sort_values(inplace=True)

# Create a time series with the timestamps
time_series = pd.Series(data.index, index=timestamps)

# Fit an ARIMA model to the time series
model = ARIMA(time_series, order=(5, 1, 0))  # Example order; you can tune this
model_fit = model.fit()

# Predict the next timestamp
forecast = model_fit.forecast(steps=1)

# Calculate the predicted timestamp
predicted_timestamp = time_series.index[-1] + pd.Timedelta(forecast.values[0], unit='D')

# Print the predicted timestamp
print("Predicted Timestamp:", predicted_timestamp)

# Calculate and print evaluation metrics
actual_timestamp = timestamps.iloc[-1]
mae = mean_absolute_error([actual_timestamp], [predicted_timestamp])
#mse = mean_squared_error([actual_timestamp], [predicted_timestamp])
#rmse = np.sqrt(mse)
mape = np.abs((actual_timestamp - predicted_timestamp).value / actual_timestamp.value) * 100

print("Mean Absolute Error (MAE):", mae)
#print("Mean Squared Error (MSE):", mse)
#print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)