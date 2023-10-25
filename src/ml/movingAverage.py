import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import timedelta

# Path of the file to read
relative_path = "../../data/alertsHistory.json"
absolute_path = os.path.dirname(__file__)
full_path = os.path.join(absolute_path, relative_path)

alerts_data = pd.read_json(full_path)

# Calculate moving avarage 

base_date = dt.datetime(2023,10,7,6,30)

alerts_data = alerts_data[alerts_data['category'] == 1]
alertsByRegion = alerts_data[alerts_data['data'].str.contains('אזור תעשייה הדרומי אשקלון')]
#alertsByRegion = alerts_data # no filteing 

alertsByRegion = alertsByRegion.iloc[::-1] #revere the data

alertsByRegion['alertDate'] = pd.to_datetime(alertsByRegion['alertDate'])
alertsByRegion['timeDifference'] = alertsByRegion['alertDate'].diff().apply(lambda x: x/np.timedelta64(1, 'm')).fillna(0).astype('int64')
#print(alertsByRegion['timeDifference'])
window_size = 3
alertsByRegion['MovingAverage'] = alertsByRegion['timeDifference'].rolling(window=window_size).mean()
alertsByRegion.dropna(inplace=True)
print(alertsByRegion['MovingAverage'] )
y_next = alertsByRegion.iloc[-1]['alertDate'] + timedelta(minutes=alertsByRegion.iloc[-1]['MovingAverage'])
print("next alert by moving avarage :", y_next)
