
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import timedelta

# Path of the file to read
relative_path = "../../data/alertsHistory.json"
absolute_path = os.path.dirname(__file__)
full_path = os.path.join(absolute_path, relative_path)

alerts_data = pd.read_json(full_path)

alerts_data.head()

# Digest the data - 1st select the region (data column)
#alertsByRegion = alerts_data.loc['ראשון' in alerts_data['data']]
#alertsByRegion = alerts_data[alerts_data['data'].str.contains('ראשון')]
alertsByRegion = alerts_data # no filteing 
# base date
base_date = dt.datetime(2023,10,7,6,30)

y = 0
alertsByRegion = alertsByRegion.iloc[::-1] #revere the data
for index, row in alertsByRegion.iterrows():
    minutes = ((dt.datetime.strptime(row['alertDate'], '%Y-%m-%dT%H:%M:%S') - base_date).total_seconds())/60
    alertsByRegion.loc[index, 'X'] = minutes
    alertsByRegion.loc[index, 'Y'] = y
    y = minutes


print(alertsByRegion)

x = alertsByRegion.X 
y = alertsByRegion.Y 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0, shuffle=False)

# Create linear regression object
regr = linear_model.LinearRegression()

x_train= x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)
# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
alerts_y_pred = regr.predict(x_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, alerts_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, alerts_y_pred))
# mean_absolute_error
print("mean_absolute_error: %.2f" % mean_absolute_error(y_test, alerts_y_pred))

#When is the next alert
y_next = regr.predict(np.array(x.iloc[-1]).reshape(1,1))
date_time = (base_date + timedelta(minutes=y_next[0])).strftime("%m/%d/%Y, %H:%M:%S")
print("Next alert is on:",date_time)

# Plot outputs
plt.scatter(x_test, y_test, color="black")
plt.plot(x_test, alerts_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()