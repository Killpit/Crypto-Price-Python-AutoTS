import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression

sns.set()
plt.style.use('seaborn-whitegrid')

data = pd.read_csv("/Users/atatekeli/PycharmProjects/LitecoinPrice/LTC-USD.csv")
print(data.head())

data.dropna()
plt.figure(figsize=(10, 4))
plt.title("SOL-USD")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(data["Close"])
plt.show()

from autots import AutoTS

model = AutoTS(forecast_length=5, frequency='infer', ensemble='simple',
               drop_data_older_than_periods=100)
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)

prediction = model.predict()
forecast = prediction.forecast
print("Solana Price Prediction")
print(forecast)
