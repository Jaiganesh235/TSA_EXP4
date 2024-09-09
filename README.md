### DEVELOPED BY: S JAIGANESH
### REGISTER NO: 212222240037
### DATE:


# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

# AIM:
To implement ARMA model in python.
# ALGORITHM:
1. Import necessary libraries including `pandas`, `numpy`, `matplotlib.pyplot`, `ArmaProcess`, `ARIMA`, and `warnings`.
2. Load the CSV dataset into a pandas DataFrame.
3. Identify the relevant time series column from the dataset.
4. Convert the time series data to numeric format using `pd.to_numeric`, and drop any `NaN` values.
5. Plot the original time series using `matplotlib`.
6. Define AR(1) and MA(1) coefficients as numpy arrays.
7. Generate a sample of 1000 data points for the ARMA(1,1) process using the `ArmaProcess` class.
8. Plot the simulated ARMA(1,1) process.
9. Display the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots for the ARMA(1,1) process.
10. Define AR(2) and MA(2) coefficients as numpy arrays.
11. Generate a sample of 10000 data points for the ARMA(2,2) process using the `ArmaProcess` class.
12. Plot the simulated ARMA(2,2) process.
13. Display the ACF and PACF plots for the ARMA(2,2) process.
14. Fit an ARMA(1,1) model to the time series data using the `ARIMA` class.
15. Print the summary of the ARMA(1,1) model.
16. Fit an ARMA(2,2) model to the time series data using the `ARIMA` class.
17. Print the summary of the ARMA(2,2) model.
18. Plot the original time series along with the fitted values from both ARMA(1,1) and ARMA(2,2) models for comparison.
19. Display the plot with legends and titles.
20. End the process with the visualizations and model summaries for analysis.### PROGRAM:

# PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('coffee_Sales.csv')

# Check the available columns in the dataset
print("Available columns in the dataset:", data.columns)

# Replace 'Sales' with the correct column name based on your dataset
# Assuming the time series column is the first column (adjust as necessary)
time_series_column_name = data.columns[0]
time_series = data[time_series_column_name]

# Convert the time series data to numeric, coercing any errors
time_series = pd.to_numeric(time_series, errors='coerce').dropna()

# Plot the original time series
plt.figure(figsize=(10, 5))
plt.plot(time_series)
plt.title('Original Time Series')
plt.show()

# Simulate ARMA(1,1) Process
ar1 = np.array([1, 0.33])
ma1 = np.array([1, 0.9])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.title('ARMA(1,1) Autocorrelation')
plt.show()

plot_pacf(ARMA_1)
plt.title('ARMA(1,1) Partial Autocorrelation')
plt.show()

# Simulate ARMA(2,2) Process
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.title('ARMA(2,2) Autocorrelation')
plt.show()

plot_pacf(ARMA_2)
plt.title('ARMA(2,2) Partial Autocorrelation')
plt.show()

# Fit ARMA(1,1) model to the provided dataset
model_1_1 = ARIMA(time_series, order=(1, 0, 1)).fit()
print("ARMA(1,1) Model Summary:")
print(model_1_1.summary())

# Fit ARMA(2,2) model to the provided dataset
model_2_2 = ARIMA(time_series, order=(2, 0, 2)).fit()
print("\nARMA(2,2) Model Summary:")
print(model_2_2.summary())

# Plotting the original time series vs fitted values for ARMA(1,1) and ARMA(2,2)
plt.figure(figsize=(10, 5))
plt.plot(time_series, label='Original')
plt.plot(model_1_1.fittedvalues, color='red', label='ARMA(1,1) Fitted')
plt.plot(model_2_2.fittedvalues, color='green', label='ARMA(2,2) Fitted')
plt.title('Original vs Fitted Time Series')
plt.legend()
plt.show()
```


# OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
![image](https://github.com/user-attachments/assets/0e763447-da5a-4593-9ecd-a8833634795f)

Partial Autocorrelation
![image](https://github.com/user-attachments/assets/1274a09d-ae47-4eb3-a9ad-9f17cdfc1678)

Autocorrelation
![image](https://github.com/user-attachments/assets/7992cbd3-0d7f-416a-8a4b-46614c78a4f0)


SIMULATED ARMA(2,2) PROCESS:
![image](https://github.com/user-attachments/assets/1a41d7de-cabc-4fe2-9a51-32d450ab1674)


Partial Autocorrelation
![image](https://github.com/user-attachments/assets/ce263902-be01-4163-9030-c571fcd6372e)


Autocorrelation
![image](https://github.com/user-attachments/assets/75025fb8-23fd-4885-b16b-6a871adb1e7e)


# RESULT:
Thus, a python program is created to fit ARMA Model for Time Series successfully.
