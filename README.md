### DEVELOPED BY: S JAIGANESH
### REGISTER NO: 212222240037
### DATE:


# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

# AIM:
To implement ARMA model in python.
# ALGORITHM:
1. Import necessary libraries such as `numpy`, `matplotlib`, `pandas`, and `statsmodels`.
2. Load the dataset using `pandas.read_csv()`.
3. Extract the relevant column from the dataset (e.g., `DAYTON_MW`).
4. Define the ARMA(1,1) process with AR(1) and MA(1) coefficients.
5. Generate a sample of 1000 data points using the `ArmaProcess` class for ARMA(1,1).
6. Plot the generated ARMA(1,1) time series using `matplotlib`.
7. Display the ACF and PACF for the ARMA(1,1) process using `plot_acf()` and `plot_pacf()`.
8. Define the ARMA(2,2) process with AR(2) and MA(2) coefficients.
9. Generate a sample of 10,000 data points for the ARMA(2,2) process.
10. Plot the generated ARMA(2,2) time series.
11. Display the ACF and PACF for the ARMA(2,2) process.
# PROGRAM:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the provided dataset
file_path = '/mnt/data/MLTempDataset.csv'
data = pd.read_csv(file_path)

# Extract the 'DAYTON_MW' column for modeling
data_values = data['DAYTON_MW'].values

# 1. ARMA(1,1) Process

# Define AR and MA coefficients for ARMA(1,1)
ar1 = np.array([1, -0.5])  # AR(1) coefficient
ma1 = np.array([1, 0.5])   # MA(1) coefficient

# Generate a sample of 1000 data points
arma11_process = ArmaProcess(ar1, ma1)
arma11_sample = arma11_process.generate_sample(nsample=1000)

# Plot the ARMA(1,1) time series
plt.figure(figsize=(10, 6))
plt.plot(arma11_sample)
plt.title('Generated ARMA(1,1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xlim(0, 1000)
plt.grid(True)
plt.show()

# Display ACF and PACF plots for ARMA(1,1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(arma11_sample, lags=35, ax=plt.gca())
plt.subplot(122)
plot_pacf(arma11_sample, lags=35, ax=plt.gca())
plt.suptitle('ACF and PACF for ARMA(1,1)')
plt.tight_layout()
plt.show()

# 2. ARMA(2,2) Process

# Define AR and MA coefficients for ARMA(2,2)
ar2 = np.array([1, -0.7, 0.3])  # AR(2) coefficients
ma2 = np.array([1, 0.5, 0.4])   # MA(2) coefficients

# Generate a sample of 10000 data points
arma22_process = ArmaProcess(ar2, ma2)
arma22_sample = arma22_process.generate_sample(nsample=10000)

# Plot the ARMA(2,2) time series
plt.figure(figsize=(10, 6))
plt.plot(arma22_sample)
plt.title('Generated ARMA(2,2) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xlim(0, 10000)
plt.grid(True)
plt.show()

# Display ACF and PACF plots for ARMA(2,2)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(arma22_sample, lags=35, ax=plt.gca())
plt.subplot(122)
plot_pacf(arma22_sample, lags=35, ax=plt.gca())
plt.suptitle('ACF and PACF for ARMA(2,2)')
plt.tight_layout()
plt.show()

```


# OUTPUT:
## SIMULATED ARMA(1,1) PROCESS:
![image](https://github.com/user-attachments/assets/4d2720a6-f472-41cf-85cf-f9d2be78e97b)


## Partial Autocorrelation
![image](https://github.com/user-attachments/assets/134a1932-6f1f-43ef-8720-61f5ab07998c)


## Autocorrelation
![image](https://github.com/user-attachments/assets/8e922e76-36d8-447b-98b3-3c712b0bba04)



## SIMULATED ARMA(2,2) PROCESS:
![image](https://github.com/user-attachments/assets/bda9d763-c0fa-44cc-9b7a-0eaf36fcae34)


## Partial Autocorrelation
![image](https://github.com/user-attachments/assets/7610ecc0-4d59-4fa7-bc44-d9e4be533836)



## Autocorrelation
![image](https://github.com/user-attachments/assets/579d4bcc-8bd9-4e46-9958-dd0af5bd9f8c)



# RESULT:
Thus, a python program is created to fit ARMA Model for Time Series successfully.
