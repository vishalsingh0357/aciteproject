import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("vechile.csv")
print(df.info())


print("\n--- Null Values Count ---")
print(df.isnull().sum())


# Calculate Q1 and Q3
Q1 = df['Percent Electric Vehicles'].quantile(0.25)
Q3 = df['Percent Electric Vehicles'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print('\n--- Outlier Bounds ---')
print('lower_bound:', lower_bound)
print('upper_bound:', upper_bound)

# Identify outliers
outliers = df[(df['Percent Electric Vehicles'] < lower_bound) | (df['Percent Electric Vehicles'] > upper_bound)]

# no of outlier
print("\nNumber of outliers in 'Percent Electric Vehicles':", len(outliers))


print("\nOutliers:")
print(outliers)
print("\n--- Converting 'Date' to Datetime ---")
print("Original DataFrame shape:", df.shape)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
print("After converting 'Date' to datetime and coercing errors.")

# Removes rows where "Date" conversion failed
initial_rows = df.shape[0]
df = df[df['Date'].notnull()]
rows_removed_date = initial_rows - df.shape[0]
print(f"Removed {rows_removed_date} rows due to invalid 'Date' conversion.")
print("DataFrame shape after date null removal:", df.shape)



initial_rows_ev = df.shape[0]
df = df[df['Electric Vehicle (EV) Total'].notnull()]
rows_removed_ev_total = initial_rows_ev - df.shape[0]
print(f"Removed {rows_removed_ev_total} rows where 'Electric Vehicle (EV) Total' was missing.")
print("DataFrame shape after EV Total null removal:", df.shape)


# Fill missing values
print("\n--- Filling Missing Values ---")
df['County'] = df['County'].fillna('Unknown')
df['State'] = df['State'].fillna('Unknown')
print("Filled missing values in 'County' and 'State' with 'Unknown'.")

# Confirm remaining nulls
print("\n--- Missing values after fill (County, State) ---")
print(df[['County', 'State']].isnull().sum())

# Display the head of the DataFrame after initial cleaning
print("\n--- DataFrame Head after Initial Cleaning ---")
print(df.head())


print("\n--- Recalculating Outlier Bounds ---")
Q1 = df['Percent Electric Vehicles'].quantile(0.25)
Q3 = df['Percent Electric Vehicles'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print('New lower_bound for capping:', lower_bound)
print('New upper_bound for capping:', upper_bound)


print("\n--- Capping Outliers in 'Percent Electric Vehicles' ---")
df['Percent Electric Vehicles'] = np.where(df['Percent Electric Vehicles'] > upper_bound, upper_bound,
                                 np.where(df['Percent Electric Vehicles'] < lower_bound, lower_bound, df['Percent Electric Vehicles']))


outliers_after_capping = df[(df['Percent Electric Vehicles'] < lower_bound) | (df['Percent Electric Vehicles'] > upper_bound)]
print("Number of outliers in 'Percent Electric Vehicles' after capping:", outliers_after_capping.shape[0])


print("\n--- DataFrame Head after Outlier Capping ---")
print(df.head())