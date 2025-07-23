import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor # New import for KNN
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Load Dataset ---
# Ensure 'vechile.csv' is in the same directory as this script.
df = pd.read_csv("vechile.csv")
print("Dataset 'vechile.csv' loaded successfully.")

# --- 2. Explore and Understand the Data ---
print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Null Values Count (Initial) ---")
print(df.isnull().sum())

# --- 3. Data Preprocessing ---

# Convert 'Date' column to datetime objects
print("\n--- Converting 'Date' to Datetime ---")
print("Original DataFrame shape:", df.shape)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
print("After converting 'Date' to datetime and coercing errors.")

# Remove rows where "Date" conversion failed
initial_rows = df.shape[0]
df = df[df['Date'].notnull()]
rows_removed_date = initial_rows - df.shape[0]
print(f"Removed {rows_removed_date} rows due to invalid 'Date' conversion.")
print("DataFrame shape after date null removal:", df.shape)

# Remove rows where the target ('Electric Vehicle (EV) Total') is missing
initial_rows_ev = df.shape[0]
df = df[df['Electric Vehicle (EV) Total'].notnull()]
rows_removed_ev_total = initial_rows_ev - df.shape[0]
print(f"Removed {rows_removed_ev_total} rows where 'Electric Vehicle (EV) Total' was missing.")
print("DataFrame shape after EV Total null removal:", df.shape)

# Fill missing values in 'County' and 'State'
print("\n--- Filling Missing Values ---")
df['County'] = df['County'].fillna('Unknown')
df['State'] = df['State'].fillna('fillna') # Assuming 'fillna' was intended based on previous discussion
print("Filled missing values in 'County' and 'State' with 'Unknown'/'fillna'.")

# Confirm remaining nulls for County and State
print("\n--- Missing values after fill (County, State) ---")
print(df[['County', 'State']].isnull().sum())

# Display the head of the DataFrame after initial cleaning
print("\n--- DataFrame Head after Initial Cleaning ---")
print(df.head())

# Convert all relevant count columns to numeric
cols_to_convert = [
    'Battery Electric Vehicles (BEVs)',
    'Plug-In Hybrid Electric Vehicles (PHEVs)',
    'Electric Vehicle (EV) Total',
    'Non-Electric Vehicle Total',
    'Total Vehicles',
    'Percent Electric Vehicles'
]

print("\n--- Converting Count Columns to Numeric ---")
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0) # Fill NaNs resulting from coercion with 0 for count data

print("Converted relevant columns to numeric and filled NaNs from coercion with 0.")
print("\n--- DataFrame Describe after Numeric Conversion ---")
print(df[cols_to_convert].describe())


# Outlier Detection and Capping for 'Percent Electric Vehicles'
print("\n--- Outlier Detection and Capping for 'Percent Electric Vehicles' ---")
Q1 = df['Percent Electric Vehicles'].quantile(0.25)
Q3 = df['Percent Electric Vehicles'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print('lower_bound:', lower_bound)
print('upper_bound:', upper_bound)

outliers_before_capping = df[(df['Percent Electric Vehicles'] < lower_bound) | (df['Percent Electric Vehicles'] > upper_bound)]
print("Number of outliers in 'Percent Electric Vehicles' before capping:", outliers_before_capping.shape[0])

df['Percent Electric Vehicles'] = np.where(df['Percent Electric Vehicles'] > upper_bound, upper_bound,
                                 np.where(df['Percent Electric Vehicles'] < lower_bound, lower_bound, df['Percent Electric Vehicles']))

outliers_after_capping = df[(df['Percent Electric Vehicles'] < lower_bound) | (df['Percent Electric Vehicles'] > upper_bound)]
print("Number of outliers in 'Percent Electric Vehicles' after capping:", outliers_after_capping.shape[0])
print("\n--- DataFrame Head after Outlier Capping ---")
print(df.head())

# --- 4. Feature Engineering ---
print("\n--- Performing Feature Engineering ---")

# Extract year, month, and numeric_date
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['numeric_date'] = df['Date'].dt.year * 12 + df['Date'].dt.month  # For trend

# Encode County
le = LabelEncoder()
df['county_encoded'] = le.fit_transform(df['County'])
print("County column encoded.")

# --- Save the LabelEncoder for later use in the Streamlit app ---
joblib.dump(le, 'label_encoder.pkl')
print("LabelEncoder saved to 'label_encoder.pkl'")
# --- End LabelEncoder Save ---

# Sort for lag creation - ESSENTIAL for time series features
df = df.sort_values(['County', 'Date']).reset_index(drop=True)
print("DataFrame sorted by County and Date for lag feature creation.")

# Create lag features (1–3 months) and rolling average
df['months_since_start'] = df.groupby('County').cumcount()

for lag in [1, 2, 3]:
    df[f'ev_total_lag{lag}'] = df.groupby('County')['Electric Vehicle (EV) Total'].shift(lag)

df['ev_total_roll_mean_3'] = df.groupby('County')['Electric Vehicle (EV) Total'] \
                               .transform(lambda x: x.shift(1).rolling(3).mean())

# Percent change
df['ev_total_pct_change_1'] = df.groupby('County')['Electric Vehicle (EV) Total'] \
                                .pct_change(periods=1, fill_method=None)
df['ev_total_pct_change_3'] = df.groupby('County')['Electric Vehicle (EV) Total'] \
                                .pct_change(periods=3, fill_method=None)

# Clean up any infs/NaNs from percent change (e.g., division by zero)
df['ev_total_pct_change_1'] = df['ev_total_pct_change_1'].replace([np.inf, -np.inf], np.nan).fillna(0)
df['ev_total_pct_change_3'] = df['ev_total_pct_change_3'].replace([np.inf, -np.inf], np.nan).fillna(0)
print("Lag, rolling mean, and percentage change features created.")

# Cumulative EV count and growth slope
df['cumulative_ev'] = df.groupby('County')['Electric Vehicle (EV) Total'].cumsum()
df['ev_growth_slope'] = df.groupby('County')['cumulative_ev'].transform(
    lambda x: x.rolling(6).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == 6 else np.nan, raw=False))
print("Cumulative EV and growth slope features created.")

# Drop early rows with NaN values resulting from lag features
initial_rows_after_fe = df.shape[0]
df = df.dropna().reset_index(drop=True)
rows_removed_after_fe = initial_rows_after_fe - df.shape[0]
print(f"Removed {rows_removed_after_fe} rows with NaN values after feature engineering.")
print("DataFrame shape after feature engineering and NaN removal:", df.shape)
print("\n--- DataFrame Head after Feature Engineering ---")
print(df.head())

# --- Save preprocessed data ---
df.to_csv('preprocessed_ev_data.csv', index=False)
print("\nPreprocessed data saved to 'preprocessed_ev_data.csv'")
# --- End preprocessed data save ---

# --- 5. Model Selection and Training ---

# Define features and target
features = [
    'months_since_start',
    'county_encoded',
    'ev_total_lag1',
    'ev_total_lag2',
    'ev_total_lag3',
    'ev_total_roll_mean_3',
    'ev_total_pct_change_1',
    'ev_total_pct_change_3',
    'ev_growth_slope',
]
target = 'Electric Vehicle (EV) Total'

X = df[features]
y = df[target]

print(f"\nFeatures selected: {features}")
print(f"Target selected: {target}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Train-test split (shuffle=False for time-series data)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1, random_state=42)
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Function to evaluate model performance
def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2Score = r2_score(y_true, y_pred)
    print(f"\n--- {model_name} Evaluation ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2Score:.4f}")
    return mae, rmse, r2Score

# --- Model 1: Random Forest Regressor ---
print("\n--- Training Random Forest Regressor ---")
rf = RandomForestRegressor(random_state=42)
param_dist_rf = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2', None]
}
random_search_rf = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist_rf,
    n_iter=30,  # 30 random combos
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=0, # Set to 0 to reduce output during search
    random_state=42
)
random_search_rf.fit(X_train, y_train)
best_rf_model = random_search_rf.best_estimator_
print("Random Forest Best Parameters:", random_search_rf.best_params_)
y_pred_rf = best_rf_model.predict(X_test)
mae_rf, rmse_rf, r2_rf = evaluate_model("Random Forest Regressor", y_test, y_pred_rf)

# --- Model 2: KNeighbors Regressor (KNN) ---
print("\n--- Training KNeighbors Regressor (KNN) ---")
knn = KNeighborsRegressor()
param_dist_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13], # Number of neighbors
    'weights': ['uniform', 'distance'], # Weight function used in prediction
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], # Algorithm used to compute nearest neighbors
    'p': [1, 2] # Power parameter for the Minkowski metric (1 for Manhattan, 2 for Euclidean)
}
random_search_knn = RandomizedSearchCV(
    estimator=knn,
    param_distributions=param_dist_knn,
    n_iter=20, # Adjust as needed for computation time
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=0,
    random_state=42
)
random_search_knn.fit(X_train, y_train)
best_knn_model = random_search_knn.best_estimator_
print("KNeighbors Best Parameters:", random_search_knn.best_params_)
y_pred_knn = best_knn_model.predict(X_test)
mae_knn, rmse_knn, r2_knn = evaluate_model("KNeighbors Regressor", y_test, y_pred_knn)

# --- Model 3: Decision Tree Regressor ---
print("\n--- Training Decision Tree Regressor ---")
dtr = DecisionTreeRegressor(random_state=42)
param_dist_dtr = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2', None]
}
random_search_dtr = RandomizedSearchCV(
    estimator=dtr,
    param_distributions=param_dist_dtr,
    n_iter=15, # Even fewer iterations
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=0,
    random_state=42
)
random_search_dtr.fit(X_train, y_train)
best_dtr_model = random_search_dtr.best_estimator_
print("Decision Tree Best Parameters:", random_search_dtr.best_params_)
y_pred_dtr = best_dtr_model.predict(X_test)
mae_dtr, rmse_dtr, r2_dtr = evaluate_model("Decision Tree Regressor", y_test, y_pred_dtr)

# --- Model Comparison Summary ---
print("\n--- Model Performance Comparison ---")
print(f"{'Model':<25} {'MAE':<10} {'RMSE':<10} {'R2 Score':<10}")
print("-" * 55)
print(f"{'Random Forest Regressor':<25} {mae_rf:<10.4f} {rmse_rf:<10.4f} {r2_rf:<10.4f}")
print(f"{'KNeighbors Regressor':<25} {mae_knn:<10.4f} {rmse_knn:<10.4f} {r2_knn:<10.4f}")
print(f"{'Decision Tree Regressor':<25} {mae_dtr:<10.4f} {rmse_dtr:<10.4f} {r2_dtr:<10.4f}")

# Select the best model for forecasting (based on R2 score)
models = {
    "Random Forest": {"model": best_rf_model, "r2": r2_rf, "y_pred": y_pred_rf},
    "KNeighbors": {"model": best_knn_model, "r2": r2_knn, "y_pred": y_pred_knn},
    "Decision Tree": {"model": best_dtr_model, "r2": r2_dtr, "y_pred": y_pred_dtr}
}
best_model_name = max(models, key=lambda name: models[name]["r2"])
model_for_forecasting = models[best_model_name]["model"]
y_pred_best_model = models[best_model_name]["y_pred"] # Get predictions from the best model for plotting
print(f"\nSelected '{best_model_name}' as the best model for forecasting based on R2 score.")

# --- 6. Model Evaluation Visualizations (for the best model) ---
# Using the best_model_name for plotting
print(f"\n--- Plotting Actual vs Predicted for {best_model_name} Regressor ---")
comparison_df = pd.DataFrame({
    'Actual EV Count': y_test.values,
    'Predicted EV Count': y_pred_best_model # Using predictions from the best model
})
comparison_df['Predicted EV Count'] = comparison_df['Predicted EV Count'].round(2)
comparison_df.reset_index(drop=True, inplace=True)
print(comparison_df.head(10))

plt.figure(figsize=(8, 5)) # Adjusted size
plt.plot(y_test.values, label='Actual EV Count', color='blue', alpha=0.7)
plt.plot(y_pred_best_model, label=f'Predicted EV Count ({best_model_name})', color='red', linestyle='--', alpha=0.7)
plt.title(f"Actual vs Predicted EV Count ({best_model_name} Regressor)")
plt.xlabel("Sample Index")
plt.ylabel("EV Count")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot Feature Importance for Random Forest Regressor (if it was the best, or as an example)
# KNN and Decision Tree have different ways of showing importance, but Random Forest's is direct.
# We will show it for Random Forest as a representative tree-based model.
if best_model_name == "Random Forest":
    print("\n--- Plotting Feature Importance for Random Forest Regressor (Selected as Best) ---")
    importances = best_rf_model.feature_importances_
    feature_names = X.columns # Get feature names from the DataFrame

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(7, 4)) # Adjusted size
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title('Feature Importance - Random Forest Regressor Model')
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print(f"\n--- Skipping Feature Importance Plot: {best_model_name} does not have direct feature_importances_ attribute like Random Forest. ---")


# --- 7. Forecasting with the Best Model ---
print(f"\n--- Starting Forecasting with {best_model_name} Model ---")

# Forecasting total electric vehicles for a specific county (e.g., Clark)
# Ensure the county you choose exists in your dataset.
county_to_forecast = "Kings" # You can change this to any county in your dataset

try:
    county_code_forecast = le.transform([county_to_forecast])[0]
    print(f"County '{county_to_forecast}' encoded as {county_code_forecast}.")
except ValueError:
    print(f"Error: County '{county_to_forecast}' not found in LabelEncoder. Please choose an existing county.")
    exit()

# Filter historical data for the chosen county
county_df_forecast = df[df['county_encoded'] == county_code_forecast].sort_values("numeric_date")

if county_df_forecast.empty:
    print(f"Warning: No historical data found for county '{county_to_forecast}'. Cannot forecast.")
    exit()
if county_df_forecast.shape[0] < 6:
    print(f"Warning: Not enough historical data for county '{county_to_forecast}' to create 6-month lags/slopes. Need at least 6 data points.")
    exit()

# Prepare EV history for forecasting loop
historical_ev_for_forecast = list(county_df_forecast['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev_for_forecast = list(county_df_forecast['cumulative_ev'].values[-6:])

months_since_start_forecast = county_df_forecast['months_since_start'].max()

historical_plot_df = county_df_forecast[['Date', 'Electric Vehicle (EV) Total', 'months_since_start']].copy()
historical_plot_df['Source'] = 'Historical'
historical_plot_df['County'] = county_to_forecast

future_rows_forecast = []
forecast_horizon = 36 # Forecast for next 3 years (36 months)

for i in range(1, forecast_horizon + 1):
    last_date_in_history = historical_plot_df['Date'].max() if future_rows_forecast == [] else future_rows_forecast[-1]['Date']
    next_date = last_date_in_history + pd.DateOffset(months=1)

    months_since_start_forecast += 1

    # Safely get lag features
    lag1, lag2, lag3 = historical_ev_for_forecast[-1] if len(historical_ev_for_forecast) >= 1 else 0, \
                       historical_ev_for_forecast[-2] if len(historical_ev_for_forecast) >= 2 else 0, \
                       historical_ev_for_forecast[-3] if len(historical_ev_for_forecast) >= 3 else 0

    roll_mean = np.mean([lag1, lag2, lag3])

    # Handle division by zero for percentage change
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0

    # Compute slope using the last 6 cumulative values
    recent_cumulative_for_slope = cumulative_ev_for_forecast[-6:]
    if len(recent_cumulative_for_slope) == 6:
        ev_growth_slope = np.polyfit(range(len(recent_cumulative_for_slope)), recent_cumulative_for_slope, 1)[0]
    else:
        ev_growth_slope = 0

    # Construct new row for prediction
    new_row_for_pred = {
        'months_since_start': months_since_start_forecast,
        'county_encoded': county_code_forecast,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    # Predict the next EV Total
    X_new_pred = pd.DataFrame([new_row_for_pred])[features]
    pred_ev_total = model_for_forecasting.predict(X_new_pred)[0]
    pred_ev_total = max(0, pred_ev_total) # Ensure non-negative

    # Update rolling histories for the next iteration
    historical_ev_for_forecast.append(pred_ev_total)
    if len(historical_ev_for_forecast) > 6:
        historical_ev_for_forecast.pop(0)

    cumulative_ev_for_forecast.append(cumulative_ev_for_forecast[-1] + pred_ev_total)
    if len(cumulative_ev_for_forecast) > 6:
        cumulative_ev_for_forecast.pop(0)

    # Store forecast result
    future_rows_forecast.append({
        'Date': next_date,
        'Electric Vehicle (EV) Total': pred_ev_total,
        'months_since_start': months_since_start_forecast,
        'County': county_to_forecast,
        'Source': 'Forecast'
    })

forecast_df_single_county = pd.DataFrame(future_rows_forecast)

# Combine historical and forecast for plotting
combined_single_county = pd.concat([historical_plot_df, forecast_df_single_county], ignore_index=True)
combined_single_county = combined_single_county.sort_values("Date").reset_index(drop=True)

# Plot Actual vs. Predicted for the single county
plt.figure(figsize=(9, 5)) # Adjusted size
for source, group in combined_single_county.groupby('Source'):
    plt.plot(group['Date'], group['Electric Vehicle (EV) Total'], label=source,
             marker='o' if source == 'Forecast' else '.', linestyle='-' if source == 'Forecast' else '--', alpha=0.8)
plt.title(f"EV Adoption Forecast vs Historical - {county_to_forecast} County (Monthly Forecast for 3 Years)")
plt.xlabel("Date")
plt.ylabel("EV Count")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title="Data Source")
plt.tight_layout()
plt.show()

# Plot Cumulative EV Adoption for the single county
combined_single_county['Cumulative EVs'] = combined_single_county['Electric Vehicle (EV) Total'].cumsum()

plt.figure(figsize=(9, 5)) # Adjusted size
for source, group in combined_single_county.groupby('Source'):
    plt.plot(group['Date'], group['Cumulative EVs'], label=f'{source} (Cumulative)',
             marker='o' if source == 'Forecast' else '.', linestyle='-' if source == 'Forecast' else '--', alpha=0.8)
plt.title(f"Cumulative EV Adoption - {county_to_forecast} County")
plt.xlabel("Date")
plt.ylabel("Cumulative EV Count")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title="Data Source")
plt.tight_layout()
plt.show()

# --- Forecasting for Top-5 Counties ---
print("\n--- Forecasting for Top 5 Counties ---")
all_combined_forecasts = []
unique_counties_in_data = df['County'].dropna().unique()

# To get top counties, we need to re-evaluate the total EVs after preprocessing
# and ensure we have enough data points for forecasting.
# Let's re-calculate top counties based on the 'Electric Vehicle (EV) Total' sum from the processed df
top_counties_overall = df.groupby('County')['Electric Vehicle (EV) Total'].sum().sort_values(ascending=False).head(10).index.tolist()
print(f"Top 10 counties by total EV count (from preprocessed data): {top_counties_overall}")

# Filter to ensure we only try to forecast for counties with enough historical data
eligible_counties = []
for county_name in top_counties_overall:
    temp_df = df[df['County'] == county_name]
    if temp_df.shape[0] >= 6: # Need at least 6 data points for initial lags/slope
        eligible_counties.append(county_name)

# Take the top 5 eligible counties
top_5_counties_for_forecast = eligible_counties[:5]
print(f"Top 5 eligible counties for forecasting: {top_5_counties_for_forecast}")


for county_name in top_5_counties_for_forecast:
    print(f"Forecasting for {county_name}...")
    try:
        county_code_current = le.transform([county_name])[0]
    except ValueError:
        print(f"Skipping {county_name}: not found in LabelEncoder.")
        continue

    county_df_current = df[df['county_encoded'] == county_code_current].sort_values("numeric_date")
    if county_df_current.empty or county_df_current.shape[0] < 6:
        print(f"Skipping {county_name}: Not enough historical data for forecasting.")
        continue

    months_since_start_current = county_df_current['months_since_start'].max()

    historical_plot_df_current = county_df_current[['Date', 'Electric Vehicle (EV) Total', 'months_since_start']].copy()
    historical_plot_df_current['Source'] = 'Historical'
    historical_plot_df_current['County'] = county_name

    historical_ev_current = list(county_df_current['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev_current = list(county_df_current['cumulative_ev'].values[-6:])

    future_rows_current = []

    for i in range(1, forecast_horizon + 1):
        last_date_in_history_current = historical_plot_df_current['Date'].max() if future_rows_current == [] else future_rows_current[-1]['Date']
        next_date_current = last_date_in_history_current + pd.DateOffset(months=1)

        months_since_start_current += 1

        # Safely get lag features
        lag1_c = historical_ev_current[-1] if len(historical_ev_current) >= 1 else 0
        lag2_c = historical_ev_current[-2] if len(historical_ev_current) >= 2 else 0
        lag3_c = historical_ev_current[-3] if len(historical_ev_current) >= 3 else 0

        roll_mean_c = np.mean([lag1_c, lag2_c, lag3_c])

        pct_change_1_c = (lag1_c - lag2_c) / lag2_c if lag2_c != 0 else 0
        pct_change_3_c = (lag1_c - lag3_c) / lag3_c if lag3_c != 0 else 0

        recent_cumulative_for_slope_c = cumulative_ev_current[-6:]
        if len(recent_cumulative_for_slope_c) == 6:
            ev_growth_slope_c = np.polyfit(range(len(recent_cumulative_for_slope_c)), recent_cumulative_for_slope_c, 1)[0]
        else:
            ev_growth_slope_c = 0

        new_row_for_pred_c = {
            'months_since_start': months_since_start_current,
            'county_encoded': county_code_current,
            'ev_total_lag1': lag1_c,
            'ev_total_lag2': lag2_c,
            'ev_total_lag3': lag3_c,
            'ev_total_roll_mean_3': roll_mean_c,
            'ev_total_pct_change_1': pct_change_1_c,
            'ev_total_pct_change_3': pct_change_3_c,
            'ev_growth_slope': ev_growth_slope_c
        }

        X_new_pred_c = pd.DataFrame([new_row_for_pred_c])[features]
        pred_ev_total_c = model_for_forecasting.predict(X_new_pred_c)[0]
        pred_ev_total_c = max(0, pred_ev_total_c) # Ensure non-negative predictions

        historical_ev_current.append(pred_ev_total_c)
        if len(historical_ev_current) > 6:
            historical_ev_current.pop(0)

        cumulative_ev_current.append(cumulative_ev_current[-1] + pred_ev_total_c)
        if len(cumulative_ev_current) > 6:
            cumulative_ev_current.pop(0)

        future_rows_current.append({
            'Date': next_date_current,
            'Electric Vehicle (EV) Total': pred_ev_total_c,
            'months_since_start': months_since_start_current,
            'County': county_name,
            'Source': 'Forecast'
        })

    forecast_df_current = pd.DataFrame(future_rows_current)
    combined_current = pd.concat([historical_plot_df_current, forecast_df_current], ignore_index=True)
    combined_current = combined_current.sort_values("Date").reset_index(drop=True)
    combined_current['Cumulative EVs'] = combined_current['Electric Vehicle (EV) Total'].cumsum()
    all_combined_forecasts.append(combined_current)

if not all_combined_forecasts:
    print("\nNo eligible counties found for top 5 forecasting. Please check your data and county eligibility criteria.")
else:
    full_df_forecast_top5 = pd.concat(all_combined_forecasts, ignore_index=True)

    plt.figure(figsize=(12, 7)) # Adjusted size
    for county, group in full_df_forecast_top5.groupby('County'):
        plt.plot(group['Date'], group['Cumulative EVs'], label=county, marker='.', markersize=4, alpha=0.9)

    plt.title("Top 5 Counties by Cumulative EV Adoption (Historical + 3-Year Forecast)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative EV Count", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title="County", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(
        ticks=pd.date_range(start=full_df_forecast_top5['Date'].min(), end=full_df_forecast_top5['Date'].max(), freq='YS'),
        labels=[str(d.year) for d in pd.date_range(start=full_df_forecast_top5['Date'].min(), end=full_df_forecast_top5['Date'].max(), freq='YS')],
        rotation=45, ha='right'
    )
    plt.yticks(fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make room for legend
    plt.show()

# --- 8. Final Model Testing and Save the Model ---
print("\n--- Saving and Loading the Best Model ---")

# Save the best trained model to file
model_filename = 'forecasting_ev_model.pkl' # This will save in the same directory as vechile.py
joblib.dump(model_for_forecasting, model_filename)
print(f"Model saved to '{model_filename}'")

# Load model from file
loaded_model = joblib.load(model_filename)
print("Model loaded successfully.")

# Make predictions with the loaded model
print(f"\n🔍 Testing saved model '{best_model_name}' on 1 sample:")
sample = X_test.iloc[[0]]  # use one row as test
true_value = y_test.iloc[0]
predicted_value = loaded_model.predict(sample)[0]

print(f"Actual EVs: {true_value:.2f}, Predicted EVs: {predicted_value:.2f}")
