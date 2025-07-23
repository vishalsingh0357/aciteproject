import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# --- 1. Load Model and LabelEncoder ---
try:
    model = joblib.load('forecasting_ev_model.pkl')
    le = joblib.load('label_encoder.pkl')
    # For demonstration, let's load the preprocessed data to get historical context
    # Ensure 'preprocessed_ev_data.csv' is saved by uncommenting the line in vechile.py
    full_df_processed = pd.read_csv('preprocessed_ev_data.csv')
    full_df_processed['Date'] = pd.to_datetime(full_df_processed['Date'])
    # st.success("Model, LabelEncoder, and preprocessed data loaded successfully!") # REMOVED THIS LINE
except FileNotFoundError:
    st.error("Error: Model (forecasting_ev_model.pkl), LabelEncoder (label_encoder.pkl), or preprocessed data (preprocessed_ev_data.csv) not found. Please run vechile.py first to generate these files.")
    st.stop() # Stop the app if files are missing

# --- 2. Streamlit App Layout ---
st.set_page_config(page_title="EV Adoption Forecaster", layout="wide")

# Custom CSS for styling - REMOVED FOR DEFAULT STYLING
# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

#     html, body, [class*="st-"] {
#         font-family: 'Inter', sans-serif;
#         color: #333;
#     }

#     .stApp {
#         background-color: #f0f2f6; /* Light gray background */
#         padding-top: 20px;
#     }

#     .css-1d391kg { /* Sidebar background */
#         background-color: #FFFFCC; /* Light yellow background for sidebar */
#         border-right: 1px solid #e0e0e0;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#         color: #000000; /* Black text for sidebar content */
#     }

#     /* Ensure sidebar headers and labels are also black */
#     .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4,
#     .css-1d391kg .stSelectbox label, .css-1d391kg .stDateInput label {
#         color: #000000; /* Black text for labels and headers in sidebar */
#     }


#     .css-1lcbmhc { /* Main content area */
#         padding: 20px 40px;
#     }

#     h1, h2, h3, h4, h5, h6 {
#         color: #2c3e50; /* Darker blue for headers in main content */
#     }

#     .stMetric {
#         background-color: #e6f7ff; /* Light blue background for metric */
#         border-left: 5px solid #007bff; /* Blue left border */
#         padding: 15px;
#         border-radius: 8px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.05);
#         margin-bottom: 20px;
#     }

#     .stButton>button {
#         background-color: #ADD8E6; /* Light blue button background */
#         color: #000000; /* Black text for buttons */
#         border-radius: 8px;
#         padding: 10px 20px;
#         font-weight: 600;
#         border: none;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.2);
#         transition: background-color 0.3s ease;
#     }

#     .stButton>button:hover {
#         background-color: #87CEEB; /* Slightly darker light blue on hover */
#     }

#     /* --- Slider Styling (kept for general sliders if any are added later, but not for date) --- */
#     .stSlider > div > div > div > div { /* Slider track */
#         background-color: #d0d0d0; /* Light gray track for better contrast */
#     }

#     .stSlider > div > div > div > div > div { /* Slider handle/thumb */
#         background-color: #007bff; /* Blue handle */
#         border: 1px solid #0056b3; /* Slightly darker border for handle */
#     }

#     /* Attempt to make slider value text white for dark handles */
#     .stSlider .st-bd, .stSlider .st-b5 {
#         color: white !important;
#         font-weight: 600;
#     }
#     /* --- End Slider Styling --- */

#     .stSuccess {
#         background-color: #d4edda;
#         color: #155724;
#         border-color: #c3e6cb;
#         padding: 10px;
#         border-radius: 5px;
#     }

#     .stError {
#         background-color: #f8d7da;
#         color: #721c24;
#         border-color: #f5c6cb;
#         padding: 10px;
#         border-radius: 5px;
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.title("Electric Vehicle Adoption Forecaster")
st.write("Predict future EV adoption for a selected county.")

# --- Default Image ---
# Retained use_container_width=True to avoid deprecation warning and ensure image fits column
st.image("https://placehold.co/1200x300/e0e0e0/000000?text=Electric+Vehicle+Forecasting",
         caption="Forecasting the future of Electric Vehicle Adoption",
         use_container_width=True)

# --- Sidebar for Inputs ---
st.sidebar.header("Forecast Settings")

# Get unique counties from the loaded LabelEncoder classes
unique_counties = sorted(le.classes_)
selected_county = st.sidebar.selectbox("Select County:", unique_counties, index=unique_counties.index("Kings") if "Kings" in unique_counties else 0)

# Input for specific forecast date using st.date_input (calendar picker)
latest_historical_date = full_df_processed['Date'].max()
# Default to 1 year from the latest historical date, ensuring it's in the future
default_forecast_date = latest_historical_date + pd.DateOffset(years=1)
if default_forecast_date <= latest_historical_date: # Fallback in case historical data is very recent
    default_forecast_date = latest_historical_date + pd.DateOffset(months=1)

forecast_date = st.sidebar.date_input(
    "Select Forecast End Date:",
    value=default_forecast_date,
    min_value=latest_historical_date + pd.DateOffset(days=1), # Ensure user can only pick future dates
    max_value=latest_historical_date + pd.DateOffset(years=10) # Limit forecast to 10 years from last historical date
)

# Convert forecast_date to datetime object for calculations (already a datetime.date object, convert to pd.Timestamp)
forecast_datetime = pd.to_datetime(forecast_date)

# --- Main Content Area ---
st.header(f"Forecasting for {selected_county} County")

# Check if forecast_datetime is valid and in the future before allowing forecast
date_is_valid_for_forecast = True
if forecast_datetime <= latest_historical_date:
    st.warning(f"Please select a forecast end date that is after the latest historical date ({latest_historical_date.strftime('%Y-%m-%d')}).")
    date_is_valid_for_forecast = False

if st.sidebar.button("Generate Forecast") and date_is_valid_for_forecast:
    st.subheader("Forecast Results")

    with st.spinner("Generating forecast... Please wait."): # Loading spinner
        try:
            county_code = le.transform([selected_county])[0]
        except ValueError:
            st.error(f"Error: County '{selected_county}' not recognized. Please select a valid county.")
            st.stop()

        # Filter historical data for the selected county
        county_historical_df = full_df_processed[full_df_processed['county_encoded'] == county_code].sort_values("Date")

        if county_historical_df.empty or county_historical_df.shape[0] < 6:
            st.warning(f"Not enough historical data ({county_historical_df.shape[0]} months) for '{selected_county}' to generate a reliable forecast. Need at least 6 months.")
            st.stop()

        # Get the latest historical data points for feature engineering
        last_6_months_data = county_historical_df.tail(6)
        historical_ev = list(last_6_months_data['Electric Vehicle (EV) Total'].values)
        cumulative_ev = list(last_6_months_data['cumulative_ev'].values) # Use actual cumulative for slope calculation

        # Initialize months_since_start for forecasting
        months_since_start = county_historical_df['months_since_start'].max()

        # Prepare historical data for plotting
        plot_data = county_historical_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        plot_data['Source'] = 'Historical'
        plot_data['Cumulative EVs'] = plot_data['Electric Vehicle (EV) Total'].cumsum() # Recalculate cumulative for plotting

        forecast_rows = []
        current_date_for_loop = latest_historical_date # Start from the last historical date

        # Generate forecasts month by month until the forecast_datetime is reached
        while current_date_for_loop < forecast_datetime:
            current_date_for_loop += pd.DateOffset(months=1)
            months_since_start += 1

            # Replicate feature engineering for the next month
            lag1 = historical_ev[-1] if len(historical_ev) >= 1 else 0
            lag2 = historical_ev[-2] if len(historical_ev) >= 2 else 0
            lag3 = historical_ev[-3] if len(historical_ev) >= 3 else 0

            roll_mean = np.mean([lag1, lag2, lag3])

            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0

            recent_cumulative_for_slope = cumulative_ev[-6:]
            if len(recent_cumulative_for_slope) == 6:
                ev_growth_slope = np.polyfit(range(len(recent_cumulative_for_slope)), recent_cumulative_for_slope, 1)[0]
            else:
                ev_growth_slope = 0

            new_features = {
                'months_since_start': months_since_start,
                'county_encoded': county_code,
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct_change_1,
                'ev_total_pct_change_3': pct_change_3,
                'ev_growth_slope': ev_growth_slope
            }

            # Predict
            X_new = pd.DataFrame([new_features])
            pred_ev_total = model.predict(X_new)[0]
            pred_ev_total = max(0, pred_ev_total) # Ensure non-negative

            # Update historical lists for next iteration
            historical_ev.append(pred_ev_total)
            if len(historical_ev) > 6: historical_ev.pop(0)

            cumulative_ev.append(cumulative_ev[-1] + pred_ev_total)
            if len(cumulative_ev) > 6: cumulative_ev.pop(0)

            forecast_rows.append({
                'Date': current_date_for_loop,
                'Electric Vehicle (EV) Total': pred_ev_total
            })

        forecast_df = pd.DataFrame(forecast_rows)
        forecast_df['Source'] = 'Forecast'
        # Ensure cumulative sum for forecast starts correctly from the last historical cumulative value
        if not plot_data.empty:
            last_historical_cumulative = plot_data['Cumulative EVs'].iloc[-1]
            forecast_df['Cumulative EVs'] = forecast_df['Electric Vehicle (EV) Total'].cumsum() + last_historical_cumulative
        else:
            forecast_df['Cumulative EVs'] = forecast_df['Electric Vehicle (EV) Total'].cumsum()


        # Combine historical and forecast data for plotting
        combined_plot_df = pd.concat([plot_data, forecast_df], ignore_index=True)

        # Display the final prediction for the selected date
        # Find the row corresponding to the exact forecast_datetime
        final_prediction_row = combined_plot_df[combined_plot_df['Date'] == forecast_datetime]
        if not final_prediction_row.empty:
            final_prediction_value = final_prediction_row['Electric Vehicle (EV) Total'].iloc[0]
            st.metric(label=f"Predicted EV Total for {forecast_datetime.strftime('%B %Y')}", value=f"{int(final_prediction_value):,}")
        else:
            st.warning("Could not find a prediction for the exact end date. Displaying prediction for the last forecasted month.")
            last_forecast_date = combined_plot_df[combined_plot_df['Source'] == 'Forecast']['Date'].max()
            last_forecast_value = combined_plot_df[combined_plot_df['Date'] == last_forecast_date]['Electric Vehicle (EV) Total'].iloc[0]
            st.metric(label=f"Predicted EV Total for {last_forecast_date.strftime('%B %Y')} (last forecasted month)", value=f"{int(last_forecast_value):,}")


        # Plotting
        st.subheader("EV Adoption Trend")
        fig, ax = plt.subplots(figsize=(10, 6)) # Adjusted plot size
        sns.lineplot(data=combined_plot_df, x='Date', y='Electric Vehicle (EV) Total', hue='Source', ax=ax, marker='o')
        ax.set_title(f"EV Total: Historical vs. Forecast for {selected_county}")
        ax.set_ylabel("Electric Vehicle (EV) Total")
        ax.set_xlabel("Date")
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig)

        st.subheader("Cumulative EV Adoption Trend")
        fig_cum, ax_cum = plt.subplots(figsize=(10, 6)) # Adjusted plot size
        sns.lineplot(data=combined_plot_df, x='Date', y='Cumulative EVs', hue='Source', ax=ax_cum, marker='o')
        ax_cum.set_title(f"Cumulative EV: Historical vs. Forecast for {selected_county}")
        ax_cum.set_ylabel("Cumulative EV Total")
        ax_cum.set_xlabel("Date")
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_cum)

        # Download button for forecast data
        download_filename = f"{selected_county}_EV_Forecast_to_{forecast_datetime.strftime('%Y%m%d')}.csv"
        st.download_button(
            label="Download Forecast Data as CSV",
            data=forecast_df.to_csv(index=False).encode('utf-8'),
            file_name=download_filename,
            mime="text/csv",
        )
