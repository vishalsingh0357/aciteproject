import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns  # Ensure this is imported for styling the plots

# === Set Streamlit page config first thing ===
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Styling ===
st.markdown("""
    <style>
    /* Main background with linear gradient */
    .stApp {
        background: linear-gradient(to right, #c2d3f2, #7f848a);
        color: white; /* Default text color for the app */
    }

    /* Headers in the app */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }

    /* Text elements like st.write */
    p, label, .css-1y4p8pa {
        color: #FFFFFF;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1c1c1c; /* Dark gray for sidebar background */
        color: white;
        border-right: 1px solid #7f848a;
    }

    /* Input widgets in sidebar (dropdowns, etc.) */
    .stSelectbox, .stMultiselect {
        color: black;
    }

    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiselect div[data-baseweb="select"] > div {
        background-color: white;
        color: black;
    }

    /* Multiselect checkboxes */
    .stMultiselect [data-testid="stCheckbox"] {
        color: black;
    }

    /* Make the selectbox label white */
    .stSelectbox label, .stMultiselect label {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# === Load data & model ===
# Use st.cache_data to speed up data loading on subsequent runs
@st.cache_data
def load_data_and_model():
    try:
        df = pd.read_csv("preprocessed_ev_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        model = joblib.load('forecasting_ev_model.pkl')
        return df, model
    except FileNotFoundError:
        st.error(
            "Missing essential files. Please ensure 'preprocessed_ev_data.csv' and 'forecasting_ev_model.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during file loading: {e}")
        st.stop()


df, model = load_data_and_model()

# === Main Title and Subtitle ===
st.markdown("""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: #FFFFFF; margin-top: 20px;'>
        🔮 EV Adoption Forecaster for a County in Washington State
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; font-size: 22px; font-weight: bold; padding-top: 10px; margin-bottom: 25px; color: #FFFFFF;'>
        Welcome to the Electric Vehicle (EV) Adoption Forecast tool.
    </div>
""", unsafe_allow_html=True)

# === Image ===
st.image("ev-car-factory.jpg", use_container_width=True)

# === Instruction line ===
st.markdown("""
    <div style='text-align: left; font-size: 22px; padding-top: 10px; color: #FFFFFF;'>
        Select a county and see the forecasted EV adoption trend for the next 3 years.
    </div>
""", unsafe_allow_html=True)

# === Single County Forecast ===
county_list = sorted(df['County'].dropna().unique().tolist())
county = st.selectbox("Select a County", county_list)

if county:
    # Filter data and get necessary info for the selected county
    county_df = df[df['County'] == county].sort_values("Date")
    if county_df.shape[0] < 6:
        st.warning(
            f"Not enough historical data for '{county}' to generate a reliable forecast. Need at least 6 months.")
        st.stop()
    county_code = county_df['county_encoded'].iloc[0]
    historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
    historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

    # --- Forecasting logic (same as your previous code) ---
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()
    future_rows = []
    forecast_horizon = 36  # Hardcoded 3 years as per your reference code

    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        recent_cumulative = cumulative_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(
            recent_cumulative) == 6 else 0
        new_row = {
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
        pred = model.predict(pd.DataFrame([new_row]))[0]
        future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
        historical_ev.append(pred)
        if len(historical_ev) > 6: historical_ev.pop(0)
        cumulative_ev.append(cumulative_ev[-1] + pred)
        if len(cumulative_ev) > 6: cumulative_ev.pop(0)

    forecast_df = pd.DataFrame(future_rows)
    forecast_df['Source'] = 'Forecast'
    forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

    combined = pd.concat([
        historical_cum[['Date', 'Cumulative EV']].assign(Source='Historical'),
        forecast_df[['Date', 'Cumulative EV', 'Source']]
    ], ignore_index=True)

    # === Plot Cumulative Graph ===
    st.subheader(f"📊 Cumulative EV Forecast for {county} County")
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, data in combined.groupby('Source'):
        ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')
    ax.set_title(f"Cumulative EV Trend - {county} (3 Years Forecast)", fontsize=14, color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Cumulative EV Count", color='white')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1c1c1c")
    fig.patch.set_facecolor('#1c1c1c')
    ax.tick_params(colors='white')
    ax.legend()
    st.pyplot(fig)

    # === Compare historical and forecasted cumulative EVs ===
    historical_total = historical_cum['Cumulative EV'].iloc[-1]
    forecasted_total = forecast_df['Cumulative EV'].iloc[-1]
    if historical_total > 0:
        forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
        trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
        st.success(
            f"Based on the graph, EV adoption in **{county}** is expected to show a **{trend} of {forecast_growth_pct:.2f}%** over the next 3 years.")
    else:
        st.warning("Historical EV total is zero, so percentage forecast change can't be computed.")

# === New: Compare up to 3 counties ===
st.markdown("---")
st.header("Compare EV Adoption Trends for up to 3 Counties")

multi_counties = st.multiselect("Select up to 3 counties to compare", county_list, max_selections=3)

if multi_counties:
    comparison_data = []

    for cty in multi_counties:
        cty_df = df[df['County'] == cty].sort_values("Date")
        if cty_df.shape[0] < 6:
            st.warning(f"Skipping '{cty}': Not enough historical data for a 3-year forecast.")
            continue
        cty_code = cty_df['county_encoded'].iloc[0]

        hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
        cum_ev = list(np.cumsum(hist_ev))
        months_since = cty_df['months_since_start'].max()
        last_date = cty_df['Date'].max()

        future_rows_cty = []
        for i in range(1, forecast_horizon + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            months_since += 1
            lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            recent_cum = cum_ev[-6:]
            ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0
            new_row = {
                'months_since_start': months_since,
                'county_encoded': cty_code,
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct_change_1,
                'ev_total_pct_change_3': pct_change_3,
                'ev_growth_slope': ev_slope
            }
            pred = model.predict(pd.DataFrame([new_row]))[0]
            future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
            hist_ev.append(pred)
            if len(hist_ev) > 6: hist_ev.pop(0)
            cum_ev.append(cum_ev[-1] + pred)
            if len(cum_ev) > 6: cum_ev.pop(0)

        hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()

        fc_df = pd.DataFrame(future_rows_cty)
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

        combined_cty = pd.concat([
            hist_cum[['Date', 'Cumulative EV']].assign(Source='Historical'),
            fc_df[['Date', 'Cumulative EV']].assign(Source='Forecast')
        ], ignore_index=True)

        combined_cty['County'] = cty
        comparison_data.append(combined_cty)

    # Combine all counties data for plotting
    comp_df = pd.concat(comparison_data, ignore_index=True)

    # Plot
    st.subheader("📈 Comparison of Cumulative EV Adoption Trends")
    fig, ax = plt.subplots(figsize=(14, 7))
    for cty, group in comp_df.groupby('County'):
        ax.plot(group['Date'], group['Cumulative EV'], marker='o', label=cty)
    ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=16, color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Cumulative EV Count", color='white')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1c1c1c")
    fig.patch.set_facecolor('#1c1c1c')
    ax.tick_params(colors='white')
    ax.legend(title="County")
    st.pyplot(fig)

    # Display % growth for selected counties ===
    growth_summaries = []
    for cty in multi_counties:
        # Filter for the current county's historical data from the combined comparison DataFrame
        historical_data_cty = comp_df[(comp_df['Source'] == 'Historical') & (comp_df['County'] == cty)]

        if not historical_data_cty.empty:
            historical_total = historical_data_cty['Cumulative EV'].iloc[-1]
            # Find the last value for the current county in the comparison DataFrame
            forecasted_total = comp_df[comp_df['County'] == cty]['Cumulative EV'].iloc[-1]

            growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
            growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
        else:
            growth_summaries.append(f"{cty}: N/A (no historical data to compare)")

    # Join all in one sentence and show with st.success
    growth_sentence = " | ".join(growth_summaries)
    st.success(f"Forecasted EV adoption growth over next 3 years — {growth_sentence}")

# === New: Download Forecast Button for Single County ===
st.markdown("---")
st.header("Download Data")

# Create a Download button for the single county forecast
if 'forecast_df' in locals():
    csv_data = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Download {county} Forecast CSV",
        data=csv_data,
        file_name=f"{county}_ev_forecast_3_years.csv",
        mime="text/csv"
    )
    st.info("The CSV contains the 3-year forecast data for the selected county.")
else:
    st.info("Please select a county and a forecast will appear here.")
