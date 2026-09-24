import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Turmeric Price Prediction",
    page_icon="🌿",
    layout="wide"
)

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "turmeric_ridge_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "duggirala_turmeric_finger_clean.csv"
)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
    return df

model = load_model()
df = load_data()

# ---------------------------------------------------------
# Main page
# ---------------------------------------------------------

st.title("🌿 Turmeric Price Prediction")
st.subheader("Duggirala APMC, Andhra Pradesh")

st.write(
    "Machine learning dashboard for predicting the next observed "
    "turmeric market-day modal price."
)

st.success("Model and cleaned turmeric dataset loaded successfully.")

# Basic project information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Rows", len(df))

with col2:
    st.metric("Latest Modal Price", f"₹{df['Modal_Price'].iloc[-1]:,.0f}")

with col3:
    st.metric(
        "Latest Observation",
        df["Arrival_Date"].iloc[-1].strftime("%d %b %Y")
    )

st.write("### Recent Turmeric Prices")

st.dataframe(
    df[
        [
            "Arrival_Date",
            "Market",
            "Variety",
            "Grade",
            "Min_Price",
            "Max_Price",
            "Modal_Price"
        ]
    ].tail(10),
    use_container_width=True
)


# ---------------------------------------------------------
# Price history chart
# ---------------------------------------------------------

st.write("### 📈 Turmeric Modal Price History")

chart_df = df[
    ["Arrival_Date", "Modal_Price"]
].copy()

chart_df = chart_df.sort_values("Arrival_Date")

st.line_chart(
    chart_df,
    x="Arrival_Date",
    y="Modal_Price",
    height=400
)

st.caption(
    "Historical modal price of Turmeric Finger at Duggirala APMC."
)


# ---------------------------------------------------------
# Model performance
# ---------------------------------------------------------

st.write("### 🤖 Model Performance")

st.write(
    "The models were evaluated using a chronological test set "
    "to preserve the time-series nature of the problem."
)

performance_data = pd.DataFrame({
    "Model": [
        "Previous Price Baseline",
        "Local Linear Regression",
        "Local + NCDEX Linear Regression",
        "Tuned Ridge + NCDEX"
    ],
    "MAE (₹)": [
        487.38,
        521.47,
        509.96,
        509.15
    ],
    "RMSE (₹)": [
        829.07,
        784.29,
        769.19,
        768.97
    ],
    "R²": [
        0.7291,
        0.7576,
        0.7668,
        0.7670
    ]
})

st.dataframe(
    performance_data,
    use_container_width=True,
    hide_index=True
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        "Final Model MAE",
        "₹509.15"
    )

with metric_col2:
    st.metric(
        "Final Model RMSE",
        "₹768.97"
    )

with metric_col3:
    st.metric(
        "Final Model R²",
        "0.7670"
    )

st.caption(
    "Final model: Ridge Regression with NCDEX features, "
    "tuned using historical training data (alpha = 1000)."
)


# ---------------------------------------------------------
# Forward forecast
# ---------------------------------------------------------

st.write("### 🔮 Next Observed Market-Day Forecast")

FORECAST_PATH = os.path.join(
    BASE_DIR,
    "results",
    "forward_forecast.csv"
)

forecast_df = pd.read_csv(FORECAST_PATH)

forecast_row = forecast_df.iloc[0]

latest_date = pd.to_datetime(
    forecast_row["Latest_Observed_Date"]
)

latest_price = float(
    forecast_row["Latest_Observed_Modal_Price"]
)

baseline_forecast = float(
    forecast_row["Naive_Baseline_Forecast"]
)

ridge_forecast = float(
    forecast_row["Ridge_NCDEX_Forecast"]
)

forecast_difference = float(
    forecast_row["Forecast_Difference"]
)

forecast_difference_pct = float(
    forecast_row["Forecast_Difference_Pct"]
)

forecast_col1, forecast_col2, forecast_col3 = st.columns(3)

with forecast_col1:
    st.metric(
        "Latest Observed Price",
        f"₹{latest_price:,.2f}",
        latest_date.strftime("%d %b %Y")
    )

with forecast_col2:
    st.metric(
        "Baseline Forecast",
        f"₹{baseline_forecast:,.2f}"
    )

with forecast_col3:
    st.metric(
        "Ridge + NCDEX Forecast",
        f"₹{ridge_forecast:,.2f}",
        f"{forecast_difference:+,.2f} ({forecast_difference_pct:+.2f}%)"
    )

st.info(
    "Forecast target: the next observed Duggirala APMC market-day "
    "modal price. This is a model forecast, not a known future market price."
)

st.write("#### NCDEX Information Used")

ncdex_col1, ncdex_col2, ncdex_col3 = st.columns(3)

with ncdex_col1:
    st.metric(
        "NCDEX Latest Close",
        f"₹{float(forecast_row['NCDEX_Close']):,.0f}"
    )

with ncdex_col2:
    st.metric(
        "NCDEX Volume",
        f"{float(forecast_row['NCDEX_Volume']):,.0f}"
    )

with ncdex_col3:
    st.metric(
        "NCDEX Open Interest",
        f"{float(forecast_row['NCDEX_Open_Interest']):,.0f}"
    )

st.caption(
    f"Latest NCDEX observation: "
    f"{pd.to_datetime(forecast_row['NCDEX_Latest_Date']).strftime('%d %b %Y')} "
    f"| Expiry: "
    f"{pd.to_datetime(forecast_row['NCDEX_Expiry']).strftime('%d %b %Y')}"
)


# ---------------------------------------------------------
# Model comparison chart
# ---------------------------------------------------------

st.write("### 📊 Model Comparison — MAE")

mae_chart_df = performance_data[
    ["Model", "MAE (₹)"]
].copy()

# Use shorter labels for the chart while keeping
# the complete model names in the performance table.
mae_chart_df["Model"] = [
    "Baseline",
    "Local Linear",
    "Local + NCDEX",
    "Ridge + NCDEX"
]

mae_chart_df = mae_chart_df.set_index("Model")

st.bar_chart(
    mae_chart_df["MAE (₹)"],
    height=350
)

st.caption(
    "Lower MAE indicates smaller average absolute prediction error "
    "on the chronological test set."
)


# ---------------------------------------------------------
# Methodology
# ---------------------------------------------------------

st.write("### 🧠 How the Prediction Works")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.write("#### 1️⃣ Data")
    st.write(
        "Daily Turmeric Finger market-price data is collected "
        "for Duggirala APMC."
    )

with step2:
    st.write("#### 2️⃣ Features")
    st.write(
        "Historical prices, rolling statistics and selected "
        "NCDEX futures information are used as model features."
    )

with step3:
    st.write("#### 3️⃣ Model")
    st.write(
        "A Ridge Regression model is trained using a "
        "chronological time-series split."
    )

with step4:
    st.write("#### 4️⃣ Forecast")
    st.write(
        "The model estimates the next observed Duggirala "
        "market-day modal price."
    )

st.write("#### 🔧 Key Modeling Choices")

st.markdown(
    """
    - **Target:** Next observed Duggirala APMC modal price
    - **Commodity:** Turmeric Finger
    - **Local market:** Duggirala APMC, Andhra Pradesh
    - **External market information:** NCDEX Turmeric Futures
    - **Final model:** Ridge Regression
    - **Ridge alpha:** 1000
    - **Validation:** Chronological / walk-forward validation
    - **Primary metrics:** MAE, RMSE and R²
    """
)


# ---------------------------------------------------------
# Limitations
# ---------------------------------------------------------

st.write("### ⚠️ Limitations")

st.markdown(
    """
    - **Market-day data:** The target represents the next observed
      market day rather than the next calendar day.
    - **Price shocks:** Sudden market movements can produce larger
      prediction errors.
    - **Regime changes:** Historical price patterns may change over time,
      which can reduce model accuracy during new market conditions.
    - **NCDEX coverage:** NCDEX information was available for most,
      but not all, Duggirala observations used in the modeling analysis.
    - **Forecast uncertainty:** The displayed forecast is a model estimate
      and should not be interpreted as a guaranteed future market price.
    """
)

st.write("### 🚀 Future Improvements")

st.markdown(
    """
    - Add weather and rainfall information.
    - Include crop arrival and production data.
    - Incorporate additional market-level supply and demand indicators.
    - Experiment with advanced time-series and machine-learning models.
    - Add automated data updates.
    - Monitor model performance as new market observations become available.
    """
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("🌿 Turmeric AI")

st.sidebar.markdown(
    "### Price Prediction Dashboard"
)

st.sidebar.markdown("---")

st.sidebar.write("**Commodity:** Turmeric Finger")
st.sidebar.write("**Market:** Duggirala APMC")
st.sidebar.write("**Location:** Andhra Pradesh")
st.sidebar.write("**Model:** Ridge Regression")
st.sidebar.write("**External Data:** NCDEX Futures")

st.sidebar.markdown("---")

st.sidebar.write("### 📅 Data Period")

data_start = df["Arrival_Date"].min().strftime("%d %b %Y")
data_end = df["Arrival_Date"].max().strftime("%d %b %Y")

st.sidebar.write(f"{data_start} → {data_end}")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Developed as a machine-learning portfolio project "
    "for agricultural price forecasting."
)


# ---------------------------------------------------------
# Project information / footer
# ---------------------------------------------------------

st.markdown("---")

st.write("### 📌 Project Information")

footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.markdown(
        """
        **Turmeric Price Prediction**

        A machine-learning portfolio project focused on
        forecasting the next observed modal price of Turmeric
        Finger at Duggirala APMC, Andhra Pradesh.
        """
    )

with footer_col2:
    st.markdown(
        """
        **Data & Disclaimer**

        Market-price data is based on the Government of India
        Open Government Data / AGMARKNET dataset. NCDEX futures
        information is used as an external market feature.

        Forecasts shown by this application are model estimates
        and are not guaranteed future market prices.
        """
    )

st.caption(
    "Turmeric Price Prediction | Machine Learning & Data Analytics Portfolio Project"
)
