# 🌿 Turmeric Price Prediction — Duggirala APMC

A machine-learning project for forecasting the next observed market-day modal price of Turmeric Finger at Duggirala APMC, Andhra Pradesh, using historical mandi prices and selected NCDEX Turmeric Futures information.

---

## 📌 Project Overview

This project studies historical Turmeric Finger prices at Duggirala APMC and develops a machine-learning forecasting workflow.

The project includes:
- Real-world government mandi price data
- Data cleaning and validation
- Exploratory data analysis
- Time-series feature engineering
- NCDEX Turmeric Futures integration
- Chronological model evaluation
- Ridge Regression
- Walk-forward validation
- Forward forecasting
- Streamlit dashboard

## 🎯 Problem Statement

Predict the next observed Duggirala APMC market-day modal price of Turmeric Finger.

The target represents the next available observation in the market-price dataset rather than the next calendar day.

## 📊 Data Sources

### Government Mandi Price Data

Primary source: Government of India Open Government Data / AGMARKNET.

Project filtering:
- State: Andhra Pradesh
- District: Guntur
- Market: Duggirala / Duggirala APMC
- Commodity: Turmeric
- Variety: Finger

Official resource:
https://data.gov.in/resource/variety-wise-daily-market-prices-data-commodity

### NCDEX Turmeric Futures

NCDEX Turmeric Futures symbol: `TMCFGRNZM`

Official product:
https://www.ncdex.com/index.php/products/TMCFGRNZM

## 🧹 Data Preparation

The cleaned Turmeric Finger dataset contains:
- 783 observations
- 14 columns
- Date range: 09 Apr 2021 → 07 Sep 2026
- No missing values
- Duggirala APMC market
- Turmeric Finger variety

Price fields include Minimum Price, Maximum Price, and Modal Price.

An invalid price-logic observation was removed during data cleaning.

## 🎯 Target Definition

Target variable: `Target_Modal_Price`

It represents the modal price of the next observed Duggirala market day.

## ⚙️ Feature Engineering

Local-market features:
- Lag 1, Lag 2, Lag 3
- Previous minimum price
- Previous maximum price
- Previous price range
- 7-day rolling mean
- 14-day rolling mean
- 30-day rolling mean
- 7-day rolling standard deviation
- Month

NCDEX features:
- Same-contract NCDEX close lag
- Same-contract NCDEX volume lag
- Same-contract NCDEX open-interest lag

Contract switching was explicitly considered during NCDEX feature engineering.

## 🧪 Machine Learning Approach

The project evaluated:
1. Previous Price Baseline
2. Linear Regression using local features
3. Linear Regression using local + NCDEX features
4. Ridge Regression using local + NCDEX features
5. Decision Tree
6. Random Forest
7. Gradient Boosting

A chronological split was used, with training observations before 01 Jan 2026 and 2026 observations used as the chronological test period.

## 🏆 Final Model

Final portfolio model: **Ridge Regression + NCDEX features**

Ridge hyperparameter: `alpha = 1000`

## 📈 Final Test Results

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| Previous Price Baseline | 487.38 | 829.07 | 0.7291 |
| Local Linear Regression | 521.61 | 784.27 | 0.7576 |
| Local + NCDEX Linear Regression | 510.74 | 769.02 | 0.7669 |
| Tuned Ridge + NCDEX | 509.90 | 768.78 | 0.7671 |

The Previous Price Baseline produced the lowest MAE on the 2026 test set.
The Tuned Ridge + NCDEX model produced the lowest RMSE and highest R² among the evaluated approaches in the final comparison.

## 🔄 Walk-Forward Validation

Average results:

| Model | Average MAE (₹) | Average RMSE (₹) |
|---|---:|---:|
| Previous Price Baseline | 475.80 | 701.19 |
| Local + NCDEX Linear Regression | 496.20 | 697.91 |
| Tuned Ridge + NCDEX | 473.52 | 676.70 |

Across five walk-forward folds, Ridge had lower MAE than the baseline in 2 of 5 folds and lower RMSE in 3 of 5 folds.

## 🔮 Forward Forecast

Latest Duggirala observation: **07 Sep 2026**

Latest observed modal price: **₹14,650**

| Method | Forecast |
|---|---:|
| Previous Price Baseline | ₹14,650.00 |
| Ridge + NCDEX | ₹15,086.24 |

The Ridge + NCDEX forecast is approximately ₹436.24 higher than the previous-price baseline, or +2.98%.

Latest NCDEX observation: **01 Sep 2026**

Selected expiry: **16 Oct 2026**

NCDEX close: **₹20,018**

NCDEX volume: **4,715**

NCDEX open interest: **36,225**

The forward forecast is a model estimate, not a guaranteed future market price.

## 📊 Error Analysis

The final model captures broad price movement but can experience larger errors during sudden market movements.

The 2026 residual analysis showed larger errors around sharp price increases and decreases.

## ⚠️ Limitations

- The target represents the next observed market day rather than the next calendar day.
- Sudden market movements can produce larger prediction errors.
- Historical price relationships can change under new market regimes.
- NCDEX information was not available for every Duggirala observation.
- Weather, rainfall, production, arrivals, and supply-demand indicators were not included in the current model.
- The forward forecast is a model estimate and should not be interpreted as a guaranteed future market price.

## 🚀 Future Improvements

- Weather and rainfall integration
- Crop production information
- Market arrival volumes
- Supply and demand indicators
- Additional market-level features
- Advanced time-series models
- Automated data collection
- Automated model retraining
- Model performance monitoring
- Dashboard data refresh automation

## 🖥️ Streamlit Dashboard

The project includes a Streamlit dashboard containing:
- Latest turmeric price
- Recent market observations
- Historical price chart
- Model performance comparison
- Forward forecast
- NCDEX information
- Methodology
- Limitations
- Future improvements

Run locally with:

```bash
streamlit run app/app.py
```

## 📁 Project Structure

```text
turmeric-price-prediction/
├── app/
│   └── app.py
├── data/
│   ├── duggirala_turmeric_finger_clean.csv
│   └── ncdex_turmeric_features.csv
├── models/
│   └── turmeric_ridge_model.pkl
├── results/
│   ├── final_model_results.csv
│   ├── walk_forward_validation.csv
│   ├── walk_forward_summary.csv
│   └── forward_forecast.csv
├── notebook/
│   └── Turmeric_Price_Prediction_Duggirala.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Ridge Regression
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Google Colab
- NCDEX data
- Government OGD / AGMARKNET data

## 📚 Reproducibility

The project follows a chronological modeling approach to reduce future-information leakage.

The production Ridge model was trained using the available complete observations containing both local-market and NCDEX features.

Saved model: `models/turmeric_ridge_model.pkl`

## 📌 Disclaimer

This project is intended for educational, analytical, and portfolio purposes.

The forecasts generated by the model are estimates based on historical data and selected market features. They are not guaranteed future prices and should not be treated as financial or trading advice.

## 👨‍💻 Project Focus

**Machine Learning + Data Analytics + Agricultural Price Forecasting**

Built as a real-world portfolio project using publicly available agricultural market and commodity-futures data.