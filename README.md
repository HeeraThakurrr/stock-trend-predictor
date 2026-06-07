# Stock Trend Predictor

I built this project to predict whether a stock will go up or down the next day, using historical price data and machine learning. The idea came from my interest in finance and stock markets, and I wanted to build something end-to-end — from raw data all the way to a deployed web app.

The project uses Reliance Industries (RELIANCE.NS) as the primary stock, but the Streamlit app lets you enter any NSE ticker and get a prediction.

---

## What it does

- Pulls 7 years of historical stock data using the yfinance API
- Does a detailed EDA — closing price trends, volume analysis, year-by-year breakdown
- Engineers features like 7-day and 21-day moving averages, daily returns, and volatility
- Trains and compares 4 ML models: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting
- Uses K-Means clustering to group stock behavior into 3 patterns (uptrend, downtrend, neutral)
- Deploys everything as an interactive web app using Streamlit

---

## Tech Stack

- **Python** — Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning** — Scikit-learn (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, K-Means)
- **Data** — yfinance API
- **Deployment** — Streamlit

---

## Results

After comparing all 4 models using precision, recall, and F1-score, Gradient Boosting performed the best. Stock market prediction is genuinely hard — the model achieves around 55-60% accuracy, which is realistic given that it only uses price-based features and can't account for news, events, or policy changes.

---

## Live Demo

[]

---

## Project Structure

```
stock-trend-predictor/
│
├── app.py                  # Streamlit web app
├── Resume_Project.ipynb    # Full EDA + model training notebook
├── requirements.txt        # Dependencies
└── README.md
```

---

## Limitations

The model only looks at historical price patterns. It can't predict black swan events like COVID-19, company-specific news like product launches, or macroeconomic changes like interest rate hikes — all of which heavily impact stock prices in the real world.

---

## About

I'm Devraj Anand, a B.Tech CSE (Data Science) student with an interest in finance and ML. This is one of my portfolio projects built while preparing for data science internships.
