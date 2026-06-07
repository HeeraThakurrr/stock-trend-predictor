import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

st.title("Stock Market Trend Predictor")
ticker = st.text_input("Enter stock ticker", "RELIANCE.NS")

#Downloading the dataset
df = yf.download(ticker, period="7y")
df.columns = df.columns.droplevel(1)

## Line Chart Presentation
st.line_chart(df, y='Close', x_label="Date", y_label="Closing Price")

## Creating Features
df['Ma_7'] = df['Close'].rolling(7).mean()
df['Ma_21'] = df['Close'].rolling(21).mean()
df['Return'] = df['Close'].pct_change()
df['Volatility'] = df['Return'].rolling(7).std()

## Target Column created
df['Target'] = np.where(df['Return'].shift(-1) < 0, 0, 1)

## Dropped Null Rows
df = df.dropna()

if df.empty:
    st.error("No data found for this ticker. Please check the ticker symbol.")
    st.stop()

if len(df) < 50:
    st.error("Not enough data to train the model for this ticker.")
    st.stop()

# Input and Target Columns
x = ['Return', 'Ma_7', 'Ma_21', 'Volatility']
y = df['Target']

## Training the Model
model = GradientBoostingClassifier()
model.fit(df[x], y)
val = df[x].iloc[-1].values.reshape(1, -1)
prediction = model.predict(val)

if(prediction == 1):
    st.write("Stock is predicted to go UP tomorrow")
else:
    st.write("Stock is predicted to go DOWN tomorrow")
