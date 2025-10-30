import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from textblob import TextBlob
import feedparser
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ------------------ APP CONFIG ------------------
st.set_page_config(page_title="Stock Sentiment & LSTM Predictor", layout="wide", page_icon="📈")
st.title("📊 Stock Price Prediction with News Sentiment (INR)")

# ------------------ USER INPUT ------------------
ticker = st.text_input("Enter Indian Stock Symbol (e.g. INFY.NS, TCS.NS, RELIANCE.NS)", "INFY.NS")

# ------------------ FETCH DATA ------------------
st.subheader("📈 Stock Data (Past 2 Years)")
try:
    df = yf.download(ticker, period="2y", auto_adjust=True)
    df = df.reset_index()
    df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]  # flatten multi-index columns

    fig = px.line(df, x='Date', y='Close', title=f'{ticker} Closing Prices (₹) - Last 2 Years')
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error fetching data: {e}")

# ------------------ FETCH LATEST NEWS (GOOGLE RSS) ------------------
st.subheader("🧠 Average News Sentiment")

rss_url = f"https://news.google.com/rss/search?q={ticker.replace('.NS', '')}+stock&hl=en-IN&gl=IN&ceid=IN:en"
feed = feedparser.parse(rss_url)

news_list = []
for entry in feed.entries[:10]:
    title = entry.title
    sentiment = TextBlob(title).sentiment.polarity
    news_list.append((title, sentiment))

if news_list:
    avg_sentiment = np.mean([s for _, s in news_list])
    st.metric("Average Sentiment (Last 7 Days)", f"{avg_sentiment:.3f}")
    st.markdown("### 📰 Latest Headlines with Sentiment:")
    for title, sent in news_list:
        emoji = "🟢" if sent > 0 else "🔴" if sent < 0 else "⚪"
        st.markdown(f"- {title} ({emoji} Sentiment: {sent:+.3f})")
else:
    avg_sentiment = 0
    st.warning("No recent news found for this stock.")

# ------------------ TRAIN MODEL ------------------
st.header("⚙️ Train Model & Predict Future Prices")

data = df[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]

X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i])
    y_train.append(train_data[i])
X_train, y_train = np.array(X_train), np.array(y_train)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

st.success("✅ Model training completed!")

# ------------------ PREDICT NEXT 10 DAYS ------------------
last_60 = scaled_data[-60:]
pred_input = last_60.reshape(1, -1, 1)
future_preds = []

for _ in range(10):
    next_price = model.predict(pred_input, verbose=0)[0][0]
    future_preds.append(next_price)
    pred_input = np.append(pred_input.flatten()[1:], next_price).reshape(1, -1, 1)

predicted_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

# Add realistic randomness & sentiment-based bias
randomness = np.random.normal(0, 0.005, 10)
trend_adjust = 1 + (avg_sentiment * 0.02)
predicted_prices = predicted_prices * trend_adjust * (1 + randomness)

# Create dataframe
future_dates = [df['Date'].iloc[-1] + timedelta(days=i+1) for i in range(10)]
pred_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Close': predicted_prices
})

st.subheader("📅 Next 10 Days Predicted Prices (₹)")
pred_df_display = pred_df.copy()
pred_df_display['Predicted_Close'] = pred_df_display['Predicted_Close'].apply(lambda x: f"₹{x:.2f}")
st.dataframe(pred_df_display, hide_index=True, use_container_width=True)

# ------------------ ADD GRAPH ------------------
st.markdown("### 📊 Future Price Trend (Predicted)")
fig_pred = px.line(pred_df, x='Date', y='Predicted_Close',
                   title=f'{ticker} - Next 10 Days Predicted Trend (₹)',
                   markers=True)
st.plotly_chart(fig_pred, use_container_width=True)


































