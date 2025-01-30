from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
import numpy as np


app = Flask(__name__)

# Load Data
df = pd.read_csv("Finaldatset.csv", parse_dates=["date"], index_col="date")
df = df.reset_index().rename(columns={"date": "ds"})

def forecast_language(language, periods=24):
    if language not in df.columns:
        return {"error": "Language not found in dataset"}
    
    data = df[["ds", language]].rename(columns={language: "y"})
    data = data.dropna()
    
    model = Prophet()
    model.fit(data)
    
    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)
    
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    return result.to_dict(orient="records")


def forecast_arima(language, periods=24):
    if language not in df.columns:
        return {"error": "Language not found in dataset"}
    
    data = df[['ds', language]].dropna()
    model = ARIMA(data[language], order=(5,1,0))  # Example order, tune for better performance
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=periods)
    future_dates = pd.date_range(start=data['ds'].iloc[-1], periods=periods+1, freq='M')[1:]
    
    return [{"ds": str(date), "yhat": pred} for date, pred in zip(future_dates, forecast)]


# def prepare_lstm_data(data, look_back=12):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# def forecast_lstm(language, periods=24):
    if language not in df.columns:
        return {"error": "Language not found in dataset"}
    
    data = df[language].dropna().values.reshape(-1, 1)
    look_back = 12
    X, y = prepare_lstm_data(data, look_back)

    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=8, verbose=0)

    predictions = []
    input_seq = data[-look_back:].reshape(1, look_back, 1)
    for _ in range(periods):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.roll(input_seq, -1)
        input_seq[0, -1, 0] = pred
    
    future_dates = pd.date_range(start=df['ds'].iloc[-1], periods=periods+1, freq='M')[1:]
    return [{"ds": str(date), "yhat": pred} for date, pred in zip(future_dates, predictions)]



@app.route("/forecast", methods=["POST"])
def get_forecast():
    request_data = request.get_json()
    language = request_data.get("language")
    model_type = request_data.get("model", "prophet")

    if not language:
        return jsonify({"error": "Language is required"}), 400
    
    if model_type == "arima":
        forecast_data = forecast_arima(language)
    # elif model_type == "lstm":
    #     forecast_data = forecast_lstm(language)
    else:
        forecast_data = forecast_language(language)

    return jsonify(forecast_data)

if __name__ == "__main__":
    app.run(debug=True)
