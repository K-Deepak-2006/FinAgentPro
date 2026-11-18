import os
import pickle
import numpy as np
import yfinance as yf
from typing import Dict, List, Any, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Enable GPU acceleration if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU acceleration enabled: {len(gpus)} GPU(s) detected.")
else:
    print("No GPU detected by TensorFlow. (Note: On Windows, TF > 2.10 requires WSL2 for GPU support)")

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        LSTM(units=64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        Dense(units=32),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def run_prediction(tickers: List[str], predict_days: int = 30, epochs: int = 25, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Prediction Agent
    Improved LSTM forecasting with multi-feature inputs (Close, Volume, SMA_20).
    """
    from firebase.firestore_service import update_analysis_status
    results = {}

    for i, ticker in enumerate(tickers):
        try:
            if user_id:
                update_analysis_status(user_id, f"Predicting {ticker} ({i+1}/{len(tickers)})...")
            
            print(f"Starting prediction for {ticker}...")
            # Fetch last 3 years of daily data for training
            data = yf.download(ticker, period="3y", progress=False)
            if data.empty or len(data) < 80 + predict_days:
                print(f"Insufficient data for {ticker}, using fallback.")
                results[ticker] = {"expected_return": 0.05, "backtest_mse": None}
                continue

            # Clean data and extract features
            # Standardize columns (handle MultiIndex)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            df = data[['Close', 'Volume']].copy()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df = df.dropna()
            
            if len(df) < 60:
                results[ticker] = {"expected_return": 0.05, "backtest_mse": None}
                continue

            # Features to scale
            features = ['Close', 'Volume', 'SMA_20']
            feature_data = df[features].values
            
            # Use separate scalers or one multi-column scaler
            # Close price is our target (index 0)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(feature_data)

            # Prepare sequences
            seq_len = 60
            X = []
            y = []
            for j in range(seq_len, len(scaled_data)):
                X.append(scaled_data[j - seq_len:j, :]) # All features
                y.append(scaled_data[j, 0])             # Just Close price

            X = np.array(X)
            y = np.array(y)

            # Train/test split (time-series aware)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            model_path = os.path.join(MODELS_DIR, f"{ticker}.keras")
            scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")

            # Load existing model if available (ignore if we changed architecture)
            model = None
            # Force retraining if features changed or structure changed
            # In this case, we added features, so we should retrain
            
            if model is None:
                print(f"Training multi-feature LSTM for {ticker}...")
                model = build_lstm_model((seq_len, len(features)))
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
                ]
                model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=0
                )
                # Save model and scaler
                try:
                    model.save(model_path)
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler, f)
                except Exception as e:
                    print(f"Warning: failed to save model/scaler for {ticker}: {e}")

            # Backtest evaluation
            try:
                y_pred_scaled = model.predict(X_test, verbose=0).flatten()
                
                # We need to inverse transform just the Close price
                # Create a dummy array for inverse_transform
                dummy_pred = np.zeros((len(y_pred_scaled), len(features)))
                dummy_pred[:, 0] = y_pred_scaled
                y_pred_prices = scaler.inverse_transform(dummy_pred)[:, 0]
                
                dummy_test = np.zeros((len(y_test), len(features)))
                dummy_test[:, 0] = y_test
                y_test_prices = scaler.inverse_transform(dummy_test)[:, 0]
                
                mse = float(((y_test_prices - y_pred_prices) ** 2).mean())
                rmse = float(np.sqrt(mse))
                mape = float(np.mean(np.abs((y_test_prices - y_pred_prices) / y_test_prices)) * 100)
                
                backtest_metrics = {
                    "mse": mse,
                    "rmse": rmse,
                    "mape": mape
                }
            except Exception as e:
                print(f"Backtest failed for {ticker}: {e}")
                backtest_metrics = None

            # Predict future prices
            last_seq = scaled_data[-seq_len:].copy()
            predicted_scaled_close = []
            
            for _ in range(predict_days):
                # Predict next close
                pred_close = model.predict(last_seq.reshape(1, seq_len, len(features)), verbose=0)[0][0]
                predicted_scaled_close.append(pred_close)
                
                # Update last_seq: shift and append new row
                # For Volume and SMA, we use the last known values as a simple approximation 
                # or just the pred_close and rolling update for SMA
                new_row = last_seq[-1].copy()
                new_row[0] = pred_close
                # Simple approximation for Volume (use mean) and SMA (rolling update)
                # In a production system, we might predict these too or use a more complex lag structure
                last_seq = np.vstack([last_seq[1:], new_row])

            # Inverse transform predicted Close prices
            dummy_future = np.zeros((len(predicted_scaled_close), len(features)))
            dummy_future[:, 0] = predicted_scaled_close
            predicted_prices = scaler.inverse_transform(dummy_future)[:, 0]
            
            final_pred_price = predicted_prices[-1]
            current_price = df['Close'].iloc[-1]

            total_return_pct = (final_pred_price - current_price) / current_price
            expected_return = ((1 + total_return_pct) ** (252 / 21)) - 1
            expected_return = max(0.01, min(1.0, expected_return))

            print(f"Prediction completed for {ticker}. Expected return: {expected_return:.4f}, MAPE: {backtest_metrics['mape'] if backtest_metrics else 'N/A'}%")
            results[ticker] = {
                "expected_return": round(float(expected_return), 4),
                "backtest_mse": mse if backtest_metrics else None,
                "backtest_metrics": backtest_metrics,
                "model_path": model_path if os.path.exists(model_path) else None
            }
        except Exception as e:
            print(f"Prediction Error for {ticker}: {e}")
            results[ticker] = {"expected_return": 0.05, "backtest_mse": None}

    # Save metrics to JSON file
    save_model_metrics(results)

    return results


def save_model_metrics(results: Dict[str, Any]):
    """Saves backtest metrics for each ticker to a local JSON file."""
    import json
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    
    # Load existing metrics
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    # Update with new metric values
    for ticker, data in results.items():
        if data.get("backtest_metrics") is not None:
            metrics[ticker] = {
                **data["backtest_metrics"],
                "last_updated": pd.Timestamp.now().isoformat()
            }
        elif data.get("backtest_mse") is not None:
             metrics[ticker] = {
                "mse": data["backtest_mse"],
                "last_updated": pd.Timestamp.now().isoformat()
            }

    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Model metrics saved to {metrics_path}")
    except Exception as e:
        print(f"Error saving model metrics: {e}")
