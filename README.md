# FinAgent Pro

FinAgent Pro is a multi-agent personalized financial advisor prototype. It ingests live market and news data, profiles users, trains lightweight forecasting models, optimizes portfolios, and generates human-friendly explanations.

Quick start

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Set required environment variables (example `.env`):

- `FIREBASE_CREDENTIALS` — path to serviceAccountKey.json for Firestore
- `NEWSAPI_KEY` — NewsAPI or newsdata.io key (optional; fallback to mock sentiment)
- `GEMINI_API_KEY` — optional for LLM explanations and optional LLM sentiment scoring

3. Run API:

```bash
uvicorn main:app --reload
```

4. Open `frontend/index.html` in a browser (or serve via a simple static file server) and use the UI to register, create a profile, and run Analysis.

Reproducibility & models

- Models and scalers are saved to `models/` per-ticker after training. Each saved model path is recorded in the API response and persisted in Firestore under `model_versions` for traceability.
- Random seeds are set in the prediction agent for basic reproducibility (`tf.random.set_seed(42)`, `numpy` seed).
- For production-grade reproducibility add model versioning (e.g., Git SHA or dataset snapshot IDs) and scheduled retraining pipelines.

Evaluation & backtesting

- Each analysis run now returns `backtest_metrics` and `model_versions` in the API response and persists them in Firestore. `backtest_metrics` contains per-ticker backtest MSE (from LSTM holdout) and portfolio-level historical metrics (total return, annual return, volatility, Sharpe, max drawdown) computed over the last year.

Key improvements implemented

- LSTM model persistence and holdout backtest MSE per ticker
- Ledoit‑Wolf covariance shrinkage for portfolio optimization
- VADER-based sentiment scoring plus optional LLM-based sentiment scoring and comparison
- Portfolio-level historical backtest metrics saved with each portfolio

Next steps (recommended for final report)

- Add scheduled retraining and model version tags
- Implement a robust backtesting harness that simulates past allocations over rolling windows
- Compare forecasting models (LSTM vs ARIMA/XGBoost) and sentiment methods (VADER vs LLM) and include results in your report
- Add unit/integration tests and a short demo video

If you want, I can add a short demo script or integrate backtest plots into a notebook.

---
Generated changes include new model persistence, backtest metrics, Ledoit‑Wolf covariance, VADER+LLM sentiment, and updated API/schema to persist and return these artifacts.
