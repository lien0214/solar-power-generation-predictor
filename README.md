# Solar Power Generation Predictor API

FastAPI-based REST API for predicting solar power generation using machine learning models.

## 🏗️ Project Structure

```
repo/
├── app/                      # Main application package
│   ├── api/                  # API endpoints
│   │   └── v1/              # API version 1
│   │       ├── prediction.py  # Prediction endpoints
│   │       └── __init__.py
│   ├── core/                 # Core configuration
│   │   ├── config.py        # Pydantic settings
│   │   └── __init__.py
│   ├── models/               # ML model training
│   │   ├── weather_trainer.py  # Weather model
│   │   ├── solar_trainer.py    # Solar model
│   │   ├── model_store.py      # Model persistence
│   │   └── __init__.py
│   ├── schemas/              # Pydantic DTOs
│   │   ├── prediction.py    # Response schemas
│   │   └── __init__.py
│   ├── services/             # Business logic
│   │   ├── model_manager.py  # Model lifecycle
│   │   ├── prediction.py     # Prediction logic
│   │   └── __init__.py
│   ├── utils/                # Shared utilities
│   ├── main.py              # FastAPI app instance
│   └── __init__.py
├── doc/                      # Documentation
├── models/                   # Trained model storage
├── .env.example             # Example environment config
├── requirements.txt         # Python dependencies
├── run.py                   # Server entry point
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd repo
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run the Server

#### Option A: Load Pre-trained Models
```bash
# Make sure STARTUP_MODE=load_models in .env
python run.py
```

#### Option B: Train Models on Startup
```bash
# Set STARTUP_MODE=train_now in .env
python run.py
```

#### Option C: Using Uvicorn Directly
```bash
uvicorn app.main:app --reload
```

### 4. Access the API

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root**: http://localhost:8000/

## 📡 API Endpoints

### Prediction Endpoints

#### Daily Predictions
```
GET /v1/predict/day
```
Query Parameters:
- `lon`: Longitude (-180 to 180)
- `lat`: Latitude (-90 to 90)
- `startDate`: Start date (YYYY-MM-DD)
- `endDate`: End date (YYYY-MM-DD)
- `pmp`: Panel Maximum Power in Watts (default: 1000)

Example:
```bash
curl "http://localhost:8000/v1/predict/day?lon=119.588339&lat=23.530236&startDate=2025-01-01&endDate=2025-01-31&pmp=1000"
```

#### Monthly Predictions
```
GET /v1/predict/month
```
Query Parameters:
- `lon`: Longitude
- `lat`: Latitude
- `startDate`: Start month (YYYY-MM)
- `endDate`: End month (YYYY-MM)
- `pmp`: Panel Maximum Power (W)

Example:
```bash
curl "http://localhost:8000/v1/predict/month?lon=119.588339&lat=23.530236&startDate=2025-01&endDate=2025-12&pmp=1000"
```

#### Yearly Predictions
```
GET /v1/predict/year
```
Query Parameters:
- `lon`: Longitude
- `lat`: Latitude
- `year`: Year (2000-2100)
- `pmp`: Panel Maximum Power (W)

Example:
```bash
curl "http://localhost:8000/v1/predict/year?lon=119.588339&lat=23.530236&year=2025&pmp=1000"
```

## 🔧 Configuration

All configuration is managed through environment variables (`.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `STARTUP_MODE` | `train_now` or `load_models` | `load_models` |
| `MODEL_DIR` | Model storage directory | `./models` |
| `WEATHER_HIST_FILE` | Historical weather data | `../code/data/23.530236_119.588339.csv` |
| `WEATHER_PRED_FILE` | Predicted weather data | `../code/data/weather-pred.csv` |
| `WEATHER_WINDOW_SIZE` | Weather model window (days) | `30` |
| `SOLAR_TEST_MONTHS` | Solar test period (months) | `6` |
| `LOG_LEVEL` | Logging level | `INFO` |

See `.env.example` for all available options.

## 🧪 Model Training

The application supports two startup modes:

### Load Pre-trained Models
```bash
export STARTUP_MODE=load_models
python run.py
```
Loads models from `MODEL_DIR`.

### Train on Startup
```bash
export STARTUP_MODE=train_now
python run.py
```
Trains both weather and solar models on startup. This matches the exact implementation from the reference XGBoost code:
- **Weather Model**: 30-day window, multi-output XGBoost
- **Solar Model**: Merged dataset with one-hot encoding

## 🐳 Docker Support

The app/ structure is designed for easy Docker containerization:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ app/
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📚 Architecture

This project follows **FastAPI best practices** with clear **Separation of Concerns**:

- **`app/api/`**: HTTP layer (routing, validation)
- **`app/services/`**: Business logic
- **`app/models/`**: ML model training and management
- **`app/schemas/`**: Data contracts (DTOs)
- **`app/core/`**: Configuration management

## 🔗 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- See `doc/` directory for detailed component documentation
