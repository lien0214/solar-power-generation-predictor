# System Architecture Documentation

**Last Updated**: December 15, 2025  
**Version**: 1.0.0

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Component Details](#component-details)
4. [Request Flow](#request-flow)
5. [Data Flow](#data-flow)
6. [Class Diagrams](#class-diagrams)
7. [Sequence Diagrams](#sequence-diagrams)
8. [Design Patterns](#design-patterns)
9. [Technology Stack](#technology-stack)

---

## System Overview

The Solar Power Prediction System is a FastAPI-based web service that predicts solar power generation using trained XGBoost machine learning models. The system processes weather data to forecast daily, monthly, and yearly solar energy output.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  (HTTP Clients, Web Apps, Mobile Apps, External Services)       │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/JSON
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI Application (app/main.py)                       │   │
│  │  - CORS Middleware                                       │   │
│  │  - Error Handling                                        │   │
│  │  - Lifecycle Management                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Core Logic & Strategies                                 │   │
│  │  - Hybrid Weather Pipeline (API + Rolling)               │   │
│  │  - Last Valid Date Determination (-999 check)            │   │
│  │  - Strategies: Merged vs Seperated                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  API v1 Router (app/api/v1/prediction.py)               │   │
│  │  - GET /v1/predict/day                                   │   │
│  │  - GET /v1/predict/month                                 │   │
│  │  - GET /v1/predict/year                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Service Layer                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  PredictionService (app/services/prediction.py)          │   │
│  │  - Date range generation                                 │   │
│  │  - Business logic                                        │   │
│  │  - Aggregation logic                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ModelManagerService (app/services/model_manager.py)     │   │
│  │  - Model lifecycle management                            │   │
│  │  - Train/Load models                                     │   │
│  │  - Model state tracking                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Model Layer                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Weather Trainer (app/models/weather_trainer.py)         │   │
│  │  - XGBoost weather forecasting                           │   │
│  │  - 30-day sequence windows                               │   │
│  │  - Multi-output regression                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Solar Trainer (app/models/solar_trainer.py)            │   │
│  │  - XGBoost solar generation forecasting                 │   │
│  │  - One-hot encoding for sites                           │   │
│  │  - Weather feature integration                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Model Store (app/models/model_store.py)                │   │
│  │  - Save/Load model bundles (pickle)                     │   │
│  │  - Model metadata management                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  File System                                             │   │
│  │  - Weather CSV files (NASA POWER format)                │   │
│  │  - Solar generation CSV files (Dynamic Discovery)       │   │
│  │  - Trained model bundles (.pkl)                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Logic & Strategies

### Data Pipeline Logic
A key feature is the hybrid approach to weather data:
1.  **Last Valid Date Determination**: On startup, the system scans `WEATHER_HIST_FILE` for the value `-999` (MISSING_VALUE). The last date before this marker is cached as the "Last Valid Date".
2.  **Prediction Routing**:
    *   **Historical Dates** (<= `last_valid_date`): The system simulates a fetch from an external weather API.
    *   **Future Dates** (> `last_valid_date`): The system uses its own trained weather model to generate a "rolling forecast".
3.  **Solar Prediction**: Weather data is combined with panel info (`pmp`) and fed into the selected solar model.

### Prediction Strategies
The system supports two strategies for solar power prediction, selectable via the `strategy` query parameter:

*   **`merged` (The Generalist)**:
    *   Uses a single model trained on *all* available solar site data combined.
    *   Best for general predictions or unknown locations.
*   **`seperated` (The Committee)**:
    *   Uses an ensemble of models, where each model is trained on a specific solar site's data.
    *   The final prediction is the **average (mean)** of all individual model outputs.
    *   Best for stability and robustness.

---

## Architecture Diagrams

### 1. Layered Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                          │
│  - HTTP Endpoints (FastAPI Routes)                             │
│  - Request/Response Serialization (Pydantic)                   │
│  - Input Validation                                            │
└────────────────────────────────────────────────────────────────┘
                             ↕
┌────────────────────────────────────────────────────────────────┐
│                      Business Logic Layer                       │
│  - PredictionService: Orchestrates predictions                 │
│  - ModelManagerService: Manages ML model lifecycle            │
│  - Date range calculation                                      │
│  - Aggregation logic (day → month → year)                     │
└────────────────────────────────────────────────────────────────┘
                             ↕
┌────────────────────────────────────────────────────────────────┐
│                      ML Model Layer                             │
│  - Weather Forecasting Model (XGBoost)                        │
│  - Solar Generation Model (XGBoost)                           │
│  - Feature Engineering                                         │
│  - Prediction Logic                                            │
└────────────────────────────────────────────────────────────────┘
                             ↕
┌────────────────────────────────────────────────────────────────┐
│                      Data Access Layer                          │
│  - CSV File I/O                                                │
│  - Model Serialization (pickle)                               │
│  - Configuration Management                                    │
└────────────────────────────────────────────────────────────────┘
```

### 2. Startup Flow

```
Application Start
       │
       ▼
┌──────────────────────────┐
│   Load Configuration     │
│   (app/core/config.py)   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Create FastAPI App      │
│  (app/main.py)           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐     YES    ┌──────────────────────────┐
│  startup_mode ==         ├───────────►│  Train Models            │
│  "train_now"?            │            │  - Weather Model         │
└────────────┬─────────────┘            │  - Solar Model           │
             │ NO                        └──────────┬───────────────┘
             │                                      │
             ▼                                      │
┌──────────────────────────┐                       │
│  Load Existing Models    │◄──────────────────────┘
│  from model_dir          │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Initialize Services     │
│  - PredictionService     │
│  - ModelManagerService   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Register API Routes     │
│  /v1/predict/*           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Server Ready            │
│  Accepting Requests      │
└──────────────────────────┘
```

---

## Component Details

### 1. API Layer (`app/api/`)

**Purpose**: HTTP interface for client interactions

**Components**:
- **v1/prediction.py**: API endpoints for predictions
  - `GET /v1/predict/day`: Day-range predictions
  - `GET /v1/predict/month`: Month-range predictions
  - `GET /v1/predict/year`: Yearly predictions

**Responsibilities**:
- Parse HTTP query parameters (camelCase)
- Validate input using Pydantic schemas
- Delegate to PredictionService
- Format responses as JSON
- Handle HTTP errors (400, 422, 503)

**Key Patterns**:
- Dependency Injection (FastAPI's Depends)
- Request/Response Models (Pydantic)
- Async/Await for non-blocking I/O

### 2. Service Layer (`app/services/`)

**Purpose**: Business logic and orchestration

#### PredictionService

**Responsibilities**:
- Generate date ranges
- Coordinate weather + solar predictions
- Aggregate daily → monthly → yearly
- Validate model readiness

**Key Methods**:
```python
async predict_day_range(lon, lat, start_date, end_date, pmp) -> List[DayPrediction]
async predict_month_range(lon, lat, start_date, end_date, pmp) -> List[MonthPrediction]
async predict_year(lon, lat, year, pmp) -> float
```

#### ModelManagerService

**Responsibilities**:
- Model lifecycle management
- Train or load models on startup
- Track model state (ready/not ready)
- Provide model bundles to PredictionService

**Key Methods**:
```python
async initialize(mode, model_dir, ...) -> None
is_ready() -> bool
get_weather_model() -> Dict[str, Any]
get_solar_model() -> Dict[str, Any]
```

### 3. Model Layer (`app/models/`)

**Purpose**: Machine learning model training and prediction

#### Weather Trainer

**Algorithm**: XGBoost Multi-Output Regressor
**Features**:
- 30-day sequence windows
- Seasonality encoding (sin/cos)
- Multi-day ahead forecasting (1-30 days)
- 500 estimators, learning_rate=0.05

**Key Function**:
```python
train_weather_model(csv_path, output_dir, targets, win, train_end, val_end, mode) -> Dict
```

#### Solar Trainer

**Algorithm**: XGBoost Regressor
**Features**:
- One-hot encoding for multiple sites
- Weather feature integration
- 6-month test period, 1-month validation
- 600 estimators, learning_rate=0.1

**Key Function**:
```python
train_solar_model(solar_files, weather_hist_file, weather_pred_file, output_dir, test_months, valid_months) -> Dict
```

### 4. Schema Layer (`app/schemas/`)

**Purpose**: Data validation and serialization

**Models**:
- `Location`: Lat/lon coordinates (±90, ±180)
- `DayPrediction`: Single day prediction (date, kWh)
- `MonthPrediction`: Monthly aggregation (date, kWh)
- `DayPredictionResponse`: API response for day predictions
- `MonthPredictionResponse`: API response for month predictions
- `YearPredictionResponse`: API response for year predictions

---

## Request Flow

### Day Prediction Flow

```
Client
  │
  │ GET /v1/predict/day?lon=119.588&lat=23.530&startDate=2025-01-01&endDate=2025-01-31&pmp=1000
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ API Layer (prediction.py)                                   │
│ 1. Parse query params (camelCase → snake_case)             │
│ 2. Validate with Pydantic (lat/lon bounds, date formats)   │
│ 3. Call prediction_service.predict_day_range()             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ PredictionService (prediction.py)                           │
│ 4. Check model_manager.is_ready()                          │
│ 5. Parse start_date and end_date                           │
│ 6. Generate date range (start → end)                       │
│ 7. For each date:                                          │
│    a. Predict weather (T2M, CLOUD, radiation, etc.)       │
│    b. Predict solar generation using weather features      │
│    c. Scale by PMP                                         │
│ 8. Return List[DayPrediction]                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ ModelManagerService (model_manager.py)                      │
│ 9. Provide weather_model_bundle                            │
│ 10. Provide solar_model_bundle                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ ML Models                                                    │
│ 11. Weather Model: Predict T2M, CLOUD_AMT, etc.            │
│ 12. Solar Model: Predict kWh based on weather + location   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                        Response JSON
                    [{date: "2025-01-01", kwh: 150.2}, ...]
```

### Month Prediction Flow

```
Client → API Layer
         │
         ▼
PredictionService.predict_month_range()
         │
         ├─► For each month in range:
         │   ├─► predict_day_range(first_day, last_day)
         │   └─► Aggregate: sum(daily_kwh)
         │
         └─► Return List[MonthPrediction]
```

### Year Prediction Flow

```
Client → API Layer
         │
         ▼
PredictionService.predict_year()
         │
         ├─► Generate 12 months (Jan-Dec)
         │
         ├─► predict_month_range(Jan, Dec)
         │
         └─► Aggregate: sum(monthly_kwh)
         │
         └─► Return float (total_kwh)
```

---

## Data Flow

### Training Data Flow

```
┌──────────────────────┐
│  Weather CSV Files   │
│  (NASA POWER)        │
│  - YEAR, MO, DY      │
│  - T2M, RH2M, etc.   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Weather Trainer     │
│  1. Load CSV         │
│  2. Create sequences │
│  3. Feature eng.     │
│  4. Train XGBoost    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐      ┌──────────────────────┐
│  Weather Model       │      │  Solar CSV Files     │
│  (XGBoost + Scaler)  │      │  - date, KWh, PMP    │
└──────────┬───────────┘      └──────────┬───────────┘
           │                             │
           │    ┌────────────────────────┘
           │    │
           ▼    ▼
       ┌──────────────────────┐
       │  Solar Trainer       │
       │  1. Load CSV         │
       │  2. Merge weather    │
       │  3. One-hot encode   │
       │  4. Train XGBoost    │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  Solar Model         │
       │  (XGBoost + Scaler)  │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  Model Bundles       │
       │  (Pickle files)      │
       │  - models/           │
       └──────────────────────┘
```

### Prediction Data Flow

```
API Request (lon, lat, date, pmp)
       │
       ▼
┌──────────────────────┐
│  Date Range Gen      │
│  [2025-01-01,        │
│   2025-01-02, ...]   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Weather Prediction  │
│  For each date:      │
│  - Load sequence     │
│  - Predict T2M, etc. │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Feature Vector      │
│  [T2M, CLOUD,        │
│   radiation, ...]    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Solar Prediction    │
│  - One-hot encode    │
│  - XGBoost predict   │
│  - Scale by PMP      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  kWh Output          │
│  {date, kwh}         │
└──────────┬───────────┘
           │
           ▼
    JSON Response
```

---

## Class Diagrams

### Core Classes

```
┌───────────────────────────────────────────────────────────────┐
│                         FastAPI App                            │
│  + lifespan: asynccontextmanager                              │
│  + app: FastAPI                                               │
│  + include_router(api_router)                                │
└───────────────────────────┬───────────────────────────────────┘
                            │ uses
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                      PredictionService                         │
│  - model_manager: ModelManagerService                         │
│  + predict_day_range(lon, lat, start, end, pmp)              │
│  + predict_month_range(lon, lat, start, end, pmp)            │
│  + predict_year(lon, lat, year, pmp)                         │
└───────────────────────────┬───────────────────────────────────┘
                            │ depends on
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    ModelManagerService                         │
│  - weather_model_bundle: Dict[str, Any]                      │
│  - solar_model_bundle: Dict[str, Any]                        │
│  - _initialized: bool                                         │
│  + initialize(mode, model_dir, ...)                          │
│  + is_ready() -> bool                                         │
│  + get_weather_model() -> Dict                               │
│  + get_solar_model() -> Dict                                 │
└───────────────────────────┬───────────────────────────────────┘
                            │ uses
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                      Training Functions                        │
│  + train_weather_model(csv_path, output_dir, ...)            │
│  + train_solar_model(solar_files, weather_files, ...)        │
│  + save_model_bundle(model, scaler, path)                    │
│  + load_model_bundle(path) -> Dict                           │
└───────────────────────────────────────────────────────────────┘
```

### Pydantic Schemas

```
┌───────────────────────────────────────────────────────────────┐
│                         BaseModel                              │
└───────────────────────────┬───────────────────────────────────┘
                            │ inherits
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌─────────────────┐  ┌─────────────┐  ┌──────────────────┐
│   Location      │  │DayPrediction│  │MonthPrediction   │
│  - lat: float   │  │ - date: str │  │ - date: str      │
│  - lon: float   │  │ - kwh: float│  │ - kwh: float     │
└─────────────────┘  └─────────────┘  └──────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌────────────────────┐  ┌────────────────────┐  ┌──────────────────┐
│DayPredictionResp   │  │MonthPredictionResp │  │YearPredictionResp│
│ - location: Loc    │  │ - location: Loc    │  │ - location: Loc  │
│ - startDate: str   │  │ - startDate: str   │  │ - year: int      │
│ - endDate: str     │  │ - endDate: str     │  │ - totalKwh: float│
│ - predictions: []  │  │ - predictions: []  │  └──────────────────┘
└────────────────────┘  └────────────────────┘
```

---

## Sequence Diagrams

### 1. Application Startup

```
User       FastAPI     ModelManager    WeatherTrainer    SolarTrainer
 │             │             │               │                │
 │  Start App  │             │               │                │
 ├────────────►│             │               │                │
 │             │  lifespan() │               │                │
 │             ├────────────►│               │                │
 │             │             │  initialize() │                │
 │             │             │               │                │
 │             │             │  mode="train_now"?             │
 │             │             ├───────────────┼───────────────►│
 │             │             │               │  train_weather│
 │             │             │               ◄───────────────┤
 │             │             │               │  model.pkl    │
 │             │             │               │                │
 │             │             ├───────────────┼───────────────►│
 │             │             │               │  train_solar  │
 │             │             │               ◄───────────────┤
 │             │             │               │  model.pkl    │
 │             │             │               │                │
 │             │             │  set_ready()  │                │
 │             │             ◄───────────────┤                │
 │             │  ready      │               │                │
 │             ◄─────────────┤               │                │
 │  Server Ready              │               │                │
 ◄─────────────┤             │               │                │
```

### 2. Day Prediction Request

```
Client    API Router    PredictionService    ModelManager    ML Models
  │            │                │                  │             │
  │  GET /day  │                │                  │             │
  ├───────────►│                │                  │             │
  │            │  validate()    │                  │             │
  │            │  predict_day_range()              │             │
  │            ├───────────────►│                  │             │
  │            │                │  is_ready()?     │             │
  │            │                ├─────────────────►│             │
  │            │                │  True            │             │
  │            │                ◄─────────────────┤             │
  │            │                │                  │             │
  │            │                │  get_weather_model()           │
  │            │                ├─────────────────►│             │
  │            │                │  model_bundle    │             │
  │            │                ◄─────────────────┤             │
  │            │                │                  │             │
  │            │                │  predict()       │             │
  │            │                ├─────────────────┼────────────►│
  │            │                │  weather_features│             │
  │            │                ◄─────────────────┼─────────────┤
  │            │                │                  │             │
  │            │                │  get_solar_model()             │
  │            │                ├─────────────────►│             │
  │            │                │  model_bundle    │             │
  │            │                ◄─────────────────┤             │
  │            │                │                  │             │
  │            │                │  predict()       │             │
  │            │                ├─────────────────┼────────────►│
  │            │                │  kwh             │             │
  │            │                ◄─────────────────┼─────────────┤
  │            │                │                  │             │
  │            │  predictions[] │                  │             │
  │            ◄────────────────┤                  │             │
  │            │                │                  │             │
  │  JSON      │                │                  │             │
  ◄───────────┤                │                  │             │
```

### 3. Month Aggregation

```
Client    API Router    PredictionService
  │            │                │
  │  GET /month│                │
  ├───────────►│                │
  │            │  predict_month_range()
  │            ├───────────────►│
  │            │                │
  │            │                │  for each month:
  │            │                │  ┌──────────────┐
  │            │                │  │ start = 1st  │
  │            │                │  │ end = last   │
  │            │                │  └──────────────┘
  │            │                │       │
  │            │                │       ▼
  │            │                │  predict_day_range(start, end)
  │            │                │       │
  │            │                │       ▼
  │            │                │  sum(daily_kwh)
  │            │                │       │
  │            │                │       ▼
  │            │                │  MonthPrediction(date, kwh)
  │            │                │
  │            │  predictions[] │
  │            ◄────────────────┤
  │  JSON      │                │
  ◄───────────┤                │
```

---

## Design Patterns

### 1. Singleton Pattern
**Component**: ModelManagerService
**Purpose**: Single instance manages all ML models
**Implementation**:
```python
_model_manager_instance = None

def get_model_manager() -> ModelManagerService:
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManagerService()
    return _model_manager_instance
```

### 2. Dependency Injection
**Component**: FastAPI routes
**Purpose**: Loose coupling between layers
**Implementation**:
```python
from fastapi import Depends

def get_prediction_service() -> PredictionService:
    return PredictionService()

@router.get("/day")
async def predict_day(
    service: PredictionService = Depends(get_prediction_service)
):
    return await service.predict_day_range(...)
```

### 3. Facade Pattern
**Component**: PredictionService
**Purpose**: Simplify complex ML model interactions
**Implementation**: PredictionService provides simple methods that hide complexity of:
- Model loading
- Feature engineering
- Multi-step predictions
- Aggregation logic

### 4. Strategy Pattern
**Component**: ModelManagerService initialization
**Purpose**: Different startup modes (train vs load)
**Implementation**:
```python
async def initialize(mode: str, ...):
    if mode == "train_now":
        self._train_models(...)
    elif mode == "load_models":
        self._load_models(...)
```

### 5. Builder Pattern
**Component**: Model training functions
**Purpose**: Complex model construction
**Implementation**: Step-by-step model building:
1. Load data
2. Feature engineering
3. Train/test split
4. Model training
5. Bundle creation

---

## Technology Stack

### Backend Framework
- **FastAPI 0.104.1+**: Modern async web framework
- **Uvicorn**: ASGI server
- **Pydantic 2.5+**: Data validation

### Machine Learning
- **XGBoost 2.0.2**: Gradient boosting framework
- **Scikit-learn**: Preprocessing, scalers
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation

### Development Tools
- **Pytest 7.4+**: Testing framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting

### Configuration
- **Python-dotenv**: Environment variables
- **Pydantic Settings**: Type-safe config

### Data Formats
- **CSV**: Training data (NASA POWER format)
- **Pickle**: Model serialization
- **JSON**: API requests/responses

---

## Directory Structure

```
repo/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── api/
│   │   └── v1/
│   │       └── prediction.py  # API endpoints
│   ├── core/
│   │   ├── config.py          # Configuration
│   │   └── exceptions.py      # Custom exceptions
│   ├── models/
│   │   ├── weather_trainer.py # Weather ML model
│   │   ├── solar_trainer.py   # Solar ML model
│   │   └── model_store.py     # Save/load utilities
│   ├── services/
│   │   ├── model_manager.py   # Model lifecycle
│   │   └── prediction.py      # Business logic
│   └── schemas/
│       └── prediction.py      # Pydantic models
├── tests/                   # Test suite
├── doc/                     # Documentation
├── requirements.txt         # Python dependencies
└── .env                     # Environment config
```

---

## API Endpoints Summary

| Endpoint | Method | Purpose | Parameters |
|----------|--------|---------|------------|
| `/v1/predict/day` | GET | Day-range predictions | lon, lat, startDate, endDate, pmp |
| `/v1/predict/month` | GET | Month-range predictions | lon, lat, startDate, endDate, pmp |
| `/v1/predict/year` | GET | Yearly prediction | lon, lat, year, pmp |
| `/health` | GET | Health check | None |

---

## Key Configuration

**Environment Variables** (`.env`):
```bash
# Startup
STARTUP_MODE=train_now|load_models
MODEL_DIR=./test_models

# Weather Data
WEATHER_HIST_FILE=./data/weather.csv
WEATHER_PRED_FILE=./data/weather-pred.csv

# Solar Data
SOLAR_FILES={"site1": "path1.csv", ...}

# Model Parameters
WEATHER_WINDOW_SIZE=30
SOLAR_TEST_MONTHS=6
```

---

## Performance Considerations

### Training
- **Weather Model**: ~2-5 minutes (500 estimators)
- **Solar Model**: ~1-3 minutes (600 estimators)
- **Startup Time**: 3-8 minutes with train_now, <5 seconds with load_models

### Prediction
- **Single Day**: ~50-100ms
- **Month (30 days)**: ~1-2 seconds
- **Year (365 days)**: ~10-15 seconds

### Optimization
- Use `load_models` for production
- Cache weather predictions for repeated locations
- Consider batch predictions for multiple requests

---

## Error Handling

| Error | Status Code | Cause |
|-------|-------------|-------|
| `ValidationError` | 422 | Invalid input (lat/lon, date format) |
| `ValueError` | 400 | Invalid date range, format |
| `RuntimeError` | 503 | Models not ready |
| `FileNotFoundError` | 500 | Missing data files |

---

## References

- [API Contract](./api-contract.md) - Complete API specification
- [Testing Guide](./testing.md) - Comprehensive test documentation
- [Model Manager](./components/model-manager.md) - Model lifecycle details
- [Prediction Engine](./components/prediction-engine.md) - Service layer details

---

**Last Updated**: December 15, 2025  
**Maintainer**: Development Team  
**Version**: 1.0.0
