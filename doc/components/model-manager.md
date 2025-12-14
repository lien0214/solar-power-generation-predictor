# Model Manager Component (Updated)

**Last Updated**: December 15, 2025  
**Status**: ✅ Aligned with actual implementation in `app/services/model_manager.py` and `app/models/`

---

## Purpose

The Model Manager Service handles the complete lifecycle of ML models:
- Training weather and solar forecasting models from scratch
- Loading pre-trained models from disk
- Keeping models in memory for fast predictions
- Providing model access to prediction services

---

## Architecture

### Components
- **ModelManagerService**: Main service class for model lifecycle
- **WeatherTrainer**: Trains weather forecasting models
- **SolarTrainer**: Trains solar generation models
- **ModelStore**: Handles model persistence (load/save)

### Model Types
1. **Weather Model**: Predicts future weather variables (temperature, cloud cover, radiation, etc.)
2. **Solar Model**: Predicts solar generation based on weather predictions and panel specifications

---

## API Reference

### ModelManagerService Class

```python
class ModelManagerService:
    """Service for managing ML models (weather and solar forecasting)."""
    
    def __init__(self):
        """Initialize model manager service."""
        self.weather_model_bundle: Optional[Dict[str, Any]] = None
        self.solar_model_bundle: Optional[Dict[str, Any]] = None
        self._initialized: bool = False
```

#### Methods

##### `async initialize()`
```python
async def initialize(
    self,
    mode: str,
    model_dir: str,
    weather_hist_file: str,
    weather_pred_file: str,
    solar_files: Dict[str, str],
    weather_window: int = 30,
    weather_mode: str = "multi",
    solar_test_months: int = 6,
    solar_valid_months: int = 1
) -> None:
    """
    Initialize models based on startup mode.
    
    Args:
        mode: "train_now" or "load_models"
        model_dir: Directory for model storage
        weather_hist_file: Historical weather data path
        weather_pred_file: Predicted weather data path
        solar_files: Dict of solar dataset paths (dataset_name -> csv_path)
        weather_window: Window size for weather model (default: 30 days)
        weather_mode: "single" or "multi" output (default: "multi")
        solar_test_months: Test period months for solar model (default: 6)
        solar_valid_months: Validation period months (default: 1)
        
    Raises:
        Exception: If training or loading fails
    """
```

**Modes**:
- `train_now`: Trains both models from scratch, saves to disk, loads into memory
- `load_models`: Loads pre-trained models from disk into memory

**Behavior**:
- Sets `_initialized` flag to True after successful completion
- Populates `weather_model_bundle` and `solar_model_bundle`
- Logs detailed progress with emoji indicators 🤖📊☀️

##### `is_ready()`
```python
def is_ready(self) -> bool:
    """
    Check if models are loaded and ready for predictions.
    
    Returns:
        True if both weather and solar models are loaded, False otherwise
    """
```

##### `get_weather_model()`
```python
def get_weather_model(self) -> Optional[Dict[str, Any]]:
    """
    Get loaded weather model bundle.
    
    Returns:
        Weather model bundle dict or None if not loaded
        
    Bundle structure:
        {
            "model": XGBoost model object(s),
            "scaler": StandardScaler object,
            "targets": List of weather variable names,
            "window": int (30),
            "mode": "single" or "multi",
            "metadata": Dict with training info
        }
    """
```

##### `get_solar_model()`
```python
def get_solar_model(self) -> Optional[Dict[str, Any]]:
    """
    Get loaded solar model bundle.
    
    Returns:
        Solar model bundle dict or None if not loaded
        
    Bundle structure:
        {
            "model": XGBoost model object,
            "scaler": StandardScaler object,
            "encoders": Dict of label encoders for categorical features,
            "feature_names": List of feature names,
            "datasets": List of dataset names,
            "metadata": Dict with training info
        }
    """
```

---

## Training Functions

### Weather Model Training

```python
def train_weather_model(
    csv_path: str,
    output_dir: str,
    targets: Optional[List[str]] = None,
    win: int = 30,
    train_end: Optional[str] = None,
    val_end: Optional[str] = None,
    mode: str = "single",
) -> Dict[str, Any]:
    """
    Train weather forecasting model.
    
    Args:
        csv_path: Path to historical weather CSV file
        output_dir: Directory to save trained model bundle
        targets: Weather variables to predict (default: all 8)
                 ["T2M", "T2M_MAX", "TS", "CLOUD_AMT_DAY", "CLOUD_OD",
                  "ALLSKY_SFC_SW_DWN", "RH2M", "ALLSKY_SFC_SW_DIRH"]
        win: Window size in days (default: 30)
        train_end: Train period end date (YYYY-MM-DD, default: auto-split)
        val_end: Validation period end date (YYYY-MM-DD, default: auto-split)
        mode: "single" (one model per target) or "multi" (one multi-output model)
        
    Returns:
        Dict with:
            - bundle_path: Path to saved model bundle pickle file
            - model_type: "weather"
            - targets: List of predicted variables
            - window: Window size used
            - mode: Model mode
            
    Raises:
        ValueError: If CSV missing required columns
        Exception: If training fails
    """
```

**Default Targets**: All 8 NASA POWER weather variables
- `T2M`: Temperature at 2 meters
- `T2M_MAX`: Maximum temperature
- `TS`: Earth skin temperature
- `CLOUD_AMT_DAY`: Cloud amount during day
- `CLOUD_OD`: Cloud optical depth
- `ALLSKY_SFC_SW_DWN`: All-sky surface shortwave downward irradiance
- `RH2M`: Relative humidity at 2 meters
- `ALLSKY_SFC_SW_DIRH`: All-sky surface shortwave direct normal irradiance

**Model Architecture**:
- Algorithm: XGBoost Regressor
- Parameters: `n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42`
- Feature Engineering: 30-day rolling windows (30 days * 8 variables = 240 features)
- Scaling: StandardScaler on input features

**Output Files**:
- `weather_model_bundle.pkl`: Complete model bundle (model + scaler + metadata)

---

### Solar Model Training

```python
def train_solar_model(
    solar_files: Dict[str, str],
    weather_hist_file: str,
    weather_pred_file: str,
    output_dir: str,
    test_months: int = 6,
    valid_months: int = 1,
) -> Dict:
    """
    Train merged solar forecasting model.
    
    Args:
        solar_files: Dict mapping dataset names to CSV file paths
                     Example: {"CT安集01": "path/to/ct01.csv", ...}
        weather_hist_file: Path to historical weather CSV
        weather_pred_file: Path to predicted weather CSV 
                          (must have _true and _pred columns)
        output_dir: Directory to save trained model bundle
        test_months: Number of 30-day months for test period (default: 6)
        valid_months: Number of 30-day months for validation (default: 1)
        
    Returns:
        Dict with:
            - bundle_path: Path to saved model bundle pickle file
            - model_type: "solar"
            - datasets: List of dataset names
            - test_months: Test period size
            - valid_months: Validation period size
            
    Raises:
        ValueError: If CSV missing required columns
        Exception: If training fails
    """
```

**Expected CSV Columns**:
- **Solar files**: `date`, `PMP` (panel power), `KWh` (generation)
- **Weather files**: `Date`, weather variable columns

**Model Architecture**:
- Algorithm: XGBoost Regressor
- Parameters: `n_estimators=600, learning_rate=0.1, max_depth=8, random_state=42`
- Feature Engineering:
  - Weather features (8 variables from weather_pred)
  - One-hot encoding for dataset/site identification
  - PMP (panel power) as continuous feature
- Scaling: StandardScaler on continuous features

**Training Strategy**:
- Merged training: All datasets combined into single model
- Site encoding: One-hot encoded to capture site-specific patterns
- Data split: Chronological (train → valid → test)
- Period calculation: test_months * 30 days, valid_months * 30 days

**Output Files**:
- `solar_model_bundle.pkl`: Complete model bundle (model + scaler + encoders + metadata)

---

## Model Loading

```python
def load_weather_model(bundle_path: str) -> Dict[str, Any]:
    """Load weather model bundle from pickle file."""

def load_solar_model(bundle_path: str) -> Dict[str, Any]:
    """Load solar model bundle from pickle file."""
```

Both functions use joblib to load pickle files and return the complete model bundle.

---

## Usage Examples

### Example 1: Train Models on Startup

```python
from app.services.model_manager import ModelManagerService

manager = ModelManagerService()

await manager.initialize(
    mode="train_now",
    model_dir="./models",
    weather_hist_file="data/weather.csv",
    weather_pred_file="data/weather-pred.csv",
    solar_files={
        "CT安集01": "data/ct01.csv",
        "CT安集02": "data/ct02.csv",
        "元晶": "data/yuanjing.csv"
    },
    weather_window=30,
    weather_mode="multi",
    solar_test_months=6,
    solar_valid_months=1
)

# Check if ready
if manager.is_ready():
    print("✅ Models ready for predictions")
    
# Get models
weather_bundle = manager.get_weather_model()
solar_bundle = manager.get_solar_model()
```

### Example 2: Load Pre-trained Models

```python
from app.services.model_manager import ModelManagerService

manager = ModelManagerService()

await manager.initialize(
    mode="load_models",
    model_dir="./models",
    weather_hist_file="",  # Not needed for load mode
    weather_pred_file="",
    solar_files={}
)

if manager.is_ready():
    print("✅ Pre-trained models loaded")
```

### Example 3: Standalone Training (without service)

```python
from app.models import train_weather_model, train_solar_model

# Train weather model
weather_result = train_weather_model(
    csv_path="data/weather.csv",
    output_dir="./models",
    win=30,
    mode="multi"
)
print(f"Weather model saved: {weather_result['bundle_path']}")

# Train solar model
solar_result = train_solar_model(
    solar_files={
        "Site_A": "data/site_a.csv",
        "Site_B": "data/site_b.csv"
    },
    weather_hist_file="data/weather.csv",
    weather_pred_file="data/weather_pred.csv",
    output_dir="./models",
    test_months=6,
    valid_months=1
)
print(f"Solar model saved: {solar_result['bundle_path']}")
```

### Example 4: Check Model Status

```python
manager = get_model_manager()

if not manager.is_ready():
    raise RuntimeError("Models not loaded. Please wait for initialization.")

# Safe to use models
weather = manager.get_weather_model()
solar = manager.get_solar_model()

print(f"Weather targets: {weather['targets']}")
print(f"Solar datasets: {solar['datasets']}")
```

---

## Model Bundle Structure

### Weather Model Bundle
```python
{
    "model": {
        "T2M": XGBRegressor(...),      # If mode="single"
        "T2M_MAX": XGBRegressor(...),
        # ... or ...
        "multi": XGBRegressor(...)     # If mode="multi"
    },
    "scaler": StandardScaler(...),
    "targets": ["T2M", "T2M_MAX", ...],
    "window": 30,
    "mode": "single" or "multi",
    "metadata": {
        "trained_at": "2025-12-15T10:30:00",
        "train_samples": 5000,
        "val_samples": 500,
        "feature_count": 240,
        "version": "1.0"
    }
}
```

### Solar Model Bundle
```python
{
    "model": XGBRegressor(...),
    "scaler": StandardScaler(...),
    "encoders": {
        "dataset": LabelEncoder(...)
    },
    "feature_names": ["T2M", "T2M_MAX", ..., "PMP", "dataset_Site_A", ...],
    "datasets": ["CT安集01", "CT安集02", "元晶"],
    "metadata": {
        "trained_at": "2025-12-15T10:35:00",
        "train_samples": 10000,
        "val_samples": 1000,
        "test_samples": 3000,
        "feature_count": 15,
        "version": "1.0"
    }
}
```

---

## Configuration

Models are configured via `app/core/config.py` Settings:

```python
class Settings(BaseSettings):
    # Model Manager Config
    startup_mode: str = "load_models"  # or "train_now"
    model_dir: str = "./models"
    
    # Weather Model Config
    weather_hist_file: str = "data/weather.csv"
    weather_pred_file: str = "data/weather_pred.csv"
    weather_window: int = 30
    weather_mode: str = "multi"
    
    # Solar Model Config
    solar_files: Dict[str, str] = {
        "CT安集01": "data/ct01.csv",
        "CT安集02": "data/ct02.csv",
        "元晶": "data/yuanjing.csv"
    }
    solar_test_months: int = 6
    solar_valid_months: int = 1
```

---

## Lifecycle

### Startup Sequence (train_now mode)
1. 🤖 ModelManagerService initialized
2. 📊 Weather model training started
   - Load historical weather data
   - Create 30-day rolling window features
   - Train XGBoost model
   - Save bundle to disk
3. ✅ Weather model loaded into memory
4. ☀️ Solar model training started
   - Load solar generation data (all sites)
   - Merge with weather predictions
   - One-hot encode site identifiers
   - Train XGBoost model
   - Save bundle to disk
5. ✅ Solar model loaded into memory
6. ✅ ModelManagerService ready

**Total Time**: ~5-10 minutes (depends on data size)

### Startup Sequence (load_models mode)
1. 🤖 ModelManagerService initialized
2. 📦 Load weather model bundle from disk
3. 📦 Load solar model bundle from disk
4. ✅ ModelManagerService ready

**Total Time**: ~1-2 seconds

---

## Error Handling

### Common Errors

**FileNotFoundError**: Model files not found
```python
# Occurs when: mode="load_models" but files don't exist in model_dir
# Solution: Train models first with mode="train_now"
```

**ValueError**: Missing required CSV columns
```python
# Occurs when: Input CSVs don't have expected columns
# Solution: Check CSV format matches expected schema
```

**RuntimeError**: Models not ready
```python
# Occurs when: Predictions attempted before initialization complete
# Solution: Check manager.is_ready() before using models
```

---

## Performance

### Memory Usage
- Weather model bundle: ~10-50 MB
- Solar model bundle: ~20-100 MB
- Total memory overhead: ~100-200 MB

### Training Time
- Weather model: 2-5 minutes (depends on data size and window)
- Solar model: 3-7 minutes (depends on number of sites and data size)

### Prediction Time
- Single prediction: < 10ms
- Batch predictions (365 days): < 50ms

---

## Testing

### Unit Tests
```python
# tests/services/test_model_manager_service.py
def test_is_ready_before_initialization()
def test_get_weather_model_before_init()
def test_get_solar_model_before_init()
```

### Integration Tests
```python
# tests/integration/test_app_lifecycle.py
@pytest.mark.slow
async def test_full_training_cycle()
```

---

## Future Enhancements

- [ ] Support for incremental training (update existing models)
- [ ] Model versioning and rollback
- [ ] A/B testing support (load multiple model versions)
- [ ] Automatic retraining on schedule
- [ ] Model performance monitoring
- [ ] Export models to ONNX format for deployment
