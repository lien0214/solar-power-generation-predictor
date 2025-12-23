# Rolling Prediction Analysis & Questions

**Date**: December 15, 2025  
**Status**: ⚠️ Analysis Complete - Implementation Gaps Identified

---

## Question 1: Rolling Prediction Logic for Future Years

### Current Implementation Status: ⚠️ NOT IMPLEMENTED

**Question**: What happens when I enter `/predict/year` for 2030 (5 years in the future)?

**Answer**: Currently, the system **does NOT implement rolling predictions**. It's using **mock/placeholder data**.

---

## Current Code Analysis

### 1. API Layer (✅ Functional)

**File**: `app/services/prediction.py`

The prediction service currently has:

```python
async def predict_year(self, lon: float, lat: float, year: int, pmp: float) -> float:
    """Predict total solar generation for a year."""
    if not self.model_manager.is_ready():
        raise RuntimeError("Models not loaded. Please wait for initialization.")
    
    # TODO: Replace with actual model prediction
    # For now, using mock data
    total_kwh = round(1200.0 + (hash(f"{year}{lat}{lon}") % 500), 2)
    return total_kwh
```

**Issue**: All prediction methods have `TODO` comments:
- ✅ `predict_day_range()` - Mock data with simple hash
- ✅ `predict_month_range()` - Mock data with simple hash
- ✅ `predict_year()` - Mock data with simple hash

**What happens for 2030**:
- The API will accept the request
- It will return a mock value: `1200.0 + (hash("2030{lat}{lon}") % 500)`
- **No actual ML model prediction occurs**
- **No rolling/recursive forecasting**

---

## Expected Rolling Prediction Logic

Based on the reference implementation (`code/xgb-weather-forecaster.py`), here's how rolling prediction **should** work:

### Rolling Prediction Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                  Rolling Weather Prediction                      │
└─────────────────────────────────────────────────────────────────┘

Historical Data (30-day window)
├── Day -30
├── Day -29
├── ...
└── Day -1
      │
      ▼
┌─────────────────┐
│ Weather Model   │  Predicts Day 0 weather
│ (XGBoost)       │  (T2M, CLOUD_AMT, radiation, etc.)
└────────┬────────┘
         │
         ▼
    Predicted Day 0 ──► Add to window, remove Day -30
         │
         ▼
┌─────────────────┐
│ Weather Model   │  Predicts Day 1 weather
│ (XGBoost)       │  (using Day 0 prediction)
└────────┬────────┘
         │
         ▼
    Predicted Day 1 ──► Add to window, remove Day -29
         │
         ▼
    ... (repeat for each day)
         │
         ▼
    Day 365 (full year predicted)
```

### Key Steps for Rolling Prediction

From `xgb-weather-forecaster.py` lines 460-540:

1. **Initialize Window** (30 days of historical data)
   ```python
   first_t = test_idx[0]
   window_feats = feats[first_t - win:first_t, :].copy()
   ```

2. **For Each Future Day**:
   ```python
   for t in test_idx:
       next_date = last_date + pd.Timedelta(days=1)
       
       # a. Build features from window
       X_flat = window_feats.reshape(-1)  # Flatten 30 days of features
       
       # b. Add location + seasonality
       X_row = np.concatenate([
           X_flat,
           np.array([lon, lat, sin_doy, cos_doy])
       ])
       
       # c. Predict weather for next day
       y_pred = model.predict(X_row)  # Predicts T2M, CLOUD_AMT, etc.
       
       # d. Roll window forward (crucial step!)
       new_feat_row = window_feats[-1, :].copy()  # Start with persistence
       
       # e. Update features that are also targets with predictions
       for target in targets:
           if target in features:
               new_feat_row[target_index] = y_pred[target_index]
       
       # f. Slide window: remove oldest day, add newest predicted day
       window_feats = np.vstack([window_feats[1:, :], new_feat_row])
   ```

3. **Use Weather Predictions for Solar**:
   ```python
   # Weather predictions → Solar model
   solar_kwh = solar_model.predict([
       T2M_pred, CLOUD_AMT_pred, radiation_pred, ...
   ])
   ```

---

## Implementation Gap Analysis

### What's Missing in Current Code

#### 1. Weather Rolling Prediction (❌ Not Implemented)

**Location**: `app/services/prediction.py` needs:
```python
async def _predict_weather_rolling(
    self,
    lon: float,
    lat: float,
    start_date: datetime,
    days_ahead: int
) -> pd.DataFrame:
    """
    Rolling weather prediction using 30-day window.
    
    Steps:
    1. Load last 30 days of historical weather
    2. For each future day:
       a. Build feature vector from window
       b. Predict next day weather
       c. Roll window forward
       d. Repeat
    
    Returns:
        DataFrame with predicted weather features for each day
    """
    # NOT IMPLEMENTED YET
    pass
```

#### 2. Solar Prediction from Weather Features (❌ Not Implemented)

```python
async def _predict_solar_from_weather(
    self,
    weather_df: pd.DataFrame,
    pmp: float
) -> List[float]:
    """
    Predict solar generation using weather features.
    
    Args:
        weather_df: Predicted weather (T2M, CLOUD_AMT, radiation, etc.)
        pmp: Panel Maximum Power
    
    Returns:
        List of daily kWh predictions
    """
    # NOT IMPLEMENTED YET
    pass
```

#### 3. Historical Weather Data Access (❌ Not Available)

The system needs:
- Last 30 days of actual weather data as "seed"
- For 2030 prediction, needs 2029-12-02 to 2029-12-31 weather data
- Options:
  - Store historical predictions
  - Use NASA POWER API for historical data
  - Maintain a rolling database

---

## Critical Issues for Future Predictions

### Issue 1: Seed Data Gap

**Problem**: To predict 2030, you need December 2029 weather data (30-day window).

**Solutions**:

**Option A: Store Past Predictions**
```
Prediction History Database
├── 2025 predictions → Seed for 2026
├── 2026 predictions → Seed for 2027
├── ...
└── 2029 predictions → Seed for 2030
```

**Option B: Use External Weather Forecasts**
```
NASA POWER API
└── Fetch historical weather up to latest available date
    └── Use as seed for future predictions
```

**Option C: Hybrid Approach**
```
1. Fetch latest available weather (e.g., up to today)
2. Use model to predict remaining days in rolling fashion
3. Store predictions for future use as seeds
```

### Issue 2: Prediction Accuracy Degradation

**Problem**: Rolling predictions accumulate errors over time.

```
Day 1:   ±2% error
Day 30:  ±5% error
Day 90:  ±10% error
Day 365: ±20-30% error (5-year horizon: ±50%+)
```

**Reality Check**:
- Short-term (1-7 days): Good accuracy (±5%)
- Medium-term (1-3 months): Moderate accuracy (±15%)
- Long-term (1 year): Low accuracy (±30%)
- **Multi-year (5 years): Very uncertain (±50%+)**

**Best Practice**: 
- For 5-year predictions, use **climatological averages** rather than day-by-day rolling
- Consider seasonal patterns and historical trends
- Provide confidence intervals

### Issue 3: Computational Cost

**Problem**: Rolling prediction is expensive.

```
Predict 5 years (1,825 days):
├── 1,825 weather predictions
│   Each requires:
│   ├── Feature vector construction (30 days × N features)
│   ├── XGBoost inference (~10-50ms)
│   └── Window rolling update
│
└── 1,825 solar predictions
    Each requires:
    ├── Feature vector construction
    └── XGBoost inference (~5-20ms)

Total: ~30-130 seconds for 5-year prediction
```

**Optimization Options**:
- Batch predictions (predict multiple days at once)
- Cache intermediate results
- Use GPU acceleration
- Async processing

---

## Recommended Implementation Approach

### Phase 1: Implement Basic Rolling Prediction

```python
class PredictionService:
    async def predict_day_range(
        self,
        lon: float,
        lat: float,
        start_date: str,
        end_date: str,
        pmp: float
    ) -> List[DayPrediction]:
        """Predict with rolling logic."""
        
        # 1. Check if we have seed data
        seed_start = (start_date - 30 days)
        weather_seed = await self._get_weather_seed(lon, lat, seed_start, start_date)
        
        # 2. Rolling weather prediction
        weather_predictions = await self._rolling_weather_predict(
            weather_seed, lon, lat, start_date, end_date
        )
        
        # 3. Solar prediction from weather
        solar_predictions = await self._predict_solar(
            weather_predictions, pmp
        )
        
        return solar_predictions
```

### Phase 2: Handle Seed Data

```python
async def _get_weather_seed(
    self,
    lon: float,
    lat: float,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Get 30-day weather seed.
    
    Priority:
    1. Database (if we have predictions)
    2. NASA POWER API (if within historical range)
    3. Climatological average (fallback)
    """
    # Try database first
    db_weather = await self.db.get_weather(lon, lat, start_date, end_date)
    if db_weather is not None:
        return db_weather
    
    # Try NASA POWER API
    try:
        nasa_weather = await self.fetch_nasa_power(lon, lat, start_date, end_date)
        return nasa_weather
    except Exception:
        pass
    
    # Fallback: climatological average
    return self._get_climatology(lon, lat, start_date, end_date)
```

### Phase 3: Add Confidence Intervals

```python
@dataclass
class DayPrediction:
    date: str
    kwh: float
    kwh_min: float  # Lower bound (90% confidence)
    kwh_max: float  # Upper bound (90% confidence)
    uncertainty: float  # ±% uncertainty
    horizon_days: int  # Days from seed (affects uncertainty)
```

---

## Answers to Your Question

### For `/predict/year?year=2030` Request:

**Current Behavior (as of now)**:
1. ✅ API accepts request
2. ✅ Returns HTTP 200
3. ❌ Returns **mock data** (1200 + random hash)
4. ❌ Does **NOT** use ML models
5. ❌ Does **NOT** perform rolling prediction

**Expected Behavior (after implementation)**:
1. ✅ API accepts request
2. ✅ Check if we have seed data (Dec 2029)
3. ✅ Perform rolling weather prediction (1,095 days from Jan 1, 2030 to Dec 31, 2032 if doing 3 years ahead)
4. ✅ Predict solar for each day
5. ✅ Aggregate to yearly total
6. ✅ Return prediction with confidence interval

**Timeline Reality**:
- ⚠️ 5-year predictions are **highly uncertain**
- 📊 Consider using seasonal averages instead
- 🎯 Best for: 1-90 day horizons
- 🔮 Acceptable for: 1-year horizon (with caveats)
- ❓ Questionable for: 5-year horizon

---

## Recommendations

### For Short-Term Predictions (1-90 days)
✅ Implement full rolling prediction with weather model

### For Medium-Term Predictions (1 year)
✅ Use rolling prediction with uncertainty estimates
✅ Provide confidence intervals
⚠️ Warn users about increasing uncertainty

### For Long-Term Predictions (5 years)
❌ **Don't use day-by-day rolling prediction**
✅ Use climatological models or seasonal averages
✅ Consider trend analysis (e.g., climate change effects)
✅ Provide wide confidence intervals (±50%)

---

## Next Steps

1. **Implement Weather Rolling Prediction** (Priority: HIGH)
   - Add `_rolling_weather_predict()` method
   - Integrate with trained weather model
   - Handle window management

2. **Implement Solar Prediction** (Priority: HIGH)
   - Add `_predict_solar_from_weather()` method
   - Integrate with trained solar model
   - Scale by PMP

3. **Add Seed Data Management** (Priority: MEDIUM)
   - Database for storing predictions
   - NASA POWER API integration
   - Climatology fallback

4. **Add Uncertainty Quantification** (Priority: MEDIUM)
   - Calculate confidence intervals
   - Track prediction horizon
   - Warn users of uncertainty

5. **Optimize Performance** (Priority: LOW)
   - Batch predictions
   - Caching
   - Async processing

---

## References

- **Reference Implementation**: `code/xgb-weather-forecaster.py` (lines 460-540)
- **Current Implementation**: `app/services/prediction.py` (lines 23-157)
- **Architecture Doc**: `doc/architecture.md`

---

**Status**: ⚠️ Critical Gap Identified - Actual prediction logic not implemented  
**Impact**: Current API returns mock data only  
**Priority**: HIGH - Implement rolling prediction for production use
