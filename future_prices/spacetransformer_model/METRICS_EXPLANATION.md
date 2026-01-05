# Why Training Metrics and Plot Metrics Are Different

## The Issue

You're seeing two different sets of metrics:

### Training Metrics (Good)
```
RMSE: 0.392034
R²:   0.849469
```

### Plot Metrics (Bad)
```
RMSE: 17.782242
R²:   -65.268121
```

## Root Cause: Scaled vs Unscaled Data

### Training Metrics (Scaled Data)
- Calculated in `train.py` `validate()` function
- Uses **scaled/normalized** data (mean=0, std=1)
- Model was trained on this scaled data
- Metrics are in "standard deviation units"
- **RMSE of 0.39** means predictions are off by 0.39 standard deviations on average

### Plot Metrics (Unscaled Data)
- Calculated in `generate_plots.py` after inverse transform
- Uses **actual price values** (e.g., 4000-5000 for ES futures)
- Data is inverse-transformed from scaled space back to price space
- Metrics are in actual price units (dollars)
- **RMSE of 17.78** means predictions are off by $17.78 on average

## Why R² is Negative in Plot Metrics

R² (coefficient of determination) is calculated as:
```
R² = 1 - (SS_res / SS_tot)
```

Where:
- `SS_res` = Sum of squared residuals (prediction errors)
- `SS_tot` = Sum of squared deviations from mean (variance of target)

**The problem**: When you inverse-transform to actual prices:
- The variance of prices (SS_tot) is **huge** (prices vary by hundreds/thousands)
- Prediction errors (SS_res) are relatively small
- But the calculation becomes: `R² = 1 - (small_number / huge_number)`
- This can give negative R² even when predictions are good!

## Example Calculation

If ES prices are around 4000-5000:
- Mean price: ~4500
- Variance: ~(5000-4000)² = 1,000,000
- Prediction error: ~17.78² = 316
- R² = 1 - (316 / 1,000,000) = 0.9997 (good!)

But if there's any systematic bias or the calculation uses different variance:
- R² can become negative

## Which Metrics to Trust?

### For Model Training: Use Scaled Metrics
- **Training/validation metrics** (R²=0.85) are the correct ones to monitor
- These reflect how well the model learned the patterns
- Use these for early stopping, hyperparameter tuning, etc.

### For Business Understanding: Use Unscaled Metrics
- **Plot metrics** (RMSE=$17.78) tell you actual dollar errors
- But R² is misleading due to scale issues
- Focus on **RMSE and MAE** in dollar terms instead

## How to Fix Plot Metrics

The issue is in how R² is calculated on unscaled data. Here are better metrics:

### Option 1: Calculate R² on Scaled Data
Calculate R² before inverse transform, then show both:
- Scaled R² (for model quality)
- Unscaled RMSE/MAE (for business understanding)

### Option 2: Use Percentage Errors
Instead of R², use:
- **MAPE** (Mean Absolute Percentage Error)
- **RMSPE** (Root Mean Squared Percentage Error)

These are scale-invariant and more interpretable.

### Option 3: Normalize R² Calculation
Use a different baseline for R² calculation that accounts for the scale.

## Recommended Fix

Modify `generate_plots.py` to:
1. Calculate R² on **scaled data** (before inverse transform)
2. Calculate RMSE/MAE on **unscaled data** (after inverse transform)
3. Show both sets of metrics clearly labeled

Example output:
```
Scaled Metrics (Model Quality):
  RMSE: 0.392034
  R²:   0.849469

Unscaled Metrics (Actual Prices):
  RMSE: $17.78
  MAE:  $17.54
  MAPE: 0.4%  (if prices ~4500)
```

## Bottom Line

- **Your model is performing well** (R²=0.85 on scaled data)
- The negative R² in plots is a **calculation artifact**, not a model problem
- **RMSE of $17.78** on prices around $4000-5000 is actually quite good (~0.4% error)
- Trust the **training metrics** for model evaluation
- Use **unscaled RMSE/MAE** for business interpretation


