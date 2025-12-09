# Why Negative R² Occurs and How to Fix It

## What is R²?

R² (coefficient of determination) measures how well your model performs compared to a simple baseline (predicting the mean).

- **R² = 1.0**: Perfect predictions
- **R² = 0.0**: Model performs as well as predicting the mean
- **R² < 0.0**: Model performs WORSE than predicting the mean (negative R²)

## Why Negative R² Happens

### 1. **Severe Underfitting**
- Model hasn't learned meaningful patterns
- Predictions are random or constant
- Worse than just predicting the mean

### 2. **Normalization/Scaling Issues**
- Targets are normalized but predictions are way off
- Inverse transform might amplify errors
- Mismatch between training and evaluation scaling

### 3. **Model Not Trained Enough**
- Early stopping triggered too early
- Learning rate too low or too high
- Model didn't converge

### 4. **Data Issues**
- Features not predictive
- Target leakage or data quality issues
- Sequence length too short to capture patterns

### 5. **Target Constraint Issues**
- High/Low constraint might be causing issues
- Predictions might be getting swapped incorrectly

## How to Fix Negative R²

### Immediate Fixes:

1. **Check if model is actually learning**
   - Monitor training loss - should decrease
   - Check validation loss - should track training loss

2. **Verify normalization**
   - Ensure targets are properly scaled
   - Check inverse transform is correct

3. **Increase training**
   - More epochs
   - Better learning rate
   - Longer sequences

4. **Check predictions**
   - Are predictions reasonable?
   - Are they in the right range?
   - Do they make sense?

5. **Model capacity**
   - Model might be too small
   - Need more parameters
   - Better architecture

