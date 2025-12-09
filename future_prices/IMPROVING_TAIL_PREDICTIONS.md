# Strategies to Improve Predictions of Values Farther from Zero

## 1. **Enhanced Loss Weighting** ✅ IMPLEMENTED
- **What**: Weight MSE loss more heavily for samples farther from the mean
- **How**: Use `mse_loss(..., weight_tails=True, tail_weight_max=5.0)`
- **Why**: Forces model to focus on getting extreme values right
- **Status**: Already implemented in `mse_loss()` function

## 2. **Increase Tail Weight in NLL Loss**
- **Current**: `tail_weight = 1.0 + clamp(abs(target_centered) / batch_std, max=2.0)` (max 3x weight)
- **Improvement**: Increase max to 5.0-10.0 for stronger emphasis
- **Location**: `negative_log_likelihood_loss()` function, line ~199
- **Code**: Change `max=2.0` to `max=5.0` or higher

## 3. **Quantile-Based Loss Functions**
- **What**: Use quantile loss (pinball loss) for different percentiles
- **How**: Train separate heads or use quantile regression
- **Benefits**: Better captures tail behavior without assuming Gaussian distribution
- **Example**: 
  ```python
  def quantile_loss(pred, target, quantile=0.95):
      error = target - pred
      return torch.max(quantile * error, (quantile - 1) * error).mean()
  ```

## 4. **Focal Loss for Regression**
- **What**: Adapt focal loss concept to regression (down-weight easy samples)
- **How**: Weight by prediction error magnitude
- **Formula**: `loss = (1 + |pred - target|)^gamma * mse_loss`
- **Benefits**: Automatically focuses on hard-to-predict samples

## 5. **Separate Models/Heads for Different Ranges**
- **What**: Train separate models or heads for small vs large values
- **How**: 
  - Split data by target magnitude (e.g., |target| < threshold vs >= threshold)
  - Train separate models or use a gating network
  - Or use separate heads with magnitude-based routing
- **Benefits**: Each model can specialize in its range

## 6. **Data Augmentation/Oversampling**
- **What**: Oversample or duplicate samples with large absolute values
- **How**: 
  - Weight sampling probability by `1 + |target| / target_std`
  - Or duplicate tail samples in dataset
- **Benefits**: Model sees more extreme examples during training

## 7. **Feature Engineering for Magnitude**
- **What**: Add features that indicate expected magnitude
- **How**:
  - Add `abs(target_lag)` features (historical magnitude)
  - Add volatility indicators
  - Add features from separate magnitude prediction model
- **Benefits**: Model can learn to scale predictions based on context

## 8. **Hierarchical/Multi-Scale Prediction**
- **What**: Predict magnitude and direction separately
- **How**:
  - Head 1: Predict `log(|target| + epsilon)` (magnitude)
  - Head 2: Predict `sign(target)` (direction)
  - Combine: `pred = sign_pred * exp(magnitude_pred)`
- **Benefits**: Magnitude prediction can be more stable

## 9. **Adaptive Variance Prediction**
- **What**: Make variance head predict larger variances for extreme values
- **How**: 
  - Add auxiliary loss: `variance_loss = -log(predicted_var) * (target - mean)^2`
  - Or use heteroscedastic model where variance depends on predicted mean
- **Benefits**: Model learns uncertainty is higher for extreme values

## 10. **Curriculum Learning**
- **What**: Start with easy samples, gradually add harder ones
- **How**:
  - Epoch 1-10: Only train on |target| < 1 std
  - Epoch 11-20: Add |target| < 2 std
  - Epoch 21+: All samples
- **Benefits**: Model learns basics first, then refines on extremes

## 11. **Robust Loss Functions**
- **What**: Use loss functions less sensitive to outliers but still penalize them
- **Options**:
  - **Huber Loss**: Linear for large errors, quadratic for small
  - **Tukey's Biweight**: Down-weights extreme errors but still penalizes
  - **Cauchy Loss**: Robust to outliers
- **Benefits**: Prevents model from being pulled too much by outliers while still learning them

## 12. **Ensemble with Tail-Specialized Models**
- **What**: Train multiple models, some specialized for tails
- **How**:
  - Model 1: Standard training
  - Model 2: Trained only on |target| > 1 std
  - Model 3: Trained with heavy tail weighting
  - Combine predictions (weighted average or stacking)
- **Benefits**: Specialized models can capture tail behavior better

## 13. **Post-Processing Calibration**
- **What**: Calibrate predictions using quantile mapping
- **How**:
  - Map predicted quantiles to actual quantiles from training data
  - Use isotonic regression to calibrate
- **Benefits**: Ensures predicted distribution matches actual distribution

## 14. **Regularization Adjustments**
- **What**: Reduce regularization that pulls predictions toward zero
- **How**:
  - Lower weight decay for output heads
  - Use different regularization for center vs range heads
  - Consider L1 instead of L2 for output heads (less shrinkage)
- **Benefits**: Allows model to predict larger values without penalty

## 15. **Multi-Task Learning with Tail Metrics**
- **What**: Add auxiliary task of predicting tail probability
- **How**:
  - Additional head: `P(|target| > threshold)`
  - Joint training with main prediction task
- **Benefits**: Model learns to identify when extreme values are likely

## Implementation Priority

**Quick Wins (Easy to implement):**
1. ✅ Enhanced MSE loss weighting (DONE)
2. Increase NLL tail weight max from 2.0 to 5.0-10.0
3. Use Huber loss instead of MSE for mean-only phase
4. Reduce weight decay on output heads

**Medium Effort:**
5. Implement focal loss for regression
6. Add magnitude-based features
7. Curriculum learning with progressive sample inclusion

**Advanced:**
8. Separate models/heads for different ranges
9. Quantile regression
10. Ensemble methods

