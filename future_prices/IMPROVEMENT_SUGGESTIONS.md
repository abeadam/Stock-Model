# Model Performance Improvement Suggestions

Based on current configuration and observed issues (negative R², prediction collapse), here are prioritized improvements:

## 🔴 **CRITICAL (Address First)**

### 1. **Increase R² Switch Threshold**
- **Current**: `R2_SWITCH_THRESHOLD = 0.05`
- **Suggested**: `R2_SWITCH_THRESHOLD = 0.1` or `0.15`
- **Why**: Switching too early means model hasn't learned basic patterns. Higher threshold ensures better mean predictions before variance learning.

### 2. **Adjust Learning Rate Schedule**
- **Current**: `LEARNING_RATE = 0.00005`, warmup starts at 10%
- **Options**:
  - **Option A**: Increase base LR to `0.0001` (2x) for faster learning
  - **Option B**: Keep `0.00005` but reduce warmup to 20-30 epochs
- **Why**: Current LR might be too conservative, preventing variance learning.

### 3. **Increase Variance Penalty Weight**
- **Current**: `0.1 * variance_penalty`
- **Suggested**: `0.5 * variance_penalty` or `1.0 * variance_penalty`
- **Why**: Current penalty might be too weak to prevent prediction collapse.

### 4. **Reduce Gradient Clipping**
- **Current**: `max_norm=0.1` (very aggressive)
- **Suggested**: `max_norm=0.5` or `1.0`
- **Why**: Too aggressive clipping might prevent necessary large updates for variance learning.

## 🟡 **IMPORTANT (Address Next)**

### 5. **Architecture Improvements**

#### 5a. **Increase Model Capacity**
- **Current**: `HIDDEN_DIMS = [128, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]`
- **Suggested**: 
  ```python
  HIDDEN_DIMS = [256, 256, 128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4]
  ```
- **Why**: More capacity in early layers helps learn complex patterns. Current architecture might be too narrow.

#### 5b. **Normalization with SNNs (SELU) - NOT RECOMMENDED**
- **Important**: Your model uses **SELU (Self-Normalizing Neural Networks)**
- **Batch Normalization**: Generally **NOT recommended** with SELU because:
  - SELU is designed to be self-normalizing (automatically normalizes activations)
  - BatchNorm is redundant and can interfere with SELU's self-normalizing property
  - Can actually hurt performance by changing the distribution SELU expects
  
- **Alternatives if normalization is needed**:
  - **Layer Normalization**: Better choice than BatchNorm with SELU (normalizes across features, not batch)
  - **Weight Normalization**: Normalizes weights instead of activations
  - **Fix underlying issues first**: If self-normalizing isn't working, check:
    - Weight initialization (should use LeCun normal - already done ✅)
    - Network depth (very deep networks might need help)
    - Learning rate (too high can break self-normalizing property)

### 6. **Loss Function Improvements**

#### 6a. **Increase Tail Weight in Quantile Loss**
- **Current**: `emphasize_extremes=True` with weight `4.0`
- **Suggested**: Increase extreme weight to `9.0` or `10.0`
- **Why**: Better captures tail behavior, prevents underprediction of extremes.

#### 6b. **Adjust Focal Loss Parameters**
- **Current**: `gamma=0.5` (just changed)
- **Monitor**: If still collapsing, try `gamma=0.3` or use simple Huber loss
- **Why**: Lower gamma = less down-weighting, more uniform learning.

### 7. **Training Strategy**

#### 7a. **Reduce Warmup Epochs**
- **Current**: `WARMUP_EPOCHS = 50`
- **Suggested**: `WARMUP_EPOCHS = 20` or `30`
- **Why**: Faster ramp-up means less time at very low LR, allows variance learning sooner.

#### 7b. **Increase Early Stopping Patience**
- **Current**: `EARLY_STOPPING_PATIENCE = 100`
- **Suggested**: `EARLY_STOPPING_PATIENCE = 200` or `300`
- **Why**: Model might need more time to learn variance after mean-only phase.

### 8. **Regularization Adjustments**

#### 8a. **Reduce Dropout During Mean-Only Phase**
- **Current**: `DROPOUT_RATE = 0.15` (applied throughout)
- **Suggested**: Use `0.05` during mean-only, `0.15` after switching
- **Why**: Less regularization during initial learning helps model learn faster.

#### 8b. **Adjust Weight Decay**
- **Current**: Backbone `1e-4`, Output `1e-6`
- **Monitor**: If overfitting, increase; if underfitting, decrease
- **Why**: Balance between generalization and capacity.

## 🟢 **OPTIONAL (Nice to Have)**

### 9. **Data Preprocessing**

#### 9a. **Feature Engineering**
- Add interaction features (e.g., price × volume)
- Add lagged features (previous N periods)
- Add rolling statistics (mean, std over windows)

#### 9b. **Target Transformation**
- Try log transformation if targets are skewed
- Try Box-Cox transformation
- Normalize targets differently (robust scaling)

### 10. **Advanced Techniques**

#### 10a. **Exponential Moving Average (EMA)**
- Maintain EMA of model weights
- Use EMA weights for validation/testing
- **Why**: More stable predictions, often better generalization.

#### 10b. **Learning Rate Finder**
- Implement LR range test
- Find optimal learning rate automatically
- **Why**: Ensures LR is in optimal range for your data/model.

#### 10c. **Cyclical Learning Rates**
- Use `OneCycleLR` or cosine annealing
- **Why**: Can escape local minima, find better solutions.

### 11. **Monitoring & Debugging**

#### 11a. **Enhanced Logging**
- Log prediction distributions (histograms)
- Log gradient norms per layer
- Log learning rate schedule
- **Why**: Better understanding of training dynamics.

#### 11b. **Visualization**
- Plot prediction vs target scatter plots
- Plot residual distributions
- Plot learning curves with multiple metrics
- **Why**: Identify issues early.

## 📊 **Recommended Configuration Changes**

### Quick Wins (Try These First):
```python
R2_SWITCH_THRESHOLD = 0.1  # From 0.05
WARMUP_EPOCHS = 30  # From 50
variance_penalty_weight = 0.5  # From 0.1 (in calculate_loss_predict_both)
grad_clip_max_norm = 0.5  # From 0.1
EARLY_STOPPING_PATIENCE = 200  # From 100
```

### Medium-Term Improvements:
```python
LEARNING_RATE = 0.0001  # From 0.00005 (2x increase)
HIDDEN_DIMS = [256, 256, 128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4]  # Wider early layers
DROPOUT_RATE = 0.10  # From 0.15 (less aggressive)
```

### Advanced (If Still Struggling):
- Add batch normalization
- Implement EMA for weights
- Use learning rate finder
- Try different optimizers (AdamW, RMSprop)

## 🔍 **Diagnostic Steps**

1. **Check Prediction Statistics**:
   - Are predictions collapsing? (std → 0)
   - Are predictions in right range?
   - Are extreme values being predicted?

2. **Check Training Dynamics**:
   - Is loss decreasing?
   - Are gradients healthy? (not too small/large)
   - Is learning rate appropriate?

3. **Check Data Quality**:
   - Are features predictive?
   - Is target distribution reasonable?
   - Are there data leaks?

## 🎯 **Success Metrics**

Aim for:
- **R² > 0.3** on validation (good)
- **R² > 0.5** on validation (excellent)
- **Prediction std ≈ Target std** (variance learned)
- **Stable training** (no divergence)
- **Positive R² from early epochs**

## 📝 **Testing Order**

1. ✅ Increase R² switch threshold to 0.1
2. ✅ Increase variance penalty weight to 0.5
3. ✅ Reduce gradient clipping to 0.5
4. ✅ Reduce warmup epochs to 30
5. Test and monitor
6. If still negative R²: Increase base LR to 0.0001
7. If still collapsing: Increase variance penalty to 1.0
8. If still struggling: Try architecture changes

