"""
Futures Price Prediction Model using MVE (Mean-Variance Estimation)
Predicts PctChange_ToMaxHigh_5 using a deep neural network
"""

import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, SequentialLR, LambdaLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Any
from datetime import datetime
# Import plotting utils from parent directory
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from futures_ploting_utils import plot_results


class FuturesDataset(Dataset):
    """Dataset class for futures price data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class MVEModel(nn.Module):
    """
    Mean-Variance Estimation (MVE) Model using Self-Normalizing Neural Networks (SNNs)
    Predicts both mean and variance of the target distribution
    Uses SELU activation for self-normalizing properties
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], dropout_rate: float = 0.0,
                 predict_both: bool = False):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization (0.0 = no dropout)
            predict_both: If True, predict both high and low using center + range approach
        """
        super(MVEModel, self).__init__()
        self.predict_both = predict_both
        
        # Build shared backbone network with SNN (SELU activation)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SELU())  # Self-Normalizing activation
            if dropout_rate > 0.0:
                layers.append(nn.AlphaDropout(dropout_rate))  # AlphaDropout works better with SELU
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        if predict_both:
            # Quantile regression: directly predict high (90th quantile) and low (10th quantile)
            # This eliminates the need for center+range derivation and prevents misalignment issues
            self.quantile_10_head = nn.Linear(prev_dim, 1)  # Low (10th percentile)
            self.quantile_90_head = nn.Linear(prev_dim, 1)  # High (90th percentile)
            # Variance heads for uncertainty quantification
            self.variance_head_high = nn.Linear(prev_dim, 1)
            self.variance_head_low = nn.Linear(prev_dim, 1)
        else:
            # Mean prediction head
            self.mean_head = nn.Linear(prev_dim, 1)
            # Variance prediction head (outputs log variance for numerical stability)
            self.variance_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights with LeCun normal for SELU
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using LeCun normal initialization for SELU activation.
        This is the recommended initialization for Self-Normalizing Neural Networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # LeCun normal initialization: std = 1 / sqrt(fan_in)
                # For SELU, this helps maintain self-normalizing properties
                # This is the recommended initialization from the SELU paper (Klambauer et al., 2017)
                # Formula: std = 1 / sqrt(fan_in) where fan_in is the number of input features
                # This ensures variance of activations stays close to 1, maintaining self-normalizing property
                nn.init.normal_(module.weight, mean=0.0, std=np.sqrt(1.0 / module.in_features))
                if module.bias is not None:
                    # Special initialization for variance heads: start with reasonable log variance
                    # Initialize to log(1.0) = 0, which corresponds to variance of 1.0 (more neutral)
                    if self.predict_both:
                        if module in [self.variance_head_high, self.variance_head_low]:
                            nn.init.constant_(module.bias, 0.0)  # log(1.0) = 0
                        elif module == self.quantile_10_head:
                            # Initialize to zero (will be set to empirical 10th quantile in main)
                            nn.init.zeros_(module.bias)
                        elif module == self.quantile_90_head:
                            # Initialize to zero (will be set to empirical 90th quantile in main)
                            nn.init.zeros_(module.bias)
                        else:
                            nn.init.zeros_(module.bias)
                    else:
                        if module == self.variance_head:
                            nn.init.constant_(module.bias, 0.0)  # log(1.0) = 0
                        else:
                            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass
        Returns:
            If predict_both=False: (mean, log_variance)
            If predict_both=True: ((mean_high, mean_low), (log_variance_high, log_variance_low))
                where high = center + range/2, low = center - range/2
        """
        # Shared backbone
        features = self.backbone(x)
        
        if self.predict_both:
            # Quantile regression: directly predict quantiles
            # No derivation needed - eliminates center misalignment issues
            quantile_10 = self.quantile_10_head(features)  # Low (10th percentile)
            quantile_90 = self.quantile_90_head(features)  # High (90th percentile)
            
            # Direct predictions - no need to derive from center+range
            mean_low = quantile_10
            mean_high = quantile_90
            
            # Variance predictions for uncertainty quantification
            log_variance_high = self.variance_head_high(features)
            log_variance_low = self.variance_head_low(features)
            
            return (mean_high, mean_low), (log_variance_high, log_variance_low)
        else:
            # Separate heads for mean and log variance
            mean = self.mean_head(features)
            log_variance = self.variance_head(features)
            
            # Clamp log variance to reasonable range for numerical stability
            # Wider range to allow model to learn appropriate variances
            # log_variance = torch.clamp(log_variance, min=-15.0, max=15.0)
            
            return mean, log_variance


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float, 
                  emphasize_extremes: bool = True) -> torch.Tensor:
    """
    Quantile loss (pinball loss) for quantile regression
    Optionally emphasizes extreme values to prevent collapse to narrow range
    
    Args:
        pred: Predicted quantile value (batch_size, 1)
        target: True target values (batch_size,)
        quantile: Quantile level (0.0 to 1.0), e.g., 0.1 for 10th percentile, 0.9 for 90th
        emphasize_extremes: If True, add extra weight for large target values
    
    Returns:
        Quantile loss (mean across batch)
    
    Formula:
        If error >= 0 (underprediction): loss = quantile * error
        If error < 0 (overprediction): loss = (quantile - 1) * error
    """
    target = target.unsqueeze(1) if target.dim() == 1 else target  # (batch_size, 1)
    if pred.dim() == 1:
        pred = pred.unsqueeze(1)
    error = target - pred  # Positive when target > pred (underprediction)
    
    # Quantile loss: asymmetric penalty based on quantile level
    base_loss = torch.max(
        quantile * error,           # Penalty for underprediction (when error > 0)
        (quantile - 1) * error       # Penalty for overprediction (when error < 0)
    )
    
    # if emphasize_extremes:
    #     # Add extra weight for large target values to prevent collapse
    #     # Weight increases with target magnitude
    #     target_abs = torch.abs(target)
    #     target_std = torch.std(target) + 1e-6
    #     # Weight is 1.0 for small targets, increases up to 5.0 for large targets (reduced from 10.0)
    #     # Lower max weight prevents over-prediction of extreme values, which causes explosion
    #     # 5.0 is still strong enough to capture extremes without causing divergence
    #     extreme_weight = 1.0 + 4.0 * torch.clamp(target_abs / (target_std + 1e-6), min=0.0, max=1.0)
    #     base_loss = base_loss * extreme_weight
    
    return base_loss.mean()


def negative_log_likelihood_loss(mean_pred: torch.Tensor, log_var_pred: torch.Tensor, 
                                  target: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood loss for Laplace distribution
    Laplace NLL = log(2b) + |x - μ| / b, where b is the scale parameter
    We interpret log_var_pred as log_scale_pred (log of scale parameter b)
    
    Includes tail weighting for robustness. Loss value is NOT clipped - only gradients
    are clipped during optimizer step.
    
    Args:
        mean_pred: Predicted location/mean (batch_size, 1)
        log_var_pred: Predicted log scale (batch_size, 1) - interpreted as log(b) for Laplace
        target: True target values (batch_size,)
    
    Returns:
        Weighted negative log-likelihood loss (not clipped)
    """
    target = target.unsqueeze(1) if target.dim() == 1 else target  # (batch_size, 1)
    
    # Clamp log scale to prevent numerical issues
    # For Laplace, scale should be positive, so log_scale can be any real number
    log_scale_pred = torch.clamp(log_var_pred, min=-15.0, max=15.0)
    
    # Convert log scale to scale parameter b
    # Clamp scale to reasonable range for numerical stability
    scale_pred = torch.exp(log_scale_pred)
    
    # Debug: Check if scale predictions are collapsing to constant
    log_scale_std = torch.std(log_scale_pred).item()
    scale_std = torch.std(scale_pred).item()
    if log_scale_std < 1e-4 or scale_std < 1e-4:
        print(f"\n  WARNING: Scale predictions collapsing to constant!")
        print(f"    log_scale_pred: mean={torch.mean(log_scale_pred).item():.6f}, std={log_scale_std:.6f}")
        print(f"    scale_pred: mean={torch.mean(scale_pred).item():.6f}, std={scale_std:.6f}")
    
    # Compute Laplace NLL: log(2b) + |x - μ| / b
    # where b = scale_pred, μ = mean_pred, x = target
    error = target - mean_pred
    abs_error = torch.abs(error)
    
    # Laplace NLL = log(2 * scale) + |error| / scale
    # = log(2) + log(scale) + |error| / scale
    # = log(2) + log_scale_pred + |error| / scale_pred
    # Create log(2) tensor on same device/dtype for proper broadcasting
    log_2 = torch.log(torch.tensor(2.0, device=log_scale_pred.device, dtype=log_scale_pred.dtype))
    nll = log_2 + log_scale_pred + abs_error / scale_pred
    
    # Check for NaN/Inf (replace with large finite value for safety, but don't clamp the loss itself)
    nll = torch.where(torch.isfinite(nll), nll, torch.tensor(100.0, device=nll.device, dtype=nll.dtype))

    return nll.mean()
    # Weight loss to emphasize samples more than 1 std from the batch mean
    # target_centered = target - target.mean()
    # batch_std = torch.std(target_centered) + 1e-6
    # tail_weight = 1.0 + torch.clamp(torch.abs(target_centered) / batch_std, max=5.0)
    
    # weighted_nll = (nll * tail_weight).mean()
    # return weighted_nll


def focal_loss(mean_pred: torch.Tensor, target: torch.Tensor,
               gamma: float = 3.0, alpha: float = 1.0, base_loss_type: str = 'mse',
               weight_tails: bool = True, tail_weight_max: float = 10.0) -> torch.Tensor:
    """
    Focal loss for regression: automatically focuses on hard-to-predict samples
    Down-weights easy samples (small errors) and emphasizes hard samples (large errors)
    
    Args:
        mean_pred: Predicted mean (batch_size, 1)
        target: True target values (batch_size,)
        gamma: Focusing parameter (higher = more focus on hard samples, typically 0.5-5.0)
        alpha: Smoothing parameter for normalized focal loss (prevents division issues)
        base_loss_type: Base loss to use ('mse' or 'huber')
        weight_tails: If True, weight samples farther from mean more heavily
        tail_weight_max: Maximum weight multiplier for tail samples
    
    Returns:
        Focal loss (optionally weighted)
    """
    target = target.unsqueeze(1)  # (batch_size, 1)
    error = mean_pred - target
    error_squared = error ** 2
    
    # Compute base loss
    if base_loss_type == 'huber':
        # Use Huber as base loss
        # Reduce delta from 1.0 to 0.1 for better sensitivity to small errors
        # This prevents predictions from collapsing to zero by being more sensitive to small deviations
        delta = 0.1
        abs_error = torch.abs(error)
        quadratic = 0.5 * error_squared
        linear = delta * (abs_error - 0.5 * delta)
        base_loss = torch.where(abs_error <= delta, quadratic, linear)
    else:  # 'mse'
        base_loss = error_squared
    
    # Compute modulating factor: (error^2 / (error^2 + alpha))^gamma
    # This is close to 0 for easy samples (small errors) and close to 1 for hard samples (large errors)
    # Normalize error by batch std for scale-invariance
    error_normalized = error_squared / (error_squared + alpha)
    # Clamp error_normalized before power operation to prevent NaN gradients
    error_normalized = torch.clamp(error_normalized, min=0.0, max=1.0)
    modulating_factor = error_normalized ** gamma
    
    # Focal loss: modulating_factor * base_loss
    # Easy samples (small error) get down-weighted, hard samples (large error) get emphasized
    focal = modulating_factor * base_loss
    
    if weight_tails:
        # Additional tail weighting: weight by distance from batch mean
        target_centered = target - target.mean()
        batch_std = torch.std(target_centered) + 1e-6
        tail_weight = 1.0 + (tail_weight_max - 1.0) * torch.clamp(
            torch.abs(target_centered) / batch_std, min=0.0, max=1.0
        )
        weighted_focal = (focal * tail_weight).mean()
        return weighted_focal
    else:
        return focal.mean()


def huber_loss(mean_pred: torch.Tensor, target: torch.Tensor,
               delta: float = 1.0, weight_tails: bool = True, tail_weight_max: float = 5.0) -> torch.Tensor:
    """
    Huber loss (robust to outliers, less sensitive than MSE for large errors)
    Optionally weights tail values more heavily to improve predictions of extreme values
    
    Args:
        mean_pred: Predicted mean (batch_size, 1)
        target: True target values (batch_size,)
        delta: Threshold for switching between quadratic and linear loss
        weight_tails: If True, weight samples farther from mean more heavily
        tail_weight_max: Maximum weight multiplier for tail samples
    
    Returns:
        Huber loss (optionally weighted)
    """
    target = target.unsqueeze(1)  # (batch_size, 1)
    error = mean_pred - target
    abs_error = torch.abs(error)
    
    # Huber loss: quadratic for small errors, linear for large errors
    # For |error| <= delta: 0.5 * error^2
    # For |error| > delta: delta * (|error| - 0.5 * delta)
    quadratic = 0.5 * error ** 2
    linear = delta * (abs_error - 0.5 * delta)
    huber = torch.where(abs_error <= delta, quadratic, linear)
    
    if weight_tails:
        # Weight by distance from batch mean (emphasize tail values)
        target_centered = target - target.mean()
        batch_std = torch.std(target_centered) + 1e-6
        # Weight increases with distance from mean, capped at tail_weight_max
        tail_weight = 1.0 + (tail_weight_max - 1.0) * torch.clamp(
            torch.abs(target_centered) / batch_std, min=0.0, max=1.0
        )
        
        weighted_huber = (huber * tail_weight).mean()
        return weighted_huber
    else:
        return huber.mean()


def mse_loss(mean_pred: torch.Tensor, target: torch.Tensor, 
             weight_tails: bool = False, tail_weight_max: float = 5.0) -> torch.Tensor:
    """
    Mean Squared Error loss (primarily for evaluation/metrics)
    Optionally weights tail values more heavily
    
    Args:
        mean_pred: Predicted mean (batch_size, 1)
        target: True target values (batch_size,)
        weight_tails: If True, weight samples farther from mean more heavily
        tail_weight_max: Maximum weight multiplier for tail samples
    
    Returns:
        MSE loss (optionally weighted)
    """
    target = target.unsqueeze(1)  # (batch_size, 1)
    
    if weight_tails:
        # Calculate squared error
        squared_error = (mean_pred - target) ** 2
        
        # Weight by distance from batch mean (emphasize tail values)
        target_centered = target - target.mean()
        batch_std = torch.std(target_centered) + 1e-6
        # Weight increases with distance from mean, capped at tail_weight_max
        tail_weight = 1.0 + (tail_weight_max - 1.0) * torch.clamp(
            torch.abs(target_centered) / batch_std, min=0.0, max=1.0
        )
        
        weighted_mse = (squared_error * tail_weight).mean()
        return weighted_mse
    else:
        return nn.MSELoss()(mean_pred, target)


def prepare_data(data_path: str, target_col: str = 'PctChange_ToMaxHigh_5',
                 test_size: float = 0.2, random_state: int = 42, predict_both: bool = False,
                 golden_test: bool = False, batch_size: int = 256) -> Tuple:
    """
    Load and prepare data for training
    
    Args:
        data_path: Path to CSV file with features and targets
        target_col: Name of target column
        test_size: Proportion of data for testing
        random_state: Random seed
        predict_both: If True, prepare both high and low targets
        golden_test: If True, only read minimal data needed for testing (batch_size + small validation)
        batch_size: Batch size for golden test mode (only used if golden_test=True)
    
    Returns:
        If predict_both=False: (X_train, X_test, y_train, y_test, scaler, feature_names)
        If predict_both=True: (X_train, X_test, (y_train_high, y_train_low), (y_test_high, y_test_low), scaler, feature_names)
    """
    print(f"Loading data from {data_path}...")
    
    # For golden test, only read the rows we need
    if golden_test:
        # Read only: batch_size for training + small validation set (e.g., 10% of batch_size)
        # Add some buffer for test set
        val_samples = max(1, batch_size // 10)
        test_samples = max(1, batch_size // 10)
        nrows_to_read = batch_size + val_samples + test_samples
        print(f"GOLDEN TEST MODE: Reading only first {nrows_to_read} rows from CSV")
        df = pd.read_csv(data_path, nrows=nrows_to_read)
    else:
        df = pd.read_csv(data_path)
    
    # Select feature columns (exclude target, date columns, and other non-feature columns)
    if predict_both:
        exclude_cols = ['Date', 'DateTime', 'DateTime_ET', 'PctChange_ToMaxHigh_5', 'PctChange_ToMinLow_5']
    else:
        exclude_cols = ['Date', 'DateTime', 'DateTime_ET', target_col, 'PctChange_ToMinLow_5']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features")
    
    # Extract features and target(s)
    X = df[feature_cols].values
    
    # Check for null values in features before handling
    X_array = np.asarray(X)
    nan_count = np.isnan(X_array).sum()
    nan_percentage = (nan_count / X_array.size) * 100 if X_array.size > 0 else 0
    
    if nan_count > 0:
        print(f"Found {nan_count:,} NaN values in features ({nan_percentage:.2f}% of total values)")
    else:
        print("No NaN values found in features")
    
    # Optional: Apply transformation to targets to amplify differences
    # Options: None, 'exp', 'square', 'cube', 'scale', 'log_scale'
    TARGET_TRANSFORM = 'log_scale'  # Set to 'exp', 'square', 'cube', 'scale', or 'log_scale' to enable
    
    def apply_transform(y_tr, y_te):
        """Apply transformation to targets"""
        if TARGET_TRANSFORM == 'exp':
            y_tr = np.exp(y_tr)
            y_te = np.exp(y_te)
            print(f"Applied exponential transform to targets")
        elif TARGET_TRANSFORM == 'square':
            y_tr = np.sign(y_tr) * np.square(y_tr)
            y_te = np.sign(y_te) * np.square(y_te)
            print(f"Applied square transform to targets")
        elif TARGET_TRANSFORM == 'cube':
            # Power of 3 transformation: preserves sign and amplifies differences
            # This helps prevent the mean from collapsing to zero by making extreme values more prominent
            y_tr = np.sign(y_tr) * np.power(np.abs(y_tr), 3)
            y_te = np.sign(y_te) * np.power(np.abs(y_te), 3)
            print(f"Applied cube (power of 3) transform to targets")
        elif TARGET_TRANSFORM == 'scale':
            y_tr = y_tr * 100
            y_te = y_te * 100
            print(f"Applied scale transform (x100) to targets")
        elif TARGET_TRANSFORM == 'log_scale':
            y_tr = np.sign(y_tr) * np.log1p(np.abs(y_tr)) * 100
            y_te = np.sign(y_te) * np.log1p(np.abs(y_te)) * 100
            print(f"Applied log scale transform to targets")
        return y_tr, y_te
    
    # Split data and apply transformations
    if predict_both:
        # Extract and split with both targets
        y_high = df['PctChange_ToMaxHigh_5'].values
        y_low = df['PctChange_ToMinLow_5'].values
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train_high, y_test_high = y_high[:split_idx], y_high[split_idx:]
        y_train_low, y_test_low = y_low[:split_idx], y_low[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Apply transformations
        y_train_high, y_test_high = apply_transform(y_train_high, y_test_high)
        y_train_low, y_test_low = apply_transform(y_train_low, y_test_low)
        print(f"High target statistics - Mean: {np.mean(y_train_high):.4f}, Std: {np.std(y_train_high):.4f}")  # type: ignore
        print(f"Low target statistics - Mean: {np.mean(y_train_low):.4f}, Std: {np.std(y_train_low):.4f}")  # type: ignore
        return X_train, X_test, (y_train_high, y_train_low), (y_test_high, y_test_low), scaler, feature_cols
    else:
        y = df[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Apply transformations
        y_train, y_test = apply_transform(y_train, y_test)
        print(f"Target statistics - Mean: {np.mean(y_train):.4f}, Std: {np.std(y_train):.4f}")
        print(f"Target range - Min: {np.min(y_train):.4f}, Max: {np.max(y_train):.4f}")
        return X_train, X_test, y_train, y_test, scaler, feature_cols


def calculate_loss_predict_both(mean_pred_high: torch.Tensor, mean_pred_low: torch.Tensor,
                                targets_high: torch.Tensor, targets_low: torch.Tensor,
                                log_var_pred_high: Optional[torch.Tensor] = None,
                                log_var_pred_low: Optional[torch.Tensor] = None,
                                mean_only_mode: bool = False) -> torch.Tensor:
    """
    Calculate loss for predict_both=True case (high and low predictions)
    
    Args:
        mean_pred_high: Predicted high values
        mean_pred_low: Predicted low values
        targets_high: Target high values
        targets_low: Target low values
        log_var_pred_high: Predicted log variance for high (optional, for variance phase)
        log_var_pred_low: Predicted log variance for low (optional, for variance phase)
        mean_only_mode: If True, use focal loss; if False, use quantile + NLL
    
    Returns:
        Combined loss tensor
    """
    if mean_only_mode:
        # Use focal loss for both high and low predictions
        # Focal loss automatically focuses on hard-to-predict samples and is robust to outliers
        # Reduce gamma from 1.0 to 0.5 to prevent excessive down-weighting of easy samples
        # Lower gamma allows model to learn variance around mean, preventing prediction collapse
        loss_high = focal_loss(mean_pred_high, targets_high, gamma=0.5, alpha=1.0,
                              base_loss_type='huber', weight_tails=True, tail_weight_max=5.0)
        loss_low = focal_loss(mean_pred_low, targets_low, gamma=0.5, alpha=1.0,
                             base_loss_type='huber', weight_tails=True, tail_weight_max=5.0)
        # Combine high and low losses (equal weight, or could weight high more)
        # REMOVED loss scaling - was causing gradient explosion and training divergence
        # Different loss scales are fine - optimizer will adapt. Scaling amplifies gradients too much.
        # If you need matching scales, adjust learning rate instead of scaling loss
        loss = (loss_high + loss_low) / 2.0
        
        # Add variance encouragement penalty to prevent predictions from collapsing to constant
        # Penalize when prediction std is too low relative to target std
        pred_std_high = torch.std(mean_pred_high)
        pred_std_low = torch.std(mean_pred_low)
        target_std_high = torch.std(targets_high.unsqueeze(1))
        target_std_low = torch.std(targets_low.unsqueeze(1))
        
        # CRITICAL: When std=0, torch.std() has zero gradient, so we need an alternative penalty
        # Use variance of predictions directly (has non-zero gradient even when mean is constant)
        # Variance = mean((x - mean(x))^2) = mean(x^2) - mean(x)^2
        pred_var_high = torch.var(mean_pred_high)  # This has non-zero gradient even when std=0
        pred_var_low = torch.var(mean_pred_low)
        target_var_high = torch.var(targets_high.unsqueeze(1))
        target_var_low = torch.var(targets_low.unsqueeze(1))
        
        # Penalize if prediction variance is less than 1% of target variance
        # Using variance instead of std ensures we have gradients even when predictions collapse
        min_var_high = 0.01 * target_var_high
        min_var_low = 0.01 * target_var_low
        variance_penalty_high = torch.clamp(min_var_high - pred_var_high, min=0.0) ** 2
        variance_penalty_low = torch.clamp(min_var_low - pred_var_low, min=0.0) ** 2
        variance_penalty = (variance_penalty_high + variance_penalty_low) / 2.0
        
        # Increase penalty weight when predictions are collapsing (std very low)
        # If std is less than 1% of target, use stronger penalty (2.0 weight for recovery)
        # Otherwise use moderate penalty (0.5 weight)
        collapse_threshold = 0.01
        is_collapsed_high = pred_std_high < collapse_threshold * target_std_high
        is_collapsed_low = pred_std_low < collapse_threshold * target_std_low
        if is_collapsed_high or is_collapsed_low:
            penalty_weight = 2.0  # Very strong penalty when collapsed to encourage recovery
        else:
            penalty_weight = 0.5  # Moderate penalty otherwise
        loss = loss + penalty_weight * variance_penalty
    else:
        # After mean-only phase: use quantile loss for mean, NLL for variance
        # Keep extreme emphasis to prevent collapse
        loss_high = quantile_loss(mean_pred_high, targets_high, quantile=0.9, emphasize_extremes=True)
        loss_low = quantile_loss(mean_pred_low, targets_low, quantile=0.1, emphasize_extremes=True)
        # NLL for variance estimation
        if log_var_pred_high is not None and log_var_pred_low is not None:
            nll_high = negative_log_likelihood_loss(mean_pred_high, log_var_pred_high, targets_high)
            nll_low = negative_log_likelihood_loss(mean_pred_low, log_var_pred_low, targets_low)
            # Combined: quantile loss for mean (weighted), NLL for variance
            # Divide by 10 to match mean-only loss scale for better stability and comparability
            # This reduces effective LR by 10x, but since full loss is typically 10x larger,
            # this brings gradients to similar scale as mean-only phase
            loss = ((loss_high + loss_low) / 2.0 + 0.5 * (nll_high + nll_low) / 2.0) / 10.0
        else:
            # Fallback if variance not provided
            loss = (loss_high + loss_low) / 2.0 / 10.0
    
    return loss


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler: Optional[Any],
                    epoch: int, history: dict, train_r2: float, val_r2: float, filepath: str):
    """
    Save a checkpoint of the model, optimizer, scheduler, and training history
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: The scheduler state (optional)
        epoch: Current epoch number
        history: Training history dictionary
        train_r2: Current training R²
        val_r2: Current validation R²
        filepath: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'train_r2': train_r2,
        'val_r2': val_r2,
    }
    
    # Save scheduler state if provided
    if scheduler is not None:
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau doesn't have state_dict, save manually
            checkpoint['scheduler_type'] = 'ReduceLROnPlateau'
            checkpoint['scheduler_best'] = scheduler.best if hasattr(scheduler, 'best') else None
            checkpoint['scheduler_num_bad_epochs'] = scheduler.num_bad_epochs if hasattr(scheduler, 'num_bad_epochs') else None
        else:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint['scheduler_type'] = type(scheduler).__name__
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"\n  *** Checkpoint saved to {filepath} ***")
    print(f"      Epoch: {epoch}, Train R²: {train_r2:.6f}, Val R²: {val_r2:.6f}")


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler: Optional[Any],
                    filepath: str, device: str = 'cpu') -> Tuple[int, dict, float, float]:
    """
    Load a checkpoint of the model, optimizer, scheduler, and training history
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        scheduler: The scheduler to load state into (optional)
        filepath: Path to the checkpoint file
        device: Device to load the checkpoint on
    
    Returns:
        Tuple of (epoch, history, train_r2, val_r2)
    """
    # weights_only=False is needed because checkpoint contains numpy scalars in history dict
    # This is safe since we're loading our own checkpoint files
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore scheduler state if provided
    if scheduler is not None and 'scheduler_type' in checkpoint:
        if checkpoint['scheduler_type'] == 'ReduceLROnPlateau':
            # ReduceLROnPlateau doesn't have state_dict, and best/num_bad_epochs are read-only
            # These will be reset naturally as training continues
            pass
        elif 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    history = checkpoint['history']
    train_r2 = checkpoint.get('train_r2', 0.0)
    val_r2 = checkpoint.get('val_r2', 0.0)
    
    print(f"\n  *** Checkpoint loaded from {filepath} ***")
    print(f"      Epoch: {epoch}, Train R²: {train_r2:.6f}, Val R²: {val_r2:.6f}")
    
    return epoch, history, train_r2, val_r2


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int, lr: float = 0.001, device: str = 'cpu',
                optimize_mean_only_steps: int = 5, predict_both: bool = False,
                target_mean: Optional[float] = None, plot_dir: Optional[str] = None,
                early_stopping_patience: int = 500, golden_test: bool = False,
                enable_mean_only_warmup: bool = True, warmup_epochs: int = 10,
                warmup_start_multiplier: float = 0.1, r2_switch_threshold: float = 0.0,
                weight_decay_backbone: float = 1e-3, weight_decay_output: float = 1e-6,
                checkpoint_path: Optional[str] = None,
                load_checkpoint_path: Optional[str] = None) -> dict:
    """
    Train the MVE model
    
    Args:
        model: MVE model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        optimize_mean_only_steps: Number of initial steps to optimize only the mean
        predict_both: If True, predict both high and low
        target_mean: Target mean for initialization
        plot_dir: Directory to save plots
        early_stopping_patience: Patience for early stopping
        golden_test: If True, disable regularization for golden test mode
        enable_mean_only_warmup: If True, use mean-only warm-up phase; if False, train full model from start
        warmup_epochs: Number of epochs for learning rate warmup
        warmup_start_multiplier: Starting LR multiplier for warmup
        r2_switch_threshold: Minimum R² to switch from mean-only to normal training
        weight_decay_backbone: Weight decay for backbone layers
        weight_decay_output: Weight decay for output heads
        checkpoint_path: Path to save checkpoint when switching from mean-only to normal training (optional)
        load_checkpoint_path: Path to load checkpoint from to resume training (optional)
    
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    # Early stopping: stop if validation loss doesn't improve for N epochs
    epochs_without_improvement = 0
    
    # Separate parameter groups: lower weight decay for output heads
    # This prevents output heads from shrinking toward zero, helping predict extreme values
    backbone_params = []
    output_params = []
    
    for name, param in model.named_parameters():
        # Identify output heads by name
        if predict_both:
            if 'quantile' in name or 'variance_head_high' in name or 'variance_head_low' in name:
                output_params.append(param)
            else:
                backbone_params.append(param)
        else:
            if 'mean_head' in name or 'variance_head' in name:
                output_params.append(param)
            else:
                backbone_params.append(param)
    
    # Create optimizer with separate parameter groups
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': lr, 'weight_decay': weight_decay_backbone},
        {'params': output_params, 'lr': lr, 'weight_decay': weight_decay_output}
    ])
    
    print(f"Optimizer setup: {len(backbone_params)} backbone params (wd={weight_decay_backbone}), "
          f"{len(output_params)} output params (wd={weight_decay_output})")
    
    
    # Learning rate warmup: gradually increase LR from small value to target LR
    # This helps larger models train more stably by starting with small updates
    # LambdaLR multiplies base LR by the lambda result, so we need to return a multiplier
    # Start at warmup_start_multiplier% of target LR (increased from 1% to allow faster learning during warmup)
    # This helps model learn variance faster and prevents prediction collapse
    warmup_lambda = lambda epoch: warmup_start_multiplier + (1.0 - warmup_start_multiplier) * (epoch / warmup_epochs)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Track if warmup should continue (will be disabled on divergence)
    # Declare as list to allow modification in nested scopes
    warmup_active = [True]
    
    # Main scheduler: ReduceLROnPlateau to reduce LR when validation loss plateaus
    # This helps when error rate goes down then back up by reducing LR when validation stops improving
    # Less aggressive settings to prevent LR from dropping too fast:
    # - factor=0.8 (20% reduction instead of 50%) - smaller steps
    # - patience=50 (wait longer before reducing) - more patience
    # - threshold=1e-5 (less sensitive to tiny changes) - only reduce on meaningful improvements
    # - min_lr=1e-5 (higher minimum) - don't let LR get too small
    main_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50, 
                                      min_lr=1e-5, threshold=1e-5)
    
    warmup_start_lr = lr * warmup_start_multiplier  # For display purposes
    print(f"Learning rate schedule: Warmup ({warmup_epochs} epochs) + ReduceLROnPlateau")
    print(f"  Warmup: LR increases linearly from {warmup_start_lr:.6f} to {lr:.6f} over {warmup_epochs} epochs")
    print(f"  Main: ReduceLROnPlateau (factor=0.8, patience=50, min_lr=1e-5, threshold=1e-5)")
    print(f"  LR will reduce by 20% when validation loss doesn't improve for 50 epochs")
    
    # Freeze variance head(s) for initial steps (only if warm-up is enabled)
    if enable_mean_only_warmup:
        if predict_both:
            variance_head_high = model.variance_head_high
            variance_head_low = model.variance_head_low
            if isinstance(variance_head_high, nn.Module):
                for param in variance_head_high.parameters():
                    param.requires_grad = False
            if isinstance(variance_head_low, nn.Module):
                for param in variance_head_low.parameters():
                    param.requires_grad = False
        else:
            variance_head = model.variance_head
            if isinstance(variance_head, nn.Module):
                for param in variance_head.parameters():
                    param.requires_grad = False
    
    # Load checkpoint if provided
    start_epoch = 0
    if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
        start_epoch, history, train_r2_loaded, val_r2_loaded = load_checkpoint(
            model, optimizer, main_scheduler, load_checkpoint_path, device
        )
        print(f"Resuming training from epoch {start_epoch + 1}")
        # Skip mean-only mode if we're loading from a checkpoint (assumes mean-only is already done)
        if enable_mean_only_warmup:
            print("  Note: Checkpoint loaded - mean-only warmup will be skipped")
            enable_mean_only_warmup = False
    else:
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'train_r2': [],
            'val_r2': [],
            'learning_rate': []
        }
    
    # Track best model
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    model_update_counter = 0  # Track number of times best model is updated
    
    # Track rollback attempts to prevent infinite loops
    rollback_count = 0
    max_rollbacks = 3  # Stop training if we rollback too many times
    
    print(f"\nTraining model...")
    if enable_mean_only_warmup:
        print(f"Mean-only training: will continue until both Train R² and Val R² are above 0.1")
        print(f"Then switching to normal training (mean + variance)")
    else:
        print(f"Mean-only warm-up DISABLED: training full model (mean + variance) from start")
    
    # Track whether we're still in mean-only mode (based on R², not fixed epochs)
    # If warm-up is disabled, start in full training mode
    mean_only_mode = enable_mean_only_warmup
    variance_heads_unfrozen = not enable_mean_only_warmup  # Start unfrozen if warm-up disabled
    variance_debug_checked = [False]  # Use list to allow modification in nested scope
    
    for epoch in range(start_epoch, num_epochs):
        # Check if we should unfreeze variance heads based on R²
        # This will be checked after we compute R² values in the validation phase
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_count = 0
        train_predictions = []
        train_targets = []
        
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            model_output = model(features)
            if predict_both:
                (mean_pred_high, mean_pred_low), (log_var_pred_high, log_var_pred_low) = model_output
                targets_high = targets[:, 0] if targets.dim() > 1 else targets
                targets_low = targets[:, 1] if targets.dim() > 1 else targets
                # For compatibility, use high for single prediction vars
                mean_pred = mean_pred_high
                log_var_pred = log_var_pred_high
            else:
                mean_pred, log_var_pred = model_output
                targets_high = targets
                targets_low = targets
                # Initialize dual prediction vars for compatibility
                mean_pred_high = mean_pred
                mean_pred_low = mean_pred
                log_var_pred_high = log_var_pred
                log_var_pred_low = log_var_pred
            
            # Debug: Check variance diversity during training (first batch of epoch when variance heads are unfrozen)
            if variance_heads_unfrozen and train_count == 0 and not variance_debug_checked[0]:
                print(f"\n  === DEBUG: First batch after unfreezing variance head ===")
                print(f"  Batch size: {features.shape[0]}")
                # Check input feature diversity first
                with torch.no_grad():
                    input_features_std = torch.std(features, dim=0).mean().item()
                    print(f"  Input features std (mean across features): {input_features_std:.6f}")
                    if input_features_std < 1e-6:
                        print("  CRITICAL: Input features are constant! This will cause constant backbone output.")
                
                log_var_std = torch.std(log_var_pred).item()
                log_var_mean = torch.mean(log_var_pred).item()
                log_var_min = torch.min(log_var_pred).item()
                log_var_max = torch.max(log_var_pred).item()
                print(f"  Variance head output stats: Mean={log_var_mean:.6f}, Std={log_var_std:.6f}, "
                      f"Range=[{log_var_min:.6f}, {log_var_max:.6f}]")
                # Check if features going into variance head are diverse
                with torch.no_grad():
                    # Get features from backbone (same as in forward pass)
                    features_for_var = model.backbone(features)  # type: ignore
                    print(f"  Backbone output shape: {features_for_var.shape}")
                    # Check diversity across samples (dim=0) and features (dim=1)
                    features_std_per_feature = torch.std(features_for_var, dim=0)  # Std across samples for each feature
                    features_std_mean = features_std_per_feature.mean().item()
                    features_std_min = features_std_per_feature.min().item()
                    features_std_max = features_std_per_feature.max().item()
                    print(f"  Backbone features std (input to variance head): "
                          f"Mean={features_std_mean:.6f}, Min={features_std_min:.6f}, Max={features_std_max:.6f}")
                    # Check if features are actually different across samples
                    features_mean_per_sample = features_for_var.mean(dim=1)  # Mean across features for each sample
                    features_mean_std = torch.std(features_mean_per_sample).item()
                    print(f"  Backbone features mean per sample std: {features_mean_std:.6f}")
                    if features_std_mean < 1e-6:
                        print("  WARNING: Backbone features are constant! This will cause constant variance.")
                    if features_mean_std < 1e-6:
                        print("  WARNING: All samples have the same mean features! This will cause constant variance.")
                    # Also check variance head weights
                    var_head_weight = None
                    var_head_bias = None
                    if predict_both:
                        if isinstance(model.variance_head_high, nn.Linear):
                            var_head_weight = model.variance_head_high.weight.data
                            var_head_bias = model.variance_head_high.bias.data
                    else:
                        if isinstance(model.variance_head, nn.Linear):
                            var_head_weight = model.variance_head.weight.data
                            var_head_bias = model.variance_head.bias.data
                    if var_head_weight is not None and var_head_bias is not None:
                        print(f"  Variance head weight shape: {var_head_weight.shape}")
                        print(f"  Variance head weight stats: Mean={var_head_weight.mean().item():.6f}, "
                              f"Std={var_head_weight.std().item():.6f}, Bias={var_head_bias.item():.6f}")
                        # Test: manually compute variance head output to verify it can produce diversity
                        test_output = torch.matmul(features_for_var, var_head_weight.t()) + var_head_bias
                        test_output_std = torch.std(test_output).item()
                        print(f"  Manual variance head output std: {test_output_std:.6f}")
                        if test_output_std < 1e-6:
                            print("  CRITICAL: Variance head cannot produce diverse outputs even with diverse inputs!")
                    print(f"  === END DEBUG ===\n")
                    # Mark that we've checked variance debug
                    variance_debug_checked[0] = True
            
            # Use MSE loss for initial steps, NLL for later steps
            if predict_both:
                # Quantile regression: directly predict high (90th quantile) and low (10th quantile)
                # No need for center/range derivation - eliminates misalignment issues
                
                # Use shared loss calculation function
                loss = calculate_loss_predict_both(
                    mean_pred_high, mean_pred_low, targets_high, targets_low,
                    log_var_pred_high if not mean_only_mode else None,
                    log_var_pred_low if not mean_only_mode else None,
                    mean_only_mode=mean_only_mode
                )
                # For monitoring, use high target (primary)
                mean_pred = mean_pred_high
                targets = targets_high
            else:
                if mean_only_mode:
                    loss = focal_loss(mean_pred, targets, gamma=2.0, alpha=1.0, 
                                     base_loss_type='huber', weight_tails=True, tail_weight_max=5.0)
                else:
                    loss = negative_log_likelihood_loss(mean_pred, log_var_pred, targets)
            
            # Check BEFORE backward()
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ERROR: Loss is NaN/Inf before backward! Loss: {loss.item()}")
                print(f"    This will cause NaN gradients. Skipping batch.")
                continue

            loss.backward()
            
            # Check gradients for NaN/Inf BEFORE clipping to identify problem parameters
            grad_has_nan = False
            grad_has_inf = False
            nan_param_names = []
            inf_param_names = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        grad_has_nan = True
                        nan_param_names.append(name)
                    if torch.isinf(param.grad).any():
                        grad_has_inf = True
                        inf_param_names.append(name)
            
            # Gradient clipping to prevent exploding gradients that can cause NaN
            # Reduced from 0.5 to 0.1 for more aggressive clipping to prevent divergence
            # Tighter clipping prevents large parameter updates that cause predictions to explode
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            except RuntimeError as e:
                # If clip_grad_norm_ fails, gradients are likely NaN/Inf
                print(f"  ERROR: clip_grad_norm_ failed: {e}")
                grad_norm = torch.tensor(float('nan'))
            
            # Check for NaN or inf in gradients or loss
            if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"  WARNING: NaN or Inf detected! Loss: {loss.item()}, Grad norm: {grad_norm.item()}")
                if grad_has_nan:
                    print(f"    Parameters with NaN gradients (first 5): {nan_param_names[:5]}")
                if grad_has_inf:
                    print(f"    Parameters with Inf gradients (first 5): {inf_param_names[:5]}")
                # Zero out NaN/Inf gradients to prevent optimizer from using them
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            param.grad.zero_()
                print(f"  Skipping this batch to prevent training instability")
                continue
            
            optimizer.step()
            # Note: scheduler.step() moved to after validation (ReduceLROnPlateau needs validation loss)
            
            # Calculate MSE for monitoring
            mse = mse_loss(mean_pred, targets).item()
            
            train_loss += loss.item()
            train_mse += mse
            train_count += 1
            
            # Collect predictions and targets for R² calculation
            train_predictions.append(mean_pred.detach().cpu().numpy().flatten())
            train_targets.append(targets.cpu().numpy())
        
        avg_train_loss = train_loss / train_count
        avg_train_mse = train_mse / train_count
        
        # Calculate R² for training set (using last batch as approximation)
        if train_predictions:
            train_predictions_all = np.concatenate(train_predictions)
            train_targets_all = np.concatenate(train_targets)
            ss_res_train = np.sum((train_targets_all - train_predictions_all) ** 2)
            ss_tot_train = np.sum((train_targets_all - np.mean(train_targets_all)) ** 2)
            train_r2 = 1 - (ss_res_train / ss_tot_train) if ss_tot_train > 0 else 0.0
            
            # Debug: Print prediction/target statistics when R² is very negative
            if epoch < 10 or (epoch % 50 == 0 and train_r2 < -0.5):
                pred_mean = float(np.mean(train_predictions_all))
                pred_std = float(np.std(train_predictions_all))
                target_mean = float(np.mean(train_targets_all))
                target_std = float(np.std(train_targets_all))
                print(f"  Debug Epoch {epoch+1}: Pred mean={pred_mean:.6f}, std={pred_std:.6f}, "
                      f"Target mean={target_mean:.6f}, std={target_std:.6f}, R²={train_r2:.6f}")
        else:
            train_r2 = 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_count = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                model_output = model(features)
                if predict_both:
                    (mean_pred_high, mean_pred_low), (log_var_pred_high, log_var_pred_low) = model_output
                    targets_high = targets[:, 0] if targets.dim() > 1 else targets
                    targets_low = targets[:, 1] if targets.dim() > 1 else targets
                    
                    # Use shared loss calculation function (matches training)
                    loss = calculate_loss_predict_both(
                        mean_pred_high, mean_pred_low, targets_high, targets_low,
                        log_var_pred_high if not mean_only_mode else None,
                        log_var_pred_low if not mean_only_mode else None,
                        mean_only_mode=mean_only_mode
                    )
                    mean_pred = mean_pred_high
                    targets = targets_high
                else:
                    mean_pred, log_var_pred = model_output
                    if mean_only_mode:
                        loss = focal_loss(mean_pred, targets, gamma=2.0, alpha=1.0,
                                         base_loss_type='huber', weight_tails=True, tail_weight_max=5.0)
                    else:
                        loss = negative_log_likelihood_loss(mean_pred, log_var_pred, targets)
                
                mse = mse_loss(mean_pred, targets).item()
                
                val_loss += loss.item()
                val_mse += mse
                val_count += 1
                
                # Collect predictions and targets for R² calculation
                val_predictions.append(mean_pred.cpu().numpy().flatten())
                val_targets.append(targets.cpu().numpy())
        
        avg_val_loss = val_loss / val_count
        avg_val_mse = val_mse / val_count
        
        # Calculate R² for validation set
        val_predictions_all = np.concatenate(val_predictions)
        val_targets_all = np.concatenate(val_targets)
        ss_res_val = np.sum((val_targets_all - val_predictions_all) ** 2)
        ss_tot_val = np.sum((val_targets_all - np.mean(val_targets_all)) ** 2)
        val_r2 = 1 - (ss_res_val / ss_tot_val) if ss_tot_val > 0 else 0.0
        
        # Compute target statistics (needed for variance head initialization)
        target_std = float(np.std(val_targets_all))
        
        # Debug: Print prediction/target statistics when R² is very negative
        if epoch < 10 or (epoch % 50 == 0 and val_r2 < -0.5):
            pred_mean = float(np.mean(val_predictions_all))
            pred_std = float(np.std(val_predictions_all))
            target_mean = float(np.mean(val_targets_all))
            print(f"  Debug Val Epoch {epoch+1}: Pred mean={pred_mean:.6f}, std={pred_std:.6f}, "
                  f"Target mean={target_mean:.6f}, std={target_std:.6f}, R²={val_r2:.6f}")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mse'].append(avg_train_mse)
        history['val_mse'].append(avg_val_mse)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        
        # Check if we should switch from mean-only to normal training based on R²
        # Switch when both train and validation R² are above the threshold
        if mean_only_mode and not variance_heads_unfrozen:
            if train_r2 > r2_switch_threshold and val_r2 > r2_switch_threshold:
                mean_only_mode = False
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}: Both Train R² ({train_r2:.6f}) and Val R² ({val_r2:.6f}) are above {r2_switch_threshold:.6f}")
                print(f"Switching to normal training: unfreezing variance head(s)")
                print(f"{'='*60}")
                
                # Compute target variance for better bias initialization
                # Initialize bias to log(target_variance) so variance starts closer to target
                # This prevents variance from starting at exp(0)=1.0 when target variance is much larger
                if predict_both:
                    # For predict_both, compute target_std for high and low separately
                    if len(val_targets) > 0 and val_targets[0].ndim > 1:
                        val_targets_all_2d = np.concatenate(val_targets)
                        target_std_high = float(np.std(val_targets_all_2d[:, 0]))
                        target_std_low = float(np.std(val_targets_all_2d[:, 1]))
                        target_log_var_high = float(np.log(target_std_high ** 2 + 1e-6))
                        target_log_var_low = float(np.log(target_std_low ** 2 + 1e-6))
                    else:
                        # Fallback: use overall target_std
                        target_log_var_high = target_log_var_low = float(np.log(target_std ** 2 + 1e-6))
                    target_log_var = 0.0  # Not used for predict_both, but initialize to avoid unbound error
                else:
                    target_log_var = float(np.log(target_std ** 2 + 1e-6))
                    target_log_var_high = target_log_var_low = 0.0  # Not used for single target, but initialize to avoid unbound error
                
                # Unfreeze variance head(s)
                if predict_both:
                    variance_head_high = model.variance_head_high
                    variance_head_low = model.variance_head_low
                    # Reinitialize variance heads
                    for head_name, variance_head, init_bias in [("high", variance_head_high, target_log_var_high), 
                                                                 ("low", variance_head_low, target_log_var_low)]:
                        if isinstance(variance_head, nn.Module):
                            print(f"  Reinitializing {head_name} variance head parameters...")
                            for name, param in variance_head.named_parameters():
                                old_value = param.data.clone()
                                if 'weight' in name:
                                    nn.init.normal_(param, mean=0.0, std=0.5)
                                    print(f"    {head_name}.{name}: reinitialized")
                                elif 'bias' in name:
                                    # Initialize to log(target_variance) for better starting point
                                    # This means variance starts at exp(log_var) ≈ target_variance
                                    nn.init.constant_(param, init_bias)
                                    print(f"    {head_name}.{name}: reinitialized to {init_bias:.6f} (log(target_var)={init_bias:.6f})")
                                param.requires_grad = True
                else:
                    variance_head = model.variance_head
                    if isinstance(variance_head, nn.Module):
                        # Reinitialize variance head to break out of constant value
                        print("  Reinitializing variance head parameters...")
                        for name, param in variance_head.named_parameters():
                            old_value = param.data.clone()
                            if 'weight' in name:
                                nn.init.normal_(param, mean=0.0, std=0.5)
                                print(f"    {name}: reinitialized from shape {param.shape}, "
                                      f"old mean={old_value.mean().item():.6f}, new mean={param.data.mean().item():.6f}")
                            elif 'bias' in name:
                                # Initialize to log(target_variance) for better starting point
                                # This means variance starts at exp(log_var) ≈ target_variance
                                nn.init.constant_(param, target_log_var)
                                print(f"    {name}: reinitialized from {old_value.item():.6f} to {param.data.item():.6f} (log(target_var)={target_log_var:.6f})")
                            param.requires_grad = True
                        # Verify variance head is unfrozen
                        is_frozen = all(not p.requires_grad for p in variance_head.parameters())
                        if is_frozen:
                            print("ERROR: Variance head is still frozen!")
                        else:
                            print("Variance head successfully unfrozen and reinitialized")
                
                mean_only_mode = False
                variance_heads_unfrozen = True
                # Reset variance debug flag so it prints on first batch after unfreezing
                variance_debug_checked[0] = False
                
                # Save checkpoint when successfully switching from mean-only to normal training
                if checkpoint_path is not None:
                    save_checkpoint(
                        model, optimizer, main_scheduler, epoch, history, train_r2, val_r2, checkpoint_path
                    )
        
        # Save best model based on validation loss
        # Only consider epochs after variance head is unfrozen for best model selection
        if variance_heads_unfrozen and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0  # Reset counter
            model_update_counter += 1  # Increment counter for new best model
            print(f"  *** New best model at epoch {best_epoch} (Val Loss: {avg_val_loss:.6f}) ***")
            
            # Generate plot for this new best model
            if plot_dir is not None:
                # Temporarily load best model state to evaluate
                temp_model_state = model.state_dict()
                model.load_state_dict(best_model_state)
                model.eval()
                
                # Evaluate on validation set to get predictions for plotting
                val_predictions_list = []
                val_targets_list = []
                val_variances_list = []
                val_predictions_high_list = []
                val_predictions_low_list = []
                val_targets_high_list = []
                val_targets_low_list = []
                
                with torch.no_grad():
                    for features, targets in val_loader:
                        features = features.to(device)
                        targets = targets.to(device)
                        
                        model_output = model(features)
                        if predict_both:
                            (mean_pred_high, mean_pred_low), (log_var_pred_high, log_var_pred_low) = model_output
                            targets_high = targets[:, 0] if targets.dim() > 1 else targets
                            targets_low = targets[:, 1] if targets.dim() > 1 else targets
                            # Use high for primary plotting
                            mean_pred = mean_pred_high
                            targets_plot = targets_high
                            log_var_pred = log_var_pred_high
                            # Store both high and low for center+range plotting
                            val_predictions_high_list.append(mean_pred_high.cpu().numpy().flatten())
                            val_predictions_low_list.append(mean_pred_low.cpu().numpy().flatten())
                            val_targets_high_list.append(targets_high.cpu().numpy())
                            val_targets_low_list.append(targets_low.cpu().numpy())
                        else:
                            mean_pred, log_var_pred = model_output
                            targets_plot = targets
                        
                        val_predictions_list.append(mean_pred.cpu().numpy().flatten())
                        val_targets_list.append(targets_plot.cpu().numpy())
                        val_variances_list.append(torch.exp(log_var_pred).cpu().numpy().flatten())
                
                # Concatenate all predictions
                val_predictions = np.concatenate(val_predictions_list)
                val_targets = np.concatenate(val_targets_list)
                val_variances = np.concatenate(val_variances_list)
                
                # Create plot with numbered suffix in snapshots folder
                plot_filename = f'future_model_MVE_SNNs_results_model_{model_update_counter:03d}.png'
                snapshots_dir = os.path.join(plot_dir, 'snapshots')
                os.makedirs(snapshots_dir, exist_ok=True)  # Ensure snapshots folder exists
                plot_path = os.path.join(snapshots_dir, plot_filename)
                
                # Prepare metrics dict for plotting (if predict_both)
                plot_metrics = None
                if predict_both and len(val_predictions_high_list) > 0:
                    predictions_high = np.concatenate(val_predictions_high_list)
                    predictions_low = np.concatenate(val_predictions_low_list)
                    targets_high = np.concatenate(val_targets_high_list)
                    targets_low = np.concatenate(val_targets_low_list)
                    
                    center = (predictions_high + predictions_low) / 2.0
                    range_pred = predictions_high - predictions_low
                    
                    plot_metrics = {
                        'predictions_high': predictions_high,
                        'predictions_low': predictions_low,
                        'targets_high': targets_high,
                        'targets_low': targets_low,
                        'center': center,
                        'range': range_pred
                    }
                
                # Generate plot
                plot_results(history, val_predictions, val_targets, variances=val_variances, 
                           save_path=plot_path, metrics=plot_metrics)
                print(f"  *** Generated plot: {plot_filename} ***")
                
                # Restore model state for continued training
                model.load_state_dict(temp_model_state)
                model.train()
        elif variance_heads_unfrozen:
            epochs_without_improvement += 1
        
        # Check for training divergence (validation loss exploding or R² becoming very negative)
        # More sensitive thresholds to catch divergence earlier:
        # - val_loss > 50 and increased 3x (was 5x) - catch loss explosions sooner
        # - R² < -1.0 (was -10.0) - R² < -1.0 means model is worse than predicting mean, severe divergence
        # - R² getting worse rapidly: current R² < previous R² - 0.2 (rapid degradation)
        val_loss_exploded = (epoch > 0 and avg_val_loss > 50.0 and 
                            avg_val_loss > history['val_loss'][-1] * 3.0)
        r2_diverged = (val_r2 < -1.0)  # R² < -1.0 indicates severe divergence (model worse than mean)
        r2_rapidly_worsening = (epoch > 0 and len(history['val_r2']) > 0 and 
                                val_r2 < history['val_r2'][-1] - 0.2)  # R² dropped by >0.2 in one epoch
        
        if val_loss_exploded or r2_diverged or r2_rapidly_worsening:
            print(f"  WARNING: Training divergence detected!")
            if val_loss_exploded:
                print(f"    Val loss: {avg_val_loss:.2f} (previous: {history['val_loss'][-1]:.2f}, {avg_val_loss/history['val_loss'][-1]:.1f}x increase)")
            if r2_diverged:
                print(f"    Val R²: {val_r2:.2f} (severe divergence, R² < -1.0)")
            if r2_rapidly_worsening:
                prev_r2 = history['val_r2'][-1] if len(history['val_r2']) > 0 else 0.0
                print(f"    Val R²: {val_r2:.2f} (previous: {prev_r2:.2f}, dropped by {prev_r2 - val_r2:.2f})")
            
            # Check if we've rolled back too many times - stop training if recovery isn't working
            rollback_count += 1
            if rollback_count > max_rollbacks:
                print(f"\n  CRITICAL: Training has diverged {rollback_count} times. Rollback recovery is not working.")
                print(f"  Best model is from epoch {best_epoch} (Val Loss: {best_val_loss:.6f})")
                print(f"  Stopping training to prevent further degradation.")
                print(f"  Consider:")
                print(f"    - Lowering base learning rate further")
                print(f"    - Enabling mean-only warmup")
                print(f"    - Checking data quality and preprocessing")
                print(f"    - Reviewing model architecture")
                break
            
            # Stop warmup permanently - don't continue increasing LR after divergence
            if warmup_active[0]:
                warmup_active[0] = False
                print(f"  Warmup stopped permanently to prevent further LR increase")
            
            # Rollback to best model if available (better starting point for recovery)
            # Only rollback if best model is actually good (has positive or near-zero R²)
            if best_model_state is not None:
                # Check if best model is actually good enough to rollback to
                # If best model also has very negative R², rolling back won't help
                # best_epoch is 1-indexed, so we need to check history[best_epoch - 1]
                best_val_r2 = None
                if best_epoch > 0 and len(history['val_r2']) >= best_epoch:
                    best_val_r2 = history['val_r2'][best_epoch - 1]
                
                if best_val_r2 is not None and best_val_r2 < -0.5:
                    print(f"  WARNING: Best model from epoch {best_epoch} also has poor R² ({best_val_r2:.2f})")
                    print(f"  Skipping rollback - would not help. Reducing LR more aggressively instead.")
                    # More aggressive LR reduction if best model is also bad
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.1  # 10x reduction if best model is also bad
                else:
                    print(f"  Rolling back to best model from epoch {best_epoch} (Val Loss: {best_val_loss:.6f})")
                    if best_val_r2 is not None:
                        print(f"    Best model R²: {best_val_r2:.6f}")
                    model.load_state_dict(best_model_state)
            else:
                print(f"  WARNING: No best model available for rollback!")
            
            # Reset optimizer state to clear bad momentum/state
            # This is critical - Adam maintains running averages that contain bad gradient info
            # Clear state for all parameters
            for param in optimizer.param_groups[0]['params']:
                if param in optimizer.state:
                    del optimizer.state[param]
            for param in optimizer.param_groups[1]['params']:
                if param in optimizer.state:
                    del optimizer.state[param]
            print(f"  Optimizer state reset to clear bad momentum")
            
            # Emergency LR reduction (more gradual: 2x instead of 10x for better recovery)
            # But if we've rolled back multiple times, be more aggressive
            lr_reduction = 0.1 if rollback_count >= 2 else 0.5  # More aggressive after 2+ rollbacks
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_reduction
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning rate reduced by {1/lr_reduction:.1f}x to: {current_lr:.6f}")
            
            # Ensure LR doesn't go below minimum (too small = too slow learning)
            min_lr = 1e-5
            if current_lr < min_lr:
                current_lr = min_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = min_lr
                print(f"  Learning rate clamped to minimum: {min_lr:.6f}")
        else:
            # Step scheduler: during warmup, use step() without args; after warmup, use step(avg_val_loss)
            # SequentialLR handles switching between schedulers, but ReduceLROnPlateau needs validation loss
            if epoch < warmup_epochs and warmup_active[0]:
                # Warmup phase: LambdaLR doesn't need validation loss
                # Only step if warmup is still active (disabled on divergence)
                warmup_scheduler.step()
            elif epoch >= warmup_epochs or not warmup_active[0]:
                # Main phase: ReduceLROnPlateau needs validation loss
                # Also use main scheduler if warmup was disabled due to divergence
                main_scheduler.step(avg_val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        # Print after every epoch with current LR
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"Train MSE: {avg_train_mse:.6f}, Val MSE: {avg_val_mse:.6f}, "
              f"Train R²: {train_r2:.6f}, Val R²: {val_r2:.6f}, "
              f"LR: {current_lr:.6f}")
        
        # Early stopping (only check after variance head is unfrozen)
        if variance_heads_unfrozen and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
            print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
            break
    
    # Load best model if found, otherwise use final model
    if best_model_state is not None:
        print(f"\nLoading best model from epoch {best_epoch} (Val Loss: {best_val_loss:.6f})")
        model.load_state_dict(best_model_state)
    else:
        print(f"\nNo best model found (variance head was frozen throughout), using final model")
        best_epoch = num_epochs
        best_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
    
    # Store best model info in a separate metadata dict to avoid type issues
    history['best_model_info'] = {  # type: ignore
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    
    return history


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu', 
                   predict_both: bool = False) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on test set
    
    Returns:
        Tuple of (metrics dictionary, predictions, variances, targets)
        If predict_both=True, predictions contains high predictions and targets contains high targets
        Additional info stored in metrics dict: 'predictions_high', 'predictions_low', 'targets_high', 'targets_low', 'center', 'range'
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_variances = []
    all_targets = []
    all_predictions_high = []
    all_predictions_low = []
    all_targets_high = []
    all_targets_low = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            model_output = model(features)
            if predict_both:
                (mean_pred_high, mean_pred_low), (log_var_pred_high, log_var_pred_low) = model_output
                # Store both high and low
                all_predictions_high.append(mean_pred_high.cpu().numpy())
                all_predictions_low.append(mean_pred_low.cpu().numpy())
                
                # Extract targets
                targets_high = targets[:, 0] if targets.dim() > 1 else targets
                targets_low = targets[:, 1] if targets.dim() > 1 else targets
                all_targets_high.append(targets_high.cpu().numpy())
                all_targets_low.append(targets_low.cpu().numpy())
                
                # Use high for primary evaluation
                mean_pred = mean_pred_high
                log_var_pred = log_var_pred_high
                targets = targets_high
            else:
                mean_pred, log_var_pred = model_output
            
            # Debug: Check if variance predictions are constant
            if len(all_variances) == 0:  # Only check first batch
                log_var_std = torch.std(log_var_pred).item()
                log_var_mean = torch.mean(log_var_pred).item()
                print(f"\nDebug - Log variance stats (first batch): Mean={log_var_mean:.6f}, Std={log_var_std:.6f}")
                if log_var_std < 1e-6:
                    print(f"WARNING: Log variance is nearly constant! All values ≈ {log_var_mean:.6f}")
                    print(f"This suggests the variance head is not learning properly.")
            
            # Convert log variance to variance for metrics and plotting
            var_pred = torch.exp(log_var_pred)
            
            all_predictions.append(mean_pred.cpu().numpy())
            all_variances.append(var_pred.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    predictions = np.concatenate(all_predictions).flatten()
    variances = np.concatenate(all_variances).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    # Store additional info in metrics for plotting
    if predict_both:
        predictions_high = np.concatenate(all_predictions_high).flatten()
        predictions_low = np.concatenate(all_predictions_low).flatten()
        targets_high = np.concatenate(all_targets_high).flatten()
        targets_low = np.concatenate(all_targets_low).flatten()
        center = (predictions_high + predictions_low) / 2.0
        range_pred = predictions_high - predictions_low
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # Calculate R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Store additional info for plotting if predict_both
    if predict_both:
        # Variables are guaranteed to be defined when predict_both is True
        assert 'predictions_high' in locals() and 'predictions_low' in locals(), "predict_both variables should be defined"
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_variance': np.mean(variances),
            'std_variance': np.std(variances),
            'predictions_high': predictions_high,  # type: ignore
            'predictions_low': predictions_low,  # type: ignore
            'targets_high': targets_high,  # type: ignore
            'targets_low': targets_low,  # type: ignore
            'center': center,  # type: ignore
            'range': range_pred  # type: ignore
        }
    else:
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_variance': np.mean(variances),
            'std_variance': np.std(variances)
        }
    
    # Tail metrics (>1 std away from mean target)
    target_mean = np.mean(targets)
    target_std = np.std(targets)
    if target_std > 0:
        tail_mask = np.abs(targets - target_mean) > target_std
        tail_count = int(np.sum(tail_mask))
        if tail_count > 0:
            tail_errors = predictions[tail_mask] - targets[tail_mask]
            tail_rmse = np.sqrt(np.mean(tail_errors ** 2))
            tail_mae = np.mean(np.abs(tail_errors))
        else:
            tail_rmse = np.nan
            tail_mae = np.nan
    else:
        tail_mask = np.zeros_like(targets, dtype=bool)
        tail_count = 0
        tail_rmse = np.nan
        tail_mae = np.nan
    
    if tail_count > 0:
        print(f"\nTail metrics (>1σ from mean): count={tail_count}, "
              f"RMSE={tail_rmse:.6f}, MAE={tail_mae:.6f}")
    else:
        print("\nTail metrics (>1σ from mean): insufficient tail samples to report (count=0)")
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_variance': np.mean(variances),
        'std_variance': np.std(variances),
        'tail_rmse': tail_rmse,
        'tail_mae': tail_mae,
        'tail_count': tail_count,
        'tail_threshold': target_std
    }
    
    # Add center + range info for plotting if predict_both
    if predict_both:
        # Variables are guaranteed to be defined when predict_both is True
        assert 'predictions_high' in locals() and 'predictions_low' in locals(), "predict_both variables should be defined"
        metrics['predictions_high'] = predictions_high  # type: ignore
        metrics['predictions_low'] = predictions_low  # type: ignore
        metrics['targets_high'] = targets_high  # type: ignore
        metrics['targets_low'] = targets_low  # type: ignore
        metrics['center'] = center  # type: ignore
        metrics['range'] = range_pred  # type: ignore
    
    return metrics, predictions, variances, targets


def save_training_results(config: dict, model: nn.Module, history: dict, 
                         metrics: dict, input_dim: int, num_features: int,
                         train_size: int, test_size: int, results_path: str):
    """
    Save training run information and results to a text file
    
    Args:
        config: Dictionary with training configuration
        model: Trained model
        history: Training history dictionary
        metrics: Test set evaluation metrics
        input_dim: Number of input features
        num_features: Total number of features used
        train_size: Number of training samples
        test_size: Number of test samples
        results_path: Path to save the results file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get final training metrics
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0.0
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0.0
    final_train_mse = history['train_mse'][-1] if history['train_mse'] else 0.0
    final_val_mse = history['val_mse'][-1] if history['val_mse'] else 0.0
    
    # Get best metrics
    best_val_loss = min(history['val_loss']) if history['val_loss'] else 0.0
    best_val_mse = min(history['val_mse']) if history['val_mse'] else 0.0
    best_epoch = history['val_loss'].index(best_val_loss) + 1 if history['val_loss'] else 0
    
    with open(results_path, 'a') as f:  # Append mode
        f.write("="*80 + "\n")
        f.write(f"TRAINING RUN - {timestamp}\n")
        f.write("="*80 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Target Column: {config.get('target_col', 'N/A')}\n")
        f.write(f"Batch Size: {config.get('batch_size', 'N/A')}\n")
        f.write(f"Number of Epochs: {config.get('num_epochs', 'N/A')}\n")
        f.write(f"Learning Rate: {config.get('learning_rate', 'N/A')}\n")
        f.write(f"Optimize Mean Only Steps: {config.get('optimize_mean_only_steps', 'N/A')}\n")
        f.write(f"Hidden Dimensions: {config.get('hidden_dims', 'N/A')}\n")
        f.write(f"Device: {config.get('device', 'N/A')}\n")
        f.write("\n")
        
        # Model Architecture
        f.write("MODEL ARCHITECTURE\n")
        f.write("-"*80 + "\n")
        f.write(f"Input Dimension: {input_dim}\n")
        f.write(f"Number of Features: {num_features}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Model Type: MVE (Mean-Variance Estimation) with SNN (SELU)\n")
        f.write("\n")
        
        # Data Information
        f.write("DATA INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Training Samples: {train_size:,}\n")
        f.write(f"Test Samples: {test_size:,}\n")
        f.write(f"Total Samples: {train_size + test_size:,}\n")
        f.write("\n")
        
        # Training History Summary
        f.write("TRAINING HISTORY SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Final Training Loss: {final_train_loss:.6f}\n")
        f.write(f"Final Validation Loss: {final_val_loss:.6f}\n")
        f.write(f"Final Training MSE: {final_train_mse:.6f}\n")
        f.write(f"Final Validation MSE: {final_val_mse:.6f}\n")
        f.write(f"\nBest Validation Loss: {best_val_loss:.6f} (Epoch {best_epoch})\n")
        f.write(f"Best Validation MSE: {best_val_mse:.6f} (Epoch {best_epoch})\n")
        f.write("\n")
        
        # Test Set Metrics
        f.write("TEST SET METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"MSE:  {metrics.get('mse', 0.0):.6f}\n")
        f.write(f"MAE:  {metrics.get('mae', 0.0):.6f}\n")
        f.write(f"RMSE: {metrics.get('rmse', 0.0):.6f}\n")
        f.write(f"R²:   {metrics.get('r2', 0.0):.6f}\n")
        f.write(f"Mean Variance: {metrics.get('mean_variance', 0.0):.6f}\n")
        f.write(f"Std Variance:  {metrics.get('std_variance', 0.0):.6f}\n")
        f.write("\n")
        
        # Training Loss History (last 10 epochs)
        f.write("TRAINING LOSS HISTORY (Last 10 Epochs)\n")
        f.write("-"*80 + "\n")
        if len(history['train_loss']) > 0:
            start_idx = max(0, len(history['train_loss']) - 10)
            for i in range(start_idx, len(history['train_loss'])):
                epoch = i + 1
                f.write(f"Epoch {epoch:3d}: Train Loss={history['train_loss'][i]:.6f}, "
                       f"Val Loss={history['val_loss'][i]:.6f}, "
                       f"Train MSE={history['train_mse'][i]:.6f}, "
                       f"Val MSE={history['val_mse'][i]:.6f}\n")
        f.write("\n")
        
        f.write("="*80 + "\n\n")


def main():
    """Main function to train and evaluate the MVE model"""
    
    # Configuration
    # Data file is in parent directory (future_prices folder)
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'es_with_indicators.csv')
    TARGET_COL = 'PctChange_ToMaxHigh_5'
    PREDICT_BOTH = True  # If True, predict both high and low using center + range approach
    BATCH_SIZE = 128
    NUM_EPOCHS = 10000
    LEARNING_RATE = 0.0005
    OPTIMIZE_MEAN_ONLY_STEPS = 20
    ENABLE_MEAN_ONLY_WARMUP = True  # If False, skip mean-only warm-up and train full model from start
    R2_SWITCH_THRESHOLD = 0.1  # Minimum R² for both train and validation to switch from mean-only to normal training
    # Increased from 0.1 to 0.3 to ensure model is more stable before switching
    # Switching too early (R² barely positive) causes prediction explosion in normal training
    WARMUP_EPOCHS = 50  # Number of epochs for learning rate warmup (gradually increase LR from small to target)
    WARMUP_START_MULTIPLIER = 0.1  # Start at 10% of base LR (increased to prevent prediction collapse)
    EARLY_STOPPING_PATIENCE = 100  # Stop after N epochs without improvement
    HIDDEN_DIMS = [256, 256, 128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4]
    DROPOUT_RATE = 0.05  # Increased from 0.10: stronger dropout to prevent overfitting
    WEIGHT_DECAY_BACKBONE = 1e-3  # Increased from 1e-6: stronger regularization for backbone layers
    WEIGHT_DECAY_OUTPUT = 1e-6  # Increased from 1e-8: moderate regularization for output heads
    
    # Quantile head weight scaling parameters for initialization
    # These control the initial weight scale to balance between preventing explosion and allowing variance
    # Shared by both quantile_10 (low) and quantile_90 (high) heads
    # Current issue: multiplier=1.0 causes pred_std (0.0127) >> target_std (0.0064), leading to R²=-3.23
    # Need smaller multiplier to match prediction variance to target variance
    QUANTILE_WEIGHT_SCALE_MIN = 0.0005  # Minimum weight scale for quantile heads
    QUANTILE_WEIGHT_SCALE_MAX = 0.5   # Maximum weight scale for quantile heads (reduced from 1.0)
    QUANTILE_WEIGHT_SCALE_MULTIPLIER = 0.10  # Multiplier: weight_scale = target_std * this (reduced from 1.0)
    # With target_std=0.0064, weight_scale = 0.0064*0.15 = 0.00096, clamped to [0.0005, 0.5] = 0.00096
    # This should reduce pred_std to better match target_std
    QUANTILE_WEIGHT_SCALE_FALLBACK = 0.05  # Fallback when target_std is 0 (reduced from 0.1)
    
    Golden_Test = False
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Prepare data
    data_result = prepare_data(
        DATA_PATH, target_col=TARGET_COL, predict_both=PREDICT_BOTH,
        golden_test=Golden_Test, batch_size=BATCH_SIZE
    )
    
    # Initialize y_train to satisfy linter (will be assigned in else branch if PREDICT_BOTH is False)
    y_train: Optional[np.ndarray] = None
    
    if PREDICT_BOTH:
        X_train, X_test, (y_train_high, y_train_low), (y_test_high, y_test_low), scaler, feature_names = data_result
        # Create datasets with both targets
        train_dataset = FuturesDataset(X_train, np.column_stack([y_train_high, y_train_low]))
        test_dataset = FuturesDataset(X_test, np.column_stack([y_test_high, y_test_low]))
    else:
        X_train, X_test, y_train, y_test, scaler, feature_names = data_result
        assert y_train is not None  # Type narrowing for linter
        train_dataset = FuturesDataset(X_train, y_train)
        test_dataset = FuturesDataset(X_test, y_test)
    
    # Golden Test: Limit to single batch size for quick testing
    if Golden_Test:
        print(f"\n{'='*60}")
        print("GOLDEN TEST MODE: Using only single batch size of data")
        print(f"{'='*60}")
        # Limit training dataset to BATCH_SIZE samples
        if len(train_dataset) > BATCH_SIZE:
            train_indices = list(range(BATCH_SIZE))
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            print(f"Training dataset limited to {BATCH_SIZE} samples")
        
        # For golden test, use all BATCH_SIZE samples for training, minimal validation
        # Use a small fixed number for validation (e.g., 10% or at least 1 sample)
        # val_size = max(1, min(10, len(train_dataset) // 10))
        # train_size = len(train_dataset) - val_size
        
        # For golden test, create proper train/val split from limited dataset
        val_size = max(1, min(10, len(train_dataset) // 10))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        # val_subset = train_subset
        test_dataset = val_subset  # Use validation as test for golden test
        print(f"Training subset: {len(train_subset)} samples, Validation subset: {len(val_subset)} samples")
    else:
        # Split training data for validation (normal mode)
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # Create data loaders
    # Note: shuffle=False for sequential financial data to preserve temporal order
    # Shuffling can cause data leakage and unrealistic performance metrics
    train_loader_subset = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader_subset = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    input_dim = X_train.shape[1]
    # In Golden Test mode, disable dropout to allow model to memorize data
    # (Regularization prevents perfect memorization even when train/val/test are the same)
    dropout_rate = 0.0 if Golden_Test else DROPOUT_RATE
    if Golden_Test:
        print(f"GOLDEN TEST MODE: Dropout disabled (was {DROPOUT_RATE}) to allow memorization")
    model = MVEModel(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout_rate=dropout_rate, 
                     predict_both=PREDICT_BOTH)
    
    # Check if we'll be loading from checkpoint (skip initialization if so)
    # Checkpoint path is created later, but we check for the default path here
    plot_dir = os.path.dirname(__file__) if __file__ else os.getcwd()
    checkpoint_dir = os.path.join(plot_dir, 'checkpoints')
    default_checkpoint_path = os.path.join(checkpoint_dir, 'mean_only_checkpoint.pth')
    will_load_checkpoint = os.path.exists(default_checkpoint_path)
    
    if will_load_checkpoint:
        print(f"\nCheckpoint found at {default_checkpoint_path} - skipping all initialization")
        print("  Model weights will be loaded from checkpoint instead")
        
        # Set target_center_mean to a default value (will be overwritten when checkpoint loads)
        target_center_mean = 0.0
        model = model.to(device)
    else:
        # Initialize model heads based on empirical statistics from training data
        # This initialization is skipped when loading from checkpoint
        if PREDICT_BOTH:
            # Calculate empirical quantiles from training data
            quantile_10 = np.percentile(y_train_low, 10)  # type: ignore
            quantile_90 = np.percentile(y_train_high, 90)  # type: ignore
            
            # Print target statistics for debugging
            print(f"\nTarget statistics for initialization:")
            print(f"  High targets - Min: {np.min(y_train_high):.6f}, Max: {np.max(y_train_high):.6f}, Mean: {np.mean(y_train_high):.6f}, 90th percentile: {quantile_90:.6f}")  # type: ignore
            print(f"  Low targets - Min: {np.min(y_train_low):.6f}, Max: {np.max(y_train_low):.6f}, Mean: {np.mean(y_train_low):.6f}, 10th percentile: {quantile_10:.6f}")  # type: ignore
            
            # Initialize quantile heads to empirical quantiles with better weight scaling
            # Goal: Start with predictions closer to target distribution for better initial R²
            target_mean_high = float(np.mean(y_train_high))  # type: ignore
            target_mean_low = float(np.mean(y_train_low))  # type: ignore
            target_std_high = float(np.std(y_train_high))  # type: ignore
            target_std_low = float(np.std(y_train_low))  # type: ignore
            
            # Initialize quantile_10_head (low) - start at target mean for better initial R²
            # Model will learn to predict quantile_10 during training via quantile loss
            if isinstance(model.quantile_10_head, nn.Linear) and model.quantile_10_head.bias is not None:
                # Initialize bias to target mean (not quantile) for better initial R²
                # This helps model start with predictions closer to actual target distribution
                nn.init.constant_(model.quantile_10_head.bias, float(target_mean_low))
            with torch.no_grad():
                # Weight scaling: balance between preventing explosion and allowing initial variance
                # Increased from 0.01-0.1 to 0.05-0.3 to allow initial prediction variance
                # Too small (0.01) causes all predictions to collapse to bias (std=0)
                # Too large (>0.5) can cause prediction explosion
                # 0.05-0.3 range allows variance while preventing explosion
                weight_scale = min(QUANTILE_WEIGHT_SCALE_MAX, max(QUANTILE_WEIGHT_SCALE_MIN, target_std_low * QUANTILE_WEIGHT_SCALE_MULTIPLIER)) if target_std_low > 0 else QUANTILE_WEIGHT_SCALE_FALLBACK
                model.quantile_10_head.weight.data *= weight_scale
                print(f"Initialized quantile_10_head: bias={target_mean_low:.6f} (target mean, will learn quantile_10={quantile_10:.6f}), weight_scale={weight_scale:.4f}")
                print(f"  Target mean_low={target_mean_low:.6f}, std_low={target_std_low:.6f}, quantile_10={quantile_10:.6f}")
            
            # Initialize quantile_90_head (high) - start at target mean for better initial R²
            # Model will learn to predict quantile_90 during training via quantile loss
            if isinstance(model.quantile_90_head, nn.Linear) and model.quantile_90_head.bias is not None:
                # Initialize bias to target mean (not quantile_90) for better initial R²
                # This helps model start with predictions closer to actual target distribution
                nn.init.constant_(model.quantile_90_head.bias, float(target_mean_high))
            with torch.no_grad():
                # Weight scaling: balance between preventing explosion and allowing initial variance
                # Increased from 0.01-0.1 to 0.05-0.3 to allow initial prediction variance
                # Too small (0.01) causes all predictions to collapse to bias (std=0)
                # Too large (>0.5) can cause prediction explosion
                # 0.05-0.3 range allows variance while preventing explosion
                weight_scale = min(QUANTILE_WEIGHT_SCALE_MAX, max(QUANTILE_WEIGHT_SCALE_MIN, target_std_high * QUANTILE_WEIGHT_SCALE_MULTIPLIER)) if target_std_high > 0 else QUANTILE_WEIGHT_SCALE_FALLBACK
                model.quantile_90_head.weight.data *= weight_scale
                print(f"Initialized quantile_90_head: bias={target_mean_high:.6f} (target mean, will learn quantile_90={quantile_90:.6f}), weight_scale={weight_scale:.4f}")
                print(f"  Target mean_high={target_mean_high:.6f}, std_high={target_std_high:.6f}, quantile_90={quantile_90:.6f}")
                print(f"  Target mean_low={target_mean_low:.6f}, std_low={target_std_low:.6f}")
                
                # Verify initialization is reasonable
                # Check if quantile_90 is reasonable relative to target distribution
                # For percentage changes, values can be small (e.g., 0.06 = 6%), so use relative check
                quantile_90_to_mean_ratio = quantile_90 / abs(target_mean_high) if target_mean_high != 0 else float('inf')
                if quantile_90_to_mean_ratio < 1.5 and quantile_90 < 0.05:
                    print(f"  WARNING: 90th percentile ({quantile_90:.6f}) is very small relative to mean ({target_mean_high:.6f})!")
                    print(f"    Ratio: {quantile_90_to_mean_ratio:.2f}x - Model may struggle to predict larger values.")
                elif quantile_90 > 0.3:
                    print(f"  INFO: 90th percentile ({quantile_90:.6f}) is large - model should learn to predict values up to this range.")
                else:
                    print(f"  INFO: 90th percentile ({quantile_90:.6f}) is {quantile_90_to_mean_ratio:.2f}x the mean - reasonable range for predictions.")
            
            # Calculate target mean for training (for compatibility)
            target_center_mean = float(np.mean((y_train_high + y_train_low) / 2.0))  # type: ignore
            
            # Move model to device before checking initial predictions
            model = model.to(device)
            
            # Verify initial predictions are reasonable (check on a small sample)
            # If predictions are way out of scale, adaptively rescale weights
            model.eval()
            with torch.no_grad():
                sample_features = torch.FloatTensor(X_train[:min(100, len(X_train))]).to(device)
                sample_output = model(sample_features)
                (mean_pred_high, mean_pred_low), _ = sample_output
                initial_pred_mean_high = float(mean_pred_high.mean().item())
                initial_pred_mean_low = float(mean_pred_low.mean().item())
                initial_pred_std_high = float(mean_pred_high.std().item())
                initial_pred_std_low = float(mean_pred_low.std().item())
                
                print(f"\nInitial prediction check (on {min(100, len(X_train))} samples):")
                print(f"  Pred high mean: {initial_pred_mean_high:.6f} (target: {target_mean_high:.6f}), std: {initial_pred_std_high:.6f}")
                print(f"  Pred low mean: {initial_pred_mean_low:.6f} (target: {target_mean_low:.6f}), std: {initial_pred_std_low:.6f}")
                
                # Adaptive rescaling if predictions are way out of scale
                # Check if prediction mean is more than 2x away from target mean
                high_ratio = abs(initial_pred_mean_high) / (abs(target_mean_high) + 1e-6)
                low_ratio = abs(initial_pred_mean_low) / (abs(target_mean_low) + 1e-6)
                
                if high_ratio > 2.0 or initial_pred_mean_high > target_mean_high + 5.0:
                    # Predictions are too large - rescale weights down
                    rescale_factor = min(0.5, target_mean_high / (abs(initial_pred_mean_high) + 1e-6))
                    print(f"  WARNING: Initial high prediction mean ({initial_pred_mean_high:.6f}) is far from target ({target_mean_high:.6f})!")
                    print(f"    Rescaling quantile_90_head weights by {rescale_factor:.4f}")
                    model.quantile_90_head.weight.data *= rescale_factor
                    # Re-check after rescaling
                    sample_output = model(sample_features)
                    (mean_pred_high, _), _ = sample_output
                    new_pred_mean_high = float(mean_pred_high.mean().item())
                    print(f"    After rescaling: {new_pred_mean_high:.6f}")
                
                if low_ratio > 2.0 or initial_pred_mean_low > target_mean_low + 5.0:
                    # Predictions are too large - rescale weights down
                    rescale_factor = min(0.5, target_mean_low / (abs(initial_pred_mean_low) + 1e-6))
                    print(f"  WARNING: Initial low prediction mean ({initial_pred_mean_low:.6f}) is far from target ({target_mean_low:.6f})!")
                    print(f"    Rescaling quantile_10_head weights by {rescale_factor:.4f}")
                    model.quantile_10_head.weight.data *= rescale_factor
                    # Re-check after rescaling
                    sample_output = model(sample_features)
                    (_, mean_pred_low), _ = sample_output
                    new_pred_mean_low = float(mean_pred_low.mean().item())
                    print(f"    After rescaling: {new_pred_mean_low:.6f}")
                
                model.train()
        else:
            # Calculate target mean for training
            assert y_train is not None, "y_train must be assigned when PREDICT_BOTH is False"
            target_center_mean = float(np.mean(y_train))
            
            # Initialize mean_head to target mean for better starting R²
            if isinstance(model.mean_head, nn.Linear) and model.mean_head.bias is not None:
                nn.init.constant_(model.mean_head.bias, target_center_mean)
                with torch.no_grad():
                    # Scale weights to allow model to produce outputs in target range
                    # Goal: weight * normalized_feature should be able to span ~±target_std
                    # Since normalized features have std≈1, we want weights that can produce ~target_std range
                    # But we start conservatively to avoid overshooting
                    target_std = float(np.std(y_train))
                    # Use target_std directly (not divided by 2) to allow proper variance learning
                    # But cap at reasonable values to avoid extreme weights
                    weight_scale = min(1.0, max(0.1, target_std)) if target_std > 0 else 0.1
                    model.mean_head.weight.data *= weight_scale
                print(f"Initialized mean_head: bias={target_center_mean:.6f}, weight_scale={weight_scale:.4f} (target_std={target_std:.6f})")
            
            # Move model to device before checking initial predictions
            model = model.to(device)
            
            # Verify initial predictions
            model.eval()
            with torch.no_grad():
                sample_features = torch.FloatTensor(X_train[:min(100, len(X_train))]).to(device)
                mean_pred, _ = model(sample_features)
                initial_pred_mean = float(mean_pred.mean().item())
                print(f"\nInitial prediction check (on {min(100, len(X_train))} samples):")
                print(f"  Pred mean: {initial_pred_mean:.6f} (target: {target_center_mean:.6f})")
                if abs(initial_pred_mean - target_center_mean) > 0.1:
                    print(f"  WARNING: Initial prediction mean is far from target! May need better initialization.")
                model.train()
    
       
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    # Get directory for saving plots
    plot_dir = os.path.dirname(__file__) if __file__ else os.getcwd()
    
    # Create/clean snapshots folder at start of training
    snapshots_dir = os.path.join(plot_dir, 'snapshots')
    if os.path.exists(snapshots_dir):
        shutil.rmtree(snapshots_dir)
        print(f"Deleted existing snapshots folder: {snapshots_dir}")
    os.makedirs(snapshots_dir, exist_ok=True)
    print(f"Created snapshots folder: {snapshots_dir}")
    
    # Checkpoint paths for mean-only training checkpoint
    checkpoint_dir = os.path.join(plot_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'mean_only_checkpoint.pth')
    load_checkpoint_path = checkpoint_path  # Set to checkpoint_path if you want to resume from mean-only checkpoint
    
    history = train_model(
        model=model,
        train_loader=train_loader_subset,
        val_loader=val_loader_subset,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        optimize_mean_only_steps=OPTIMIZE_MEAN_ONLY_STEPS,
        predict_both=PREDICT_BOTH,
        target_mean=target_center_mean,
        plot_dir=plot_dir,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        golden_test=Golden_Test,
        enable_mean_only_warmup=ENABLE_MEAN_ONLY_WARMUP,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_start_multiplier=WARMUP_START_MULTIPLIER,
        r2_switch_threshold=R2_SWITCH_THRESHOLD,
        weight_decay_backbone=WEIGHT_DECAY_BACKBONE,
        weight_decay_output=WEIGHT_DECAY_OUTPUT,
        checkpoint_path=checkpoint_path,
        load_checkpoint_path=load_checkpoint_path
    )
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    metrics, predictions, variances, targets = evaluate_model(model, test_loader, device, predict_both=PREDICT_BOTH)
    
    print(f"\nTest Set Metrics:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    print(f"  Mean Variance: {metrics['mean_variance']:.6f}")
    print(f"  Std Variance:  {metrics['std_variance']:.6f}")
    
    # Plot results
    plot_path = os.path.join(os.path.dirname(__file__), 'future_model_MVE_SNNs_results.png')
    plot_results(history, predictions, targets, variances=variances, save_path=plot_path, metrics=metrics)
    
    # Save model (this is the best model, loaded at end of training)
    model_path = os.path.join(os.path.dirname(__file__), 'futures_model.pt')
    best_model_info = history.get('best_model_info', {})
    best_epoch = best_model_info.get('best_epoch', NUM_EPOCHS)
    best_val_loss = best_model_info.get('best_val_loss', history['val_loss'][-1] if history['val_loss'] else 0.0)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_names': feature_names,
        'input_dim': input_dim,
        'hidden_dims': HIDDEN_DIMS,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }, model_path)
    print(f"\nModel saved to {model_path} (Best model from epoch {best_epoch}, Val Loss: {best_val_loss:.6f})")
    
    # Save training results
    results_path = os.path.join(os.path.dirname(__file__), 'training_results.txt')
    config = {
        'target_col': TARGET_COL,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'optimize_mean_only_steps': OPTIMIZE_MEAN_ONLY_STEPS,
        'hidden_dims': HIDDEN_DIMS,
        'device': device
    }
    save_training_results(
        config=config,
        model=model,
        history=history,
        metrics=metrics,
        input_dim=input_dim,
        num_features=len(feature_names),
        train_size=len(X_train),
        test_size=len(X_test),
        results_path=results_path
    )
    print(f"Training results saved to {results_path}")


if __name__ == "__main__":
    main()

