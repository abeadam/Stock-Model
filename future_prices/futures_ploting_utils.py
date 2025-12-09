"""
Plotting utilities for futures price prediction model
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_results(history: dict, predictions: np.ndarray, targets: np.ndarray,
                 variances: Optional[np.ndarray] = None, save_path: Optional[str] = None,
                 metrics: Optional[dict] = None):
    """
    Plot training history and predictions
    
    Args:
        history: Training history dictionary
        predictions: Predicted values (or high predictions if predict_both)
        targets: Actual target values (or high targets if predict_both)
        variances: Predicted variances (optional)
        save_path: Path to save the plot
        metrics: Metrics dictionary (optional, may contain center+range info if predict_both)
    """
    predict_both = metrics is not None and 'center' in metrics
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot 1: Training history
    ax1 = axes[0, 0]
    ax1.plot(history['train_loss'], label='Train Loss', alpha=0.7)
    ax1.plot(history['val_loss'], label='Val Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MSE history
    ax2 = axes[0, 1]
    ax2.plot(history['train_mse'], label='Train MSE', alpha=0.7)
    ax2.plot(history['val_mse'], label='Val MSE', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Training History - MSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Predictions vs Targets
    ax3 = axes[1, 0]
    ax3.scatter(targets, predictions, alpha=0.5, s=10)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    ax3.set_xlabel('True Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title('Predictions vs Targets')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    ax4 = axes[1, 1]
    residuals = targets - predictions
    ax4.scatter(predictions, residuals, alpha=0.5, s=10)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Predicted Values')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residual Plot')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Last 100 Predicted vs Actual (Time Series) with Center and Confidence Intervals
    ax5 = axes[2, 0]
    n_points = min(100, len(predictions))
    last_indices = np.arange(len(predictions) - n_points, len(predictions))
    
    if predict_both and metrics is not None:
        # Use center + range structure
        predictions_high = metrics['predictions_high']
        predictions_low = metrics['predictions_low']
        targets_high = metrics['targets_high']
        targets_low = metrics['targets_low']
        center = metrics['center']
        range_pred = metrics['range']
        
        last_high = predictions_high[-n_points:]
        last_low = predictions_low[-n_points:]
        last_center = center[-n_points:]
        last_range = range_pred[-n_points:]
        last_targets_high = targets_high[-n_points:]
        last_targets_low = targets_low[-n_points:]
        
        # Plot range as shaded area (high to low)
        ax5.fill_between(last_indices, last_low, last_high,
                        alpha=0.2, color='purple', label='Predicted Range (High-Low)', zorder=1)
        
        # Plot confidence intervals around center if variances available
        if variances is not None and len(variances) >= n_points:
            last_variances = variances[-n_points:]
            std_dev = np.sqrt(last_variances)
            
            # Plot 1σ confidence interval around center
            ax5.fill_between(last_indices,
                            last_center - std_dev,
                            last_center + std_dev,
                            alpha=0.15, color='green', label='Center ±1σ (68.3%)', zorder=2)
        
        # Plot high and low prediction lines
        ax5.plot(last_indices, last_high, label='Predicted High', 
                color='red', linewidth=2, linestyle='--', alpha=0.8, zorder=4)
        ax5.plot(last_indices, last_low, label='Predicted Low', 
                color='blue', linewidth=2, linestyle='--', alpha=0.8, zorder=4)
        
        # Plot center line (most prominent)
        ax5.plot(last_indices, last_center, label='Center (Predicted)', 
                color='orange', linewidth=2.5, marker='o', markersize=4, alpha=0.9, zorder=5)
        
        # Plot actual high and low
        ax5.plot(last_indices, last_targets_high, label='Actual High', 
                color='darkred', linewidth=2, marker='s', markersize=3, alpha=0.8, zorder=3)
        ax5.plot(last_indices, last_targets_low, label='Actual Low', 
                color='darkblue', linewidth=2, marker='s', markersize=3, alpha=0.8, zorder=3)
        
        ax5.set_title(f'Last {n_points} Predictions: Center + Range Structure')
    else:
        # Original single prediction plot
        last_predictions = predictions[-n_points:]
        last_targets = targets[-n_points:]
        
        # Plot confidence intervals first (so they appear behind the center line)
        if variances is not None and len(variances) >= n_points:
            last_variances = variances[-n_points:]
            std_dev = np.sqrt(last_variances)  # Convert variance to standard deviation
            
            # Plot 3σ confidence interval (outermost, lightest)
            ax5.fill_between(last_indices,
                            last_predictions - 3 * std_dev,
                            last_predictions + 3 * std_dev,
                            alpha=0.1, color='blue', label='±3σ (99.7%)')
            
            # Plot 2σ confidence interval (middle)
            ax5.fill_between(last_indices,
                            last_predictions - 2 * std_dev,
                            last_predictions + 2 * std_dev,
                            alpha=0.15, color='cyan', label='±2σ (95.4%)')
            
            # Plot 1σ confidence interval (innermost, most visible)
            ax5.fill_between(last_indices,
                            last_predictions - std_dev,
                            last_predictions + std_dev,
                            alpha=0.2, color='green', label='±1σ (68.3%)')
            
            # Plot confidence interval boundaries as dashed lines
            ax5.plot(last_indices, last_predictions + 3 * std_dev, 
                    linestyle='--', color='blue', alpha=0.4, linewidth=0.8)
            ax5.plot(last_indices, last_predictions - 3 * std_dev, 
                    linestyle='--', color='blue', alpha=0.4, linewidth=0.8)
            ax5.plot(last_indices, last_predictions + 2 * std_dev, 
                    linestyle='--', color='cyan', alpha=0.5, linewidth=0.8)
            ax5.plot(last_indices, last_predictions - 2 * std_dev, 
                    linestyle='--', color='cyan', alpha=0.5, linewidth=0.8)
            ax5.plot(last_indices, last_predictions + std_dev, 
                    linestyle='--', color='green', alpha=0.6, linewidth=0.8)
            ax5.plot(last_indices, last_predictions - std_dev, 
                    linestyle='--', color='green', alpha=0.6, linewidth=0.8)
        
        # Plot center (predicted mean) as a prominent thick line
        ax5.plot(last_indices, last_predictions, label='Center (Predicted Mean)', 
                color='red', linewidth=2.5, marker='o', markersize=4, alpha=0.9, zorder=5)
        
        # Plot actual values
        ax5.plot(last_indices, last_targets, label='Actual', 
                color='black', linewidth=2, marker='s', markersize=3, alpha=0.8, zorder=4)
        
        ax5.set_title(f'Last {n_points} Predictions: Center with Confidence Intervals')
    
    ax5.set_xlabel('Sample Index')
    ax5.set_ylabel('Value')
    ax5.legend(loc='best', fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Last 100 Prediction Error
    ax6 = axes[2, 1]
    errors = targets[-n_points:] - predictions[-n_points:]
    ax6.plot(last_indices, errors, label='Prediction Error', color='red', marker='o', markersize=3, alpha=0.7, linewidth=1.5)
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax6.fill_between(last_indices, errors, 0, alpha=0.2, color='red')
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('Error (Actual - Predicted)')
    ax6.set_title(f'Last {n_points} Prediction Errors')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

