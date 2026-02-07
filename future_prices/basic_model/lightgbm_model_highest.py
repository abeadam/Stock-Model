"""
LightGBM Model for PctChange_CloseToHigh (lightgbm_model_highest)

Predicts the value of the PctChange_CloseToHigh column (percentage change from
Close to High within the same bar). Outputs: lightgbm_model_highest.pkl, etc.
"""

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

from lightgbm_utils import (
    load_data,
    prepare_features_and_target,
    train_lightgbm,
    scale_target,
    unscale_target,
    evaluate_model,
    get_feature_importance,
    plot_predictions_vs_actual,
    run_correlation_analysis,
    run_kfold_cv,
    run_top_n_feature_search,
    apply_top_n_features,
    write_metrics_file,
)


def main():
    """Main function to run the LightGBM model."""
    # Base name for all output artifacts (from script name, e.g. lightgbm_model_highest)
    MODEL_BASE = Path(__file__).resolve().stem
    
    # Configuration
    csv_path = Path(__file__).parent.parent / 'es_with_indicators.csv'
    
    # For initial testing, you can use a sample of the data
    # Set to None to use all data (will take longer)
    # Start with 1000 to quickly test if model is learning, then scale up
    SAMPLE_SIZE = None  # Start small for quick testing, then set to None for full training
    # Recommended progression: 1000 -> 10000 -> 100000 -> None (full dataset)
    
    # Model parameters (from lightgbm_best_params_random_20260202_093117.json)
    if SAMPLE_SIZE and SAMPLE_SIZE < 10000:
        # Smaller model for small datasets to prevent overfitting
        N_ESTIMATORS = 500
        MAX_DEPTH = 8
        NUM_LEAVES = 127
        LEARNING_RATE = 0.01
        MIN_DATA_IN_LEAF = 5
        BAGGING_FREQ = 1
        print(f"  Using smaller model parameters for sample size {SAMPLE_SIZE}")
    else:
        N_ESTIMATORS = 1600
        MAX_DEPTH = 8
        NUM_LEAVES = 511
        LEARNING_RATE = 0.012036847570290075
        MIN_DATA_IN_LEAF = 11
        BAGGING_FREQ = 1
    
    FEATURE_FRACTION = 0.9458480547239904
    BAGGING_FRACTION = 0.7938194534317411
    LAMBDA_L1 = 0.0515137118615616
    LAMBDA_L2 = 0.27818644082591015
    MIN_GAIN_TO_SPLIT = 0.0  # Allow splits even with tiny gains (important for small targets)
    
    # CPU parallelization (optimized for Mac)
    N_JOBS = -1  # -1 = use all CPU cores
    
    # Random seed
    RANDOM_STATE = 42
    
    # Cross-validation
    N_FOLDS = 10
    
    # Grid search over number of top features (by |correlation| with target)
    SEARCH_TOP_N_FEATURES = True   # If True, optimize USE_TOP_N_FEATURES via CV; else use fixed value below
    TOP_N_CANDIDATES = [None, 10, 15, 20, 25, 30, 40, 50]  # None = use all features
    N_FOLDS_TOP_N_SEARCH = 5       # Folds used when searching TOP_N (fewer = faster)
    
    # Used only when SEARCH_TOP_N_FEATURES is False
    USE_TOP_N_FEATURES = None  # None = use all features; 20 = train on top 20 by |corr| with target
    
    # Base train kwargs (L1/L2 may be overridden when using limited feature set)
    base_train_kwargs = dict(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        num_leaves=NUM_LEAVES,
        learning_rate=LEARNING_RATE,
        feature_fraction=FEATURE_FRACTION,
        bagging_fraction=BAGGING_FRACTION,
        bagging_freq=BAGGING_FREQ,
        lambda_l1=LAMBDA_L1,
        lambda_l2=LAMBDA_L2,
        min_data_in_leaf=MIN_DATA_IN_LEAF,
        min_gain_to_split=MIN_GAIN_TO_SPLIT,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    
    print("=" * 60)
    print("LightGBM Model (highest): PctChange_CloseToHigh")
    print("=" * 60)
    
    # Load data
    df = load_data(csv_path, sample_size=SAMPLE_SIZE)
    
    # Prepare features and target
    X, y, feature_cols, TARGET_SCALE, SCALING_METHOD, y_original = prepare_features_and_target(
        df, target_column='PctChange_CloseToHigh'
    )
    
    # Feature-target correlations (using original unscaled target)
    corr_file = Path(__file__).parent / f'{MODEL_BASE}_feature_correlations.csv'
    correlations = run_correlation_analysis(X, y_original, feature_cols, corr_file)

    # Option to use only top N features by absolute correlation (reduces noise, may help R²)
    if SEARCH_TOP_N_FEATURES:
        USE_TOP_N_FEATURES, _, _ = run_top_n_feature_search(
            X, y, correlations, feature_cols,
            TARGET_SCALE, SCALING_METHOD,
            base_train_kwargs,
            TOP_N_CANDIDATES,
            n_folds=N_FOLDS_TOP_N_SEARCH,
            sample_size=SAMPLE_SIZE,
        )
        X, feature_cols = apply_top_n_features(X, correlations, feature_cols, USE_TOP_N_FEATURES)
        if USE_TOP_N_FEATURES is not None:
            LAMBDA_L1 = 0.0
            LAMBDA_L2 = 0.05
            print(f"  Using top {USE_TOP_N_FEATURES} features; L1=0, L2={LAMBDA_L2}")
        else:
            print(f"  Using all features")
    elif USE_TOP_N_FEATURES is not None and len(correlations) >= USE_TOP_N_FEATURES:
        X, feature_cols = apply_top_n_features(X, correlations, feature_cols, USE_TOP_N_FEATURES)
        print(f"\n  Using top {USE_TOP_N_FEATURES} features by |correlation| with target:")
        print(f"    {feature_cols}")
        LAMBDA_L1 = 0.0
        LAMBDA_L2 = 0.05
        print(f"  L1=0, L2={LAMBDA_L2} (reduced regularization for limited feature set)")
    
    # Option to use hyperparameter tuning
    USE_GRID_SEARCH = False  # Set to True to enable hyperparameter tuning
    USE_RANDOM_SEARCH = True  # If True, use HalvingRandomSearchCV (better for large spaces)
                               # If False, use HalvingGridSearchCV (exhaustive but slower)
    
    if USE_GRID_SEARCH:
        print("\n" + "="*60)
        print("Hyperparameter Tuning with HalvingGridSearchCV")
        print("="*60)
        
        # Define parameter distributions or grid based on search type
        # No early stopping - want to see best possible values for each combination
        param_distributions = None  # Initialize to avoid "possibly unbound" error
        param_grid = None  # Initialize to avoid "possibly unbound" error
        
        if USE_RANDOM_SEARCH:
            # Use distributions for RandomSearchCV - can explore wider ranges efficiently
            from scipy.stats import randint, uniform
            
            # Tuned around last best (20260201): n_est 1004, depth 10, leaves 511, lr 0.017,
            # feat_frac 0.95, bag_frac 0.78, L1 0.036, L2 0.13, min_leaf 13, bag_freq 1
            param_distributions = {
                'n_estimators': randint(800, 2200),   # Narrow around 1004; early stop ~300-500
                'max_depth': randint(8, 13),          # Best 10; top CV 8-13
                'num_leaves': [255, 511, 1023],      # Best 511; drop 127
                'learning_rate': uniform(0.008, 0.018),  # Best 0.017; ~0.008-0.026
                'feature_fraction': uniform(0.88, 0.12),  # Best 0.95; 0.88-1.0
                'bagging_fraction': uniform(0.72, 0.26),  # Best 0.78; 0.72-0.98
                'bagging_freq': [0, 1, 5],
                'lambda_l1': uniform(0.01, 0.06),    # Best 0.036; low L1
                'lambda_l2': uniform(0.05, 0.35),    # Best 0.13; avoid very high L2
                'min_data_in_leaf': randint(8, 16), # Best 13; 8-15
            }
            print("  Using HalvingRandomSearchCV - exploring wider parameter ranges")
        else:
            # Grid focused around last best (20260201)
            param_grid = {
                'n_estimators': [800, 1200, 1800],
                'max_depth': [8, 10, 12],
                'num_leaves': [255, 511, 1023],
                'learning_rate': [0.008, 0.012, 0.018, 0.025],
                'feature_fraction': [0.88, 0.93, 0.98],
                'bagging_fraction': [0.75, 0.82, 0.90],
                'bagging_freq': [0, 1, 5],
                'lambda_l1': [0.01, 0.03, 0.05],
                'lambda_l2': [0.08, 0.15, 0.25],
                'min_data_in_leaf': [8, 11, 14],
            }
            print("  Using HalvingGridSearchCV - grid around last best params")
        
        # Create base model with fixed parameters
        # No early stopping - train fully to see best possible performance
        # feature_name='auto' prevents feature name warnings when using numpy arrays
        base_model = lgb.LGBMRegressor(
            boosting_type='gbdt',
            objective='regression',
            metric='rmse',
            bagging_freq=1,
            force_col_wise=True,
            random_state=RANDOM_STATE,
            n_jobs=1,  # Use 1 job per model (CV handles parallelization)
            verbose=-1
        )
        
        # Note: No early stopping - each model trains for full n_estimators
        # This allows seeing the true best possible values for each parameter combination
        
        # Scoring for hyperparameter search (higher is better; sklearn uses neg_* for losses)
        # Options: 'neg_mean_squared_error' (default), 'r2', 'neg_mean_absolute_error',
        #   'neg_median_absolute_error' (robust), 'explained_variance'
        SCORING = 'neg_mean_squared_error'
        
        # Use HalvingRandomSearchCV or HalvingGridSearchCV (limited candidates for speed)
        # factor=3 means keep top 1/3 of candidates at each iteration
        n_candidates = None  # Initialize to avoid "possibly unbound" error
        if USE_RANDOM_SEARCH:
            n_candidates = 50  # Fixed number of random combinations (faster than "exhaust")
            search_cv = HalvingRandomSearchCV(
                base_model,
                param_distributions,
                n_candidates=n_candidates,
                cv=2,  # 2-fold for speed
                scoring=SCORING,
                n_jobs=1,
                factor=3,
                min_resources=200,  # Subsample in early rounds for speed
                random_state=RANDOM_STATE,
                verbose=1
            )
            print(f"  Testing {n_candidates} random parameter combinations (2-fold CV, scoring={SCORING})")
        else:
            # HalvingGridSearchCV with limited resources (not exhaustive final refit)
            search_cv = HalvingGridSearchCV(
                base_model,
                param_grid,
                cv=2,  # 2-fold for speed
                scoring=SCORING,
                n_jobs=1,
                factor=3,
                min_resources=500,  # Subsample in early rounds; final round uses more but not full data
                random_state=RANDOM_STATE,
                verbose=1
            )
            # Calculate total combinations (approximate, since HalvingGridSearchCV uses successive halving)
            total_combinations = 1
            for param_values in param_grid.values():
                total_combinations *= len(param_values)
            print(f"  Parameter grid contains ~{total_combinations} total combinations")
        
        print(f"  Using {len(X):,} samples (full data, internal CV)")
        print(f"  No early stopping - each model trains fully to see best possible values")
        print(f"  This may take a while...")
        
        # Fit search on full data (HalvingSearchCV uses internal CV)
        search_cv.fit(X, y)
        
        print(f"\n  Best parameters found:")
        for param, value in search_cv.best_params_.items():
            print(f"    {param}: {value}")
        print(f"  Best cross-validation score ({SCORING}): {search_cv.best_score_:.6f}")
        
        # Save search results
        results_dir = Path(__file__).parent
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        search_type = "random" if USE_RANDOM_SEARCH else "grid"
        
        # Save best parameters to JSON
        best_params_file = results_dir / f'{MODEL_BASE}_best_params_{search_type}_{timestamp}.json'
        best_params_dict = {
            'best_params': search_cv.best_params_,
            'best_score': float(search_cv.best_score_),
            'best_score_std': float(search_cv.cv_results_['std_test_score'][search_cv.best_index_]) if hasattr(search_cv, 'best_index_') else None,
            'n_splits': search_cv.cv if isinstance(search_cv.cv, int) else None,
            'scoring': search_cv.scoring,
            'search_type': search_type,
            'timestamp': timestamp
        }
        if USE_RANDOM_SEARCH:
            best_params_dict['n_candidates'] = n_candidates
        with open(best_params_file, 'w') as f:
            json.dump(best_params_dict, f, indent=2)
        print(f"\n  Best parameters saved to: {best_params_file}")
        
        # Save full CV results to CSV
        cv_results_file = results_dir / f'{MODEL_BASE}_cv_results_{search_type}_{timestamp}.csv'
        cv_results_df = pd.DataFrame(search_cv.cv_results_)
        # Sort by mean test score (descending)
        if 'mean_test_score' in cv_results_df.columns:
            cv_results_df = cv_results_df.sort_values('mean_test_score', ascending=False)
        cv_results_df.to_csv(cv_results_file, index=False)
        print(f"  Full CV results saved to: {cv_results_file}")
        print(f"    Total parameter combinations tested: {len(cv_results_df)}")
        
        # Save best model
        model_file = results_dir / f'{MODEL_BASE}_best_model_{search_type}_{timestamp}.pkl'
        joblib.dump(search_cv.best_estimator_, model_file)
        print(f"  Best model saved to: {model_file}")
        
        # Use best model
        model = search_cv.best_estimator_
        scaler = None
        
        # Update parameters for display
        N_ESTIMATORS = search_cv.best_params_['n_estimators']
        MAX_DEPTH = search_cv.best_params_['max_depth']
        NUM_LEAVES = search_cv.best_params_['num_leaves']
        LEARNING_RATE = search_cv.best_params_['learning_rate']
        FEATURE_FRACTION = search_cv.best_params_['feature_fraction']
        BAGGING_FRACTION = search_cv.best_params_['bagging_fraction']
        BAGGING_FREQ = search_cv.best_params_['bagging_freq']
        LAMBDA_L1 = search_cv.best_params_['lambda_l1']
        LAMBDA_L2 = search_cv.best_params_['lambda_l2']
        MIN_DATA_IN_LEAF = search_cv.best_params_['min_data_in_leaf']
        
    else:
        # Train model with fixed parameters on full data
        model, scaler = train_lightgbm(
            X, y,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            num_leaves=NUM_LEAVES,
            sample_size=SAMPLE_SIZE,
            learning_rate=LEARNING_RATE,
            feature_fraction=FEATURE_FRACTION,
            bagging_fraction=BAGGING_FRACTION,
            bagging_freq=BAGGING_FREQ,
            lambda_l1=LAMBDA_L1,
            lambda_l2=LAMBDA_L2,
            min_data_in_leaf=MIN_DATA_IN_LEAF,
            min_gain_to_split=MIN_GAIN_TO_SPLIT,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=-1  # -1 = errors only, 0 = warnings, 1 = info
        )
    
    # Cross-validation for robust evaluation (same params as final model)
    train_kwargs = dict(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        num_leaves=NUM_LEAVES,
        learning_rate=LEARNING_RATE,
        feature_fraction=FEATURE_FRACTION,
        bagging_fraction=BAGGING_FRACTION,
        bagging_freq=BAGGING_FREQ,
        lambda_l1=LAMBDA_L1,
        lambda_l2=LAMBDA_L2,
        min_data_in_leaf=MIN_DATA_IN_LEAF,
        min_gain_to_split=MIN_GAIN_TO_SPLIT,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    metrics_mean, metrics_std, oof_y_true, oof_y_pred, _ = run_kfold_cv(
        X, y, N_FOLDS, TARGET_SCALE, SCALING_METHOD,
        sample_size=SAMPLE_SIZE, report_weak_folds=True, quiet=False, **train_kwargs
    )
    metrics = metrics_mean
    
    # Evaluate final model on full data (training set metrics)
    print("\n" + "="*60)
    print("Evaluating Final Model on Full Data (Training Fit)...")
    print("="*60)
    train_metrics, y_train_pred = evaluate_model(
        model, scaler, X, y, target_scale=TARGET_SCALE, scaling_method=SCALING_METHOD
    )
    print(f"\nTraining Set Results (Percentage Change):")
    print(f"  MSE: {train_metrics['mse']:.6f} (%²)")
    print(f"  RMSE: {train_metrics['rmse']:.6f}%")
    print(f"  MAE: {train_metrics['mae']:.6f}%")
    print(f"  R²: {train_metrics['r2']:.4f}")
    if not np.isnan(train_metrics['mape']):
        print(f"  MAPE: {train_metrics['mape']:.2f}%")
    
    # Baseline comparison using out-of-fold predictions
    print("\nBaseline Comparison (on OOF predictions):")
    y_pred_baseline = np.concatenate([[oof_y_true[0]], oof_y_true[:-1]])
    baseline_r2 = r2_score(oof_y_true, y_pred_baseline)
    baseline_mae = mean_absolute_error(oof_y_true, y_pred_baseline)
    print(f"  Baseline R² (predict previous): {baseline_r2:.4f}")
    print(f"  Baseline MAE: {baseline_mae:.6f}%")
    print(f"  Model R² (CV mean): {metrics['r2']:.4f}")
    print(f"  Model MAE (CV mean): {metrics['mae']:.6f}%")
    if metrics['r2'] > baseline_r2:
        print(f"  ✓ Model outperforms baseline by {metrics['r2'] - baseline_r2:.4f} R² points")
    else:
        print(f"  ⚠ Baseline outperforms model by {baseline_r2 - metrics['r2']:.4f} R² points")
    
    # Generate plot of predictions vs actual (OOF, unscaled; first 100 consecutive steps)
    plot_target_label = "Target value (unscaled)" if SCALING_METHOD not in ("none", "absolute") else "Target value"
    plot_path = plot_predictions_vs_actual(
        oof_y_true, oof_y_pred, n_steps=100, target_label=plot_target_label,
        save_path=Path(__file__).parent / f'{MODEL_BASE}_predictions_vs_actual.png'
    )
    print(f"Visualization saved to {plot_path}")
    
    # Get feature importances
    importance_df = get_feature_importance(model, feature_cols, top_n=20)
    
    # Save model
    model_dir = Path(__file__).parent
    model_path = model_dir / f'{MODEL_BASE}.pkl'
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    
    # Save target scaler if using StandardScaler (absolute method)
    if SCALING_METHOD == "absolute":
        target_scaler_path = model_dir / f'{MODEL_BASE}_target_scaler.pkl'
        joblib.dump(TARGET_SCALE, target_scaler_path)
        print(f"Target StandardScaler saved to {target_scaler_path}")
        print("Note: Use target scaler to inverse transform predictions")
    
    # Save feature importances
    importance_path = model_dir / f'{MODEL_BASE}_feature_importances_splits_and_gain.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importances saved to {importance_path}")
    
    # Save metrics (CV mean±std and final model on full data)
    metrics_path = model_dir / f'{MODEL_BASE}_metrics.txt'
    write_metrics_file(
        metrics_path, "LightGBM Model (highest) Metrics - PctChange_CloseToHigh",
        N_FOLDS, metrics_mean, metrics_std, train_metrics
    )
    print(f"Metrics saved to {metrics_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
