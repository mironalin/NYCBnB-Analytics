#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, skipping XGBoost models")

def main():
    # Load the featured dataset
    data_file = 'dataset/featured_AB_NYC_2019.csv'
    if not os.path.exists(data_file):
        print(f"Error: File {data_file} not found!")
        return

    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Ensure numeric columns are properly typed
    numeric_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                    'calculated_host_listings_count', 'availability_365', 'has_reviews',
                    'days_since_last_review']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove outliers and invalid data
    print("Preparing data for price prediction...")
    df = df[df['price'] > 0].copy()  # Remove free listings

    # More intelligent outlier removal - use percentiles instead of fixed cutoff
    price_cutoff = df['price'].quantile(0.99)  # Remove top 1% of prices
    df = df[df['price'] < price_cutoff].copy()
    print(f"Removed listings with price > ${price_cutoff:.2f} (99th percentile)")

    # Enhanced feature engineering
    print("Performing enhanced feature engineering...")

    # Add geographical features
    df['distance_to_center'] = np.sqrt(
        (df['latitude'] - df['latitude'].mean())**2 +
        (df['longitude'] - df['longitude'].mean())**2
    )

    # Create interaction features
    df['reviews_per_availability'] = df['number_of_reviews'] / (df['availability_365'] + 1)  # +1 to avoid division by zero

    # Neighborhood features
    neighborhood_price_map = df.groupby('neighbourhood')['price'].mean().to_dict()
    df['neighbourhood_avg_price'] = df['neighbourhood'].map(neighborhood_price_map)

    neighborhood_review_map = df.groupby('neighbourhood')['number_of_reviews'].mean().to_dict()
    df['neighbourhood_avg_reviews'] = df['neighbourhood'].map(neighborhood_review_map)

    # Room type average price
    room_price_map = df.groupby('room_type')['price'].mean().to_dict()
    df['room_type_avg_price'] = df['room_type'].map(room_price_map)

    # Create log-transformed features for skewed variables
    df['log_reviews'] = np.log1p(df['number_of_reviews'])
    df['log_price'] = np.log1p(df['price'])  # For potential log-transformed target

    # Feature selection - include all potentially useful features
    X = df[[
        'neighbourhood_group', 'neighbourhood_avg_price', 'neighbourhood_avg_reviews',
        'room_type', 'room_type_avg_price', 'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
        'has_reviews', 'days_since_last_review', 'distance_to_center',
        'reviews_per_availability', 'log_reviews'
    ]]

    # Use actual price as target
    y = df['price']

    # IMPORTANT: Use full dataset, not sample
    print(f"Using full dataset with {X.shape[0]} rows and {X.shape[1]} features")

    # Drop rows with NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Split data with stratification based on price ranges
    price_bins = pd.qcut(y, 5, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=price_bins
    )

    # Define preprocessing for numerical and categorical columns
    numeric_features = [
        'minimum_nights', 'number_of_reviews', 'reviews_per_month',
        'calculated_host_listings_count', 'availability_365', 'has_reviews',
        'days_since_last_review', 'neighbourhood_avg_price', 'neighbourhood_avg_reviews',
        'room_type_avg_price', 'distance_to_center', 'reviews_per_availability', 'log_reviews'
    ]
    categorical_features = ['neighbourhood_group', 'room_type']

    # Create preprocessing pipeline with polynomial features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define base models
    base_models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42, alpha=0.01),
        'ElasticNet': ElasticNet(random_state=42, alpha=0.01, l1_ratio=0.5)
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        base_models['XGBoost'] = xgb.XGBRegressor(random_state=42)
        print("XGBoost model added")

    results = {}

    print("\nEvaluating base models...")
    for name, model in base_models.items():
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Cross-validation
        print(f"Training {name}...")
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)

        # Train on full training set
        pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'pipeline': pipeline,
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std(),
            'test_rmse': rmse,
            'test_mae': mae,
            'test_r2': r2,
            'y_pred': y_pred
        }

        print(f"{name}:")
        print(f"  CV RMSE: ${rmse_scores.mean():.2f} ± ${rmse_scores.std():.2f}")
        print(f"  Test RMSE: ${rmse:.2f}")
        print(f"  Test MAE: ${mae:.2f}")
        print(f"  Test R²: {r2:.4f}")

    # Choose best model for hyperparameter tuning
    best_model_name = min(results, key=lambda k: results[k]['test_rmse'])
    print(f"\nBest base model: {best_model_name} with Test R²: {results[best_model_name]['test_r2']:.4f}")

    # Hyperparameter tuning with focused parameter grid
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")

    # Define hyperparameter grid based on best model
    param_grid = {}
    if best_model_name == 'RandomForest':
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 15, 30],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'GradientBoosting':
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        }
    elif best_model_name == 'XGBoost':
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        }
    elif best_model_name in ['Ridge', 'Lasso', 'ElasticNet']:
        param_grid = {
            'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'model__max_iter': [2000]
        }
        if best_model_name == 'ElasticNet':
            param_grid['model__l1_ratio'] = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Use RandomizedSearchCV for tuning
    best_pipeline = results[best_model_name]['pipeline']
    random_search = RandomizedSearchCV(
        best_pipeline,
        param_distributions=param_grid,
        n_iter=10,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("Training tuned model...")
    random_search.fit(X_train, y_train)

    print(f"Best parameters found: {random_search.best_params_}")

    # Evaluate tuned model
    tuned_model = random_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test)

    tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    tuned_r2 = r2_score(y_test, y_pred_tuned)
    tuned_mae = mean_absolute_error(y_test, y_pred_tuned)

    print(f"\nTuned {best_model_name} performance:")
    print(f"  RMSE: ${tuned_rmse:.2f}")
    print(f"  MAE: ${tuned_mae:.2f}")
    print(f"  R²: {tuned_r2:.4f}")
    print(f"  Improvement in R²: {tuned_r2 - results[best_model_name]['test_r2']:.4f}")

    # Create stacking ensemble model
    print("\nBuilding stacking ensemble model...")

    # Select best models for stacking (top 3-5)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_rmse'])
    top_models = sorted_models[:3]  # Take top 3 models

    print(f"Using top models for stacking: {[name for name, _ in top_models]}")

    # Prepare estimators for stacking
    estimators = []
    for name, result in top_models:
        # Create a fresh instance with the same parameters as the base model
        if name == 'RandomForest':
            model_params = result['pipeline'].named_steps['model'].get_params()
            estimator = RandomForestRegressor(random_state=42)
            # Only keep relevant parameters
            filtered_params = {k: v for k, v in model_params.items()
                                if k in estimator.get_params()}
            estimator.set_params(**filtered_params)
            estimators.append((name, estimator))
        elif name == 'GradientBoosting':
            model_params = result['pipeline'].named_steps['model'].get_params()
            estimator = GradientBoostingRegressor(random_state=42)
            filtered_params = {k: v for k, v in model_params.items()
                                if k in estimator.get_params()}
            estimator.set_params(**filtered_params)
            estimators.append((name, estimator))
        elif name == 'XGBoost' and XGBOOST_AVAILABLE:
            model_params = result['pipeline'].named_steps['model'].get_params()
            estimator = xgb.XGBRegressor(random_state=42)
            filtered_params = {k: v for k, v in model_params.items()
                                if k in estimator.get_params()}
            estimator.set_params(**filtered_params)
            estimators.append((name, estimator))
        elif name in ['Ridge', 'Lasso', 'ElasticNet']:
            model_params = result['pipeline'].named_steps['model'].get_params()
            if name == 'Ridge':
                estimator = Ridge(random_state=42)
            elif name == 'Lasso':
                estimator = Lasso(random_state=42)
            else:  # ElasticNet
                estimator = ElasticNet(random_state=42)
            filtered_params = {k: v for k, v in model_params.items()
                                if k in estimator.get_params()}
            estimator.set_params(**filtered_params)
            estimators.append((name, estimator))

    # Create stacking regressor with Ridge as final estimator
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(random_state=42)
    )

    # Create pipeline for stacking
    stacking_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', stacking_regressor)
    ])

    # Train and evaluate stacking model
    print("Training stacking ensemble...")
    stacking_pipeline.fit(X_train, y_train)
    y_pred_stack = stacking_pipeline.predict(X_test)

    stack_rmse = np.sqrt(mean_squared_error(y_test, y_pred_stack))
    stack_r2 = r2_score(y_test, y_pred_stack)
    stack_mae = mean_absolute_error(y_test, y_pred_stack)

    print(f"\nStacking Ensemble performance:")
    print(f"  RMSE: ${stack_rmse:.2f}")
    print(f"  MAE: ${stack_mae:.2f}")
    print(f"  R²: {stack_r2:.4f}")

    # Select final best model
    final_models = {
        'Tuned': {
            'pipeline': tuned_model,
            'rmse': tuned_rmse,
            'r2': tuned_r2,
            'mae': tuned_mae,
            'y_pred': y_pred_tuned
        },
        'Stacking': {
            'pipeline': stacking_pipeline,
            'rmse': stack_rmse,
            'r2': stack_r2,
            'mae': stack_mae,
            'y_pred': y_pred_stack
        },
        'Base': {
            'pipeline': results[best_model_name]['pipeline'],
            'rmse': results[best_model_name]['test_rmse'],
            'r2': results[best_model_name]['test_r2'],
            'mae': results[best_model_name]['test_mae'],
            'y_pred': results[best_model_name]['y_pred']
        }
    }

    best_final_model = min(final_models, key=lambda k: final_models[k]['rmse'])

    # Create directory for models if it doesn't exist
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Save the best model
    best_model_path = os.path.join(models_dir, f'price_prediction_enhanced_{best_final_model}.joblib')
    joblib.dump(final_models[best_final_model]['pipeline'], best_model_path)
    print(f"\nFinal best model ({best_final_model}) saved to {best_model_path}")
    print(f"Final R² score: {final_models[best_final_model]['r2']:.4f}")

    # Generate feature importance visualization for RandomForest (if available)
    try:
        print("\nGenerating feature importance visualization...")

        if 'RandomForest' in results:
            rf_pipeline = results['RandomForest']['pipeline']
            rf_model = rf_pipeline.named_steps['model']

            # Transform the features to get feature names
            preprocessed_features = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()

            # Get importance scores
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Plot feature importance (top 20 features)
            plt.figure(figsize=(14, 10))
            plt.title('Feature Importance for Price Prediction', fontsize=16)
            plt.bar(range(min(len(importances), 20)), importances[indices[:20]], align='center')
            plt.xticks(range(min(len(importances), 20)),
                        [preprocessed_features[i] for i in indices[:20]],
                        rotation=90)
            plt.tight_layout()

            plots_dir = 'plots'
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)

            importance_plot_path = os.path.join(plots_dir, 'price_prediction_feature_importance_enhanced.png')
            plt.savefig(importance_plot_path)
            print(f"Feature importance plot saved to {importance_plot_path}")
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

    # Compare all models in a performance chart
    all_models = {}
    for name, result in results.items():
        all_models[name] = {
            'r2': result['test_r2'],
            'rmse': result['test_rmse']
        }

    # Add the tuned and stacking models
    all_models['Tuned'] = {
        'r2': final_models['Tuned']['r2'],
        'rmse': final_models['Tuned']['rmse']
    }

    if 'Stacking' in final_models:
        all_models['Stacking'] = {
            'r2': final_models['Stacking']['r2'],
            'rmse': final_models['Stacking']['rmse']
        }

    model_names = list(all_models.keys())
    r2_scores = [all_models[name]['r2'] for name in model_names]
    rmse_scores = [all_models[name]['rmse'] for name in model_names]

    # Create bar chart of R² scores
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    bar_colors = ['#1f77b4'] * len(model_names)
    best_idx = r2_scores.index(max(r2_scores))
    bar_colors[best_idx] = '#2ca02c'  # Highlight best model
    plt.bar(model_names, r2_scores, color=bar_colors)
    plt.ylim(0, max(r2_scores) * 1.1)
    plt.title('Model Performance Comparison (R²)', fontsize=14)
    plt.ylabel('R² Score (higher is better)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(r2_scores):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

    # Create bar chart of RMSE
    plt.subplot(1, 2, 2)
    bar_colors = ['#1f77b4'] * len(model_names)
    best_idx = rmse_scores.index(min(rmse_scores))
    bar_colors[best_idx] = '#2ca02c'  # Highlight best model
    plt.bar(model_names, rmse_scores, color=bar_colors)
    plt.title('Model Performance Comparison (RMSE)', fontsize=14)
    plt.ylabel('RMSE (lower is better)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(rmse_scores):
        plt.text(i, v + 1, f'{v:.1f}', ha='center')

    plt.tight_layout()
    comparison_plot_path = os.path.join(plots_dir, 'model_comparison.png')
    plt.savefig(comparison_plot_path)
    print(f"Model comparison plot saved to {comparison_plot_path}")

    # Plot actual vs predicted for best model
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, final_models[best_final_model]['y_pred'], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(f'Actual vs Predicted Prices - Enhanced Model ({best_final_model})')

    pred_plot_path = os.path.join(plots_dir, 'price_prediction_actual_vs_predicted_enhanced.png')
    plt.savefig(pred_plot_path)
    print(f"Actual vs predicted plot saved to {pred_plot_path}")

    # Create prediction distribution plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(y_test, color='blue', kde=True, stat='density', linewidth=0)
    sns.histplot(final_models[best_final_model]['y_pred'], color='red', kde=True, stat='density', linewidth=0, alpha=0.5)
    plt.xlabel('Price ($)')
    plt.ylabel('Density')
    plt.title('Distribution of Actual vs Predicted Prices')
    plt.legend(['Actual', 'Predicted'])

    plt.subplot(1, 2, 2)
    residuals = y_test - final_models[best_final_model]['y_pred']
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual Error')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')

    plt.tight_layout()
    dist_plot_path = os.path.join(plots_dir, 'price_prediction_distributions_enhanced.png')
    plt.savefig(dist_plot_path)
    print(f"Distribution plots saved to {dist_plot_path}")

if __name__ == "__main__":
    main()