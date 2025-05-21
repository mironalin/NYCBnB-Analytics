#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading data and model...")
    # Check if model exists
    model_path = 'models/price_prediction_enhanced_Tuned.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return

    # Load the model
    model = joblib.load(model_path)

    # Load the dataset
    data_file = 'dataset/featured_AB_NYC_2019.csv'
    if not os.path.exists(data_file):
        print(f"Error: File {data_file} not found!")
        return

    df = pd.read_csv(data_file)

    # Do minimal preprocessing to get the same features used in training
    print("Preparing data for visualization...")

    # Ensure numeric columns are properly typed
    numeric_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                    'calculated_host_listings_count', 'availability_365', 'has_reviews',
                    'days_since_last_review']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove outliers and invalid data
    df = df[df['price'] > 0].copy()
    price_cutoff = df['price'].quantile(0.99)
    df = df[df['price'] < price_cutoff].copy()

    # Add geographical features
    df['distance_to_center'] = np.sqrt(
        (df['latitude'] - df['latitude'].mean())**2 +
        (df['longitude'] - df['longitude'].mean())**2
    )

    # Create interaction features
    df['reviews_per_availability'] = df['number_of_reviews'] / (df['availability_365'] + 1)

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
    df['log_price'] = np.log1p(df['price'])

    # Use the same feature set
    X = df[[
        'neighbourhood_group', 'neighbourhood_avg_price', 'neighbourhood_avg_reviews',
        'room_type', 'room_type_avg_price', 'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
        'has_reviews', 'days_since_last_review', 'distance_to_center',
        'reviews_per_availability', 'log_reviews'
    ]]
    y = df['price']

    # Handle missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Split data to get the same test set
    price_bins = pd.qcut(y, 5, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=price_bins
    )

    # Generate predictions
    print("Generating model predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model Performance:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  R²: {r2:.4f}")

    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Generate visualizations
    print("Creating visualization plots...")

    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs Predicted Prices - Enhanced Model')
    plt.grid(True, linestyle='--', alpha=0.7)

    pred_plot_path = os.path.join(plots_dir, 'price_prediction_actual_vs_predicted_enhanced.png')
    plt.savefig(pred_plot_path)
    print(f"Actual vs predicted plot saved to {pred_plot_path}")

    # 2. Distribution and Residuals Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(y_test, color='blue', kde=True, stat='density', linewidth=0)
    sns.histplot(y_pred, color='red', kde=True, stat='density', linewidth=0, alpha=0.5)
    plt.xlabel('Price ($)')
    plt.ylabel('Density')
    plt.title('Distribution of Actual vs Predicted Prices')
    plt.legend(['Actual', 'Predicted'])

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual Error')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')

    plt.tight_layout()
    dist_plot_path = os.path.join(plots_dir, 'price_prediction_distributions_enhanced.png')
    plt.savefig(dist_plot_path)
    print(f"Distribution plots saved to {dist_plot_path}")

    # 3. Price Range Accuracy
    # Create bins of prices and measure accuracy in each bin
    price_bins = pd.cut(y_test, bins=[0, 50, 100, 150, 200, 300, 1000],
                            labels=['0-50', '50-100', '100-150', '150-200', '200-300', '300+'])
    bin_metrics = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'price_bin': price_bins
    })

    bin_stats = bin_metrics.groupby('price_bin').agg({
        'actual': ['count', 'mean'],
        'predicted': 'mean'
    }).reset_index()

    bin_stats.columns = ['price_bin', 'count', 'actual_mean', 'predicted_mean']
    bin_stats['error'] = (bin_stats['predicted_mean'] - bin_stats['actual_mean']) / bin_stats['actual_mean'] * 100

    plt.figure(figsize=(12, 7))
    bar_width = 0.35
    x = np.arange(len(bin_stats))

    plt.bar(x - bar_width/2, bin_stats['actual_mean'], bar_width, label='Actual', color='blue')
    plt.bar(x + bar_width/2, bin_stats['predicted_mean'], bar_width, label='Predicted', color='red')

    plt.xlabel('Price Range ($)')
    plt.ylabel('Average Price ($)')
    plt.title('Prediction Accuracy by Price Range')
    plt.xticks(x, bin_stats['price_bin'])
    plt.legend()

    # Add error percentage labels
    for i, row in enumerate(bin_stats.itertuples()):
        plt.text(i, max(row.actual_mean, row.predicted_mean) + 10,
                f"{row.error:.1f}%", ha='center')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    price_range_path = os.path.join(plots_dir, 'price_prediction_by_range.png')
    plt.savefig(price_range_path)
    print(f"Price range accuracy plot saved to {price_range_path}")

    # 4. Feature Importance Scatter Plot
    # Plot how features relate to price
    key_features = ['neighbourhood_avg_price', 'room_type_avg_price',
                    'distance_to_center', 'availability_365', 'number_of_reviews']

    plt.figure(figsize=(18, 10))
    plt.suptitle('Relationship Between Key Features and Price', fontsize=16)

    for i, feature in enumerate(key_features):
        plt.subplot(2, 3, i+1)

        if feature in ['room_type', 'neighbourhood_group']:
            # For categorical features
            sns.boxplot(x=feature, y='price', data=df[df['price'] < price_cutoff])
            plt.xticks(rotation=45)
        else:
            # For numeric features, use scatter plot with trend line
            plt.scatter(df[feature], df['price'], alpha=0.1)
            plt.plot(np.unique(df[feature]),
                    np.poly1d(np.polyfit(df[feature], df['price'], 1))(np.unique(df[feature])),
                    color='red')

        plt.title(f'{feature} vs Price')
        plt.xlabel(feature)
        plt.ylabel('Price ($)')
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    feature_scatter_path = os.path.join(plots_dir, 'price_vs_features.png')
    plt.savefig(feature_scatter_path)
    print(f"Feature-price relationship plot saved to {feature_scatter_path}")

    print("\nAll visualization plots generated successfully!")

if __name__ == "__main__":
    main()