#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main():
    print("Creating model comparison visualizations...")

    # Define metrics for each model version
    models = {
        "Base": {
            "r2": 0.3746,      # From original GradientBoosting model
            "rmse": 89.61,
            "mae": 52.50,
            "color": "#1f77b4", # Blue
            "plots": {
                "actual_vs_pred": "price_prediction_actual_vs_predicted.png",
                "distribution": "price_prediction_distributions.png",
                "feature_importance": "price_prediction_feature_importance.png"
            }
        },
        "Improved": {
            "r2": 0.4332,      # From improved version with GradientBoosting
            "rmse": 80.19,
            "mae": 47.74,
            "color": "#ff7f0e", # Orange
            "plots": {
                "actual_vs_pred": "price_prediction_actual_vs_predicted_improved.png",
                "distribution": "price_prediction_distributions_improved.png",
                "feature_importance": "price_prediction_feature_importance_improved.png"
            }
        },
        "Enhanced": {
            "r2": 0.5141,      # From enhanced XGBoost model with tuning
            "rmse": 71.71,
            "mae": 44.12,
            "color": "#2ca02c", # Green
            "plots": {
                "actual_vs_pred": "price_prediction_actual_vs_predicted_enhanced.png",
                "distribution": "price_prediction_distributions_enhanced.png",
                "feature_importance": "price_prediction_feature_importance_enhanced.png"
            }
        }
    }

    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # 1. Generate R² comparison bar chart
    plt.figure(figsize=(12, 6))

    # Plot main comparison metrics
    plt.subplot(1, 2, 1)
    model_names = list(models.keys())
    r2_values = [models[name]["r2"] for name in model_names]
    colors = [models[name]["color"] for name in model_names]

    bars = plt.bar(model_names, r2_values, color=colors)
    plt.title('Model Performance Improvement (R²)', fontsize=14)
    plt.ylabel('R² Score (higher is better)')
    plt.ylim(0, max(r2_values) * 1.15)

    # Add percentage improvement labels
    base_r2 = models["Base"]["r2"]
    for i, (model, r2) in enumerate(zip(model_names, r2_values)):
        if i > 0:
            improvement = ((r2 - base_r2) / base_r2) * 100
            plt.text(i, r2 + 0.02, f"+{improvement:.1f}%", ha='center')
        plt.text(i, r2/2, f"{r2:.3f}", ha='center', color='white', fontweight='bold')

    # Plot RMSE comparison
    plt.subplot(1, 2, 2)
    rmse_values = [models[name]["rmse"] for name in model_names]

    bars = plt.bar(model_names, rmse_values, color=colors)
    plt.title('Model Error Reduction (RMSE)', fontsize=14)
    plt.ylabel('RMSE (lower is better)')

    # Add percentage improvement labels
    base_rmse = models["Base"]["rmse"]
    for i, (model, rmse) in enumerate(zip(model_names, rmse_values)):
        if i > 0:
            improvement = ((base_rmse - rmse) / base_rmse) * 100
            plt.text(i, rmse - 5, f"-{improvement:.1f}%", ha='center')
        plt.text(i, rmse/2, f"{rmse:.1f}", ha='center', color='white', fontweight='bold')

    plt.tight_layout()
    metrics_path = os.path.join(plots_dir, 'model_evolution_metrics.png')
    plt.savefig(metrics_path)
    print(f"Model evolution metrics comparison saved to {metrics_path}")

    # 2. Generate progress timeline visualization
    plt.figure(figsize=(14, 8))

    # Define positions on timeline
    timeline_points = [1, 2.5, 4]
    timeline_y = 0.5

    # Draw horizontal timeline
    plt.plot([0.5, 4.5], [timeline_y, timeline_y], 'k-', lw=2, alpha=0.7)

    # Create vertical dotted lines connecting events to metrics
    metrics_y = [1.5, 2.5, 3.5]

    # Plot milestones
    for i, (name, point) in enumerate(zip(model_names, timeline_points)):
        # Plot milestone circles
        plt.plot(point, timeline_y, 'o', markersize=15, color=models[name]["color"])
        plt.text(point, timeline_y - 0.2, name, ha='center', fontsize=12, fontweight='bold')

        # Draw vertical connector
        plt.plot([point, point], [timeline_y + 0.1, metrics_y[0] - 0.1], 'k:', alpha=0.5)

        # Plot metrics
        r2 = models[name]["r2"]
        rmse = models[name]["rmse"]
        mae = models[name]["mae"]

        plt.annotate(f'R² = {r2:.3f}', xy=(point, metrics_y[0]),
                    ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', fc=models[name]["color"], alpha=0.7))

        plt.annotate(f'RMSE = ${rmse:.2f}', xy=(point, metrics_y[1]),
                    ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', fc=models[name]["color"], alpha=0.7))

        plt.annotate(f'MAE = ${mae:.2f}', xy=(point, metrics_y[2]),
                    ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', fc=models[name]["color"], alpha=0.7))

        # Add description
        if name == "Base":
            desc = "GradientBoosting\nBasic features"
        elif name == "Improved":
            desc = "GradientBoosting\nExpanded feature set\nHyperparameter tuning"
        else: # Enhanced
            desc = "XGBoost\nFull dataset\nPolynomial features\nAdvanced tuning"

        plt.annotate(desc, xy=(point, 0.1),
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0', alpha=0.7))

    # Add improvement arrows
    for i in range(len(model_names)-1):
        start = timeline_points[i]
        end = timeline_points[i+1]
        mid = (start + end) / 2

        # R² improvement
        r2_start = models[model_names[i]]["r2"]
        r2_end = models[model_names[i+1]]["r2"]
        r2_imp = ((r2_end - r2_start) / r2_start) * 100

        plt.annotate(f'+{r2_imp:.1f}%', xy=(mid, timeline_y + 0.3),
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='#d0ffd0', alpha=0.9))

        # Add arrow
        plt.arrow(start + 0.15, timeline_y, end - start - 0.3, 0,
                    head_width=0.1, head_length=0.1, fc='green', ec='green', alpha=0.6)

    # Remove axis ticks and labels
    plt.axis('off')
    plt.ylim(-0.2, 4)
    plt.xlim(0, 5)
    plt.title('AirBnB Price Prediction Model Evolution', fontsize=16)

    timeline_path = os.path.join(plots_dir, 'model_evolution_timeline.png')
    plt.savefig(timeline_path)
    print(f"Model evolution timeline saved to {timeline_path}")

    # 3. Create key feature correlation comparison
    # This is based on the most important features identified in our analyses
    key_features_by_importance = {
        'Base': ['room_type', 'neighbourhood_group', 'availability_365', 'minimum_nights', 'calculated_host_listings_count'],
        'Improved': ['room_type', 'neighbourhood_group', 'availability_365', 'distance_to_center', 'reviews_per_availability'],
        'Enhanced': ['neighbourhood_avg_price', 'room_type_avg_price', 'room_type', 'distance_to_center', 'availability_365']
    }

    plt.figure(figsize=(14, 7))
    plt.title('Evolution of Feature Importance Across Model Versions', fontsize=16)

    # Create a grid to plot feature rank evolution
    all_features = set()
    for features in key_features_by_importance.values():
        all_features.update(features)

    # Convert to sorted list
    all_features = sorted(list(all_features))

    # Create a feature rank matrix
    feature_ranks = pd.DataFrame(index=model_names, columns=all_features)
    for model, features in key_features_by_importance.items():
        for i, feature in enumerate(features):
            feature_ranks.loc[model, feature] = i + 1  # Rank 1 is most important

    # Fill NaN with max rank + 1 (meaning not in top features)
    feature_ranks = feature_ranks.fillna(len(all_features) + 1)

    # Plot rank evolution
    for feature in all_features:
        plt.plot(model_names, feature_ranks[feature], 'o-', label=feature, lw=2, markersize=10)

    plt.ylabel('Feature Importance Rank (1 is most important)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Features', loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    feature_evolution_path = os.path.join(plots_dir, 'feature_importance_evolution.png')
    plt.savefig(feature_evolution_path)
    print(f"Feature importance evolution saved to {feature_evolution_path}")

    print("\nAll comparison visualizations have been generated successfully!")

if __name__ == "__main__":
    main()