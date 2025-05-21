#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def main():
    # Load the featured dataset
    data_file = 'dataset/featured_AB_NYC_2019.csv'
    if not os.path.exists(data_file):
        print(f"Error: File {data_file} not found!")
        return
    
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    print("Starting time series analysis of review patterns...")
    
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    
    # Filter out rows without review date
    df_with_reviews = df.dropna(subset=['last_review']).copy()
    print(f"Using {df_with_reviews.shape[0]} listings with review dates for time series analysis")
    
    # Extract date components
    df_with_reviews['review_year'] = df_with_reviews['last_review'].dt.year
    df_with_reviews['review_month'] = df_with_reviews['last_review'].dt.month
    df_with_reviews['review_quarter'] = df_with_reviews['last_review'].dt.quarter
    df_with_reviews['review_yearmonth'] = df_with_reviews['last_review'].dt.strftime('%Y-%m')
    
    # Create temporal aggregations
    
    # 1. Review counts by month
    monthly_reviews = df_with_reviews.groupby('review_yearmonth').size().reset_index(name='review_count')
    monthly_reviews['date'] = pd.to_datetime(monthly_reviews['review_yearmonth'] + '-01')
    monthly_reviews = monthly_reviews.sort_values('date')
    
    # 2. Average price by month
    monthly_prices = df_with_reviews.groupby('review_yearmonth')['price'].mean().reset_index()
    monthly_prices['date'] = pd.to_datetime(monthly_prices['review_yearmonth'] + '-01')
    monthly_prices = monthly_prices.sort_values('date')
    
    # 3. Room type distribution over time
    room_time_dist = df_with_reviews.groupby(['review_yearmonth', 'room_type']).size().reset_index(name='count')
    room_time_dist = room_time_dist.pivot(index='review_yearmonth', columns='room_type', values='count')
    room_time_dist = room_time_dist.fillna(0)
    room_time_dist.index = pd.to_datetime(room_time_dist.index + '-01')
    room_time_dist = room_time_dist.sort_index()
    
    # 4. Neighborhood activity
    neighborhood_time = df_with_reviews.groupby(['review_yearmonth', 'neighbourhood_group']).size().reset_index(name='count')
    neighborhood_time = neighborhood_time.pivot(index='review_yearmonth', columns='neighbourhood_group', values='count')
    neighborhood_time = neighborhood_time.fillna(0)
    neighborhood_time.index = pd.to_datetime(neighborhood_time.index + '-01')
    neighborhood_time = neighborhood_time.sort_index()
    
    # Let's create visualizations
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 1. Review activity over time
    plt.figure(figsize=(14, 7))
    ax = sns.lineplot(data=monthly_reviews, x='date', y='review_count', linewidth=2)
    
    # Add moving average trendline
    window_size = 3
    monthly_reviews['trend'] = monthly_reviews['review_count'].rolling(window=window_size).mean()
    sns.lineplot(data=monthly_reviews, x='date', y='trend', linewidth=2, color='red')
    
    plt.title('Airbnb Review Activity Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    trend_plot_path = os.path.join(plots_dir, 'review_trend_over_time.png')
    plt.savefig(trend_plot_path)
    print(f"Review trend plot saved to {trend_plot_path}")
    
    # 2. Average price trend over time
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=monthly_prices, x='date', y='price', linewidth=2)
    
    # Add moving average trendline
    monthly_prices['trend'] = monthly_prices['price'].rolling(window=window_size).mean()
    sns.lineplot(data=monthly_prices, x='date', y='trend', linewidth=2, color='red')
    
    plt.title('Average Airbnb Price Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Price ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    price_trend_path = os.path.join(plots_dir, 'price_trend_over_time.png')
    plt.savefig(price_trend_path)
    print(f"Price trend plot saved to {price_trend_path}")
    
    # 3. Room type distribution over time
    plt.figure(figsize=(14, 7))
    room_time_dist_pct = room_time_dist.div(room_time_dist.sum(axis=1), axis=0) * 100
    room_time_dist_pct.plot(kind='area', stacked=True, alpha=0.7)
    
    plt.title('Room Type Distribution Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Percentage of Reviews', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Room Type')
    plt.tight_layout()
    
    room_type_path = os.path.join(plots_dir, 'room_type_distribution_over_time.png')
    plt.savefig(room_type_path)
    print(f"Room type distribution plot saved to {room_type_path}")
    
    # 4. Neighborhood activity over time
    plt.figure(figsize=(14, 7))
    neighborhood_time_pct = neighborhood_time.div(neighborhood_time.sum(axis=1), axis=0) * 100
    neighborhood_time_pct.plot(kind='area', stacked=True, alpha=0.7)
    
    plt.title('Neighborhood Activity Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Percentage of Reviews', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Neighborhood Group')
    plt.tight_layout()
    
    neighborhood_path = os.path.join(plots_dir, 'neighborhood_activity_over_time.png')
    plt.savefig(neighborhood_path)
    print(f"Neighborhood activity plot saved to {neighborhood_path}")
    
    # 5. Seasonal patterns (reviews by month)
    plt.figure(figsize=(12, 6))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    monthly_pattern = df_with_reviews.groupby('review_month').size().reindex(range(1, 13)).fillna(0)
    ax = sns.barplot(x=list(range(1, 13)), y=monthly_pattern.values)
    
    # Customize x-axis labels
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    
    plt.title('Seasonal Review Patterns by Month', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    seasonal_path = os.path.join(plots_dir, 'seasonal_review_pattern.png')
    plt.savefig(seasonal_path)
    print(f"Seasonal pattern plot saved to {seasonal_path}")
    
    # Save time series data for potential further analysis
    output_dir = 'dataset'
    monthly_reviews.to_csv(os.path.join(output_dir, 'monthly_review_trends.csv'), index=False)
    monthly_prices.to_csv(os.path.join(output_dir, 'monthly_price_trends.csv'), index=False)
    
    print("\nTime series analysis completed successfully!")

if __name__ == "__main__":
    main() 