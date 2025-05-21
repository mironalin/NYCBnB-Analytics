#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def main():
    # Load the featured dataset
    data_file = 'dataset/featured_AB_NYC_2019.csv'
    if not os.path.exists(data_file):
        print(f"Error: File {data_file} not found!")
        return

    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    print("Preparing data for clustering...")
    # Select only numeric features for clustering
    numeric_features = ['price', 'minimum_nights', 'number_of_reviews',
                        'reviews_per_month', 'calculated_host_listings_count',
                        'availability_365', 'latitude', 'longitude']

    # Convert to numeric and handle any conversion errors
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with missing values in selected features
    df_cluster = df[numeric_features].dropna()
    print(f"Using {df_cluster.shape[0]} complete rows for clustering analysis.")

    # Standardize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)

    # Find optimal number of clusters using Elbow Method
    print("Determining optimal number of clusters...")
    inertia = []
    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    # Save elbow plot
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    elbow_plot_path = os.path.join(plots_dir, 'kmeans_elbow_plot.png')
    plt.savefig(elbow_plot_path)
    print(f"Elbow plot saved to {elbow_plot_path}")

    # Based on elbow plot, choose optimal k (typically where the curve bends)
    optimal_k = 4  # This will be determined by viewing the elbow plot

    # Perform K-means clustering with optimal k
    print(f"Performing K-means clustering with {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)

    # Add cluster labels to original data
    df_result = df.loc[df_cluster.index].copy()
    df_result['cluster'] = clusters

    # Analyze clusters - only include numeric columns to avoid errors with text data
    print("\nCluster Analysis:")
    numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
    cluster_stats = df_result.groupby('cluster')[numeric_cols].mean()
    print(cluster_stats[['price', 'minimum_nights', 'number_of_reviews',
                        'availability_365', 'reviews_per_month']])

    # Visualize clusters with PCA
    print("Creating PCA visualization of clusters...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Cluster')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Clusters of Airbnb Listings')
    cluster_plot_path = os.path.join(plots_dir, 'kmeans_clusters_pca.png')
    plt.savefig(cluster_plot_path)
    print(f"Cluster visualization saved to {cluster_plot_path}")

    # Save clustered data
    output_file = 'dataset/clustered_AB_NYC_2019.csv'
    df_result.to_csv(output_file, index=False)
    print(f"Saved clustered data to {output_file}")

    # Summarize each cluster's characteristics
    print("\nCluster Characteristics:")
    for i in range(optimal_k):
        cluster_data = df_result[df_result['cluster'] == i]
        n_listings = cluster_data.shape[0]
        pct = n_listings / df_result.shape[0] * 100

        print(f"Cluster {i}: {n_listings} listings ({pct:.1f}%)")
        print(f"  Average price: ${cluster_data['price'].mean():.2f}")

        # For categorical variables, use mode() and check if result exists
        if not cluster_data['neighbourhood_group'].empty:
            mode_value = cluster_data['neighbourhood_group'].mode()
            if not mode_value.empty:
                print(f"  Most common neighborhood group: {mode_value[0]}")

        if not cluster_data['room_type'].empty:
            mode_value = cluster_data['room_type'].mode()
            if not mode_value.empty:
                print(f"  Most common room type: {mode_value[0]}")

        print(f"  Average availability: {cluster_data['availability_365'].mean():.1f} days/year")
        print(f"  Average reviews: {cluster_data['number_of_reviews'].mean():.1f}")
        print()

if __name__ == "__main__":
    main()