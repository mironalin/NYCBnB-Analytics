#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap, MarkerCluster
import os

def main():
    # Load the featured dataset
    data_file = 'dataset/featured_AB_NYC_2019.csv'
    if not os.path.exists(data_file):
        print(f"Error: File {data_file} not found!")
        return

    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    print("Creating geospatial visualizations...")

    # Create maps directory if it doesn't exist
    maps_dir = 'maps'
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)

    # Center the map on the average coordinates of NYC
    nyc_center = [df['latitude'].mean(), df['longitude'].mean()]

    # Create base map parameters (we'll create new maps instead of copying)
    base_map_params = {
        'location': nyc_center,
        'zoom_start': 11,
        'tiles': 'cartodbpositron'
    }

    # Create a sample DataFrame to reduce marker density
    # (full dataset would be too heavy for interactive visualization)
    sample_size = 5000  # Increased from 2000 for more detailed visualization
    df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    print(f"Using {len(df_sample)} listings for marker visualization")

    # 1. Create clustered marker map
    print("Creating clustered marker map...")
    marker_map = folium.Map(**base_map_params)

    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(marker_map)

    # Define color map for room types
    room_colors = {
        'Entire home/apt': 'red',
        'Private room': 'blue',
        'Shared room': 'green'
    }

    # Add markers for each listing in the sample
    for idx, row in df_sample.iterrows():
        popup_text = f"""
        <b>{row['name']}</b><br>
        Price: ${row['price']}<br>
        Room: {row['room_type']}<br>
        Reviews: {row['number_of_reviews']}<br>
        """

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=room_colors.get(row['room_type'], 'gray'))
        ).add_to(marker_cluster)

    # Add a legend for room types
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; padding: 10px;
    background-color: white; border: 2px solid grey; border-radius: 5px">
    <p><b>Room Types</b></p>
    <p><i class="fa fa-circle" style="color:red"></i> Entire home/apt</p>
    <p><i class="fa fa-circle" style="color:blue"></i> Private room</p>
    <p><i class="fa fa-circle" style="color:green"></i> Shared room</p>
    </div>
    """
    marker_map.get_root().html.add_child(folium.Element(legend_html))

    # Save the marker map
    marker_map_path = os.path.join(maps_dir, 'airbnb_marker_map.html')
    marker_map.save(marker_map_path)
    print(f"Marker map saved to {marker_map_path}")

    # 2. Create price heatmap
    print("Creating price heatmap...")
    price_map = folium.Map(**base_map_params)

    # For heatmaps, we can use a larger subset of data since it's more optimized
    heatmap_sample_size = 10000  # Increased sample size for heatmap
    df_heat_sample = df.sample(min(heatmap_sample_size, len(df)), random_state=42)
    print(f"Using {len(df_heat_sample)} listings for heatmap visualization")

    # Prepare data for heatmap (lat, lon, price)
    heat_data = [[row['latitude'], row['longitude'], min(row['price'], 500)]
                    for _, row in df_heat_sample.iterrows() if row['price'] > 0]

    # Add heatmap layer
    HeatMap(heat_data, radius=15, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},
            min_opacity=0.3).add_to(price_map)

    # Save price heatmap
    price_map_path = os.path.join(maps_dir, 'airbnb_price_heatmap.html')
    price_map.save(price_map_path)
    print(f"Price heatmap saved to {price_map_path}")

    # 3. Create review density heatmap
    print("Creating review density heatmap...")
    review_map = folium.Map(**base_map_params)

    # Prepare data for reviews heatmap (lat, lon, number_of_reviews)
    review_heat_data = [[row['latitude'], row['longitude'], min(row['number_of_reviews'], 300)]
                        for _, row in df_heat_sample.iterrows() if row['number_of_reviews'] > 0]

    # Add heatmap layer
    HeatMap(review_heat_data, radius=15, gradient={0.2: 'green', 0.4: 'lime', 0.6: 'yellow', 1: 'red'},
            min_opacity=0.3).add_to(review_map)

    # Save reviews heatmap
    review_map_path = os.path.join(maps_dir, 'airbnb_review_heatmap.html')
    review_map.save(review_map_path)
    print(f"Review density heatmap saved to {review_map_path}")

    # 4. Create availability choropleth
    print("Creating neighborhood average availability map...")
    availability_map = folium.Map(**base_map_params)

    # Group by neighborhood and compute average availability
    neighborhood_stats = df.groupby('neighbourhood').agg({
        'availability_365': 'mean',
        'latitude': 'mean',
        'longitude': 'mean',
        'price': 'mean',
        'number_of_reviews': 'mean'
    }).reset_index()

    # Add circle markers for neighborhoods, sized by average availability
    for _, row in neighborhood_stats.iterrows():
        avg_avail = row['availability_365']
        avg_price = row['price']
        # Set color based on availability (red = low availability, green = high availability)
        color = 'red' if avg_avail < 90 else ('orange' if avg_avail < 180 else 'green')

        popup_text = f"""
        <b>{row['neighbourhood']}</b><br>
        Avg availability: {avg_avail:.1f} days/year<br>
        Avg price: ${avg_price:.2f}<br>
        Avg reviews: {row['number_of_reviews']:.1f}
        """

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=min(10, max(5, avg_avail / 36.5)),  # Scale: 1-10, based on availability
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(availability_map)

    # Add a legend for availability
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; padding: 10px;
    background-color: white; border: 2px solid grey; border-radius: 5px">
    <p><b>Average Availability</b></p>
    <p><i class="fa fa-circle" style="color:red"></i> Low (< 90 days/year)</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Medium (90-180 days/year)</p>
    <p><i class="fa fa-circle" style="color:green"></i> High (> 180 days/year)</p>
    <p>Circle size proportional to availability days</p>
    </div>
    """
    availability_map.get_root().html.add_child(folium.Element(legend_html))

    # Save availability map
    avail_map_path = os.path.join(maps_dir, 'airbnb_availability_map.html')
    availability_map.save(avail_map_path)
    print(f"Availability map saved to {avail_map_path}")

    print("\nGeospatial visualizations complete. Open the HTML files in a web browser to view interactive maps.")

if __name__ == "__main__":
    main()