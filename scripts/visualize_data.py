import pandas as pd
import matplotlib.pyplot as plt
import matplotlib # For colormaps
import seaborn as sns
import geopandas
from shapely.geometry import Point
import os
import numpy as np # For log transformation

# Define file paths
featured_file_path = "dataset/featured_AB_NYC_2019.csv"
plot_dir = "plots"

# Create plot directory if it doesn't exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

try:
    df = pd.read_csv(featured_file_path, parse_dates=['last_review'])
    print(f"Loaded '{featured_file_path}', shape: {df.shape}")

    # Plots 1-4 (assuming they were correct and saved, to reduce terminal output on re-runs)
    # If they need to be re-generated, uncomment the print statements for save confirmations.

    # 1. Histogram of price and log(price)
    if not os.path.exists(os.path.join(plot_dir, "price_histograms.png")):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(df['price'], bins=50, kde=False)
        plt.title('Histogram of Price')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        df['log_price'] = np.log1p(df['price'])
        sns.histplot(df['log_price'], bins=50, kde=False)
        plt.title('Histogram of Log(Price)')
        plt.xlabel('Log(Price)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "price_histograms.png"))
        plt.close()
        print(f"Saved: price_histograms.png")

    # 2. Bar chart of counts per neighbourhood_group
    if not os.path.exists(os.path.join(plot_dir, "neighbourhood_group_counts.png")):
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, y='neighbourhood_group', order=df['neighbourhood_group'].value_counts().index)
        plt.title('Number of Listings per Neighbourhood Group')
        plt.xlabel('Number of Listings')
        plt.ylabel('Neighbourhood Group')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "neighbourhood_group_counts.png"))
        plt.close()
        print(f"Saved: neighbourhood_group_counts.png")

    # 3. Bar chart of average price by neighbourhood_group
    if not os.path.exists(os.path.join(plot_dir, "avg_price_by_neighbourhood_group.png")):
        plt.figure(figsize=(8, 6))
        avg_price_group = df.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_price_group.values, y=avg_price_group.index, orient='h')
        plt.title('Average Price by Neighbourhood Group')
        plt.xlabel('Average Price ($)')
        plt.ylabel('Neighbourhood Group')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "avg_price_by_neighbourhood_group.png"))
        plt.close()
        print(f"Saved: avg_price_by_neighbourhood_group.png")

    # 4. Bar chart of average price by room_type
    if not os.path.exists(os.path.join(plot_dir, "avg_price_by_room_type.png")):
        plt.figure(figsize=(8, 6))
        avg_price_room = df.groupby('room_type')['price'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_price_room.values, y=avg_price_room.index, orient='h')
        plt.title('Average Price by Room Type')
        plt.xlabel('Average Price ($)')
        plt.ylabel('Room Type')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "avg_price_by_room_type.png"))
        plt.close()
        print(f"Saved: avg_price_by_room_type.png")

    # 5. GeoPandas scatter plot of listing locations
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = geopandas.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Plotting listings colored by price (without basemap)
    if not os.path.exists(os.path.join(plot_dir, "listing_locations_geopandas_price.png")):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        gdf.plot(ax=ax, column='price', cmap='viridis', s=5, legend=True,
                    legend_kwds={'label': "Price ($)", 'orientation': "horizontal"}, alpha=0.5)
        plt.title('NYC Airbnb Listing Locations Colored by Price')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "listing_locations_geopandas_price.png"))
        plt.close()
        print(f"Saved: listing_locations_geopandas_price.png")

    # Plotting listings colored by neighbourhood_group (without basemap)
    # Simplified approach using direct column plotting for neighborhood_group
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf.plot(ax=ax, column='neighbourhood_group', legend=True, s=5, alpha=0.7,
                categorical=True, cmap='tab10') # cmap='tab10' for distinct colors
    plt.title('NYC Airbnb Listing Locations by Neighbourhood Group')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "listing_locations_geopandas_group.png"))
    plt.close()
    print(f"Saved: listing_locations_geopandas_group.png")

    # 6. Histogram of days_since_last_review
    plt.figure(figsize=(8, 6))
    sns.histplot(df[df['days_since_last_review'] != 9999]['days_since_last_review'], bins=50, kde=False)
    plt.title('Histogram of Days Since Last Review (Excluding No-Review Listings)')
    plt.xlabel('Days Since Last Review')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "days_since_last_review_hist.png"))
    plt.close()
    print(f"Saved: days_since_last_review_hist.png")

    print("\nAll plots generated/checked successfully.")

except FileNotFoundError:
    print(f"Error: The file '{featured_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred during visualization: {e}")