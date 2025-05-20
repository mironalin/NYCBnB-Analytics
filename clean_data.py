import pandas as pd
import numpy as np

file_path = "dataset/AB_NYC_2019.csv"
cleaned_file_path = "dataset/cleaned_AB_NYC_2019.csv" # Output file path

try:
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")

    # 1. Handle missing 'name' and 'host_name'
    # To address the FutureWarning and ensure changes are made correctly:
    df['name'] = df['name'].fillna('Unknown')
    df['host_name'] = df['host_name'].fillna('Unknown')
    print(f"Missing names filled. Missing host_names filled.")

    # 2. Handle missing 'reviews_per_month'
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    print(f"Missing reviews_per_month filled with 0.")

    # 3. Convert 'last_review' to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    print(f"'last_review' converted to datetime.")

    # 4. Investigate price == 0
    zero_price_listings = df[df['price'] == 0]
    print(f"\n--- Listings with price == 0 ---")
    print(f"Number of listings with price == 0: {len(zero_price_listings)}")
    if not zero_price_listings.empty:
        print(zero_price_listings[['id', 'name', 'host_id', 'neighbourhood_group', 'price']])

    if not zero_price_listings.empty:
        df = df[df['price'] > 0].copy()
        print(f"\nRemoved {len(zero_price_listings)} listings with price == 0.")
        print(f"New dataset shape after removing zero price listings: {df.shape}")

    print("\n--- DataFrame Info After Initial Cleaning ---")
    df.info()

    # (Keep other print statements for verification if desired)
    # print("\n--- DataFrame Head After Initial Cleaning (First 5 rows) ---")
    # print(df.head())
    # print("\n--- Missing Values per Column After Initial Cleaning ---")
    # print(df.isnull().sum())
    # print("\n--- Descriptive Statistics (Numerical Columns) After Initial Cleaning ---")
    # print(df.describe())
    # print("\n--- Descriptive Statistics (Object Columns) After Initial Cleaning ---")
    # print(df.describe(include=['object']))
    # print("\n--- Data Types After Initial Cleaning ---")
    # print(df.dtypes)

    # 5. Save the cleaned DataFrame to a new CSV file
    df.to_csv(cleaned_file_path, index=False)
    print(f"\nCleaned data saved to '{cleaned_file_path}'")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")