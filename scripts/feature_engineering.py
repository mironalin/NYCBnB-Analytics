import pandas as pd
import numpy as np
import csv

cleaned_file_path = "dataset/cleaned_AB_NYC_2019.csv"
featured_file_path = "dataset/featured_AB_NYC_2019.csv"

try:
    df = pd.read_csv(cleaned_file_path, parse_dates=['last_review'])
    print(f"Loaded '{cleaned_file_path}', shape: {df.shape}")

    # 1. Create 'has_reviews' column
    df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)
    print("Created 'has_reviews' column.")

    # 2. Create 'days_since_last_review' column
    if not df['last_review'].isnull().all(): # Proceed if there are any non-NaT dates
        # Determine the most recent review date in the dataset as a reference
        # Add one day to make it the 'analysis date' (the day after the last known review)
        analysis_date = df['last_review'].max() + pd.Timedelta(days=1)
        print(f"Analysis date for 'days_since_last_review': {analysis_date.strftime('%Y-%m-%d')}")

        # Calculate days since last review
        df['days_since_last_review'] = (analysis_date - df['last_review']).dt.days

        # Fill NaT/NaN in 'days_since_last_review' (for listings with no reviews)
        # with a large placeholder value (e.g., 9999 days)
        df['days_since_last_review'] = df['days_since_last_review'].fillna(9999)
        print("Created 'days_since_last_review' column and filled missing values.")
    else:
        # If all last_review dates are NaT (e.g. after cleaning or in a subset)
        df['days_since_last_review'] = 9999
        print("'last_review' column has no valid dates. 'days_since_last_review' filled with 9999.")


    print("\n--- Info for new columns ---")
    if 'has_reviews' in df.columns:
        print("\nValue counts for 'has_reviews':")
        print(df['has_reviews'].value_counts())

    if 'days_since_last_review' in df.columns:
        print("\nDescriptive stats for 'days_since_last_review':")
        print(df['days_since_last_review'].describe())

    # 3. Save the DataFrame with new features
    df.to_csv(featured_file_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nDataFrame with new features saved to '{featured_file_path}', shape: {df.shape}")

except FileNotFoundError:
    print(f"Error: The file '{cleaned_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")