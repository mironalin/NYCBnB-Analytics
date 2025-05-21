import pandas as pd
import numpy as np
import os # Added for path joining if needed, though not strictly for this simple case

original_file_path = "dataset/AB_NYC_2019.csv"
cleaned_file_path = "dataset/cleaned_AB_NYC_2019.csv"

print(f"Starting data cleaning...")
print(f"Input file: {original_file_path}")
print(f"Output file: {cleaned_file_path}")

try:
    df = pd.read_csv(original_file_path)
    print(f"Original dataset loaded. Shape: {df.shape}")

    # 1. Handle missing 'name' and 'host_name'
    df['name'] = df['name'].fillna('Unknown')
    df['host_name'] = df['host_name'].fillna('Unknown')
    print(f"Missing names and host_names filled.")

    # 2. Clean embedded newlines/carriage returns from name and host_name
    # This is crucial for SAS compatibility.
    print(f"Cleaning newlines from 'name' and 'host_name' columns...")
    for col in ['name', 'host_name']:
        if col in df.columns:
            # Convert to string to ensure .str accessor works
            s = df[col].astype(str)
            # Step 1: Remove carriage returns entirely
            s = s.str.replace('\r', '', regex=False)
            # Step 2: Replace newlines with a single space
            s = s.str.replace('\n', ' ', regex=False)
            # Step 3: Replace multiple whitespace characters (including spaces, tabs, newlines that became spaces) with a single space
            s = s.str.replace('\s+', ' ', regex=True)
            # Step 4: Strip leading/trailing whitespace
            df[col] = s.str.strip()
    print("Newlines and excess whitespace cleaned from 'name' and 'host_name'.")

    # 3. Handle missing 'reviews_per_month'
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    print(f"Missing reviews_per_month filled with 0.")

    # 4. Convert 'last_review' to datetime
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    print(f"'last_review' converted to datetime (unparseable as NaT).")

    # 5. Remove price == 0 listings
    zero_price_count = len(df[df['price'] == 0])
    if zero_price_count > 0:
        df = df[df['price'] > 0].copy()
        print(f"Removed {zero_price_count} listings with price == 0.")
    print(f"Dataset shape after initial cleaning: {df.shape}")

    # 6. Save the cleaned DataFrame
    df.to_csv(cleaned_file_path, index=False, quoting=1) # quoting=1 is csv.QUOTE_ALL
    print(f"Cleaned data saved to '{cleaned_file_path}'")
    print("Cleaning script finished successfully.")

except FileNotFoundError:
    print(f"Error: The file '{original_file_path}' was not found. Please ensure it is in the dataset folder.")
except Exception as e:
    print(f"An error occurred during cleaning: {e}")
    import traceback
    traceback.print_exc()