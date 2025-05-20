import pandas as pd

# Define the file path relative to the script location or workspace root
# Assuming the script will be in the workspace root, and data is in dataset/
file_path = "dataset/AB_NYC_2019.csv"

try:
    df = pd.read_csv(file_path)

    print("--- DataFrame Info ---")
    df.info()  # Prints directly to stdout

    print("\n--- DataFrame Head (First 5 rows) ---")
    print(df.head())

    print("\n--- Missing Values per Column ---")
    print(df.isnull().sum())

    print("\n--- Descriptive Statistics (Numerical Columns) ---")
    print(df.describe())

    print("\n--- Descriptive Statistics (Object Columns) ---")
    print(df.describe(include=['object']))
    
    print("\n--- Value Counts for 'neighbourhood_group' ---")
    print(df['neighbourhood_group'].value_counts())

    print("\n--- Value Counts for 'room_type' ---")
    print(df['room_type'].value_counts())
    
    print("\n--- Data Types ---")
    print(df.dtypes)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct location.")
except Exception as e:
    print(f"An error occurred: {e}") 