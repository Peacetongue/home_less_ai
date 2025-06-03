import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Define the path to the data file
DATA_FILE_PATH = r"C:\Users\pavel\home_less_ai\data\real_estate_data.csv"
PLOTS_DIR = r"c:\Users\pavel\home_less_ai\plots"

def load_data(file_path):
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        print("and place it in the 'data' subfolder as 'real_estate_data.csv'.")
        return None
    try:
        # Specify low_memory=False if there are mixed type warnings,
        # though it's better to fix types explicitly later.
        # Specify the delimiter as tab
        df = pd.read_csv(file_path, delimiter='\t', low_memory=False)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def initial_inspection(df):
    """Performs initial data inspection."""
    if df is None:
        return
    print("\n--- First 5 Rows ---")
    print(df.head())
    print("\n--- Data Info ---")
    df.info()
    print("\n--- Descriptive Statistics ---")
    print(df.describe(include='all'))

def check_missing_values(df, stage="Before Preprocessing"):
    """Checks and displays missing values."""
    if df is None:
        return
    print(f"\n--- Missing Values per Column ({stage}) ---")
    missing_values = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_info = pd.concat([missing_values, missing_percent], axis=1, keys=['Total Missing', 'Percentage Missing'])
    print(missing_info[missing_info['Total Missing'] > 0].sort_values(by='Total Missing', ascending=False))

def preprocess_data(df):
    """Performs data cleaning, preprocessing, and feature engineering."""
    if df is None:
        return None
    
    print("\n--- Starting Data Preprocessing ---")
    df_processed = df.copy()

    # 1. Handle Missing Values
    print("\nHandling missing values...")
    # For boolean-like columns, fill NaN with False (or a suitable default)
    bool_like_cols = ['is_apartment', 'studio', 'open_plan']
    for col in bool_like_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(False)
            print(f"Filled NaNs in '{col}' with False.")

    if 'balcony' in df_processed.columns:
        df_processed['balcony'] = df_processed['balcony'].fillna(0)
        print("Filled NaNs in 'balcony' with 0.")

    # Impute numerical features with median
    numerical_cols_to_impute_median = [
        'ceiling_height', 'living_area', 'kitchen_area', 
        'airports_nearest', 'cityCenters_nearest', 
        'parks_around3000', 'parks_nearest', 
        'ponds_around3000', 'ponds_nearest', 'days_exposition'
    ]
    for col in numerical_cols_to_impute_median:
        if col in df_processed.columns:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"Filled NaNs in '{col}' with median ({median_val:.2f}).")

    # For 'floors_total', if 'floor' is present, ensure floors_total >= floor.
    # A simple median imputation for now.
    if 'floors_total' in df_processed.columns:
        median_floors_total = df_processed['floors_total'].median()
        df_processed['floors_total'].fillna(median_floors_total, inplace=True)
        print(f"Filled NaNs in 'floors_total' with median ({median_floors_total:.0f}).")
    
    # Drop rows where 'locality_name' or 'last_price' (target) is missing
    critical_cols_for_dropna = ['locality_name', 'last_price', 'total_area']
    original_rows = len(df_processed)
    for col in critical_cols_for_dropna:
        if col in df_processed.columns:
            df_processed.dropna(subset=[col], inplace=True)
    print(f"Dropped {original_rows - len(df_processed)} rows due to NaNs in critical columns ({', '.join(critical_cols_for_dropna)}).")


    # 2. Correct Data Types
    print("\nCorrecting data types...")
    if 'first_day_exposition' in df_processed.columns:
        df_processed['first_day_exposition'] = pd.to_datetime(df_processed['first_day_exposition'], errors='coerce')
        print("Converted 'first_day_exposition' to datetime.")
        # Drop rows where conversion failed if any
        df_processed.dropna(subset=['first_day_exposition'], inplace=True)


    for col in bool_like_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(bool)
            print(f"Converted '{col}' to boolean.")
            
    integer_cols = ['rooms', 'balcony', 'floors_total', 'parks_around3000', 'ponds_around3000']
    for col in integer_cols:
        if col in df_processed.columns:
            # Ensure no NaNs remain before converting to int
            if df_processed[col].isnull().sum() > 0:
                 # If previous median fill didn't catch all (e.g. for rooms if not in numerical_cols_to_impute_median)
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
            df_processed[col] = df_processed[col].astype(int)
            print(f"Converted '{col}' to integer.")
    
    # 3. Outlier Handling
    print("\nHandling outliers...")
    # Ceiling height: realistic range (e.g., 2m to 10m). Others are likely errors.
    if 'ceiling_height' in df_processed.columns:
        df_processed['ceiling_height'] = np.clip(df_processed['ceiling_height'], 2.0, 10.0)
        print("Clipped 'ceiling_height' to be between 2.0m and 10.0m.")

    # For price and area, clip extreme values using quantiles
    quantile_cols = ['last_price', 'total_area', 'living_area', 'kitchen_area']
    for col in quantile_cols:
        if col in df_processed.columns:
            q_low = df_processed[col].quantile(0.01)
            q_high = df_processed[col].quantile(0.99)
            df_processed[col] = np.clip(df_processed[col], q_low, q_high)
            print(f"Clipped '{col}' between {q_low:.2f} (1st percentile) and {q_high:.2f} (99th percentile).")

    # Ensure living_area and kitchen_area are not greater than total_area after clipping/imputation
    if 'living_area' in df_processed.columns and 'total_area' in df_processed.columns:
        df_processed['living_area'] = df_processed[['living_area', 'total_area']].min(axis=1)
    if 'kitchen_area' in df_processed.columns and 'total_area' in df_processed.columns:
        df_processed['kitchen_area'] = df_processed[['kitchen_area', 'total_area']].min(axis=1)


    # 4. Feature Engineering
    print("\nPerforming feature engineering...")
    # Price per square meter
    if 'last_price' in df_processed.columns and 'total_area' in df_processed.columns and df_processed['total_area'].min() > 0:
        df_processed['price_per_sq_meter'] = df_processed['last_price'] / df_processed['total_area']
        print("Created 'price_per_sq_meter'.")

    # Date features
    if 'first_day_exposition' in df_processed.columns:
        df_processed['exposition_year'] = df_processed['first_day_exposition'].dt.year
        df_processed['exposition_month'] = df_processed['first_day_exposition'].dt.month
        df_processed['exposition_dayofweek'] = df_processed['first_day_exposition'].dt.dayofweek
        print("Created date features: 'exposition_year', 'exposition_month', 'exposition_dayofweek'.")

    # Area ratios (handle potential division by zero if total_area can be 0 after processing)
    if 'living_area' in df_processed.columns and 'total_area' in df_processed.columns:
        df_processed['living_to_total_area_ratio'] = (df_processed['living_area'] / df_processed['total_area'].replace(0, np.nan)).fillna(0)
        print("Created 'living_to_total_area_ratio'.")
    if 'kitchen_area' in df_processed.columns and 'total_area' in df_processed.columns:
        df_processed['kitchen_to_total_area_ratio'] = (df_processed['kitchen_area'] / df_processed['total_area'].replace(0, np.nan)).fillna(0)
        print("Created 'kitchen_to_total_area_ratio'.")

    # Floor relative position
    if 'floor' in df_processed.columns and 'floors_total' in df_processed.columns:
        # Ensure floors_total is not zero and floor <= floors_total
        df_processed['floors_total'] = df_processed['floors_total'].replace(0, 1) # Avoid division by zero, ensure at least 1 floor
        df_processed['floor'] = df_processed.apply(lambda x: min(x['floor'], x['floors_total']), axis=1)
        df_processed['floor_relative_position'] = (df_processed['floor'] / df_processed['floors_total']).fillna(0)
        print("Created 'floor_relative_position'.")
    
    # Example: Is first or last floor (often less desirable)
    if 'floor' in df_processed.columns and 'floors_total' in df_processed.columns:
        df_processed['is_first_floor'] = (df_processed['floor'] == 1)
        df_processed['is_last_floor'] = (df_processed['floor'] == df_processed['floors_total'])
        print("Created 'is_first_floor' and 'is_last_floor'.")

    print("\n--- Data Preprocessing Finished ---")
    return df_processed


def create_visualizations(df, plots_dir):
    """Creates and saves basic visualizations."""
    if df is None or df.empty:
        print("DataFrame is None or empty. Skipping visualizations.")
        return

    if not os.path.exists(plots_dir):
        try:
            os.makedirs(plots_dir)
            print(f"Created directory: {plots_dir}")
        except OSError as e:
            print(f"Error creating directory {plots_dir}: {e}. Plots will not be saved.")
            return
    else:
        print(f"Plots directory already exists: {plots_dir}")

    # Histograms for some numerical features
    numerical_cols_to_plot = ['last_price', 'total_area', 'rooms', 'ceiling_height', 'price_per_sq_meter', 'living_to_total_area_ratio', 'kitchen_to_total_area_ratio']
    for col in numerical_cols_to_plot:
        if col in df.columns:
            if df[col].dropna().empty:
                print(f"Skipping histogram for '{col}' as it contains no data after dropna.")
                continue
            
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            
            file_path = os.path.join(plots_dir, f'{col}_histogram.png')
            try:
                plt.savefig(file_path)
                print(f"Saved histogram for {col} to {file_path}")
            except Exception as e:
                print(f"Error saving histogram for {col} to {file_path}: {e}")
            plt.close() # Close the figure to free memory
            
    # Correlation heatmap
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        if numeric_df.shape[1] < 2:
            print("Skipping correlation heatmap as there are less than 2 numeric columns.")
        else:
            plt.figure(figsize=(12, 10))
            try:
                correlation_matrix = numeric_df.corr()
                sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
                plt.title('Correlation Heatmap of Numerical Features')
                
                file_path = os.path.join(plots_dir, 'correlation_heatmap.png')
                plt.savefig(file_path)
                print(f"Saved correlation heatmap to {file_path}")
            except Exception as e:
                print(f"Error generating or saving correlation heatmap to {file_path}: {e}")
            plt.close()
    else:
        print("No numeric columns found for correlation heatmap.")

    # Example scatter plot: total_area vs last_price
    if 'total_area' in df.columns and 'last_price' in df.columns:
        if df[['total_area', 'last_price']].dropna().empty:
            print("Skipping scatter plot for 'total_area' vs 'last_price' as it contains no data after dropna.")
        else:
            plt.figure(figsize=(10, 6))
            try:
                sns.scatterplot(x='total_area', y='last_price', data=df, alpha=0.5, hue='exposition_year' if 'exposition_year' in df.columns else None)
                plt.title('Total Area vs. Last Price')
                plt.xlabel('Total Area (sq. m)')
                plt.ylabel('Last Price')
                
                file_path = os.path.join(plots_dir, 'total_area_vs_last_price_scatter.png')
                plt.savefig(file_path)
                print(f"Saved scatter plot for total_area vs. last_price to {file_path}")
            except Exception as e:
                print(f"Error saving scatter plot for total_area vs. last_price to {file_path}: {e}")
            plt.close() # Close the figure to free memory
    else:
        print("Skipping scatter plot: 'total_area' or 'last_price' not in DataFrame columns.")


def main():
    # Load the data
    df = load_data(DATA_FILE_PATH)
    
    # Initial inspection
    initial_inspection(df)
    
    # Check missing values
    check_missing_values(df, stage="Initial")
    
    # Preprocess the data
    df_processed = preprocess_data(df)
    
    # Check missing values after preprocessing
    check_missing_values(df_processed, stage="After Preprocessing")
    
    # Create visualizations
    create_visualizations(df_processed, PLOTS_DIR)
    
    # Display a message indicating the end of the script
    print("\n--- End of Script ---")

# The following line would typically be used to run the main function,
# but it's commented out here to prevent automatic execution on import.
if __name__ == "__main__":
    main()
