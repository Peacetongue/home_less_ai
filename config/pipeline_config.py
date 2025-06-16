# Airflow configuration for real estate data pipeline

# Database connections
REAL_ESTATE_DB_CONN = {
    'host': 'postgres',
    'port': 5432,
    'database': 'real_estate_db',
    'username': 'real_estate_user',
    'password': 'real_estate_pass'
}

AIRFLOW_DB_CONN = {
    'host': 'postgres',
    'port': 5432,
    'database': 'airflow',
    'username': 'airflow',
    'password': 'airflow'
}

# Data paths
DATA_PATHS = {
    'raw_data': '/opt/airflow/data/real_estate_data.csv',
    'preprocessed_data': '/opt/airflow/data/real_estate_data_preprocessed.csv',
    'ml_ready_data': '/opt/airflow/data/ml_ready_data.csv',
    'models_dir': '/opt/airflow/data/models',
    'plots_dir': '/opt/airflow/plots'
}

# Model configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    },
    'linear_regression': {
        'fit_intercept': True
    },
    'test_size': 0.2,
    'random_state': 42
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'numerical_features': [
        'total_area',
        'living_area', 
        'kitchen_area',
        'rooms',
        'ceiling_height'
    ],
    'target_column': 'last_price',
    'derived_features': [
        'living_to_total_ratio',
        'kitchen_to_total_ratio',
        'price_per_sq_meter_calc'
    ]
}

# Data quality thresholds
DATA_QUALITY = {
    'max_missing_percentage': 50,  # Maximum percentage of missing values per row
    'min_records': 100,  # Minimum number of records required
    'outlier_threshold': 3  # Standard deviations for outlier detection
}
