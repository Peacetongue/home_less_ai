from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os

# Default arguments for the DAG
default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'real_estate_ml_pipeline',
    default_args=default_args,
    description='Real Estate Machine Learning Pipeline',
    schedule_interval=timedelta(weeks=1),  # Run weekly
    catchup=False,
    tags=['real_estate', 'machine_learning', 'model_training'],
)

def extract_and_prepare_features(**context):
    """Extract and prepare features for machine learning"""
    data_path = '/opt/airflow/data/real_estate_data.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    # Load data
    df = pd.read_csv(data_path, delimiter='\t', low_memory=False)
    
    # Feature engineering (example - adjust based on your actual data columns)
    feature_columns = []
    target_column = 'last_price'  # Adjust based on your actual target column
    
    # Check available columns and create features accordingly
    print(f"Available columns: {list(df.columns)}")
    
    # Example feature engineering (adapt to your actual data structure)
    if 'total_area' in df.columns:
        feature_columns.append('total_area')
    if 'living_area' in df.columns:
        feature_columns.append('living_area')
        # Create ratio feature
        if 'total_area' in df.columns:
            df['living_to_total_ratio'] = df['living_area'] / df['total_area']
            feature_columns.append('living_to_total_ratio')
    if 'kitchen_area' in df.columns:
        feature_columns.append('kitchen_area')
        # Create kitchen ratio
        if 'total_area' in df.columns:
            df['kitchen_to_total_ratio'] = df['kitchen_area'] / df['total_area']
            feature_columns.append('kitchen_to_total_ratio')
    if 'rooms' in df.columns:
        feature_columns.append('rooms')
    if 'ceiling_height' in df.columns:
        feature_columns.append('ceiling_height')
    
    # Create price per square meter if not exists
    if 'total_area' in df.columns and target_column in df.columns:
        df['price_per_sq_meter_calc'] = df[target_column] / df['total_area']
    
    # Remove rows with missing target variable
    df = df.dropna(subset=[target_column])
    
    # Remove rows with missing features
    df = df.dropna(subset=feature_columns)
    
    # Save prepared data
    output_path = '/opt/airflow/data/ml_ready_data.csv'
    df.to_csv(output_path, index=False)
    
    return {
        'records_count': len(df),
        'feature_columns': feature_columns,
        'target_column': target_column,
        'output_path': output_path
    }

def train_linear_model(**context):
    """Train a linear regression model"""
    # Get feature info from previous task
    ti = context['task_instance']
    feature_info = ti.xcom_pull(task_ids='extract_and_prepare_features')
    
    # Load prepared data
    df = pd.read_csv(feature_info['output_path'])
    
    # Prepare features and target
    X = df[feature_info['feature_columns']]
    y = df[feature_info['target_column']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Save model
    model_path = '/opt/airflow/data/models/linear_regression_model.joblib'
    os.makedirs('/opt/airflow/data/models', exist_ok=True)
    joblib.dump(model, model_path)
    
    # Feature importance (coefficients for linear regression)
    feature_importance = dict(zip(feature_info['feature_columns'], model.coef_))
    
    model_results = {
        'model_name': 'LinearRegression',
        'model_path': model_path,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'feature_importance': feature_importance,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    print(f"Linear Regression Results: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")
    
    return model_results

def train_random_forest_model(**context):
    """Train a random forest model"""
    # Get feature info from previous task
    ti = context['task_instance']
    feature_info = ti.xcom_pull(task_ids='extract_and_prepare_features')
    
    # Load prepared data
    df = pd.read_csv(feature_info['output_path'])
    
    # Prepare features and target
    X = df[feature_info['feature_columns']]
    y = df[feature_info['target_column']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Save model
    model_path = '/opt/airflow/data/models/random_forest_model.joblib'
    os.makedirs('/opt/airflow/data/models', exist_ok=True)
    joblib.dump(model, model_path)
    
    # Feature importance
    feature_importance = dict(zip(feature_info['feature_columns'], model.feature_importances_))
    
    model_results = {
        'model_name': 'RandomForest',
        'model_path': model_path,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'feature_importance': feature_importance,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    print(f"Random Forest Results: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")
    
    return model_results

def compare_models_and_select_best(**context):
    """Compare models and select the best one"""
    ti = context['task_instance']
    
    # Get results from both models
    linear_results = ti.xcom_pull(task_ids='train_linear_model')
    rf_results = ti.xcom_pull(task_ids='train_random_forest_model')
    
    # Compare based on R2 score (you can use other metrics)
    if rf_results['r2'] > linear_results['r2']:
        best_model = rf_results
        print(f"Random Forest selected as best model with R2={rf_results['r2']:.3f}")
    else:
        best_model = linear_results
        print(f"Linear Regression selected as best model with R2={linear_results['r2']:.3f}")
    
    # Save model comparison results
    comparison_results = {
        'best_model': best_model['model_name'],
        'linear_regression': linear_results,
        'random_forest': rf_results,
        'selection_date': datetime.now().isoformat()
    }
    
    results_path = '/opt/airflow/data/models/model_comparison.json'
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    return best_model

def log_model_to_database(**context):
    """Log model training results to PostgreSQL"""
    ti = context['task_instance']
    best_model = ti.xcom_pull(task_ids='compare_models_and_select_best')
    
    # This would connect to PostgreSQL and log the results
    # For demonstration, we'll just print the log entry
    log_entry = {
        'model_name': best_model['model_name'],
        'training_date': datetime.now(),
        'r2_score': best_model['r2'],
        'mae_score': best_model['mae'],
        'rmse_score': best_model['rmse'],
        'model_path': best_model['model_path'],
        'feature_importance': json.dumps(best_model['feature_importance']),
        'dag_run_id': context['dag_run'].run_id
    }
    
    print(f"Model training log entry: {log_entry}")
    
    return "Model logged successfully"

def generate_model_report(**context):
    """Generate a model performance report"""
    ti = context['task_instance']
    best_model = ti.xcom_pull(task_ids='compare_models_and_select_best')
    
    # Generate a simple report
    report = f"""
# Real Estate Price Prediction Model Report

## Model Performance Summary
- **Best Model**: {best_model['model_name']}
- **R² Score**: {best_model['r2']:.3f}
- **Mean Absolute Error**: {best_model['mae']:.2f}
- **Root Mean Square Error**: {best_model['rmse']:.2f}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Feature Importance
{chr(10).join([f"- {feature}: {importance:.4f}" for feature, importance in best_model['feature_importance'].items()])}

## Training Details
- Training Samples: {best_model['training_samples']}
- Test Samples: {best_model['test_samples']}
- Model Path: {best_model['model_path']}
"""
    
    # Save report
    report_path = '/opt/airflow/data/models/model_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print("Model report generated successfully")
    return report_path

# Define tasks
extract_features_task = PythonOperator(
    task_id='extract_and_prepare_features',
    python_callable=extract_and_prepare_features,
    dag=dag,
)

train_linear_task = PythonOperator(
    task_id='train_linear_model',
    python_callable=train_linear_model,
    dag=dag,
)

train_rf_task = PythonOperator(
    task_id='train_random_forest_model',
    python_callable=train_random_forest_model,
    dag=dag,
)

compare_models_task = PythonOperator(
    task_id='compare_models_and_select_best',
    python_callable=compare_models_and_select_best,
    dag=dag,
)

log_model_task = PythonOperator(
    task_id='log_model_to_database',
    python_callable=log_model_to_database,
    dag=dag,
)

generate_report_task = PythonOperator(
    task_id='generate_model_report',
    python_callable=generate_model_report,
    dag=dag,
)

# Define task dependencies
extract_features_task >> [train_linear_task, train_rf_task]
[train_linear_task, train_rf_task] >> compare_models_task
compare_models_task >> [log_model_task, generate_report_task]
