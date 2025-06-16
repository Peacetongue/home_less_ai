from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import os

# Default arguments for the DAG
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'real_estate_data_pipeline',
    default_args=default_args,
    description='Real Estate Data Processing Pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['real_estate', 'data_processing', 'ml'],
)

def load_and_validate_data(**context):
    """Load and validate the real estate data"""
    data_path = '/opt/airflow/data/real_estate_data.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    # Load data with tab delimiter
    df = pd.read_csv(data_path, delimiter='\t', low_memory=False)
    
    # Basic validation
    if df.empty:
        raise ValueError("Data file is empty")
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Store basic stats for next task
    stats = {
        'total_records': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return stats

def preprocess_data(**context):
    """Preprocess the real estate data"""
    data_path = '/opt/airflow/data/real_estate_data.csv'
    df = pd.read_csv(data_path, delimiter='\t', low_memory=False)
    
    # Basic preprocessing steps
    initial_count = len(df)
    
    # Remove rows with too many missing values (example: more than 50% missing)
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)
    
    print(f"Removed {initial_count - len(df)} rows with excessive missing values")
    
    # Save preprocessed data
    output_path = '/opt/airflow/data/real_estate_data_preprocessed.csv'
    df.to_csv(output_path, index=False)
    
    return {'preprocessed_records': len(df), 'output_path': output_path}

def create_database_tables(**context):
    """Create database tables for storing processed data"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS real_estate_data (
        id SERIAL PRIMARY KEY,
        total_area FLOAT,
        living_area FLOAT,
        kitchen_area FLOAT,
        rooms INTEGER,
        ceiling_height FLOAT,
        last_price FLOAT,
        price_per_sq_meter FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS data_processing_log (
        id SERIAL PRIMARY KEY,
        dag_run_id VARCHAR(255),
        task_id VARCHAR(255),
        records_processed INTEGER,
        processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(50)
    );
    """
    
    # This would need proper PostgreSQL connection configuration
    # For now, just print the SQL
    print("Database tables creation SQL:")
    print(create_table_sql)

def store_data_to_postgres(**context):
    """Store processed data to PostgreSQL"""
    try:
        # Load preprocessed data
        data_path = '/opt/airflow/data/real_estate_data_preprocessed.csv'
        df = pd.read_csv(data_path)
        
        # This is where you would connect to PostgreSQL and insert data
        # For demonstration, we'll just log the operation
        print(f"Would store {len(df)} records to PostgreSQL")
        
        # Log the processing
        log_entry = {
            'dag_run_id': context['dag_run'].run_id,
            'task_id': context['task_instance'].task_id,
            'records_processed': len(df),
            'status': 'completed'
        }
        print(f"Processing log: {log_entry}")
        
    except Exception as e:
        print(f"Error storing data: {str(e)}")
        raise

def run_eda_analysis(**context):
    """Run exploratory data analysis"""
    # This would run the EDA script
    eda_command = "cd /opt/airflow/workspace && python eda.py"
    print(f"Would run EDA analysis: {eda_command}")
    
    # For now, just indicate the task completed
    return "EDA analysis completed"

# Define tasks
data_validation_task = PythonOperator(
    task_id='validate_data',
    python_callable=load_and_validate_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

create_tables_task = PythonOperator(
    task_id='create_database_tables',
    python_callable=create_database_tables,
    dag=dag,
)

store_data_task = PythonOperator(
    task_id='store_data_to_postgres',
    python_callable=store_data_to_postgres,
    dag=dag,
)

run_eda_task = PythonOperator(
    task_id='run_eda_analysis',
    python_callable=run_eda_analysis,
    dag=dag,
)

# Alternative: Use BashOperator for running the EDA script
run_eda_bash_task = BashOperator(
    task_id='run_eda_bash',
    bash_command='cd /opt/airflow/workspace && python eda.py',
    dag=dag,
)

# Define task dependencies
data_validation_task >> data_preprocessing_task >> create_tables_task >> store_data_task
data_preprocessing_task >> run_eda_task
store_data_task >> run_eda_bash_task
