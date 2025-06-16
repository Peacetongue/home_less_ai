#!/bin/bash
set -e

# Create additional database for real estate data
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE real_estate_db;
    CREATE USER real_estate_user WITH ENCRYPTED PASSWORD 'real_estate_pass';
    GRANT ALL PRIVILEGES ON DATABASE real_estate_db TO real_estate_user;
EOSQL

# Connect to the real estate database and create initial tables
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "real_estate_db" <<-EOSQL
    -- Create real estate data table
    CREATE TABLE IF NOT EXISTS real_estate_data (
        id SERIAL PRIMARY KEY,
        total_area FLOAT,
        living_area FLOAT,
        kitchen_area FLOAT,
        rooms INTEGER,
        ceiling_height FLOAT,
        last_price FLOAT,
        price_per_sq_meter FLOAT,
        building_type VARCHAR(100),
        location VARCHAR(200),
        metro_distance FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create data processing log table
    CREATE TABLE IF NOT EXISTS data_processing_log (
        id SERIAL PRIMARY KEY,
        dag_run_id VARCHAR(255),
        task_id VARCHAR(255),
        records_processed INTEGER,
        processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(50),
        notes TEXT
    );

    -- Create model training log table
    CREATE TABLE IF NOT EXISTS model_training_log (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(100),
        model_version VARCHAR(50),
        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        accuracy_score FLOAT,
        mae_score FLOAT,
        rmse_score FLOAT,
        r2_score FLOAT,
        model_path TEXT,
        hyperparameters JSONB,
        feature_importance JSONB
    );

    -- Create predictions table
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        model_id INTEGER REFERENCES model_training_log(id),
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        input_features JSONB,
        predicted_price FLOAT,
        confidence_interval JSONB
    );

    -- Grant permissions to real estate user
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO real_estate_user;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO real_estate_user;

    -- Create indexes for better performance
    CREATE INDEX IF NOT EXISTS idx_real_estate_price ON real_estate_data(last_price);
    CREATE INDEX IF NOT EXISTS idx_real_estate_area ON real_estate_data(total_area);
    CREATE INDEX IF NOT EXISTS idx_real_estate_rooms ON real_estate_data(rooms);
    CREATE INDEX IF NOT EXISTS idx_processing_log_date ON data_processing_log(processing_date);
    CREATE INDEX IF NOT EXISTS idx_model_training_date ON model_training_log(training_date);
    CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);

EOSQL

echo "Database initialization completed successfully!"
