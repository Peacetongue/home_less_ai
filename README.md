# Real Estate AI - Price Prediction Pipeline

A comprehensive machine learning pipeline for real estate price prediction using Apache Airflow and PostgreSQL in Docker containers.

## 🏗️ Architecture

This project implements a complete data science pipeline with the following components:

- **Apache Airflow**: Workflow orchestration and task scheduling
- **PostgreSQL**: Data storage and model metadata
- **Python ML Stack**: scikit-learn, pandas, numpy for machine learning
- **Docker**: Containerized deployment for consistency and scalability

## 📊 Features

- **Data Processing Pipeline**: Automated ETL processes for real estate data
- **Machine Learning Pipeline**: Automated model training and evaluation
- **Model Comparison**: Automatic selection of best performing models
- **Data Visualization**: EDA plots and model performance reports
- **Database Integration**: Structured storage of data and model results
- **Workflow Orchestration**: Scheduled and dependency-managed tasks

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available
- Ports 8080 (Airflow), 5432 (PostgreSQL) available

### Starting the Environment

#### On Windows:
```bash
# Run the startup script
start.bat

# Or manually:
docker-compose up airflow-init
docker-compose up -d
```

#### On Linux/Mac:
```bash
# Make the script executable
chmod +x start.sh

# Run the startup script
./start.sh

# Or manually:
docker-compose up airflow-init
docker-compose up -d
```

### Accessing Services

Once started, you can access:

- **Airflow Web UI**: http://localhost:8080
  - Username: `admin`
  - Password: `admin`

- **PostgreSQL Database**: `localhost:5432`
  - Airflow DB: `airflow/airflow`
  - Real Estate DB: `real_estate_user/real_estate_pass@real_estate_db`

- **Flower (Celery Monitoring)**: http://localhost:5555
  ```bash
  docker-compose --profile flower up -d
  ```

## 📁 Project Structure

```
home_less_ai/
├── dags/                          # Airflow DAGs
│   ├── real_estate_pipeline.py    # Data processing pipeline
│   └── ml_pipeline.py             # Machine learning pipeline
├── data/                          # Data files
│   ├── real_estate_data.csv       # Raw data
│   └── models/                    # Trained models
├── config/                        # Configuration files
├── plugins/                       # Airflow plugins
├── logs/                          # Airflow logs
├── plots/                         # Generated visualizations
├── init-scripts/                  # Database initialization
│   └── init-db.sh                 # PostgreSQL setup script
├── docker-compose.yml             # Docker services configuration
├── Dockerfile                     # Custom Airflow image
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
├── start.sh / start.bat          # Startup scripts
└── README.md                     # This file
```

## 🔄 Available Pipelines

### 1. Data Processing Pipeline (`real_estate_pipeline`)
- Data validation and quality checks
- Data preprocessing and cleaning
- Feature engineering
- Database storage
- EDA report generation

**Schedule**: Daily

### 2. Machine Learning Pipeline (`ml_pipeline`)
- Feature extraction and preparation
- Model training (Linear Regression, Random Forest)
- Model comparison and selection
- Performance evaluation
- Model persistence and logging

**Schedule**: Weekly

## 📈 Data Schema

### Real Estate Data Table
```sql
CREATE TABLE real_estate_data (
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
```

### Model Training Log
```sql
CREATE TABLE model_training_log (
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
```

## 🛠️ Development

### Adding New DAGs

1. Create new Python files in the `dags/` directory
2. Follow Airflow DAG patterns
3. Use the shared database connections and utilities

### Customizing Models

1. Modify `ml_pipeline.py`
2. Add new model training functions
3. Update model comparison logic

### Database Operations

Connect to PostgreSQL:
```bash
# Connect to Airflow database
docker exec -it <postgres_container> psql -U airflow -d airflow

# Connect to Real Estate database
docker exec -it <postgres_container> psql -U real_estate_user -d real_estate_db
```

## 📊 Monitoring

### Airflow Web Interface
- View DAG runs and task status
- Monitor execution logs
- Trigger manual runs
- View task dependencies

### Database Monitoring
- Query model performance history
- Track data processing statistics
- Monitor data quality metrics

### Logs
```bash
# View all services
docker-compose logs -f

# View specific service
docker-compose logs -f airflow-webserver
docker-compose logs -f postgres
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Airflow settings
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin

# Database settings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=real_estate_db
```

### Custom Python Packages
Add to `requirements.txt` and rebuild:
```bash
docker-compose build --no-cache
docker-compose up -d
```

## 🧪 Testing

### Running EDA Manually
```bash
# Connect to airflow container
docker exec -it <airflow_container> bash

# Run EDA script
cd /opt/airflow/workspace
python eda.py
```

### Testing DAGs
```bash
# Test DAG parsing
docker exec -it <airflow_container> airflow dags list

# Test specific task
docker exec -it <airflow_container> airflow tasks test real_estate_pipeline validate_data 2024-01-01
```

## 📚 Additional Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in `docker-compose.yml`
2. **Memory issues**: Increase Docker memory limit
3. **Permission errors**: Check file permissions and AIRFLOW_UID
4. **Database connection**: Verify PostgreSQL is running and accessible

### Cleanup
```bash
# Stop all services
docker-compose down

# Remove volumes (⚠️ deletes all data)
docker-compose down -v

# Remove all containers and images
docker system prune -a
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.