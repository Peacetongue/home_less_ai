FROM apache/airflow:2.7.3-python3.10

# Switch to root to install system dependencies
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy and install Python requirements
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install additional Airflow providers
RUN pip install --no-cache-dir \
    apache-airflow-providers-postgres==5.6.0 \
    apache-airflow-providers-celery==3.3.0

# Create necessary directories
RUN mkdir -p /opt/airflow/data/models
