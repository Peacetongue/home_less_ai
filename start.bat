@echo off
REM Real Estate AI - Docker Startup Script for Windows

echo Starting Real Estate AI Docker Environment...

REM Create necessary directories
if not exist "dags" mkdir dags
if not exist "logs" mkdir logs  
if not exist "plugins" mkdir plugins
if not exist "data\models" mkdir data\models

REM Initialize Airflow database
echo Initializing Airflow...
docker-compose up airflow-init

REM Start the services
echo Starting services...
docker-compose up -d

REM Wait for services to be ready
echo Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check service status
echo Checking service status...
docker-compose ps

echo.
echo ==================================================
echo Real Estate AI Environment is starting up!
echo ==================================================
echo.
echo Services:
echo - Airflow Web UI: http://localhost:8080
echo   Username: admin
echo   Password: admin
echo.
echo - PostgreSQL Database: localhost:5432
echo   Username: airflow
echo   Password: airflow
echo   Database: airflow
echo.
echo - Real Estate Database: localhost:5432
echo   Username: real_estate_user
echo   Password: real_estate_pass
echo   Database: real_estate_db
echo.
echo - Flower (Celery monitoring): http://localhost:5555
echo   (Run: docker-compose --profile flower up -d)
echo.
echo To view logs: docker-compose logs -f [service_name]
echo To stop: docker-compose down
echo To rebuild: docker-compose build --no-cache
echo.
