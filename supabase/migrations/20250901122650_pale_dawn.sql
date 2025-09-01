-- Initialize the forecasting database
-- This script runs when the PostgreSQL container starts for the first time

-- Create the database if it doesn't exist (though it should be created by POSTGRES_DB)
SELECT 'CREATE DATABASE forecasting_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'forecasting_db');

-- Connect to the forecasting database
\c forecasting_db;

-- Grant all privileges to the postgres user
GRANT ALL PRIVILEGES ON DATABASE forecasting_db TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA public TO postgres;

-- Ensure the postgres user can create tables
ALTER USER postgres CREATEDB;