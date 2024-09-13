-- Check if the database already exists
SELECT datname FROM pg_database WHERE datname = 'dipalda';

-- If the database does not exist, create it
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'dipalda') THEN
        CREATE DATABASE dipalda;
    END IF;
END $$;

-- Connect to the main database
\c dipalda;

-- Enable the PostGIS extension
CREATE EXTENSION IF NOT EXISTS POSTGIS;
CREATE EXTENSION IF NOT EXISTS dblink;

