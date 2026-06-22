CREATE DATABASE medical_db;

USE medical_db;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100),
    password VARCHAR(100)
);

INSERT INTO users(username,password)
VALUES
('admin','admin123');

CREATE TABLE patient_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,

    blood_pressure INT,
    sugar_level INT,
    heart_rate INT,
    oxygen_level INT,

    symptoms VARCHAR(100),

    predicted_disease VARCHAR(100),

    probability FLOAT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);