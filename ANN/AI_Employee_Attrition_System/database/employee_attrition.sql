CREATE DATABASE employee_attrition;

USE employee_attrition;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100),
    password VARCHAR(255)
);

INSERT INTO users(username,password)
VALUES('admin','admin123');

CREATE TABLE employees (
    id INT AUTO_INCREMENT PRIMARY KEY,

    age INT,
    monthly_income INT,
    job_satisfaction INT,
    years_at_company INT,
    performance_rating INT,

    department VARCHAR(100),

    attrition VARCHAR(10)
);