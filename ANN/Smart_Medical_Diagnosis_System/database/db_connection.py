import mysql.connector

def get_connection():

    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Aarthi123",
        database="medical_db"
    )

    return connection