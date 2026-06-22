import mysql.connector
from config import *

def get_connection():

    connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Aarthi123",
    database="employee_attrition"
    )

    return connection