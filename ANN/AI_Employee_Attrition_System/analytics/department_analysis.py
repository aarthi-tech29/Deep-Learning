import pandas as pd

df = pd.read_csv(
    "dataset/employee_attrition.csv"
)

analysis = df.groupby(
    "Department"
)["Attrition"].value_counts()

print(analysis)