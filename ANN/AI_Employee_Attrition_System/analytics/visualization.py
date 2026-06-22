import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "dataset/employee_attrition.csv"
)

department_attrition = df.groupby(
    "Department"
)["Attrition"].count()

plt.figure(figsize=(8,5))

department_attrition.plot(
    kind="bar"
)

plt.title(
    "Department Wise Employees"
)

plt.xlabel(
    "Department"
)

plt.ylabel(
    "Count"
)

plt.tight_layout()

plt.show()