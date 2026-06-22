import pandas as pd

df = pd.read_csv(
    "dataset/employee_attrition.csv"
)

correlation = df[
[
"MonthlyIncome",
"JobSatisfaction",
"YearsAtCompany",
"PerformanceRating"
]
].corr()

print(correlation)