import pandas as pd

df = pd.read_csv(
    "dataset/student_performance.csv"
)

total_students = len(df)

pass_count = len(
    df[df["Result"] == "Pass"]
)

fail_count = len(
    df[df["Result"] == "Fail"]
)

pass_percentage = round(
    (pass_count / total_students) * 100,
    2
)

avg_attendance = round(
    df["Attendance"].mean(),
    2
)

avg_internal = round(
    df["InternalMarks"].mean(),
    2
)

top_rank_students = len(
    df[df["Rank"] <= 10]
)

print("\n====================")
print("FACULTY DASHBOARD")
print("====================")

print(
    "Total Students :",
    total_students
)

print(
    "Pass Count :",
    pass_count
)

print(
    "Fail Count :",
    fail_count
)

print(
    "Pass Percentage :",
    pass_percentage,
    "%"
)

print(
    "Average Attendance :",
    avg_attendance
)

print(
    "Average Internal Marks :",
    avg_internal
)

print(
    "Top Rank Students :",
    top_rank_students
)