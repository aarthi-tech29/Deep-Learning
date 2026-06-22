attendance = int(
    input("Attendance (%): ")
)

internal_marks = int(
    input("Internal Marks: ")
)

assignment = int(
    input("Assignment Completion (%): ")
)

lab = int(
    input("Lab Performance (%): ")
)

print("\n======================")
print("AI RECOMMENDATIONS")
print("======================")

if attendance < 75:

    print(
        "- Improve Attendance"
    )

if internal_marks < 50:

    print(
        "- Focus on Theory Subjects"
    )

if assignment < 70:

    print(
        "- Complete Assignments Regularly"
    )

if lab < 50:

    print(
        "- Practice More Lab Sessions"
    )

if (
    attendance >= 75
    and internal_marks >= 50
    and assignment >= 70
    and lab >= 50
):

    print(
        "- Excellent Performance. Keep It Up!"
    )
    
# Attendance (%): 92
# Internal Marks: 88
# Assignment Completion (%): 95
# Lab Performance (%): 90

# Attendance (%): 55
# Internal Marks: 42
# Assignment Completion (%): 50
# Lab Performance (%): 45