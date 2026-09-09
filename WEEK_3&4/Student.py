import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# csv read
df = pd.read_csv("Student.csv")

# data print 
print("Student Data")
print(df)

# average marks
average = np.mean(df["Marks"])
print("Average Marks =", average)

# highest marks
highest = df["Marks"].max()
print("Highest Marks =", highest)

# lowest marks
lowest = df["Marks"].min()
print("Lowest Marks =", lowest)

# List Comprehension

bonus_marks = [mark + 5 for mark in df["Marks"]]

print("\nBonus Marks:")
print(bonus_marks)
# Dictionary Comprehension

student_dict = {name: marks for name, marks in zip(df["Name"], df["Marks"])}

print("\nStudent Dictionary:")
print(student_dict)

# Function

def grade(marks):
    if marks >= 90:
        return "A"

    elif marks >= 75:
        return "B"

    else:
        return "C"


print("\nGrade of Nandini:")
print(grade(95))


# OOP (Class and Object)

class Student:

    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def display(self):
        print("\nStudent Details")
        print("Name:", self.name)
        print("Marks:", self.marks)

student1 = Student("Nandini", 95)

student1.display()
# Matplotlib Visualization

plt.figure(figsize=(8, 5))

plt.bar(df["Name"], df["Marks"])

plt.title("Student Marks")

plt.xlabel("Students")

plt.ylabel("Marks")

plt.show() 


# Seaborn Visualization

plt.figure(figsize=(8, 5))

sns.barplot(data=df, x="Name", y="Marks")

plt.title("Student Marks using Seaborn")

plt.xlabel("Students")

plt.ylabel("Marks")

plt.show()
