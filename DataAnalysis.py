#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading
import pandas as pd

file_path = r"C:\Users\Karthika Suresh\Downloads\cursor\_\resources\app\extensions\vscode-jupyter-keymap\student-mat.csv"


df = pd.read_csv(file_path, sep=';')

# Display the first few rows to verify
print(df.head())


# 2. Data Exploration
# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Display column data types
print("\nColumn Data Types:\n", df.dtypes)

# Understand dataset size
print("\nDataset Shape:", df.shape)

# 3. Data Cleaning
# Handle missing values
if df.isnull().sum().sum() > 0:
    df = df.fillna(df.median(numeric_only=True))

# Remove duplicate entries
df = df.drop_duplicates()

# 4. Data Analysis
# 1. Average final grade (G3)
avg_g3 = df['G3'].mean()
print("\nAverage Final Grade (G3):", avg_g3)

# 2. Students scoring above 15 in final grade
above_15_count = (df['G3'] > 15).sum()
print("\nStudents scoring above 15 in G3:", above_15_count)

# 3. Correlation between study time and final grade
correlation = df[['studytime', 'G3']].corr().iloc[0, 1]
print("\nCorrelation between study time and final grade:", correlation)

# 4. Average final grade by gender
gender_avg_g3 = df.groupby('sex')['G3'].mean()
print("\nAverage Final Grade by Gender:\n", gender_avg_g3)

# 5. Visualization
plt.figure(figsize=(12, 6))

# Distribution of Final Grades
plt.subplot(1, 2, 1)
sns.histplot(df['G3'], bins=10, kde=True, color='blue')
plt.title("Distribution of Final Grades (G3)")

plt.xlabel("Final Grade (G3)")

# Study Time vs. Final Grade
plt.subplot(1, 2, 2)
sns.scatterplot(x=df['studytime'], y=df['G3'], hue=df['sex'])
plt.title("Study Time vs. Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")

plt.tight_layout()
plt.show()

