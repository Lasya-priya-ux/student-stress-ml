import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report

# Load dataset
data = pd.read_csv("student_behavior_data.csv")

# Linear Regression
features = [
    'Login_Frequency',
    'Time_Spent_Hours',
    'Attendance_Percentage',
    'Assignment_Delay_Days',
    'Quiz_Average',
    'Forum_Interactions',
    'Sleep_Hours'
]

X = data[features]
y = data['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

print("Linear Regression Results")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Decision Tree
def classify_performance(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

data['Performance_Level'] = data['Final_Score'].apply(classify_performance)

X_cls = data[features]
y_cls = data['Performance_Level']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)
dt_model.fit(X_train_cls, y_train_cls)
y_pred_cls = dt_model.predict(X_test_cls)

print("\nDecision Tree Results")
print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print(classification_report(y_test_cls, y_pred_cls))

# Isolation Forest
iso_model = IsolationForest(contamination=0.1, random_state=42)
data['Anomaly'] = iso_model.fit_predict(X)
data['Stress_Risk'] = data['Anomaly'].apply(lambda x: "High Risk" if x == -1 else "Normal")

print("\nIsolation Forest Results")
print(data[['Student_ID', 'Stress_Risk']].head())

# Graph 1
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Score")
plt.show()

# Graph 2
data['Performance_Level'].value_counts().plot(kind='bar')
plt.title("Performance Levels")
plt.show()

# Graph 3
data['Stress_Risk'].value_counts().plot(kind='bar')
plt.title("Stress Risk")
plt.show()