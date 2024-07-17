import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
data = pd.read_csv("medical data.csv")

# Verify Column Names


print(data.columns)
data.info()
data.isnull().sum()

# Handling missing values
data["DateOfBirth"] = pd.to_datetime(data["DateOfBirth"], errors="coerce")
median_date = data["DateOfBirth"].dropna().median()
data["DateOfBirth"].fillna(median_date, inplace=True)

categorical_columns = ["Gender", "Symptoms", "Causes", "Disease", "Medicine"]
for column in categorical_columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

data["Name"].fillna("Unknown", inplace=True)

# Encode categorical variables
label_encoders = {}
for column in data.columns:
    if data[column].dtype == "object":
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Exploratory Data Analysis (EDA)
sns.countplot(data=data, x="Medicine")
plt.title("Distribution of Medicine")
plt.show()

numeric_columns = ["DateOfBirth"]
sns.heatmap(data[numeric_columns].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Splitting the dataset
X = data.drop("Medicine", axis=1)
y = data["Medicine"]
X = X.drop("DateOfBirth", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the model and label encoders
joblib.dump(rf_classifier, "rf_classifier.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Model evaluation
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Feature importances from Random Forest
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X.columns[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), names)
plt.gca().invert_yaxis()
plt.show()
