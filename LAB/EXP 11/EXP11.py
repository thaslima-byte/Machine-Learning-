# Experiment: Credit Score Classification

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Create sample credit score dataset
data = {
    'Age': [25, 45, 35, 52, 23, 40, 31, 48, 29, 55, 38, 42, 26, 50, 34],
    'Annual_Income': [30000, 80000, 55000, 90000, 25000, 70000, 45000, 85000, 35000, 95000, 60000, 75000, 28000, 88000, 50000],
    'Credit_Utilization': [0.75, 0.20, 0.40, 0.15, 0.85, 0.25, 0.50, 0.18, 0.65, 0.10, 0.35, 0.22, 0.80, 0.12, 0.45],
    'Payment_History': [2, 9, 7, 10, 1, 8, 6, 9, 3, 10, 7, 8, 2, 9, 6],
    'Existing_Loans': [4, 1, 2, 1, 5, 2, 3, 1, 4, 0, 2, 1, 5, 1, 3],
    'Credit_Score': ['Poor', 'Excellent', 'Good', 'Excellent', 'Poor',
                     'Good', 'Good', 'Excellent', 'Poor', 'Excellent',
                     'Good', 'Excellent', 'Poor', 'Excellent', 'Good']
}

df = pd.DataFrame(data)

# Display dataset
print("Credit Score Dataset:")
print(df)

# Encode target labels
encoder = LabelEncoder()
df['Credit_Score'] = encoder.fit_transform(df['Credit_Score'])

# Separate features and target
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Random Forest classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot(cmap="Blues")
plt.title("Credit Score Classification")
plt.xlabel("Predicted Credit Score")
plt.ylabel("Actual Credit Score")
plt.show()
