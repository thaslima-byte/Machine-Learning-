# Experiment: Naïve Bayes Classification for Bank Loan Prediction
print("J THASLIMA NASREEN/192424152")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Create bank loan dataset
data = {
    'Age': [25, 35, 45, 28, 50, 40, 30, 55, 42, 29, 48, 33, 52, 27, 38, 46, 31, 57, 36, 44],
    'Income': [25000, 50000, 80000, 30000, 90000, 65000, 40000, 95000, 70000, 28000,
               85000, 45000, 100000, 32000, 55000, 75000, 42000, 110000, 60000, 72000],
    'LoanAmount': [10000, 20000, 30000, 15000, 25000, 22000, 18000, 30000, 20000, 12000,
                   28000, 18000, 35000, 14000, 19000, 25000, 17000, 40000, 21000, 26000],
    'CreditScore': [600, 700, 750, 620, 780, 720, 650, 800, 730, 610,
                    770, 680, 810, 630, 690, 740, 660, 820, 710, 760],
    'Loan_Status': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No',
                    'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# Display dataset
print("Bank Loan Dataset:")
print(df)

# Encode target variable
encoder = LabelEncoder()
df['Loan_Status'] = encoder.fit_transform(df['Loan_Status'])

# Separate input and output
X = df[['Age', 'Income', 'LoanAmount', 'CreditScore']]
y = df['Loan_Status']

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Naïve Bayes classifier
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Display actual and predicted values
print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

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
plt.title("Naïve Bayes - Bank Loan Prediction")
plt.xlabel("Predicted Loan Status")
plt.ylabel("Actual Loan Status")
plt.show()
