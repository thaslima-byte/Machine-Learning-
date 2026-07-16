# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Read dataset
data = pd.read_csv("/content/DT 3 - Sheet1.csv")

# Encode categorical data
encoders = {}

for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    encoders[column] = le

# Features and Target
X = data.drop("Watch Movie", axis=1)
y = data["Watch Movie"]

import numpy as np

# Function to calculate Entropy
def entropy(target):
    values, counts = np.unique(target, return_counts=True)
    ent = 0

    for count in counts:
        p = count / len(target)
        ent -= p * np.log2(p)

    return ent

# Function to calculate Information Gain
def information_gain(data, attribute, target):

    total_entropy = entropy(data[target])

    values, counts = np.unique(data[attribute], return_counts=True)

    weighted_entropy = 0

    for value, count in zip(values, counts):
        subset = data[data[attribute] == value]
        weighted_entropy += (count / len(data)) * entropy(subset[target])

    gain = total_entropy - weighted_entropy

    return gain

print("\nEntropy of Dataset =", round(entropy(data["Watch Movie"]),4))

print("\nInformation Gain")
for column in X.columns:
    print(column, "=", round(information_gain(data, column, "Watch Movie"),4))
# -------------------- END --------------------


# Train Decision Tree using ID3 (Entropy)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

# Train Decision Tree using ID3 (Entropy)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)


print("\n===== Movie Recommendation System =====")

# Get user input
print("\n===== Movie Recommendation System =====")

print("\nSelect Age")
print("1. Teen")
print("2. Adult")
print("3. Senior")
age = int(input("Enter your choice: "))

print("\nSelect Genre")
print("1. Action")
print("2. Comedy")
print("3. Drama")
print("4. Horror")
print("5. Romance")
print("6. Sci-Fi")
genre = int(input("Enter your choice: "))

print("\nIs it Weekend?")
print("1. Yes")
print("2. No")
weekend = int(input("Enter your choice: "))

print("\nSelect Mood")
print("1. Happy")
print("2. Excited")
print("3. Relaxed")
print("4. Bored")
print("5. Tired")
print("6. Scared")
mood = int(input("Enter your choice: "))

# Convert numbers to text
age_dict = {1:"Teen", 2:"Adult", 3:"Senior"}
genre_dict = {
    1:"Action",
    2:"Comedy",
    3:"Drama",
    4:"Horror",
    5:"Romance",
    6:"Sci-Fi"
}
weekend_dict = {1:"Yes", 2:"No"}
mood_dict = {
    1:"Happy",
    2:"Excited",
    3:"Relaxed",
    4:"Bored",
    5:"Tired",
    6:"Scared"
}

user = [[
    encoders["Age"].transform([age_dict[age]])[0],
    encoders["Genre"].transform([genre_dict[genre]])[0],
    encoders["Weekend"].transform([weekend_dict[weekend]])[0],
    encoders["Mood"].transform([mood_dict[mood]])[0]
]]

prediction = model.predict(user)
result = encoders["Watch Movie"].inverse_transform(prediction)

print("\nRecommendation :", result[0])
