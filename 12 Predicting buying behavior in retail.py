import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Sample data: age, income, past purchase (1=yes,0=no), and buying behavior (1=will buy, 0=won't buy)
data = {
    'age': [22, 35, 58, 45, 30, 40, 23, 50, 36, 28],
    'income': [30000, 60000, 80000, 70000, 40000, 50000, 32000, 75000, 62000, 41000],
    'past_purchase': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    'will_buy': [0, 1, 1, 0, 1, 1, 0, 1, 0, 0]
}

# Load data into DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['age', 'income', 'past_purchase']]
y = df['will_buy']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict buying behavior for a new customer
new_customer = [[40, 50000, 0]]  # age=40, income=50k, no past purchase
prediction = model.predict(new_customer)

print(f"Prediction for new customer: {'Will Buy' if prediction[0] == 1 else 'Will Not Buy'}")
