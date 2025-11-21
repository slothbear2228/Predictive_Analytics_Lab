import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'Years_at_Company': [1, 3, 5, 2, 6, 4],
    'Performance_Rating': [3, 4, 5, 2, 5, 4],
    'Previous_Promotions': [0, 1, 2, 0, 2, 1],
    'Growth': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes']  # Target variable
}

df = pd.DataFrame(data)

# Encode target variable
le = LabelEncoder()
df['Growth'] = le.fit_transform(df['Growth'])  # No=0, Yes=1

# Features and target
X = df[['Years_at_Company', 'Performance_Rating', 'Previous_Promotions']]
y = df['Growth']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Predict growth for a new employee
new_employee = [[4, 4, 1]]  # Years_at_Company=4, Performance_Rating=4, Previous_Promotions=1
growth_pred = model.predict(new_employee)[0]
growth_label = le.inverse_transform([growth_pred])[0]

print(f"Predicted growth for new employee: {growth_label}")
