import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Sample dataset (You can replace with your own CSV)
data = {
    'age': [22, 35, 26, 45, 52, 23, 40, 36, 28, 30],
    'income': [25000, 60000, 35000, 80000, 120000, 28000, 70000, 62000, 40000, 45000],
    'visits_last_month': [5, 1, 3, 2, 0, 7, 1, 2, 3, 4],
    'time_on_site': [30, 5, 15, 7, 2, 40, 8, 10, 12, 20],
    'clicked_ad': [1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
    'purchased': [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# Step 2: Feature selection
X = df[['age', 'income', 'visits_last_month', 'time_on_site', 'clicked_ad']]
y = df['purchased']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Prediction
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Feature importance
importances = model.feature_importances_
feature_names = X.columns
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.show()
