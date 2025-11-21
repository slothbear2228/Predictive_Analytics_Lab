import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Simulated dataset of behavioral symptoms and sickness
data = {
    'fever': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],          # 1 = yes, 0 = no
    'cough': [1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    'fatigue': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    'headache': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    'loss_of_appetite': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    'sick': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]           # Target variable: 1 = sick, 0 = healthy
}

df = pd.DataFrame(data)

# Step 2: Feature and target separation
X = df.drop('sick', axis=1)
y = df['sick']

# Step 3: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Feature importance visualization
importances = model.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title('Feature Importance in Predicting Sickness')
plt.xlabel('Importance Score')
plt.ylabel('Behavioral Symptom')
plt.tight_layout()
plt.show()
