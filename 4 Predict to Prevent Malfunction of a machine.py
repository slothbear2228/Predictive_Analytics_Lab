import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Sample machine sensor dataset
# 1 = malfunction, 0 = normal
data = {
    'temperature': [70, 85, 90, 95, 100, 65, 75, 110, 80, 105],
    'vibration': [0.2, 0.5, 0.6, 0.7, 1.0, 0.1, 0.3, 1.2, 0.4, 1.1],
    'sound_level': [30, 40, 45, 50, 60, 25, 35, 70, 38, 65],
    'rpm': [1000, 1100, 1200, 1300, 1400, 950, 1050, 1600, 1150, 1500],
    'malfunction': [0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Step 2: Feature and target split
X = df[['temperature', 'vibration', 'sound_level', 'rpm']]
y = df['malfunction']

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = model.predict(X_test)

# Step 6: Evaluate performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns
sns.barplot(x=importances, y=feature_names)
plt.title("Sensor Feature Importance")
plt.show()
