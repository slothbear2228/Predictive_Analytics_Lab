import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Sample network traffic data (packet_size in bytes, connection_duration in seconds)
data = {
    'packet_size': [500, 600, 1500, 2000, 100, 400, 1800, 1200, 300, 2500],
    'connection_duration': [30, 45, 10, 5, 100, 40, 8, 15, 70, 3],
    'intrusion': [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]  # 0 = normal, 1 = intrusion
}

# Load data into a DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['packet_size', 'connection_duration']]
y = df['intrusion']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example: Predict intrusion on new network traffic data
new_traffic_samples = [
    [1600, 7],   # likely intrusion
    [350, 50],   # likely normal
    [2200, 3],   # likely intrusion
]

predictions = model.predict(new_traffic_samples)

for sample, pred in zip(new_traffic_samples, predictions):
    status = "Intrusion" if pred == 1 else "Normal"
    print(f"Traffic with packet_size={sample[0]} and duration={sample[1]} is predicted as: {status}")
