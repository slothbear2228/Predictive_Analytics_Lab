import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Simulated historical ICC match data (India)
data = {
    'Opponent': ['Australia', 'England', 'Pakistan', 'South Africa', 'New Zealand', 'Australia', 'England', 'Pakistan'],
    'Venue': ['Home', 'Away', 'Neutral', 'Home', 'Away', 'Neutral', 'Home', 'Away'],
    'Toss_Won': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Toss_Decision': ['Bat', 'Bowl', 'Bat', 'Bat', 'Bowl', 'Bowl', 'Bat', 'Bowl'],
    'Match_Format': ['ODI', 'ODI', 'T20', 'T20', 'ODI', 'T20', 'ODI', 'T20'],
    'Result': ['Win', 'Lose', 'Win', 'Lose', 'Win', 'Lose', 'Win', 'Lose']
}

df = pd.DataFrame(data)

# Step 2: Encode categorical variables
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 3: Prepare features and target
X = df.drop('Result', axis=1)
y = df['Result']  # Encoded as 1=Win, 0=Lose

# Step 4: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 5: Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoders['Result'].classes_))

# Step 7: Predict performance for a new match
new_match = {
    'Opponent': 'Australia',
    'Venue': 'Neutral',
    'Toss_Won': 'Yes',
    'Toss_Decision': 'Bat',
    'Match_Format': 'ODI'
}

# Encode new match
new_match_encoded = {col: label_encoders[col].transform([val])[0] for col, val in new_match.items()}

# Predict
new_match_df = pd.DataFrame([new_match_encoded])
pred_encoded = model.predict(new_match_df)[0]
pred_label = label_encoders['Result'].inverse_transform([pred_encoded])[0]

print(f"\nPredicted result for new match: {pred_label}")
