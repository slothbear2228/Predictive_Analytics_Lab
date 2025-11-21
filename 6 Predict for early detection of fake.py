import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
import string

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 1: Sample Dataset (Replace with your own or use a CSV file)
data = {
    'text': [
        "Breaking news: Scientists discover cure for cancer!",
        "Click here to win a free iPhone now!",
        "Government confirms new economic policy changes.",
        "Lose weight in 7 days with this magical pill!",
        "NASA confirms life on Mars through rover data.",
        "This is not real: Man grows wings after drinking soda.",
        "Elections results: President reelected with 60% votes.",
        "Earn $500 a day working from your bedroom!"
    ],
    'label': [0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Real, 1 = Fake
}

df = pd.DataFrame(data)

# Step 2: Text Cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

df['clean_text'] = df['text'].apply(clean_text)

# Step 3: Feature extraction with TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Predictions and evaluation
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Make a prediction
def predict_fake(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    return "Fake" if pred == 1 else "Real"

# Example usage
new_text = "Win a brand new car just by clicking this link!"
print(f"\nPrediction: '{new_text}' -> {predict_fake(new_text)}")
