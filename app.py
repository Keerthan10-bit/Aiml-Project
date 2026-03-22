import pandas as pd
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import streamlit as st

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv("propaganda_data.csv")  # change filename if needed

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Apply cleaning
data['text'] = data['text'].astype(str)
data['clean_text'] = data['text'].apply(clean_text)

# Features and labels
X = data['clean_text']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("AI Manipulative Advertisement Detector")

st.write(f"Model Accuracy: {round(accuracy*100, 2)}%")

user_input = st.text_area("Enter Advertisement Text")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    st.subheader("Result:")
    st.write(prediction[0])