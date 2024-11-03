from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests  # For fetching news from an API
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load the trained model and vectorizer
model = joblib.load('/Users/cvhuy/Fake News Project/fake_news_detection_model.pkl')
vectorizer = joblib.load('/Users/cvhuy/Fake News Project/tfidf_vectorizer.pkl')

app = Flask(__name__)

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def fetch_news(query):
    api_key = 'b9b22c103c4f4aae8793484bc8a45d82'  # Replace with your actual API key
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    return response.json().get('articles', [])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    cleaned_text = preprocess_text(data['content'])
    X_new = vectorizer.transform([cleaned_text])
    prediction = model.predict(X_new)

    return jsonify({
        'prediction': 'Fake News' if prediction[0] == 0 else 'Real News',
        'title': data.get('title'),  # Retrieve title from request
        'url': data.get('url')       # Retrieve URL from request
    })

@app.route('/fetch-news', methods=['GET'])
def get_news():
    query = request.args.get('query')
    articles = fetch_news(query)
    
    # Format the articles for the response
    formatted_articles = []
    for article in articles:
        formatted_articles.append({
            'title': article['title'],
            'url': article['url'],
            'content': article.get('content', '')
        })
    
    return jsonify(formatted_articles)

if __name__ == '__main__':
    app.run(port=5000)
