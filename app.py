import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.util import pr
stemmer = nltk.SnowballStemmer("english")
import string
from flask import Flask, render_template, request, jsonify

stopwords_set = set(stopwords.words("english"))

df = pd.read_csv("labeled_data.csv")
print(df.head())

df['labels'] = df['class'].map({0: "Hate speech detected", 1: "Offensive language detected", 2: "No hate speech or offensive language detected"})
print(df.head())

df = df[['tweet', 'labels']]
df.head()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[' + re.escape(string.punctuation) + ']', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split(" ") if word not in stopwords_set]
    text = " ".join(text)
    return text

df['tweet'] = df['tweet'].apply(preprocess)
print(df.head())

x = np.array(df['tweet'])
y = np.array(df["labels"])

cv = CountVectorizer()
x = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_text = request.form['user_text']
        preprocessed_text = preprocess(user_text)
        vectorized_text = cv.transform([preprocessed_text]).toarray()
        prediction = clf.predict(vectorized_text)
        result = str(prediction[0])
        return jsonify({'prediction': result})
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=False)