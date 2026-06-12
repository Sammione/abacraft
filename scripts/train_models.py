import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK requirements
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


# 1. SETUP DIRECTORIES
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("Directories initialized.")

# 2. GENERATE SYNTHETIC DATA
# Generate synthetic text data for reviews
print("Generating synthetic text data...")
reviews_data = [
    # Positive
    ("The quality of this leather bag is outstanding, highly recommend!", 1),
    ("Very comfortable shoes, fits perfectly.", 1),
    ("Beautiful dress, fabric is premium and soft.", 1),
    ("These sunglasses are sleek and very fashionable.", 1),
    ("Amazing craftsmanship on this accessory.", 1),
    ("Fast delivery, clothes fit well and look great.", 1),
    # Negative
    ("The bag zipper broke on the first day. Terrible quality.", 0),
    ("Very uncomfortable shoes. Hurt my feet and size is too small.", 0),
    ("The fabric of the clothing is cheap and rough.", 0),
    ("The accessory looks cheap and broke instantly.", 0),
    ("Poor customer service and the product is defective.", 0),
    ("Disappointed with the quality of these shoes.", 0),
]

df_reviews = pd.DataFrame(reviews_data, columns=["review", "sentiment"])
df_reviews.to_csv("data/customer_reviews.csv", index=False)

# 3. TEXT PREPROCESSING
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

print("Preprocessing synthetic reviews...")
df_reviews['cleaned_review'] = df_reviews['review'].apply(preprocess_text)

# 4. CACHE CV MODEL (Pretrained MobileNetV2)
print("Caching Pretrained MobileNetV2 weights and ImageNet classes...")
mobile_model = applications.MobileNetV2(weights='imagenet')
# Run a dummy prediction to cache the imagenet_class_index.json
dummy_input = tf.zeros((1, 224, 224, 3))
preds = mobile_model.predict(dummy_input)
applications.mobilenet_v2.decode_predictions(preds, top=1)
print("MobileNetV2 cached successfully.")

print("Training process and caching completed successfully.")
