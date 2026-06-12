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


# 1. SETUP DIRECTORIES
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("Directories initialized.")

# 2. GENERATE SYNTHETIC DATA
# Generate 16 dummy images (4 per class: Bags, Shoes, Clothing, Accessories)
# Image size: 224x224x3 (EfficientNet standard input size)
CLASSES = ["Bags", "Shoes", "Clothing", "Accessories"]
NUM_CLASSES = len(CLASSES)

print("Generating synthetic image data...")
x_train_img = []
y_train_img = []

for i, cls in enumerate(CLASSES):
    for _ in range(8):  # 8 images per class
        # Create a simple synthetic image with a unique color pattern per class to allow learning
        img = np.zeros((224, 224, 3), dtype=np.float32)
        # Class-dependent pattern
        if cls == "Bags":
            img[50:170, 50:170, 0] = 1.0  # Red square
        elif cls == "Shoes":
            img[50:170, 50:170, 1] = 1.0  # Green square
        elif cls == "Clothing":
            img[50:170, 50:170, 2] = 1.0  # Blue square
        elif cls == "Accessories":
            img[50:170, 50:170, 0:2] = 1.0  # Yellow square
            
        # Add random noise
        img += np.random.normal(0, 0.05, img.shape)
        img = np.clip(img, 0.0, 1.0)
        x_train_img.append(img)
        y_train_img.append(i)

x_train_img = np.array(x_train_img)
y_train_img = np.array(y_train_img)

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

# 4. TRAIN CV MODEL (Transfer Learning with EfficientNetB0)
print("Building and training Product Classifier (EfficientNetB0)...")
# Note: we use weights=None to avoid slow external downloads.
base_model = applications.EfficientNetB0(
    weights=None, 
    include_top=False, 
    input_shape=(224, 224, 3)
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train for 2 epochs just to initialize weights and check-point
model.fit(
    x_train_img, 
    y_train_img, 
    epochs=3, 
    batch_size=8,
    verbose=1
)

# Save Keras Model
model_path = "models/product_classifier.keras"
model.save(model_path)
print(f"Product Classifier model saved to {model_path}")

# 5. TRAIN NLP MODEL (TF-IDF + Logistic Regression)
print("Training Sentiment Analyzer...")
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df_reviews['cleaned_review'])
y_sentiment = df_reviews['sentiment'].values

sentiment_model = LogisticRegression()
sentiment_model.fit(X_tfidf, y_sentiment)

# Save Vectorizer and Sentiment Model
vectorizer_path = "models/tfidf_vectorizer.pkl"
model_pkl_path = "models/sentiment_model.pkl"

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

with open(model_pkl_path, 'wb') as f:
    pickle.dump(sentiment_model, f)

print(f"Sentiment model saved to {model_pkl_path}")
print(f"TF-IDF vectorizer saved to {vectorizer_path}")
print("Training process completed successfully.")
