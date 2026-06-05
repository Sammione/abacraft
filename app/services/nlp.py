import os
import pickle
import nltk
from app.utils.text_preprocessor import preprocess_text

class SentimentAnalyzer:
    def __init__(
        self, 
        model_path: str = "models/sentiment_model.pkl", 
        vectorizer_path: str = "models/tfidf_vectorizer.pkl"
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.load_models()

    def load_models(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(
                "NLP models not found. Please run scripts/train_models.py first."
            )
            
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        print("SentimentAnalyzer and TF-IDF Vectorizer loaded successfully.")

    def analyze(self, review: str) -> str:
        if self.model is None or self.vectorizer is None:
            self.load_models()
            
        cleaned = preprocess_text(review)
        if not cleaned:
            return "Neutral"
            
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        return "Positive" if prediction == 1 else "Negative"


class KeywordExtractor:
    def __init__(self):
        # Ensure POS tagger resources are downloaded if we want to be fancy, or keep it basic
        try:
            nltk.data.find('help/tagsets')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    def extract(self, review: str) -> list[str]:
        # Preprocess text to get lemmatized/cleaned tokens
        cleaned = preprocess_text(review)
        tokens = cleaned.split()
        
        # Optionally filter for unique keywords and maintain order
        seen = set()
        keywords = []
        for token in tokens:
            if token not in seen and len(token) > 2:  # Avoid very short words
                seen.add(token)
                keywords.append(token)
        return keywords
