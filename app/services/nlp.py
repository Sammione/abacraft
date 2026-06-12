import os
import pickle
import nltk
from app.utils.text_preprocessor import preprocess_text

class SentimentAnalyzer:
    def __init__(self):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()
        print("Pretrained VADER SentimentAnalyzer loaded successfully.")

    def analyze(self, review: str) -> str:
        # VADER handles punctuation, capitalization, and emojis natively
        scores = self.analyzer.polarity_scores(review)
        
        # Determine sentiment based on compound score
        if scores['compound'] >= 0.05:
            return "Positive"
        elif scores['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"


class KeywordExtractor:
    def __init__(self):
        print("KeywordExtractor with POS tagging initialized.")

    def extract(self, review: str) -> list[str]:
        # Tokenize and tag parts of speech
        tokens = nltk.word_tokenize(review.lower())
        tags = nltk.pos_tag(tokens)
        
        # We want to extract Nouns (NN*) and Adjectives (JJ*) as key product features/descriptors
        allowed_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}
        
        seen = set()
        keywords = []
        for word, tag in tags:
            # Filter out short words and non-alphabetic tokens
            if tag in allowed_tags and len(word) > 2 and word.isalpha():
                if word not in seen:
                    seen.add(word)
                    keywords.append(word)
        return keywords
