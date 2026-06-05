from typing import List, Dict, Any
from collections import Counter
import re
from app.services.nlp import SentimentAnalyzer, KeywordExtractor

class CustomerInsightEngine:
    def __init__(self, sentiment_analyzer: SentimentAnalyzer, keyword_extractor: KeywordExtractor):
        self.sentiment_analyzer = sentiment_analyzer
        self.keyword_extractor = keyword_extractor

    def generate_insights(self, reviews: List[str]) -> Dict[str, Any]:
        """
        Processes a list of customer reviews and extracts:
        - top_complaints (frequent phrases/terms in negative reviews)
        - top_preferences (frequent phrases/terms in positive reviews)
        - trending_keywords (most common terms overall)
        """
        negative_reviews = []
        positive_reviews = []
        all_keywords = []

        for review in reviews:
            if not review.strip():
                continue
            
            # Predict Sentiment
            sentiment = self.sentiment_analyzer.analyze(review)
            
            # Extract Keywords
            keywords = self.keyword_extractor.extract(review)
            all_keywords.extend(keywords)

            # Categorize review sentences/segments
            if sentiment == "Negative":
                # Find phrases like "broke", "too small", "poor quality", "terrible", etc.
                negative_reviews.append(review)
            else:
                positive_reviews.append(review)

        # 1. Top Complaints Extraction
        # Run a simple keyword/phrase frequency on negative reviews
        complaints_counter = Counter()
        for rev in negative_reviews:
            # Look for 2-word phrases (bigrams) or fallback to keywords
            words = self.keyword_extractor.extract(rev)
            for w in words:
                complaints_counter[w] += 1
            # Also extract some simple bigrams to make it sound like real complaints
            raw_words = re.findall(r'\b\w+\b', rev.lower())
            for i in range(len(raw_words) - 1):
                phrase = f"{raw_words[i]} {raw_words[i+1]}"
                # check if contains stop words or not
                if not any(w in ["the", "is", "and", "a", "of", "to", "this", "it", "in", "was"] for w in raw_words[i:i+2]):
                    complaints_counter[phrase] += 1

        # 2. Top Preferences Extraction
        preferences_counter = Counter()
        for rev in positive_reviews:
            words = self.keyword_extractor.extract(rev)
            for w in words:
                preferences_counter[w] += 1
            raw_words = re.findall(r'\b\w+\b', rev.lower())
            for i in range(len(raw_words) - 1):
                phrase = f"{raw_words[i]} {raw_words[i+1]}"
                if not any(w in ["the", "is", "and", "a", "of", "to", "this", "it", "in", "was"] for w in raw_words[i:i+2]):
                    preferences_counter[phrase] += 1

        # Extract top elements
        # Filter out single words from complaints if they are also bigrams
        top_complaints = [item for item, count in complaints_counter.most_common(5) if len(item.split()) > 1]
        if not top_complaints:
            top_complaints = [item for item, count in complaints_counter.most_common(3)]

        top_preferences = [item for item, count in preferences_counter.most_common(5) if len(item.split()) > 1]
        if not top_preferences:
            top_preferences = [item for item, count in preferences_counter.most_common(3)]

        trending_keywords = [item for item, count in Counter(all_keywords).most_common(5)]

        return {
            "top_complaints": top_complaints,
            "top_preferences": top_preferences,
            "trending_keywords": trending_keywords
        }
