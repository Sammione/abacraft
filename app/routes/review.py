from fastapi import APIRouter, HTTPException, status
from app.schemas.api_schemas import (
    ReviewAnalysisRequest, 
    ReviewAnalysisResponse,
    KeywordExtractionRequest,
    KeywordExtractionResponse
)
from app.services.nlp import SentimentAnalyzer, KeywordExtractor

router = APIRouter(tags=["Review NLP Intelligence"])

# Initialize models
try:
    sentiment_analyzer = SentimentAnalyzer()
    keyword_extractor = KeywordExtractor()
except Exception as e:
    print(f"Warning: NLP models failed to initialize: {e}")
    sentiment_analyzer = None
    keyword_extractor = None

@router.post("/analyze-review", response_model=ReviewAnalysisResponse)
async def analyze_review(payload: ReviewAnalysisRequest):
    """
    Analyze the sentiment of a customer review.
    Returns: 'Positive' or 'Negative'.
    """
    if sentiment_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analysis model is not loaded. Please train the model first."
        )
    try:
        sentiment = sentiment_analyzer.analyze(payload.review)
        return ReviewAnalysisResponse(sentiment=sentiment)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing review: {str(e)}"
        )

@router.post("/extract-keywords", response_model=KeywordExtractionResponse)
async def extract_keywords(payload: KeywordExtractionRequest):
    """
    Extract product-related keywords from a customer review.
    """
    if keyword_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Keyword extraction engine is not available."
        )
    try:
        keywords = keyword_extractor.extract(payload.review)
        return KeywordExtractionResponse(keywords=keywords)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting keywords: {str(e)}"
        )
