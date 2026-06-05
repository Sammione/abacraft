from fastapi import APIRouter, HTTPException, status
from app.schemas.api_schemas import CustomerInsightsRequest, CustomerInsightsResponse
from app.services.nlp import SentimentAnalyzer, KeywordExtractor
from app.services.insights import CustomerInsightEngine

router = APIRouter(tags=["Business Insights Engine"])

try:
    sentiment_analyzer = SentimentAnalyzer()
    keyword_extractor = KeywordExtractor()
    insight_engine = CustomerInsightEngine(sentiment_analyzer, keyword_extractor)
except Exception as e:
    print(f"Warning: Insight engine failed to initialize: {e}")
    insight_engine = None

@router.post("/customer-insights", response_model=CustomerInsightsResponse)
async def get_customer_insights(payload: CustomerInsightsRequest):
    """
    Process multiple reviews to extract top complaints, preferences, 
    and trending keywords across all customer feedback.
    """
    if insight_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insight engine is not ready. Verify models are trained."
        )
    try:
        insights = insight_engine.generate_insights(payload.reviews)
        return CustomerInsightsResponse(**insights)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating insights: {str(e)}"
        )
