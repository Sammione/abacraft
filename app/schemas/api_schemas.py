from pydantic import BaseModel, Field
from typing import List

class ProductPredictionResponse(BaseModel):
    predicted_class: str = Field(..., example="Bag")
    confidence: float = Field(..., example=0.97)

class ReviewAnalysisRequest(BaseModel):
    review: str = Field(..., example="The quality is excellent")

class ReviewAnalysisResponse(BaseModel):
    sentiment: str = Field(..., example="Positive")

class KeywordExtractionRequest(BaseModel):
    review: str = Field(..., example="The leather bag quality is excellent")

class KeywordExtractionResponse(BaseModel):
    keywords: List[str] = Field(..., example=["leather", "bag", "quality"])

class CustomerInsightsRequest(BaseModel):
    reviews: List[str] = Field(
        ..., 
        example=[
            "The leather bag is of excellent quality.",
            "Worst customer service, the zipper broke instantly.",
            "I love the fabric and fitting of this dress."
        ]
    )

class CustomerInsightsResponse(BaseModel):
    top_complaints: List[str] = Field(..., example=["zipper broke", "worst customer service"])
    top_preferences: List[str] = Field(..., example=["leather bag", "excellent quality", "fabric"])
    trending_keywords: List[str] = Field(..., example=["leather", "quality", "zipper"])
