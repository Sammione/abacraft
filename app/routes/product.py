from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.schemas.api_schemas import ProductPredictionResponse
from app.services.classifier import ProductClassifier

router = APIRouter(tags=["Product Intelligence"])

# Initialize classifier lazily or on load
try:
    classifier = ProductClassifier()
except Exception as e:
    print(f"Warning: ProductClassifier failed to initialize: {e}")
    classifier = None

@router.post("/predict-product", response_model=ProductPredictionResponse)
async def predict_product(file: UploadFile = File(...)):
    """
    Upload an image of an Aba craft product (Bag, Shoe, Clothing, Accessory)
    to classify it.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image."
        )
        
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or configured. Please train the model."
        )

    try:
        contents = await file.read()
        predicted_class, confidence = classifier.predict(contents)
        return ProductPredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )
