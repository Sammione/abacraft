import os
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    print("Testing GET / ...")
    response = client.get("/")
    assert response.status_code == 200
    print("Root response:", response.json())

def test_predict_product():
    print("\nTesting POST /predict-product ...")
    # Generate a dummy image in memory
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    files = {"file": ("test_image.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/predict-product", files=files)
    assert response.status_code == 200
    print("Predict response:", response.json())

def test_analyze_review():
    print("\nTesting POST /analyze-review ...")
    payload = {"review": "The leather bag is of excellent quality, I love it!"}
    response = client.post("/analyze-review", json=payload)
    assert response.status_code == 200
    print("Analyze review response:", response.json())

def test_extract_keywords():
    print("\nTesting POST /extract-keywords ...")
    payload = {"review": "The leather bag is of excellent quality, I love it!"}
    response = client.post("/extract-keywords", json=payload)
    assert response.status_code == 200
    print("Extract keywords response:", response.json())

def test_customer_insights():
    print("\nTesting POST /customer-insights ...")
    payload = {
        "reviews": [
            "The leather bag is of excellent quality.",
            "Worst customer service, the zipper broke instantly.",
            "I love the fabric and fitting of this dress."
        ]
    }
    response = client.post("/customer-insights", json=payload)
    assert response.status_code == 200
    print("Customer insights response:", response.json())

if __name__ == "__main__":
    print("Starting API tests...")
    test_root()
    test_predict_product()
    test_analyze_review()
    test_extract_keywords()
    test_customer_insights()
    print("\nAll tests completed successfully!")
