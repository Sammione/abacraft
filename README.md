# AI-Powered Aba Craft Product Intelligence System

This repository contains the complete production-grade backend for the Capstone project:
**"AI-Powered Aba Craft Product Intelligence System Using Computer Vision, NLP, and FastAPI"**

The system classifies craft products from Aba (Bags, Shoes, Clothing, Accessories) using transfer learning, performs sentiment analysis on customer reviews, extracts key terms, and aggregates actionable business insights.

---

## Project Structure

```
project/
├── app/
│   ├── main.py                 # FastAPI application configuration and entrypoint
│   ├── routes/
│   │   ├── product.py          # /predict-product endpoint
│   │   ├── review.py           # /analyze-review & /extract-keywords endpoints
│   │   └── insight.py          # /customer-insights endpoint
│   ├── services/
│   │   ├── classifier.py       # Reusable ProductClassifier wrapper
│   │   ├── nlp.py              # Reusable SentimentAnalyzer & KeywordExtractor
│   │   └── insights.py         # CustomerInsightEngine aggregator
│   ├── schemas/
│   │   └── api_schemas.py      # Pydantic schema validation
│   └── utils/
│       └── text_preprocessor.py# Text cleaning (stopwords, lemmatizer, tokenization)
│
├── models/
│   ├── product_classifier.keras# Saved Keras model
│   ├── sentiment_model.pkl     # Logistic regression sentiment model
│   └── tfidf_vectorizer.pkl    # Vectorizer model
│
├── notebooks/
│   └── model_training.ipynb    # Walkthrough of dataset and training details
│
├── scripts/
│   └── train_models.py         # Automated model training/serialization
│
├── requirements.txt            # System dependencies
├── Dockerfile                  # Application dockerization file
└── README.md                   # Project documentation
```

---

## Local Setup & Quickstart

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Install the packages using `pip`:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Datasets & Train Models
Since this repository starts fresh, run the helper script to create datasets, train the EfficientNetB0 CV model, and fit the Logistic Regression + TF-IDF model:
```bash
python scripts/train_models.py
```
This command will create the `models/` directory and populate the required files:
- `models/product_classifier.keras`
- `models/sentiment_model.pkl`
- `models/tfidf_vectorizer.pkl`

### 3. Start the FastAPI Server
Run the local development server with:
```bash
uvicorn app.main:app --reload
```
The server will start at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## API Endpoints & Testing

Access the interactive Swagger UI documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test the endpoints directly from the browser.

### Endpoint 1: Classify Product
- **URL**: `/predict-product`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Input**: Form parameter `file` (image upload)
- **Response**:
```json
{
  "predicted_class": "Bags",
  "confidence": 0.97
}
```

**cURL Example**:
```bash
curl -X POST "http://127.0.0.1:8000/predict-product" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@test_image.jpg"
```

### Endpoint 2: Analyze Review Sentiment
- **URL**: `/analyze-review`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Input**:
```json
{
  "review": "The leather bag is of excellent quality and feels durable."
}
```
- **Response**:
```json
{
  "sentiment": "Positive"
}
```

**cURL Example**:
```bash
curl -X POST "http://127.0.0.1:8000/analyze-review" -H "Content-Type: application/json" -d "{\"review\": \"The leather bag is of excellent quality and feels durable.\"}"
```

### Endpoint 3: Extract Keywords
- **URL**: `/extract-keywords`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Input**:
```json
{
  "review": "The leather bag quality is excellent."
}
```
- **Response**:
```json
{
  "keywords": ["leather", "bag", "quality", "excellent"]
}
```

### Endpoint 4: Aggregate Business Insights
- **URL**: `/customer-insights`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Input**:
```json
{
  "reviews": [
    "The leather bag is of excellent quality.",
    "Worst customer service, the zipper broke instantly.",
    "I love the fabric and fitting of this dress."
  ]
}
```
- **Response**:
```json
{
  "top_complaints": ["zipper broke", "worst customer service"],
  "top_preferences": ["leather bag", "excellent quality", "fabric"],
  "trending_keywords": ["leather", "quality", "zipper"]
}
```

---

## Docker Deployment

Build and run the entire application inside a Docker container:

### 1. Build Docker Image
```bash
docker build -t aba-craft-intelligence .
```

### 2. Run Docker Container
```bash
docker run -p 8000:8000 aba-craft-intelligence
```
Visit [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
