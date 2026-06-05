# API Endpoint Reference — AI-Powered Aba Craft Product Intelligence System

**Base URL (local):** `http://127.0.0.1:8000`  
**Base URL (Docker/Production):** `http://your-server-ip:8000`  
**Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`  
**CORS:** Enabled for all origins  

---

## Endpoint 1 — Classify a Product Image

| Property | Value |
|---|---|
| URL | `/predict-product` |
| Method | `POST` |
| Content-Type | `multipart/form-data` |

### Request
Upload a product image as a form file field named `file`.

### JavaScript (fetch)
```js
const formData = new FormData();
formData.append('file', imageFile); // imageFile = File from <input type="file">

const res = await fetch('http://127.0.0.1:8000/predict-product', {
  method: 'POST',
  body: formData
});
const data = await res.json();
// { "predicted_class": "Bags", "confidence": 0.97 }
```

### Response
```json
{
  "predicted_class": "Bags",
  "confidence": 0.97
}
```
**Classes:** `Bags` | `Shoes` | `Clothing` | `Accessories`

---

## Endpoint 2 — Analyze Review Sentiment

| Property | Value |
|---|---|
| URL | `/analyze-review` |
| Method | `POST` |
| Content-Type | `application/json` |

### Request Body
```json
{
  "review": "The quality is excellent, I love it!"
}
```

### JavaScript (fetch)
```js
const res = await fetch('http://127.0.0.1:8000/analyze-review', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ review: "The quality is excellent!" })
});
const data = await res.json();
// { "sentiment": "Positive" }
```

### Response
```json
{
  "sentiment": "Positive"
}
```
**Values:** `"Positive"` or `"Negative"`

---

## Endpoint 3 — Extract Keywords from Review

| Property | Value |
|---|---|
| URL | `/extract-keywords` |
| Method | `POST` |
| Content-Type | `application/json` |

### Request Body
```json
{
  "review": "The leather bag quality is excellent"
}
```

### JavaScript (fetch)
```js
const res = await fetch('http://127.0.0.1:8000/extract-keywords', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ review: "The leather bag quality is excellent" })
});
const data = await res.json();
// { "keywords": ["leather", "bag", "quality", "excellent"] }
```

### Response
```json
{
  "keywords": ["leather", "bag", "quality", "excellent"]
}
```

---

## Endpoint 4 — Customer Insights (Batch)

| Property | Value |
|---|---|
| URL | `/customer-insights` |
| Method | `POST` |
| Content-Type | `application/json` |

### Request Body
```json
{
  "reviews": [
    "The leather bag is of excellent quality.",
    "Worst customer service, the zipper broke instantly.",
    "I love the fabric and fitting of this dress.",
    "Very comfortable shoes, fits perfectly.",
    "The product arrived damaged. Very disappointed."
  ]
}
```

### JavaScript (fetch)
```js
const res = await fetch('http://127.0.0.1:8000/customer-insights', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ reviews: ["Great bag quality!", "Terrible zipper broke on day one."] })
});
const data = await res.json();
```

### Response
```json
{
  "top_complaints": ["zipper broke", "worst customer", "arrived damaged"],
  "top_preferences": ["leather bag", "excellent quality", "fabric"],
  "trending_keywords": ["leather", "quality", "zipper", "bag", "comfortable"]
}
```

---

## Endpoint 5 — Health Check

| Property | Value |
|---|---|
| URL | `/` |
| Method | `GET` |

### Response
```json
{
  "project": "AI-Powered Aba Craft Product Intelligence System",
  "status": "online",
  "documentation": "/docs"
}
```

---

## cURL Quick Reference

```bash
# Health check
curl http://127.0.0.1:8000/

# Analyze sentiment
curl -X POST http://127.0.0.1:8000/analyze-review \
  -H "Content-Type: application/json" \
  -d '{"review": "The leather bag is excellent quality!"}'

# Extract keywords
curl -X POST http://127.0.0.1:8000/extract-keywords \
  -H "Content-Type: application/json" \
  -d '{"review": "The leather bag quality is excellent"}'

# Customer insights
curl -X POST http://127.0.0.1:8000/customer-insights \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great bag quality!", "Terrible zipper broke on day one."]}'

# Upload image to classify product
curl -X POST http://127.0.0.1:8000/predict-product \
  -F "file=@/path/to/product.jpg"
```

---

## Error Responses

All endpoints return standard HTTP error codes:

| Code | Meaning |
|---|---|
| `200` | Success |
| `400` | Bad request (e.g. non-image file uploaded) |
| `422` | Validation error (wrong request body format) |
| `500` | Internal server error |
| `503` | Model not loaded — run `python scripts/train_models.py` first |
