from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import product, review, insight

app = FastAPI(
    title="AI-Powered Aba Craft Product Intelligence System",
    description=(
        "An intelligent backend system capable of classifying Aba craft products, "
        "analyzing customer feedback sentiment, extracting keywords, and generating actionable business insights."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(product.router)
app.include_router(review.router)
app.include_router(insight.router)

@app.get("/", tags=["Root"])
def root():
    return {
        "project": "AI-Powered Aba Craft Product Intelligence System",
        "status": "online",
        "documentation": "/docs"
    }
