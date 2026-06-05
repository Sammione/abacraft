import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io

class ProductClassifier:
    def __init__(self, model_path: str = "models/product_classifier.keras"):
        self.model_path = model_path
        self.classes = ["Bags", "Shoes", "Clothing", "Accessories"]
        self.model = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please run scripts/train_models.py first."
            )
        # Load keras model
        self.model = tf.keras.models.load_model(self.model_path)
        print("ProductClassifier model loaded successfully.")

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize to 224x224 (as trained)
        image = image.resize((224, 224))
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        if self.model is None:
            self.load_model()
            
        img_array = self.preprocess_image(image_bytes)
        predictions = self.model.predict(img_array)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        
        # Map back to corresponding class name
        predicted_class = self.classes[predicted_idx]
        return predicted_class, confidence
