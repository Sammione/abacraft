import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io

class ProductClassifier:
    def __init__(self):
        self.classes = ["Bags", "Shoes", "Clothing", "Accessories"]
        self.model = None
        self.load_model()

    def load_model(self):
        # Load EfficientNetB0 pretrained on ImageNet for higher accuracy
        self.model = tf.keras.applications.EfficientNetB0(weights="imagenet")
        print("Pretrained EfficientNetB0 model loaded successfully.")

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        # EfficientNet expects specific preprocessing
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def map_imagenet_to_category(self, decoded_predictions) -> tuple[str, float]:
        # Broadened keywords for better mapping
        category_keywords = {
            "Bags": ["bag", "purse", "backpack", "satchel", "pouch", "wallet", "packet", "suitcase", "briefcase", "basket", "pack", "luggage", "tote"],
            "Shoes": ["shoe", "sandal", "boot", "sneaker", "loafer", "clog", "sock", "footwear"],
            "Clothing": ["shirt", "suit", "dress", "skirt", "jacket", "coat", "sweater", "tie", "jersey", "vest", "cardigan", "gown", "cloak", "bikini", "apron", "t-shirt", "jean", "uniform", "kimono", "pajama", "poncho", "sarong", "windbreaker", "wool", "velvet", "fleece", "garment", "apparel"],
            "Accessories": ["glasses", "watch", "necklace", "belt", "hat", "cap", "helmet", "umbrella", "sunglasses", "chain", "ring", "jewelry", "bracelet", "earring", "glove", "scarf", "sombrero", "bandana", "cravat"]
        }

        # Check top 10 predictions for a higher chance of finding a match
        for _, class_name, prob in decoded_predictions[0]:
            class_name_lower = class_name.lower().replace("_", " ")
            
            for category, keywords in category_keywords.items():
                if any(kw in class_name_lower for kw in keywords):
                    # We return the probability of the ImageNet class we matched against
                    return category, float(prob)
        
        # If no match found in top 10, default to Accessories with the top probability
        top_prob = float(decoded_predictions[0][0][2])
        return "Accessories", top_prob

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        if self.model is None:
            self.load_model()
            
        img_array = self.preprocess_image(image_bytes)
        predictions = self.model.predict(img_array)
        
        # Decode top 10 ImageNet predictions to ensure we catch variations
        decoded = tf.keras.applications.efficientnet.decode_predictions(predictions, top=10)
        
        # Map to Aba Craft categories
        predicted_class, confidence = self.map_imagenet_to_category(decoded)
        
        return predicted_class, confidence
