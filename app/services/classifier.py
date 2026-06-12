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
        # Load MobileNetV2 pretrained on ImageNet
        # This will download the weights the first time if not cached
        self.model = tf.keras.applications.MobileNetV2(weights="imagenet")
        print("Pretrained MobileNetV2 model loaded successfully.")

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        # MobileNetV2 expects inputs in range [-1, 1]
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def map_imagenet_to_category(self, decoded_predictions) -> tuple[str, float]:
        # decoded_predictions is a list of tuples: (class_id, class_name, prob)
        # Keywords for our 4 categories
        category_keywords = {
            "Bags": ["bag", "purse", "backpack", "satchel", "pouch", "wallet", "packet", "suitcase", "briefcase"],
            "Shoes": ["shoe", "sandal", "boot", "sneaker", "loafer", "clog", "sock"],
            "Clothing": ["shirt", "suit", "dress", "skirt", "jacket", "coat", "sweater", "tie", "jersey", "vest", "cardigan", "gown", "cloak", "bikini", "apron", "t-shirt"],
            "Accessories": ["glasses", "watch", "necklace", "belt", "hat", "cap", "helmet", "umbrella", "sunglasses", "chain", "ring"]
        }

        # Check top 5 predictions
        for _, class_name, prob in decoded_predictions[0]:
            class_name_lower = class_name.lower().replace("_", " ")
            
            for category, keywords in category_keywords.items():
                if any(kw in class_name_lower for kw in keywords):
                    return category, float(prob)
        
        # If no match found in top 5, default to Accessories with the top probability
        top_prob = float(decoded_predictions[0][0][2])
        return "Accessories", top_prob

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        if self.model is None:
            self.load_model()
            
        img_array = self.preprocess_image(image_bytes)
        predictions = self.model.predict(img_array)
        
        # Decode top 5 ImageNet predictions
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)
        
        # Map to Aba Craft categories
        predicted_class, confidence = self.map_imagenet_to_category(decoded)
        
        return predicted_class, confidence
