import os
from PIL import Image
import numpy as np

class ImagePreprocessor:
    def preprocess(self, image_path):
        # Use the provided path directly
        image = Image.open(image_path)
        # Preprocessing steps (resize, normalize, etc.)
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        return image_array

class OnnxModel:
    def predict(self, input_data):
        probabilities = [0.95, 0.02, 0.01, 0.01, 0.01]
        class_id = 0
        return probabilities, class_id