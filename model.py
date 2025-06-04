import onnxruntime as ort
import numpy as np
from PIL import Image

class ImagePreprocessor:
    def __init__(self):
        pass

    def preprocess(self, image_path):
        img = Image.open(image_path)
        # Check dimensions before resizing
        if img.size[0] < 224 or img.size[1] < 224:
            raise ValueError(f"Image dimensions too small: got {img.size}, minimum required is (224, 224)")
        # Convert to RGB if not already
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((224, 224), Image.Resampling.LANCZOS)  # Resize to 224x224
        img_array = np.array(img).astype(np.float32)
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Expected image shape (224, 224, 3), got {img_array.shape}")
        return np.expand_dims(img_array, axis=0)

class OnnxModel:
    def __init__(self, model_path="image_classifier.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_data):
        outputs = self.session.run(None, {self.input_name: input_data})[0]
        probabilities = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)
        class_id = np.argmax(probabilities, axis=1)[0]
        return probabilities[0], class_id