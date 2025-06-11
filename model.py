import onnxruntime
from PIL import Image
import numpy as np

class OnnxModel:
    def __init__(self):
        self.session = onnxruntime.InferenceSession("image_classifier.onnx")

    def predict(self, image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocessing steps based on convert_to_onnx.py
        # The ONNX model itself contains preprocessing layers.
        # It expects an input of shape (N, H, W, C) with H=224, W=224.
        # Values should be raw pixel values (0-255), as float32.

        image = image.resize((224, 224)) # Resize to match dummy_input dimensions
        image_array = np.array(image, dtype=np.float32) # Shape: (H, W, C), values 0-255

        # Add batch dimension. Model expects N, H, W, C.
        image_array = np.expand_dims(image_array, axis=0) # Now (1, H, W, C)

        input_name = self.session.get_inputs()[0].name
        # Ensure input type matches model expectation if issues persist (usually float32)
        # model_input_type = self.session.get_inputs()[0].type # e.g. 'tensor(float)'

        output = self.session.run(None, {input_name: image_array})

        # Output processing might need adjustment if the model's output structure is different
        # For ResNet18, output is typically a single tensor of shape (batch_size, num_classes)
        probabilities = output[0][0] # output[0] is the first output tensor, [0] for the first batch item.
        class_id = np.argmax(probabilities)

        return probabilities, class_id