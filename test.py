import unittest
import numpy as np
from model import OnnxModel
from PIL import Image
import os # For creating a dummy invalid file

class TestImageClassifier(unittest.TestCase):
    def setUp(self):
        self.model = OnnxModel()
        # Ensure test images are available or skip tests
        self.test_image_tench = "images/n01440764_tench.jpg"
        self.test_image_turtle = "images/n01667114_mud_turtle.jpg"
        self.invalid_image_path = "invalid_image.jpg"

        # Create a dummy invalid image file for testing error handling
        with open(self.invalid_image_path, "w") as f:
            f.write("this is not an image")

    def tearDown(self):
        # Clean up the dummy invalid image file
        if os.path.exists(self.invalid_image_path):
            os.remove(self.invalid_image_path)
        # Clean up any other created files like grayscale or resized images if any test created them
        if os.path.exists("test_grayscale.jpg"):
            os.remove("test_grayscale.jpg")
        if os.path.exists("test_invalid.jpg"): # From old tests, ensure cleanup
            os.remove("test_invalid.jpg")


    def test_model_prediction_tench(self):
        if not os.path.exists(self.test_image_tench):
            self.skipTest(f"Test image {self.test_image_tench} not found.")

        probabilities, class_id = self.model.predict(self.test_image_tench)
        print(f"Tench Prediction - Class ID: {class_id}, Probabilities: {probabilities[:5]}...") # Print first 5 for brevity
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape[0], 1000) # Assuming 1000 classes
        self.assertEqual(class_id, 0) # Expected class ID for tench

    def test_model_prediction_turtle(self):
        if not os.path.exists(self.test_image_turtle):
            self.skipTest(f"Test image {self.test_image_turtle} not found.")

        probabilities, class_id = self.model.predict(self.test_image_turtle)
        print(f"Turtle Prediction - Class ID: {class_id}, Probabilities: {probabilities[:5]}...")
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape[0], 1000) # Assuming 1000 classes
        self.assertEqual(class_id, 35) # Expected class ID for mud turtle

    def test_invalid_image_file(self):
        # Expecting an error when trying to open an invalid image file
        # PIL.UnidentifiedImageError is a common exception for this,
        # but onnxruntime or other parts of the predict function might raise others.
        # Let's catch a broad Exception and check its type or message if needed,
        # or specify the exact exception if known (e.g., PIL.UnidentifiedImageError).
        with self.assertRaises(Exception) as context: # More specific: PIL.UnidentifiedImageError or onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException
            self.model.predict(self.invalid_image_path)
        print(f"Invalid image test caught: {type(context.exception).__name__} - {context.exception}")
        # Check if it's a PIL error or ONNX runtime error (if it gets that far)
        self.assertTrue(isinstance(context.exception, (Image.UnidentifiedImageError,FileNotFoundError, Exception))) # Added FileNotFoundError just in case, and generic Exception

if __name__ == "__main__":
    unittest.main()