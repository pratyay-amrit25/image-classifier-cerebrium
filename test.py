import unittest
import numpy as np
from model import ImagePreprocessor, OnnxModel
from PIL import Image

class TestImageClassifier(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        self.model = OnnxModel()
        self.test_image = "images/n01440764_tench.jpg"
        self.test_image_turtle = "images/n01667114_mud_turtle.jpg"

    def test_preprocessor_rgb(self):
        img_array = self.preprocessor.preprocess(self.test_image)
        self.assertEqual(img_array.shape, (1, 224, 224, 3))
        self.assertEqual(img_array.dtype, np.float32)

    def test_preprocessor_grayscale(self):
        img = Image.open(self.test_image).convert("L")
        img.save("test_grayscale.jpg")
        img_array = self.preprocessor.preprocess("test_grayscale.jpg")
        self.assertEqual(img_array.shape, (1, 224, 224, 3))

    def test_preprocessor_invalid_shape(self):
        img = Image.new("RGB", (100, 100))
        img.save("test_invalid.jpg")
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess("test_invalid.jpg")

    def test_model_prediction(self):
        img_array = self.preprocessor.preprocess(self.test_image)
        probs, class_id = self.model.predict(img_array)
        self.assertEqual(probs.shape, (1000,))
        self.assertEqual(class_id, 0)

    def test_model_prediction_turtle(self):
        img_array = self.preprocessor.preprocess(self.test_image_turtle)
        probs, class_id = self.model.predict(img_array)
        self.assertEqual(class_id, 35)

if __name__ == "__main__":
    unittest.main()