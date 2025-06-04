from model import ImagePreprocessor, OnnxModel

def run(image_path: str, run_id: str):
    try:
        preprocessor = ImagePreprocessor()
        model = OnnxModel()
        input_data = preprocessor.preprocess(image_path)
        probabilities, class_id = model.predict(input_data)
        result = {"class_id": int(class_id), "probabilities": probabilities.tolist()}
        return {"my_result": result, "status_code": 200}
    except Exception as e:
        return {"my_result": {"error": str(e)}, "status_code": 500}