from fastapi import FastAPI, File, UploadFile, HTTPException
from model import OnnxModel
import os
import shutil

app = FastAPI()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        os.makedirs("temp_images", exist_ok=True)
        filename = os.path.basename(image.filename)
        temp_image_path = os.path.join("temp_images", filename)
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        model = OnnxModel()
        probabilities, class_id = model.predict(temp_image_path)
        os.remove(temp_image_path)
        result = {"class_id": int(class_id), "probabilities": probabilities}  # Remove .tolist()
        return {"my_result": result, "status_code": 200}
    except Exception as e:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        raise HTTPException(status_code=500, detail=str(e))