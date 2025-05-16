from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import io

MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

class_names = [
    'Tomato__Late_blight', 'Tomato_healthy', 'Grape__healthy',
    'Orange__Haunglongbing(Citrus_greening)', 'Soybean___healthy',
    'Squash__Powdery_mildew', 'Potato__healthy',
    'Corn_(maize)__Northern_Leaf_Blight', 'Tomato__Early_blight',
    'Tomato__Septoria_leaf_spot', 'Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Strawberry__Leaf_scorch', 'Peach_healthy', 'Apple__Apple_scab',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Bacterial_spot',
    'Apple__Black_rot', 'Blueberry__healthy',
    'Cherry_(including_sour)__Powdery_mildew', 'Peach__Bacterial_spot',
    'Apple__Cedar_apple_rust', 'Tomato_Target_Spot', 'Pepper,_bell__healthy',
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Potato___Late_blight',
    'Tomato__Tomato_mosaic_virus', 'Strawberry_healthy', 'Apple__healthy',
    'Grape__Black_rot', 'Potato_Early_blight', 'Cherry(including_sour)___healthy',
    'Corn_(maize)__Common_rust', 'Grape__Esca(Black_Measles)',
    'Raspberry__healthy', 'Tomato__Leaf_Mold',
    'Tomato__Spider_mites Two-spotted_spider_mite', 'Pepper,_bell__Bacterial_spot',
    'Corn_(maize)___healthy'
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(file_bytes):
    try:
        img_array = np.frombuffer(file_bytes, np.uint8)  # Convert bytes to numpy array
        x = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode image
        
        if x is None:
            raise ValueError("Failed to load image. Please provide a valid image file.")

        Resizeimage = cv2.resize(x, (100, 100))  # Resize image
        S = np.expand_dims(Resizeimage, axis=0)  # Add batch dimension

        return S

    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

@app.post("/predict-image/")
async def predict(file: UploadFile = File(...)):
    try:
        S = preprocess_image(await file.read())
        predictions = model.predict(S)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])  # Extract confidence score

        return {"classname": predicted_class, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}



    """
@app.post("/predict-image/")
async def predict(file: UploadFile = File(...)):
    try:
        S = preprocess_image(await file.read())
        predictions = model.predict(S)
        
        predicted_class = class_names[np.argmax(predictions)]
        
        return {"classname": predicted_class}
    except Exception as e:
        return {"error": str(e)}



    """





