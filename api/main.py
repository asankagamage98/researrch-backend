

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image



def get_class_response(ret):
    if(ret == "Early Blight"):
        minor=[
            "There is a minor diseases",
            "If the disease is severe enough to warrant chemical control, select one of the following fungicides: mancozeb (very good); chlorothalonil or copper fungicides (good). Follow the directions on the label."
        ]
        return minor  
    
    elif(ret == "Late Blight"):    
        major=[
             "There is a major disease",
             "If the disease is severe enough to warrant chemical control, select one of the following fungicides: chlorothalonil (very good), copper fungicide, or mancozeb (good)."
        ]
        return major
    
    elif(ret == "Healthy"):
        return "There is no disease. this is a healthy leaf"
    else: 
        return "There is an error"
    


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    

    return {
        'class': predicted_class,
        # 'details':get_class_response(predicted_class),
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=9000)
