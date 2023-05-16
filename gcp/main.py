from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None
interpreter = None
input_index = None
output_index = None

class_names = ["Early Blight", "Late Blight", "Healthy"]

BUCKET_NAME = "tomato-model-1" # Here you need to put the name of your GCP bucket


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def get_class_response(ret):
    if(ret == "Early Blight"):
        minor=[
            "There is a minor diseases",
            "1. Use fungicides to control the disease.",
            "2. Practice crop rotation."
            "3. Remove infected plants and debris from the field.",
            "4. Provide proper spacing between plants to improve air circulation."
        ]
        return minor  
    
    elif(ret == "Late Blight"):    
        major=[
            "There is a major disease",
            "1. Apply copper-based fungicides to control the disease.",
            "2. Eliminate volunteer tomato and potato plants.",
            "3. Plant resistant varieties.",
            "4. Use proper irrigation techniques to reduce leaf wetness."
        ]
        return major
    
    elif(ret == "Healthy"):
        return "There is no disease. this is a healthy leaf"
    else: 
        return "There is an error"
    


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/tomato.h5",
            "/tmp/tomato.h5",
        )
        model = tf.keras.models.load_model("/tmp/tomato.h5")

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256)) # image resizing
    )

    image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    
    details = get_class_response(predicted_class),


    return {"class": predicted_class, "confidence": confidence,'details':details}
