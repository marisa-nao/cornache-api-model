from flask import Flask, request, jsonify
from google.cloud import storage, firestore
from google.oauth2 import service_account
from load_model import best_model, class_names
import os
import datetime
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Set the maximum allowed payload to 5 megabytes
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT")
BUCKET_NAME = os.getenv("BUCKET_NAME")
PROJECT_ID = os.getenv("PROJECT_ID")

# Initialize Google Cloud Storage client
storage_client = storage.Client(project=PROJECT_ID)
# bucket_name = "cornache-bucket"
bucket = storage_client.bucket(BUCKET_NAME)

# Initialize Firestore client with service account credentials
# credentials_path = 'credentials/cornache-key.json'
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT)
firestore_client = firestore.Client(credentials=credentials, project=PROJECT_ID)

def img_url_bucket(filename):
    return f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"

def upload_to_bucket(file, fieldname):
    time_stamp = int(datetime.datetime.now().timestamp())
    img_name = f"{time_stamp}-{file.filename}"

    if fieldname == 'avatar_image':
        path = 'user-profile/'
    elif fieldname == 'predicted_image':
        path = 'predicted-image/'
    elif fieldname == 'room_image':
        path = 'room-image/'
    else:
        path = ''

    # Replace "/" with "_"
    gcsname = path + img_name.replace("/", "_")
    blob = bucket.blob(gcsname)
    blob.upload_from_string(
        file.read(),
        content_type=file.content_type
    )
    blob.make_public()

    return img_url_bucket(gcsname), gcsname

def save_metadata_to_firestore(user_id, filename, url, fieldname):
    created_at = datetime.datetime.now().strftime('%Y-%m-%d')
    print("created:", created_at)

    doc_ref = firestore_client.collection('predicts').document(user_id)  # Don't specify document ID
    doc_ref.set({
        "user_id": user_id,
        "prediction": {
            'filename': filename,
            'url': url,
            'fieldname': fieldname,
            'created_at': created_at
        }
    })

def get_user_by_id(user_id):
    try:
        if not user_id:
            return jsonify({
                "error": True,
                "message": "User ID is required"
            }), 400

        # Reference to the document in Firestore
        user_ref = firestore_client.collection('users').document(user_id)
        
        # Get the document
        user = user_ref.get()
        
        if user.exists:
            # Return the document data as a JSON response
            return jsonify(user.to_dict()), 200
        else:
            return jsonify({
                "error": True,
                "message": "User not found"
            }), 404

    except Exception as error:
        return jsonify({
            "error": True,
            "message": str(error)
        }), 500

def preprocess_image_as_array(image_file, target_size=(256, 256)):
    """
    Preprocess the input image for model prediction.

    :param image_file: The input image file (an instance of werkzeug.datastructures.FileStorage).
    :param target_size: A tuple specifying the target size (width, height).
    :return: A numpy array of the preprocessed image.
    """
    # Open and resize the image
    im = Image.open(image_file).convert('RGB')
    im = im.resize(target_size)

    # Convert the image to a numpy array
    img_array = img_to_array(im)
    
    # Expand dimensions to match the input shape of the model (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image array
    img_array = preprocess_input(img_array)
    
    return img_array

def predict_image_class(model, image_array, class_names):
    # Predict class probabilities
    predictions = model.predict(image_array)
    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions)
    print("pred", predictions)
    # Get the confidence score of the predicted class
    confidence_score = predictions[0][predicted_class_index]

    print("confidence :", confidence_score)
    # Return the class name and confidence score
    return class_names[predicted_class_index], confidence_score

@app.route("/predict", methods=["POST"])
def predict_image():
    user_id = request.form.get('user_id')

    if not user_id:
        return jsonify({
            "error": True,
            "message": "User ID is required"
        }), 400

    if 'image_predict' not in request.files:
        return jsonify({
            "error": True,
            "message": "No file part in the request"
        }), 400

    file = request.files['image_predict']
    
    if file.filename == '':
        return jsonify({
            "error": True,
            "message": "No file selected"
        }), 400
    
    try:
        public_url, gcsname = upload_to_bucket(file, 'predicted_image')
        save_metadata_to_firestore(user_id, gcsname, public_url, 'predicted_image')
        
        # Preprocess the image
        img_array = preprocess_image_as_array(file)

        # Predict the image and get the confidence score
        predicted_class, confidence_score = predict_image_class(best_model, img_array, class_names)

        return jsonify({
            "error": False,
            "message": "Image prediction successful", 
            "predicted_class": predicted_class,
            "confidence_score": float(confidence_score),
            "url": public_url
        }), 200
    
    except Exception as error:
        return jsonify({
            "error": True,
            "message": str(error)
        }), 400

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": True,
        "message": "File size exceeds the maximum limit of 5 MB"
    }), 413

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)
