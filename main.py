from flask import Flask, request, jsonify
from google.cloud import storage, firestore
from google.oauth2 import service_account
from services.load_model import best_model, class_names
from services.image_utils import preprocess_image_as_array
from services.storage_manager import img_url_bucket, upload_to_bucket
from services.firestrore_manager import save_metadata_to_firestore 
from services.model_manager import predict_image_class
import os
import datetime
from flask_swagger_ui import get_swaggerui_blueprint
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Predict testing"
    }
)

# Set the maximum allowed payload to 5 megabytes
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT")
BUCKET_NAME = os.getenv("BUCKET_NAME")
PROJECT_ID = os.getenv("PROJECT_ID")

# Initialize Google Cloud Storage client
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)


credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT)
firestore_client = firestore.Client(credentials=credentials, project=PROJECT_ID)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    if not allowed_file(file.filename):
        return jsonify({
            "error": True,
            "message": "File type not allowed. Only JPG, JPEG, and PNG are allowed."
        }), 400

    try:
        # Upload image to bucket and get public URL
        public_url = upload_to_bucket(file, 'predicted_image')

        # Preprocess the image
        img_array = preprocess_image_as_array(file)
        created_at = datetime.datetime.now().strftime('%Y-%m-%d')

        # Predict the image and get the confidence score
        predicted_class, confidence_score = predict_image_class(best_model, img_array, class_names)


        # Save metadata to Firestore
        save_metadata_to_firestore(predicted_class, confidence_score, user_id, public_url, created_at)

        data_predict = {
        "error": False,
        "message": "Berhasil memprediksi gambar",
        "history": {
            "user_id": user_id,
            "prediction": {
                "image": public_url,
                "name": predicted_class,
                "confidence_score": confidence_score,
                "created_at": created_at
                }
            }
        }

        return jsonify(data_predict), 200
    
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

app.register_blueprint(swaggerui_blueprint)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
