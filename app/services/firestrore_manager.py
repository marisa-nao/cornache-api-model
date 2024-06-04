from flask import jsonify
import datetime
from google.cloud import firestore
from google.oauth2 import service_account
import os
from dotenv import load_dotenv
load_dotenv()

SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT")
PROJECT_ID = os.getenv("PROJECT_ID")

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT)
firestore_client = firestore.Client(credentials=credentials, project=PROJECT_ID)

def save_metadata_to_firestore(predicted_class, confidence_score, user_id, url, created_at):


    data_predict = {
        "user_id": user_id,
        "prediction": {
            "image": url,
            "name": predicted_class,
            "confidence_score": confidence_score,
            "created_at": created_at
        }
    }
    
    doc_ref = firestore_client.collection('predicts').document()  # Don't specify document ID
    doc_ref.set(data_predict)

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