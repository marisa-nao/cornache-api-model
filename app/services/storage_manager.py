import os
import datetime
from google.cloud import storage

SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT")
BUCKET_NAME = os.getenv("BUCKET_NAME")
PROJECT_ID = os.getenv("PROJECT_ID")

storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

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