import os
from pymongo import MongoClient

import io
from minio import Minio
from minio.error import S3Error

from PIL import Image
import hashlib

# MongoDB
mongodb_client = None
db = None
collections = {}

# Minio
minio_client = None
bucket_name = "my-bucket"


def initialize_mongodb():
    """ Initializes the MongoDB client, connects to the database, and sets up deciated collection for each model. """
    
    global mongodb_client, db, collections
    
    try:
        mongodb_client = MongoClient(os.environ.get('MONGO_URI', 'mongodb://127.0.0.1:27017'))
        db = mongodb_client.mydatabase
        collections['layoutlmv3'] = db.entryhistory
        # Add new collection for an additional model here

    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise

def initialize_minio():
    """ Initializes the MinIO client, connects to the server, and ensures the specified bucket exists."""
    
    global minio_client, bucket_name

    try:
        minio_url = os.getenv('MINIO_URL')
        access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin') 
        
        minio_client = Minio(
            minio_url,
            access_key=access_key,
            secret_key=secret_key,
            secure=False  # If MinIO-Server uses HTTPS, set True
        )

        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f'Bucket "{bucket_name}" successfully created.')
        else:
            print(f'Bucket "{bucket_name}" already exists.')

    except S3Error as e:
        print(f"Error connecting to MinIO: {e}")
        raise


def insert_data(model_name, data):
    """ Inserts data into the specific mogno-db collection of the model. 
    
    Args:
        model_name (str): Name of the coresponding model in order to save data to the coresponding collection.
        data (dict): Data from the inference process of a specified model to be stored.
    """

    if db is None:
        initialize_mongodb()

    collection = get_collection(model_name)
    collection.insert_one(data)


def insert_image(model_name, object_name, image):
    """ Uploads an given image bound to a unique object name. 
    
    Args:
        object_name (str): consists of string '<model_name>/<MD5-Image-Hash>'
        image (bytes object): Image of a certain inference-process to be stored a referenced.
    """

    try:
        if minio_client.stat_object(bucket_name, object_name):
            
            print("[*] Database: Image already exists.", flush=True)

    except:
    
        minio_client.put_object(
            bucket_name,
            object_name,
            data=io.BytesIO(image),
            length=len(image),
            content_type='image/png'
        )


def update_feedback_type(model_name, inference_id, new_feedback_type):
    """ Updates the feedback type based on a given model name and unique inference id"""

    if db is None:
        initialize_mongodb()

    collection = get_collection(model_name)

    # define which document
    filter = {"inference_id": inference_id}

    # define what to update
    update = {"$set": {"feedback_type": new_feedback_type}}

    # update
    result = collection.update_one(filter, update)

    if result.matched_count > 0:
        print(f"[*] Database: Successfully updated the document with id: {inference_id}")
    else:
        print(f"No document found with id: {inference_id}")


def get_collection(model_name):
    """ Retrieves the specific collection for the given model specified in the initialize_mongodb() function.
    
    Args:
        model_name (str): Name of the coresponding model to be returned by the global collections dict.

    Returns:
        collection (pymongo.collection.Collection): The coresponding collection to a given model name.

    """

    if model_name in collections:
        return collections[model_name]
    else:
        raise ValueError(f"No collection found for model: {model_name}")


def get_image_by_id(model_name, inference_id):
    """ Returns the image of an entry based on its ID. """

    object_name = f"{model_name}/{inference_id}.png"

    try:
        response = minio_client.get_object(bucket_name, object_name)

        image_data = response.read()
        return image_data

    except Exception as e:
        raise Exception(f"Error retrieving image from MinIO: {str(e)}")
    

def get_feedback_type_by_id(model_name, inference_id):
    """ Returns the feedback type of an entry based on its ID. """

    if db is None:
        initialize_mongodb()

    collection = get_collection(model_name)

    try:
        entry = collection.find_one({"inference_id": inference_id})

        if not entry:
            return None 
        return entry['feedback_type']
    
    except Exception as e:
        raise Exception(f"Database error: {str(e)}")
    

def get_entries_by_id(model_name, inference_id):
    """ Returns returns all entries of a given model and inference id. """

    if 'db' not in globals():
        initialize_mongodb()

    collection = get_collection(model_name)
    
    fields = {
        "_id": 1,
        "inference_id": 1,
        "timestamp": 1,
        "question": 1,
        "image": 1,
        "words": 1,
        "input_ids": 1,
        "attention_mask": 1,
        "bbox": 1,
        "pixel_values": 1,
        "result": 1,
        "feedback_type": 1,
        "confidence_score_start" : 1,
        "confidence_score_end" : 1
    }

    try:
        entry = collection.find_one({"inference_id": inference_id}, fields)
        
        if not entry:
            return None

        
        entry['_id'] = str(entry['_id']) 
        return entry
    
    except Exception as e:
        raise Exception(f"Database error: {str(e)}")
    

def generate_image_hash(image):
    """ This function takes a PIL image object and returns the MD5 hash of the image. """
    
    image_bytes = image.tobytes()

    md5 = hashlib.md5()
    md5.update(image_bytes)

    return md5.hexdigest()

           

