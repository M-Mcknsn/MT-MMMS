
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3Processor, AutoModelForQuestionAnswering
import pytesseract

import torch.nn.functional as F
import torch

import io
from PIL import Image

import os

from datetime import datetime

# Modules
import database
import metrics

#torch.set_num_threads(24)

model_name = "layoutlmv3"

image_processor = LayoutLMv3ImageProcessor()

print("[*] Layoutlmv3: Loading Encoder", flush=True)
load_encoder_start = datetime.now()
encoder = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large", resume_download=True, force_download=True, apply_ocr=False)
load_encoder_end = datetime.now()
print("[*] Layoutlmv3: Encoder loaded", flush=True)

print("[*] Layoutlmv3: Loading Model", flush=True)
load_model_start = datetime.now()
model = AutoModelForQuestionAnswering.from_pretrained("rubentito/layoutlmv3-base-mpdocvqa", resume_download=True, low_cpu_mem_usage=False) 
load_model_end = datetime.now()
print("[*] Layoutlmv3: Model loaded", flush=True)

print("[*] Layoutlmv3: Loading Tesseract", flush=True)
pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD', '/usr/bin/tesseract')
print("[*] Layoutlmv3: Tesseract loaded", flush=True)


metrics.update_initialization_duration(model_name, "Encoder", load_encoder_start, load_encoder_start)
metrics.update_initialization_duration(model_name, "Model", load_model_start, load_model_end)


def convert_image(image):
    """
    Converts a given Stream Image into a RGB PIL-Image 

    Args:
        image (bytes object): The given image to be converted.
        
    Returns:
        image (PIL.Image): The image to be used in the inference.
    """

    pil_image = Image.open(io.BytesIO(image)).convert("RGB")
    
    return pil_image

def start_inference(question, image, inference_id):
    """
    Processes an image and performs question-answering inference using the LayoutLMv3 model.

    Args:
        question (str): The question to be answered based on the content of the image.
        image (bytes object): The raw byte data of the image file read from a request.
        inference_id (str): A unique identifier for the inference request.

    Returns:
        str: The result of the inference, which is the answer generated by the model.

    The function performs the following steps:
    1. **Image Processing**: 
       - The raw byte image is converted to a PIL image format for further processing.
       
    2. **Encoding**: 
       - Encodes the question and the image using the LayoutLMv3 model processor.
       - Extracts words and bounding boxes from the image for the model.

    3. **Model Inference**: 
       - Runs inference on the encoded features to generate the answer to the question.
       - Records confidence scores for the start and end positions of the predicted answer.

    4. **Data Storage**: 
       - Generates a hash of the image and stores it in the database along with the inference data as a refence for the object store.
       - Encodes the input data into a dictionary format suitable for storage.
       - Uploads the image and its corresponding data to the objectstore.

    5. **Metrics Calculation**: 
       - Updates metric

    6. **Return**: 
       - Returns the final inference result (the answer to the question based on the content of the document image).

    """

    print("[*] Layoutlmv3: Processing Image", flush=True)

    pil_image = convert_image(image)

    print("[*] Layoutlmv3: Encoding", flush=True)

    encoding_start = datetime.now()

    encoded_data, words, boxes = encoding(question, pil_image)

    encoding_end = datetime.now()

    print("[*] Layoutlmv3: Inference", flush=True)

    inference_start = datetime.now()

    result, confidence_score_s, confidence_score_e = inference(encoded_data)

    # Model returns empty strings with failed inferences
    if not result.strip():
        metrics.update_failed_inference_count(model_name, inference_id)

    inference_end = datetime.now()

    timestamp_now = datetime.now()
    
    image_hash = database.generate_image_hash(pil_image)

    object_name = f"{model_name}/{image_hash}.png"
    
    encoded_dict = tensor_to_json(encoded_data)

    data_input = {
        'inference_id' : inference_id,
        'timestamp': timestamp_now,
        'question': question,
        'image': object_name,
        'words' : words,
        'input_ids' : encoded_dict['input_ids'],
        'attention_mask' : encoded_dict['attention_mask'],
        'bbox' : encoded_dict['bbox'],
        'pixel_values' : encoded_dict['pixel_values'],
        'result' : result,
        'confidence_score_start' : confidence_score_s,
        'confidence_score_end' : confidence_score_e,
        'feedback_type' : "None"
    }
    
    print("[*] Layoutlmv3: Starting Database upload", flush=True)
    database.insert_data(model_name, data_input)
    print(f"[*] Layoutlmv3: Saving Image as Object: {object_name}", flush=True)
    database.insert_image(model_name, object_name, image)
    print("[*] Layoutlmv3: Database upload done", flush=True)

    image_width, image_height = pil_image.size

    print("[*] Layoutlmv3: Calculating Metrics", flush=True)
    metrics.update_image_size(model_name, image_width, image_height)
    metrics.update_confidence_score(model_name, confidence_score_s, confidence_score_e)
    metrics.calculate_bounding_box_metrics(model_name, boxes, image_width, image_height)
    metrics.update_ocr_word_count(model_name, words)
    metrics.update_question_length(model_name, question)
    metrics.update_token_distribution(model_name, encoded_dict['input_ids'][0])
    metrics.calculate_encoding_duration(model_name,encoding_start, encoding_end)
    metrics.calculate_inference_duration(model_name, inference_start, inference_end)
    metrics.update_token_ids_count(model_name, encoded_dict['input_ids'][0])
    print("[*] Layoutlmv3: Metrics done", flush=True)

    return result


def tensor_to_json(encoded_data):
    """
    Converts tensor data into a dictionary structure in order to be saved by the database.

    Args:
        encoded_data (tensor): Encoded features as tensor-structure.

    Returns:
        encoding_dict (dict): Encoded features as dictionary.

    """
    encoding_dict = dict(encoded_data) 

    for key, value in encoding_dict.items():
        if isinstance(value, torch.Tensor):
            encoding_dict[key] = value.tolist()
        elif isinstance(value, list):
            encoding_dict[key] = [v.tolist() if isinstance(v, torch.Tensor) else v for v in value]

    return encoding_dict


def encoding(question, image):
    """
    Processes a given PIL-Image in ordner to obtain OCR words and boundignj boxes.

    Args:
        question (str): The question regarding a given image.
        image (PIL.Image): The given PIL-Image for the inference. 

    Returns:
        encoding (tensor): Encoded features to be used by the model.
        words (List): List of recognized OCR-Words.
        bboc (List): List if coresponding bounding boxes.
    """
        
    print("[*] Layoutlmv3: Encoding > Preprocess Image", flush=True)
    
    processed_image = image_processor.preprocess(image)

    encoded = {
    "words": processed_image.words,
    "bbox": processed_image.boxes
    }

    words = encoded["words"][0]
    boxes = encoded['bbox'][0]

    print("[*] Layoutlmv3: Encoding > Starting Enconding", flush=True)
    encoding = encoder(image, question, words, boxes=boxes, return_tensors="pt", max_length = 512, padding="max_length", truncation=True)
    print("[*] Layoutlmv3: Encoding > Enconding finished", flush=True)
    
    return encoding, words, boxes

def inference(encoded_data):
    """
    Performs inference based on given features.

    Args:
        encoded_data (tensor): Encoded features as tensors.
        
    Returns:
        result (str): Answer to the posed question as an inference result.
        confidence_score_s (float): Models confidence score (0.0 - 1.0) for the prediction of the starting position in the given context.
        confidence_score_e (float): Models confidence score (0.0 - 1.0) for the prediction of the ending position in the given context.
    """
        
    start_positions = torch.tensor([1])
    end_positions = torch.tensor([3])

    outputs = model(**encoded_data, start_positions=start_positions, end_positions=end_positions)
    loss = outputs.loss
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    predicted_start_idx = start_logits.argmax(-1).item()
    predicted_end_idx = end_logits.argmax(-1).item()

    probabilities_s = F.softmax(outputs.start_logits, dim=-1)
    probabilities_e = F.softmax(outputs.end_logits, dim=-1)

    confidence_score_s = probabilities_s[0][predicted_start_idx].item()
    confidence_score_e = probabilities_e[0][predicted_end_idx].item()

    result = encoder.tokenizer.decode(encoded_data.input_ids.squeeze()[predicted_start_idx:predicted_end_idx+1])
    
    # Postprocessing, Inference adds a single ' ' infront of result.
    if result.startswith(' '):
        result = result.lstrip()

    return result, confidence_score_s, confidence_score_e
