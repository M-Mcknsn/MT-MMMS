
from collections import Counter as CollectionCounter
from prometheus_client import Gauge, Histogram, Counter
import re

#########################################################################
### Input Metrics
#########################################################################

IMAGE_WIDTH = Gauge('image_width_pixels', 'Width of the image in pixels', ['model_name'])
IMAGE_HEIGHT = Gauge('image_height_pixels', 'Height of the image in pixels', ['model_name'])

IMAGE_WIDTH_HISTOGRAMM = Histogram('image_width_pixels_histogram', 'Distribution of Width of the image in pixels', ['model_name'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
IMAGE_HEIGHT_HISTOGRAMM = Histogram('image_height_pixels_histogram', 'Distribution of Height of the image in pixels', ['model_name'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

BOUNDING_BOX_COVERAGE_HISTOGRAM  = Histogram('bounding_box_coverage_histogram', 'Disribution of bounding box coverage in', ['model_name'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
BOUNDING_BOX_COVERAGE   = Gauge('bounding_box_coverage', 'relativ Bounding box coverage', ['model_name'])

AVG_BOUNDING_BOX_AREA = Gauge('avg_bounding_box_area', 'average bounding box area per image', ['model_name'])
AVG_BOUNDING_BOX_AREA_HISTOGRAM = Histogram('avg_bounding_box_area_histogram', 'Distribution of average bounding box area per image', ['model_name'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

OCR_WORD_COUNT = Gauge('ocr_word_count', 'amount of recognized words per image', ['model_name'])

QUESTION_WORD_LENGTH_HISTOGRAM = Histogram('question_word_length_histogram', 'Distribution of amount of words per question', ['model_name'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
QUESTION_WORD_LENGTH = Gauge('question_word_length', 'amount of words per question', ['model_name'])

TOKEN_ID_HISTOGRAMM = Histogram('token_id_histogram', 'destibution of token ids', ['model_name'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
TOKEN_USAGE = Counter('token_usage', 'Tracks usage of tokens', ['model_name','token_id'])
TOKEN_USAGE_GAUGE = Gauge('token_usage_gauge', 'Tracks usage of tokens', ['model_name','token_id'])

#########################################################################
### System Metrics
#########################################################################

TOTAL_STARTUP_DURATION = Gauge('total_startup_duration','total duration in seconds until system is ready')
INITIALIZATION_DURATION = Gauge('initialization_duration','total duration in seconds for initialization and loading of certain part (encoder, model, databases)',['model_name', 'part'])

TOTAL_INFERENCE_DURATION = Gauge('total_inference_duration', 'Total duration (seconds) of inference starting from request sent by frontend to response', ['model_name'])
TOTAL_INFERENCE_DURATION_HISTOGRAM = Histogram('total_inference_duration_histogram', 'distribution of total duration (seconds) of inference starting from request sent by frontend to response', ['model_name'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

BACKEND_INFERENCE_DURATION = Gauge('backend_inference_duration', 'Backend inference duration (seconds) of inference starting from endpoint call to response', ['model_name'])
BACKEND_INFERENCE_DURATION_HISTOGRAM = Histogram('backend_inference_duration_histogram', 'distribution of Backend inference duration (seconds) of inference starting from endpoint call to response', ['model_name'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

ENCODING_DURATION = Gauge('encoding_duration', 'duration (seconds) of encoding including invoking function with unprocessed input to generating encoded data', ['model_name'])
ENCODING_DURATION_HISTOGRAM = Histogram('encoding_duration_histogram', 'distribution of duration (seconds) of encoding including invoking function with unprocessed input to generating encoded data', ['model_name'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

SUCCESSFUL_INFERENCES = Counter('successful_inferences', 'Total amount of successful inferences', ['model_name'])
UNSUCCESSFUL_INFERENCES = Counter('unsuccessful_inferences', 'Total amount of unsuccessful inferences', ['model_name'])

REQUEST_LATENCY = Gauge('request_latency','Latency for a certain endpoint in ms', ['model_name', 'endpoint'])
REQUEST_LATENCY_HISTOGRAM = Histogram('request_latency_histogram','distribution of Latency for a certain endpoint in ms', ['model_name', 'endpoint'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

#########################################################################
### Model Metrics
#########################################################################

INFERENCE_DURATION = Gauge('inference_duration', 'duration (seconds) of inference including invoking model with encoded_input to generating output', ['model_name'])
INFERENCE_DURATION_HISTOGRAM = Histogram('inference_duration_histogram', 'distribution of duration (seconds) of inference including invoking model with encoded_input to generating output', ['model_name'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

CONFIDENCE_SCORE = Gauge('confidence_score', 'model confidence score', ['model_name', 'score_type']) #'inference_id', 
CONFIDENCE_SCORE_HISTOGRAM = Histogram('confidence_score_histogram', 'Distribution of confidence score of ml model', ['model_name','score_type'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
CONFIDENCE_SCORE_DIFFERENCE = Gauge('confidence_score_difference', 'difference of start and end confidence score', ['model_name']) #,'inference_id'
CONFIDENCE_SCORE_DIFFERENCE_HISTOGRAM = Histogram('confidence_score_difference_histogram', 'distribution of confidence score difference', ['model_name'], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

#########################################################################
### Output Metrics
#########################################################################

USER_FEEDBACK_COUNTER = Counter('user_feedback_counter', 'total amount of positive or negative user feedback',['model_name', 'feedback_type'])

FAILED_INFERENCE_COUNTER = Counter('failed_inference_counter', 'Total amount of failed inferences by model with empty string as result', ['model_name'])







def update_user_feedback_counter(model_name, feedback_type):
    USER_FEEDBACK_COUNTER.labels(model_name=model_name, feedback_type=feedback_type).inc()


def inc_successful__inference(model_name):
    SUCCESSFUL_INFERENCES.labels(model_name=model_name).inc()


def inc_unsuccessful__inference(model_name):
    UNSUCCESSFUL_INFERENCES.labels(model_name=model_name).inc()


def update_confidence_score(model_name, start_confidence, end_confidence):

    confidence_difference = abs(start_confidence-end_confidence)  # absolute difference of start and end confidence
    
    CONFIDENCE_SCORE.labels(model_name=model_name, score_type='start').set(start_confidence)
    CONFIDENCE_SCORE.labels(model_name=model_name,  score_type='end').set(end_confidence)
    CONFIDENCE_SCORE_DIFFERENCE.labels(model_name=model_name).set(confidence_difference)
    
    CONFIDENCE_SCORE_HISTOGRAM.labels(model_name=model_name, score_type='start').observe(start_confidence)
    CONFIDENCE_SCORE_HISTOGRAM.labels(model_name=model_name, score_type='end').observe(end_confidence)
    CONFIDENCE_SCORE_DIFFERENCE_HISTOGRAM.labels(model_name=model_name).observe(confidence_difference)


def update_image_size(model_name, image_width, image_height):

    IMAGE_WIDTH.labels(model_name=model_name).set(image_width)
    IMAGE_HEIGHT.labels(model_name=model_name).set(image_height)

    IMAGE_WIDTH_HISTOGRAMM.labels(model_name=model_name).observe(image_width)
    IMAGE_HEIGHT_HISTOGRAMM.labels(model_name=model_name).observe(image_height)


def calculate_bounding_box_metrics(model_name, bounding_boxes, image_width, image_height):
    
    
    image_area = image_width * image_height
    

    total_bounding_box_area = 0
    for box in bounding_boxes:
        if len(box) == 4:
           
            total_bounding_box_area = sum(
                (xmax - xmin) * (ymax - ymin)
                for xmin, ymin, xmax, ymax in bounding_boxes
            )
        else:

            print(f"Invalid bounding box detected: {box}")
    
    coverage_percentage = (total_bounding_box_area / image_area) * 100
    average_area = total_bounding_box_area / len(bounding_boxes)

    BOUNDING_BOX_COVERAGE_HISTOGRAM.labels(model_name=model_name).observe(coverage_percentage)
    BOUNDING_BOX_COVERAGE.labels(model_name=model_name).set(coverage_percentage)

    AVG_BOUNDING_BOX_AREA_HISTOGRAM.labels(model_name=model_name).observe(average_area)
    AVG_BOUNDING_BOX_AREA.labels(model_name=model_name).set(average_area)
    

def update_ocr_word_count(model_name, words):

    ocr_word_count = len(words)
    OCR_WORD_COUNT.labels(model_name=model_name).set(ocr_word_count)


def update_question_length(model_name, question):

    question_cleaned = re.sub(r"[,.?!]", "", question)

    question_length = len(question_cleaned.split())

    QUESTION_WORD_LENGTH_HISTOGRAM.labels(model_name).observe(question_length)
    QUESTION_WORD_LENGTH.labels(model_name).set(question_length)


def update_token_distribution(model_name, input_ids):
    for token_id in input_ids:
        TOKEN_ID_HISTOGRAMM.labels(model_name).observe(token_id)

   
def calculate_total_inference_duration(model_name, start, end):
    total_duration = end-start
    TOTAL_INFERENCE_DURATION.labels(model_name=model_name).set(total_duration.total_seconds())
    TOTAL_INFERENCE_DURATION_HISTOGRAM.labels(model_name=model_name).observe(total_duration.total_seconds())


def calculate_backend_inference_duration(model_name, start, end):
    total_duration = end-start
    BACKEND_INFERENCE_DURATION.labels(model_name=model_name).set(total_duration.total_seconds())
    BACKEND_INFERENCE_DURATION_HISTOGRAM.labels(model_name=model_name).observe(total_duration.total_seconds())



def calculate_encoding_duration(model_name, start, end):
    total_duration = end-start
    ENCODING_DURATION.labels(model_name=model_name).set(total_duration.total_seconds())
    ENCODING_DURATION_HISTOGRAM.labels(model_name=model_name).observe(total_duration.total_seconds())



def calculate_inference_duration(model_name, start, end):
    total_duration = end-start
    INFERENCE_DURATION.labels(model_name=model_name).set(total_duration.total_seconds())
    INFERENCE_DURATION_HISTOGRAM.labels(model_name=model_name).observe(total_duration.total_seconds())


def update_endpoint_latency(model_name, endpoint, start, end):
    latency = start - end
    REQUEST_LATENCY.labels(model_name=model_name, endpoint=endpoint).set(latency.total_seconds())
    REQUEST_LATENCY_HISTOGRAM.labels(model_name=model_name, endpoint=endpoint).observe(latency.total_seconds())


def update_initialization_duration(model_name, part, start, end):
    total_duration = end-start
    INITIALIZATION_DURATION.labels(model_name=model_name, part=part).set(total_duration.total_seconds())


def update_failed_inference_count(model_name, inference_id):
    FAILED_INFERENCE_COUNTER.labels(model_name=model_name, inference_id=inference_id).inc()


def update_token_ids_count(model_name, token_ids):

    token_counts = CollectionCounter(token_ids)

    for token_id, count in token_counts.items():
        TOKEN_USAGE_GAUGE.labels(model_name=model_name, token_id=token_id).set(count)
        TOKEN_USAGE.labels(model_name=model_name, token_id=token_id).inc()