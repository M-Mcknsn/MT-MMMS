from flask import Flask, request, jsonify, send_file, Response
from datetime import datetime
import io
from PIL import Image
from prometheus_client import multiprocess
from prometheus_client import make_wsgi_app, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from werkzeug.middleware.dispatcher import DispatcherMiddleware

import model.layoutlmv3 as layoutlmv3

import database
import metrics





app = Flask(__name__)

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
        '/metrics': make_wsgi_app()
})

# Initialize Database
print("[*] Backend: Initialize Database", flush=True)
initialize_db_start = datetime.now()
database.initialize_mongodb()
database.initialize_minio()
initialize_db_end= datetime.now()
print("[*] Backend: Database initialized", flush=True)

metrics.update_initialization_duration("layoutlmv3", "Databases", initialize_db_start, initialize_db_end)


print("[*] Backend: Backend ready", flush=True)


@app.route("/metrics")
def get_metrics():
    """ Retrieves Prometheus metrics from the multi-process collector and returns them as a response. """

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    data = generate_latest(registry)
    return Response(data, mimetype=CONTENT_TYPE_LATEST)


#########################################################################
### LayoutLMv3 Inference Endpoints
#########################################################################


@app.route('/layoutlmv3/distinct_inference', methods=['POST'])
def distinct_inference_route():
    """
    Receives an inference POST request from the frontend containing an Image, a questiond and the coresponding inference id

    Returns:
        dict: The JSON response containing the inference result and the coresponding inference id.
    """
    inference_start = datetime.now()
    print("[*] Backend: Receiving Input", flush=True)
    try:


        question = request.form['question']
        inference_id = request.form['inference_id']
        image_file = request.files['image']
        request_timestamp_string = request.form['timestamp']
        request_timestamp = datetime.strptime(request_timestamp_string, "%Y-%m-%d %H:%M:%S")
        
        image = image_file.read()
        result = layoutlmv3.start_inference(question, image, inference_id)
        
        inference_end = datetime.now()
        
        metrics.inc_successful__inference("layoutlmv3")
        metrics.calculate_backend_inference_duration("layoutlmv3", inference_start, inference_end)
        metrics.update_endpoint_latency("layoutlmv3", "distinct_inference", inference_start, request_timestamp)
        metrics.calculate_total_inference_duration("layoulmv3", request_timestamp, inference_end)
        print("[*] Backend: Sending results", flush=True)

        return jsonify({"result": result, "inference_id": inference_id})

    except Exception as e:
        
        metrics.inc_unsuccessful__inference("layoutlmv3")
        return jsonify({"error": str(e)}), 500


@app.route('/layoutlmv3/handle_feedback', methods=['POST'])
def handle_feedback_route():
    """
    Receives an POST request in ordner to update the contained feedback type in the database.

    Returns:
        dict: Status Code
    """

    print("[*] Backend: Receiving Feedback", flush=True)
    try:
        feedback_type = request.form['feedback_type']
        inference_id = request.form['inference_id']
        request_timestamp_string = request.form['timestamp']
        request_timestamp = datetime.strptime(request_timestamp_string, "%Y-%m-%d %H:%M:%S")

        database.update_feedback_type("layoutlmv3", inference_id, feedback_type)
        
        metrics.update_user_feedback_counter("layoutlmv3", feedback_type)
        metrics.update_endpoint_latency("layoutlmv3", "handle_feedback", datetime.now(), request_timestamp)
        
        print(f"[*] Backend: updated {feedback_type} Feedback for inference {inference_id}", flush=True)

        return jsonify({"message": "Feedback received"}), 200

    except Exception as e:
        
        print(f"[*] Backend: Handling Feedback Error - {str(e)}", flush=True)
        return jsonify({"error": str(e)}), 500


#########################################################################
### Database Endpoints
#########################################################################


@app.route('/get_image_by_id/<model>/<inference_id>', methods=['GET'])
def get_image_by_id_endpoint(model, inference_id):
    try:
        image_data = database.get_image_by_id(model, inference_id)
        if not image_data:
            return jsonify({"error": "No entry found with that ID"}), 404

        img_io = io.BytesIO(image_data)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_feedback_type_by_id/<model>/<inference_id>', methods=['GET'])
def get_feedback_type_by_id_endpoint(model, inference_id):
    try:
        feedback_type = database.get_feedback_type_by_id(model, inference_id)
        if not feedback_type:
            return jsonify({"error": "No entry found with that ID"}), 404

        return jsonify({"feedback_type": feedback_type})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_entries_by_id/<model>/<inference_id>', methods=['GET'])
def get_entries_by_id_endpoint(model, inference_id):
    try:
        entries = database.get_entries_by_id(model, inference_id)
        if not entries:
            return jsonify({"error": "No entry found with that ID"}), 404

        return jsonify(entries)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)

