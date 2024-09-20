
import gradio as gr
import requests
import io
import uuid
from datetime import datetime

#########################################################################
### LayoutLMv3 Inference Functions 
#########################################################################

def run_distinct_inference(question, image):
    """
    Sends an inference request to the backend for a distinct image-based question-answering task.

    Args:
        question (str): The question to be answered based on the content of the image.
        image (PIL.Image): The image to be used in the inference.

    Returns:
        dict: The JSON response from the backend containing the inference result and the coresponding inference id.
    """
    
    print("[*] Frontend: Calling Inference", flush=True)
    url = 'http://nginx/api/layoutlmv3/distinct_inference'
    
    # Generate a unique ID for tracking the inference request
    inference_id = str(uuid.uuid4())
    print(f"[*] Frontend: Inference {inference_id} starts", flush=True)

    with io.BytesIO() as buffered:
        image.save(buffered, format="PNG")
        buffered.seek(0)
        
        # Prepare the data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        files = {'image': ('image.png', buffered, 'image/png')}
        data = {'question': question, 'inference_id': inference_id, 'timestamp': timestamp}
        
        # POST request to the backend inference API
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()

        result = response.json()
        
        if 'result' not in result:
            raise gr.Error("Invalid response: 'result' field is missing")
    
    return result['result'], inference_id


def handle_feedback(inference_id, feedback_type):
    """
    Sends a given feedback type as a request towards the dedicated endpoint. 
    
    Args:
        inference_id (str): The unique inference id to the coresponding inference process.
        feedback_type (str): "correct" or "incorrect" as feedback to be updated.
    """

    url = 'http://nginx/api/layoutlmv3/handle_feedback'

    if inference_id is not None:
        print(f"[*] Frontend: Sending '{feedback_type}' for inference_id: {inference_id}", flush=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {'feedback_type': feedback_type, 'inference_id': inference_id, 'timestamp': timestamp}
        
        try:
            response = requests.post(url, data=data)
            response.raise_for_status() 
        except requests.RequestException as e:
            print(f"Error sending feedback: {e}")


#########################################################################
### Gradio User Interface Visuals
#########################################################################

with gr.Blocks() as demo:

    with gr.Tab("Layoutlmv3"):
        gr.Markdown("# Document Question Answering")
        gr.Markdown("Upload an Image of a Document and enter a question to get an answer from the ML model")
        image_input_custom = gr.Image(type="pil", label="Upload Image")
        question_input_custom = gr.Textbox(label="Enter Question")
        button_custom = gr.Button("Submit")
        result_output_custom = gr.Textbox(label="Result")
        inference_id_label = gr.Label(label="Inference-ID", visible=True)
        
        button_custom.click(
            fn=run_distinct_inference, 
            inputs=[question_input_custom, image_input_custom], 
            outputs=[result_output_custom, inference_id_label]
        )


        with gr.Row():
            positive_feedback = gr.Button("Correct")
            negative_feedback  = gr.Button("Incorrect")
            
            positive_feedback.click(
                fn=lambda inference_id: handle_feedback(inference_id, "correct"),
                inputs=[inference_id_label],
                outputs=[]
            )

            negative_feedback.click(
                fn=lambda inference_id: handle_feedback(inference_id, "incorrect"),
                inputs=[inference_id_label],
                outputs=[]
            )
    
    # Add a new model here:
    # with gr.Tab("..."):

            
if __name__ == "__main__":
    
    demo.queue(default_concurrency_limit=None).launch(server_name="0.0.0.0", server_port=7860, show_error=True, show_api=True, max_threads = 450)
   