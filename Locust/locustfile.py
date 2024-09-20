
from locust import task, between, FastHttpUser, events
from anls_star import anls_score as anls_star_score

import pandas as pd
import queue
from PIL import Image
from gradio_client import Client, handle_file

import json
import time
import numpy as np

from datetime import datetime
import os


path = "data/T1-SP-DocVQA/val_v1.0_withQT.json"
root_dir = "data/T1-SP-DocVQA/"
output_dir = "output"

#  Loads all records from the SP-DocVQA Datasets as dataframes and adds new columns
def prepare_dataset(max_records=None):

    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data['data'])

    df = df.sample(frac = 1)

    if max_records:
        df = df.iloc[:max_records]
    
    df = remove_indices_from_df("output", df)

    df[['inference_id', 'inference_answer', 'ANLS', 'ANLS*', 'Accuracy']] = None
    df['full_image_path'] = [root_dir + image_file for image_file in df['image']]

    return df

# Removes records from dataframe which were already computed during earlier tests
def remove_indices_from_df(directory, df):

    for filename in os.listdir(directory):

        if filename.startswith("record_") and filename.endswith(".csv"):
            try:
                index = int(filename.split("_")[1].split(".")[0])
                
                if index in df.index:
                    df = df.drop(index)
                    print(f"Index {index} has been removed.", flush = True)
            except ValueError:
                print(f"File {filename} does not contain vailid indices.", flush = True)
    
    return df


def prepare_data_queue():
    for index in df.index:
        data_queue.put(index)


def string_accuracy(str1, str2):

    len1, len2 = len(str1), len(str2)
    
    max_len = max(len1, len2)
    
    matches = sum(1 for a, b in zip(str1, str2) if a == b)
    
    accuracy = matches / max_len
    
    return accuracy


def levenshtein_distance(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1


    matrix = np.zeros((len_str1, len_str2))


    for i in range(len_str1):
        matrix[i, 0] = i
    for j in range(len_str2):
        matrix[0, j] = j


    for i in range(1, len_str1):
        for j in range(1, len_str2):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i, j] = min(matrix[i - 1, j] + 1,      # Insert
                               matrix[i, j - 1] + 1,      # Remove
                               matrix[i - 1, j - 1] + cost)  # Replace

    return matrix[len_str1 - 1, len_str2 - 1]


def anls_score(str1, str2):

    lev_distance = levenshtein_distance(str1, str2)
    
    max_len = max(len(str1), len(str2))
    
    if max_len == 0:
        return 1.0
    
    normalized_lev_distance = lev_distance / max_len
    
    if normalized_lev_distance >= 0.5:
        anls = 0.0
    else:
        anls = 1 - normalized_lev_distance
    
    return anls


def call_inference(client, question, image_path):
    
    result = client.predict(
            question=question,
            image=handle_file(image_path),
            api_name="/run_distinct_inference"
    )
    	
    id_label = result[1]["label"] 

    print(f"Result: '{result[0]}' for inference: {id_label}")
    
    return result[0], id_label


def evaluate_result(answer, inference_answer):
    
    anls_star = anls_star_score(answer, inference_answer)

    anls = anls_score(answer[0], inference_answer)

    accuracy_score = string_accuracy(answer[0], inference_answer)
    
    return anls_star, anls, accuracy_score


def give_correct_feedback(client, id_label):

    client.predict(
            inference_id={"label": id_label, "confidences" : None, "confidence": None},
            api_name="/lambda"
    )

def give_incorrect_feedback(client, id_label):

    client.predict(
            inference_id={"label": id_label, "confidences" : None, "confidence": None},
            api_name="/lambda_1"
    )



class WebsiteUser(FastHttpUser):

    wait_time = between(1, 5)  

    @task
    def perform_query(self):
        if not data_queue.empty():
            index = data_queue.get()
            row = df.loc[index]

            print(f"[*] {data_queue.qsize()}/{len(df)}")

            gr_client = Client("http://localhost/")

            try:
                  
                inference_answer, inference_id = call_inference(gr_client, row['question'], row['full_image_path'])
                            
                anls_star, anls, accuracy_score = evaluate_result(row['answers'], inference_answer)

                if anls_star >= 0.8:
                    time.sleep(1)
                    give_correct_feedback(gr_client, inference_id)


                else:
                    time.sleep(1)
                    give_incorrect_feedback(gr_client, inference_id)

                df.at[index, 'inference_id'] = inference_id
                df.at[index, 'inference_answer'] = inference_answer
                df.at[index, 'ANLS'] = anls
                df.at[index, "ANLS*"] = anls_star
                df.at[index, "Accuracy"] = accuracy_score

                result = df.loc[[index]]
                result.to_csv(f'{output_dir}/record_{index}.csv', index=False)
                
            except Exception as e:
                print(f"An error has occurred: {e}", flush=True)
                data_queue.put(index)
            finally:

                data_queue.task_done()
        else:

            print("QUEUE EMPTY")
            self.stop(True)

    
@events.test_stop.add_listener
def export_data(environment, **kwargs):

    df.to_csv(f'{output_dir}/updated_data.csv', index=False)
    print("Dataset exported.")




df = prepare_dataset()
data_queue = queue.Queue()
prepare_data_queue()


#webui
#locust -f locustfile.py --host=http://localhost
#http://localhost:8089