# Introduction

This project contains monitoring a multimodal end-to-end machine learning system, demonstrated through a Document-Question-Answering model. It includes the development, containerization, and deployment of of an easily expandable machine learning system utilizing an online inference approach, providing a foundation for future exploration and evaluation of additional ML models. The monitoring system is implemented using Prometheus and Grafana to track performance and system metrics.

# Architecture

![MachineLEarningSystemArchitecture3](https://github.com/user-attachments/assets/323d5faf-114d-46d1-a841-4f5ee6943d5b)

For easy access, a Gradio frontend is employed, which is served via an NGINX web server infront of a Gunicorn WSGI server. MongoDB is used for the historization of feature, model and text data and MinIO for the storage of image data. For monitoring, a comprehensive approach is taken by utilizing system, resource, model, input, and output-specific metrics. These metrics are collected and visualized with Prometheus and Grafana, providing insights into the system's performance and reliability in real time.

# Setup

## Prerequisites
- On Windows: WSL 2 Ubuntu distribution
- docker-desktop application
  
### Install WSL
From the official [microsoft wsl documentation](https://learn.microsoft.com/en-us/windows/wsl/install): 
You can install everything you need to run WSL with a single command. Open PowerShell or Windows Command Prompt in **administrator** mode by right-clicking and selecting "Run as administrator", enter the following command, then restart your machine.
```
wsl --install
```
This command will enable the features necessary to run WSL and install the Ubuntu distribution of Linux by default. 

### Install Docker Desktop
Download Docker Desktop from the offical [Docker Desktop Website](https://www.docker.com/products/docker-desktop/).
After launching Docker Desktop, ensure that Ubuntu is enabled by navigating to Settings > Resources > WSL Integration.

### Start the System
- Clone this project into a directory of your choice.
- Open a terminal and navigate to the MT-MMMS directory within your chosen location.
- Run the following command to pull all necessary Docker images and start the system. Ensure that Docker Desktop is running before executing the command.
```
docker-compose up
```
The first download of the docker images may take a while.
After the download is complete, the system will start and the terminal will notify you with `[*] Backend: Backend ready` when the initialisation is complete.

Hint: Gunicorn is configured with 4 Worker-Instances, which defines the maximum of parallel workflows (see [Gunicorn Documentation](https://docs.gunicorn.org/en/latest/design.html) for more details).
Four worker instances are adequate for a testing and development environment, but should be increased to meet higher demands. You can adjust this by modifying the parameter following `-w` in the `docker-compose.yml`:
```
command: gunicorn -w 4 -b 0.0.0.0:5000 --timeout 1200 backend:app
```

### Grafana Configuration
- Open [http://localhost:3000](http://localhost:3000) on a webrowser of your choice.
- Enter the following inital credentials:
  - Email or username: admin
  - Password: admin
- Then set your own password.
- Now you have access to Grafana. In order to setup a connection to Prometheus you need to navigate to the menu icon on the top left corner and click on **Data sources**.
  
![grafik](https://github.com/user-attachments/assets/d640c916-4005-4ca1-955d-1fc32dfbf340)
  
- Click **add data source** and choose "Prometheus" on the following List.
- Now add `http://prometheus:9090` as the Prometheus server URL.
- Scroll down and click **save & test**
- Now navigate to the Dashboards section using the menu.
  
![grafik](https://github.com/user-attachments/assets/a3312fa8-9d74-4414-941c-728213a22df2)

- Click on **New** in the top right corner and choose **Import**.
- Drag and Drop one Grafana Dashboard located in `MT-MMMS\monitoring\grafana` of this repository. Make sure to select your previously configured prometheus data source before you click **Import**. Repeat this for each dashboard of your choice.
- Metrics will begin to be collected and displayed on the dashboards after the first few inputs are processed by the system.

### Inference

- Open [http://localhost/](http://localhost/) and upload a PNG-image of a document.
- Enter a question related to the uploaded document and click **Submit**. After a couple of seconds you should receive a reply.

## Modules

`frontend.py`:
- This module provides the foundational interface for interacting with the deployed ML model. It can be easily extended by adding a new tab to the interface for each additional model, along with configuring and implementing new functions tailored to the specific model.

`backend.py`:
- This module provides the Flask-Application containing Endpoints in order to mediate Request between User Interface and ML Model. Add a new endpoint for each ml model and direct requests to the desired ML-Module function to start the inference process.

`model/layoutlmv3.py`
- This module contains all neccesary steps for the complete inference process of the LayoutLMv3 Model. It includes the preprocessing, interactions with `database.py` or `metrics.py` as well as the model-inference itself.

`database.py`:
- In this module, both MinIO and MongoDB databases are initialized. Data insertion, updating, and retrieval are managed here. Each model is assigned its own collection within the MongoDB database and must be added to the collections dictionary in the `initialize_mongodb()` function. Each ML-Model module can call desired database-functions in order to store data during the inference process.

`metrics.py`:
- This module handles the calculation of all metrics to be displayed in Grafana. Each metric must be initialized and computed through a dedicated function, allowing it to be accessed system-wide for the calculation of various metrics.

## Components

System:
- [Gradio](https://www.gradio.app/)
- [MongoDB](https://www.mongodb.com/de-de)
- [MinIO](https://min.io/)
- [Gunicorn](https://gunicorn.org/)
- [LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)
  -  [LayoutLMv3 base fine-tuned on MP-DocVQA](https://huggingface.co/rubentito/layoutlmv3-base-mpdocvqa  )  
- [NGINX](https://nginx.org/en/)
- [Flask](https://flask.palletsprojects.com/en/3.0.x/)

Monitoring:
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
- [NGINX Exporter](https://github.com/nginxinc/nginx-prometheus-exporter)
- [cAdvisor](https://github.com/google/cadvisor)

Grafana Dasboards used:
- [NGINX-Exporter](https://grafana.com/grafana/dashboards/12708-nginx/)
- [cAdvisor](https://grafana.com/grafana/dashboards/14282-cadvisor-exporter/)
