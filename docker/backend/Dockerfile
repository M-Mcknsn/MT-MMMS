FROM python:3.10-slim

RUN apt-get update && apt-get install -y tesseract-ocr

WORKDIR /app

COPY docker/backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

ENV TESSERACT_CMD=/usr/bin/tesseract

ENV prometheus_multiproc_dir=/tmp/prometheus_multiproc_dir

RUN python -c "from transformers import LayoutLMv3Processor; LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-large', resume_download=True, force_download=True, apply_ocr=False)"

RUN python -c "from transformers import AutoModelForQuestionAnswering; AutoModelForQuestionAnswering.from_pretrained('rubentito/layoutlmv3-base-mpdocvqa', resume_download=True, low_cpu_mem_usage=False)"

# create multiprocessing directory for gunicrn + prometheus
RUN mkdir -p $prometheus_multiproc_dir

COPY app/ /app/

