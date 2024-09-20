FROM python:3.10-slim

WORKDIR /app

COPY docker/frontend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app/frontend.py /app/

CMD ["python", "frontend.py"]
