FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/huggingface /root/.cache/torch

COPY . /app

ENV PYTHONPATH=/app

CMD ["python", "hongikjiki/app/main.py"]