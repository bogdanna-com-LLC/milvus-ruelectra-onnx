﻿FROM nvcr.io/nvidia/ai-workbench/python-cuda122:1.0.6
LABEL authors="Danila"

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt 

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]