# Base Image: python 3.11 on slim linux
FROM python:3.11-slim

# set corking directory inside the container
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

# run the app
CMD ["streamlit", "run", "main.py"]