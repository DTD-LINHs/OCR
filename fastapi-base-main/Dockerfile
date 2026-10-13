FROM python:3.12.3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


RUN apt update 
RUN apt upgrade -y 
# docker.io provides the Docker CLI used by the sandbox runner.
# libgl1/libglib2.0-0 are needed by OpenCV/PaddleOCR.
RUN apt install -y nano unixodbc unixodbc-dev docker.io libgl1 libglib2.0-0

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN chmod 777 entrypoint.sh

