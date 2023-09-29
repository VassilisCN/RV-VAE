# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim
FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/random_variable_modules:$PYTHONPATH

RUN apt-get update 
RUN apt-get install -y python3 python3-pip python3-distutils-extra
RUN pip install opencv-python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install sudo

# Install pip requirements
run mkdir RV_VAE
COPY . /RV_VAE
RUN pip install -r requirements.txt
WORKDIR /RV_VAE
