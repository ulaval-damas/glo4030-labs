FROM nvidia/cuda:11.4.3-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

# WORKDIR /workspace/

# Install any python packages you need
COPY ./requirements.txt requirements.txt
COPY ./setup.py setup.py
COPY ./deeplib/ deeplib/

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install -r requirements.txt
RUN pip install .


# Set the working directory
WORKDIR /workspace/
