# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /edge
WORKDIR /edge

# Copy the current directory contents into the container at /edge
COPY . /edge

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Define environment variable
ENV NAME=anagarridoupm

# Run app.py when the container launches
CMD ["python3", "compress_cameras_sim.py"]