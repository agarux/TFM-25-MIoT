# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed dependencies for open3d
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libgomp1 \
    libgl1-mesa-glx \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 5002

# Define environment variable
ENV NAME=anagarridoupm

# Run app.py when the container launches
CMD ["python", "app.py"]