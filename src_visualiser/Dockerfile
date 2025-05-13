# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /webapp

# Copy the current directory contents into the container at /app
ADD . /webapp

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# port 
EXPOSE 5002

# Define environment variable
ENV NAME=anagarridoupm

# Run app.py when the container launches
CMD ["python", "app.py"]