# Base image
FROM python:3.12
#FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the project dependencies

RUN apt-get update 

RUN apt-get install -y gdal-bin

RUN apt-get install -y libgdal-dev

RUN apt-get install -y redis

#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt
# Copy the Django project code into the container
COPY . .

# Expose the port that the Django server will listen on
EXPOSE 8000

ENV RUNNING_IN_DOCKER=true

# Set the command to run the Django server
CMD daphne -b 0.0.0.0 -p 8000 rails.asgi:application
