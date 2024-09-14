FROM python:3.9.7-slim-buster

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and set permissions for the data directory
RUN mkdir /data && chmod 777 /data

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements.txt
COPY requirements.txt /app/

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application files
COPY . /app

EXPOSE 8000

ENV NAME sample

# Set the entrypoint command
CMD ["gunicorn", "--conf", "/app/gunicorn_conf.py", "main:app"]

COPY log_monitor.py /app/
