# Use an official Python image.
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, including Tesseract for OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# The command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "run:create_app()"]