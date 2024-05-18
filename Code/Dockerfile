# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Expose the Prometheus port
EXPOSE 8001

# Command to run the FastAPI application
CMD ["uvicorn", "bdl_proj_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]