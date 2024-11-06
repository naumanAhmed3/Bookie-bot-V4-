# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the Google Cloud service account key file into the container
COPY bookme-ai-2ec25f6f3400.json /app/google-credentials.json

# Ensure the credentials file has the correct permissions
RUN chmod 644 /app/google-credentials.json

# Install dependencies from requirements.txt (including python-dotenv)
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for Google Cloud credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="bookme-ai-2ec25f6f3400.json"
ENV GOOGLE_API_KEY="AIzaSyDvbke4TODM1nOMbkZAXXhOVGQeECSsATU"


# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]