# Start with a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files (app.py, SVR_pipeline.joblib) to the working directory in the container
COPY . /app

# Install required Python libraries (FastAPI for API, joblib for model loading, and scikit-learn for model functionality)
RUN pip install --no-cache-dir fastapi uvicorn xgboost joblib numpy scikit-learn

# Expose the port that the FastAPI app will run on
EXPOSE 5000

# Command to start the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]