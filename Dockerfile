# Start with a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files (app.py, credit_defaults_model.pkl) to the working directory in the container
COPY . /app

# Install required Python libraries (Flask for API, joblib for model loading, and scikit-learn for model functionality)
RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn

# Expose the port that the Flask app will run on
EXPOSE 5000

# Command to start the Flask app
CMD ["uvicorn", "weather_api:app", "--host", "0.0.0.0", "--port", "80"]