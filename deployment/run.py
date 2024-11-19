import os
import time
import requests
import subprocess

# Define paths and container details
project_path = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
root_path = os.path.abspath(os.path.join(project_path, '..'))  # Root project directory (AI-systemer-og-infrastruktur-miniprojekt)
image_name = "pizza-sales-prediction-model"
container_name = "pizza-sales-container"
host_port = 5000
container_port = 5000

# Change to the root project directory
print(f"Root project path: {root_path}")
try:
    os.chdir(root_path)
except FileNotFoundError:
    print(f"Error: Directory '{root_path}' not found.")
    exit(1)

# Check if Docker is running
try:
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except subprocess.CalledProcessError:
    print("Error: Docker daemon is not running. Please make sure Docker Desktop is open and running.")
    exit(1)

# Build the Docker container
try:
    # Build the Docker image from the root directory (AI-systemer-og-infrastruktur-miniprojekt)
    subprocess.run(["docker", "build", "-f", "deployment/Dockerfile", "-t", image_name, "."], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error building Docker image: {e}")
    exit(1)

# Run the Docker container (stop any existing container with the same name first)
subprocess.run(["docker", "stop", container_name], stderr=subprocess.DEVNULL)
subprocess.run(["docker", "rm", container_name], stderr=subprocess.DEVNULL)

try:
    subprocess.run(["docker", "run", "-d", "--name", container_name, "-p", f"{host_port}:{container_port}", image_name], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Docker container: {e}")
    exit(1)

print("Waiting for the container to start...")

# Retry logic with more retries and longer wait times
max_attempts = 20
attempts = 0
while attempts < max_attempts:
    try:
        # Attempt to send a GET request to see if the server is responding
        response = requests.get(f"http://localhost:{host_port}/")
        if response.status_code == 200:
            print("Container is up and running!")
            break
    except requests.exceptions.RequestException:
        print(f"Attempt {attempts+1}/{max_attempts}: Waiting for the container to become available...")
    
    attempts += 1
    time.sleep(3)  # Wait for 3 seconds before retrying

if attempts == max_attempts:
    print("Error: The container did not start in time.")
    exit(1)

# Send an inference request
payload = {
    "hour": 14,
    "day_of_week": 6,
    "month": 12
}

try:
    response = requests.post(f"http://localhost:{host_port}/predict/", json=payload)
    if response.status_code == 200:
        print("Prediction:", response.json())
    else:
        print("Error:", response.status_code, response.text)
except requests.exceptions.RequestException as e:
    print(f"Error connecting to the server: {e}")