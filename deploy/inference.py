import os
import time
import requests
import subprocess

# Define paths and container details
project_path = "deploy"
image_name = "timeseries-model"
container_name = "timeseries-container"
host_port = 8000
container_port = 80

# Step 1: Change to the project directory
os.chdir(project_path)

# Step 2: Build the Docker container
subprocess.run(["docker", "build", "-t", image_name, "."], check=True)

# Step 3: Run the Docker container (stop any existing container with the same name first)
subprocess.run(["docker", "stop", container_name], stderr=subprocess.DEVNULL)
subprocess.run(["docker", "rm", container_name], stderr=subprocess.DEVNULL)
subprocess.run(["docker", "run", "-d", "--name", container_name, "-p", f"{host_port}:{container_port}", image_name], check=True)

# Step 4: Wait for the container to start up
print("Waiting for the container to start...")
time.sleep(5)  # Wait a few seconds for the container to be ready

# Step 5: Send an inference request
payload = {
    "day": 15,
    "month": 6,
    "hour": 14,
    "Classic": 100,
    "Chicken": 50,
    "Supreme": 30,
    "Veggie": 20
}

try:
    response = requests.post(f"http://localhost:{host_port}/predict/", json=payload)
    if response.status_code == 200:
        print("Prediction:", response.json())
    else:
        print("Error:", response.status_code, response.text)
except requests.exceptions.RequestException as e:
    print(f"Error connecting to the server: {e}")
