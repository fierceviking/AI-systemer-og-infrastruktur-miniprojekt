import requests

host_port = 8000

# Define the payload
payload = {
    "day": 5,
    "month": 3,
    "hour": 11
}

# Send a POST request
response = requests.post(f"http://localhost:{host_port}/predict/", json=payload)

# Print the response
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)