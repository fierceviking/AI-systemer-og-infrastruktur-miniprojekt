import requests

host_port = 8000

# Define the payload
payload = {
    "day": 15,
    "month": 6,
    "hour": 14,
    "Classic": 100,
    "Chicken": 50,
    "Supreme": 30,
    "Veggie": 20
}

# Send a POST request
response = requests.post(f"http://localhost:{host_port}/predict/", json=payload)

# Print the response
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)