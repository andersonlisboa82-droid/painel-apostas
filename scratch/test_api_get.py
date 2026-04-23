import requests

try:
    print("Testing GET /")
    response = requests.get("http://127.0.0.1:8765/", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Headers: {response.headers}")
except Exception as e:
    print(f"Error: {e}")
