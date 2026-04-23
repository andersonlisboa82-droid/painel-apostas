import requests

try:
    response = requests.options("http://127.0.0.1:8765/api/ai-analysis")
    print(f"OPTIONS Status: {response.status_code}")
    print(f"Headers: {response.headers}")
    
    response = requests.post("http://127.0.0.1:8765/api/ai-analysis", json={"selected_date": "2026-04-23", "prompt": "test"})
    print(f"POST Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
