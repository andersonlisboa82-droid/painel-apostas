import requests

url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}
payload = {
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 10
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status: {response.status_code}")
    print(response.text)
except Exception as e:
    print(e)
