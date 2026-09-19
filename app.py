import requests

url = "http://localhost:1234/v1/chat/completions"

payload = {
    "model": "qwen2.5-7b-instruct-1m",
    "messages": [
        {
            "role": "user",
            "content": "Explain RAG in three sentences."
        }
    ],
    "temperature": 0.2
}

response = requests.post(url, json=payload)

response.raise_for_status()

data = response.json()

print(data["choices"][0]["message"]["content"])