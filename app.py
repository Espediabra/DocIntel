import requests

url = "http://localhost:1234/v1/chat/completions"

payload = {
    "model": "qwen2.5-7b-instruct-1m",
    "messages": [
        {
            "role": "system",
            "content": "You are a concise technical assistant. Answer in French."
        },
        {
            "role": "user",
            "content": "Explique ce qu'est un LLM en trois phrases."
        }
    ],
    "temperature": 0.2
}

response = requests.post(url, json=payload)

response.raise_for_status()

data = response.json()

answer = data["choices"][0]["message"]["content"]

print(answer)