from services.llm_client import LLMClient


llm = LLMClient()

messages = [
    {
        "role": "system",
        "content": "You are a concise technical assistant. Answer in French.",
    },
    {
        "role": "user",
        "content": "Explique ce qu'est un LLM en trois phrases.",
    },
]

answer = llm.generate(messages)

print(answer)