import openai

client = openai.OpenAI(base_url="https://48be-149-7-4-5.ngrok-free.app/v1", api_key="token-abc-sundai")

prompt = """Puzzle - """

out = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=[
        {"role": "user", "content": prompt}
    ],

)

print(out.choices[0].message.content)
print(out.choices[0].message.reasoning_content)
