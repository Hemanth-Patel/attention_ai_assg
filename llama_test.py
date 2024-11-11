import ollama
response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': 'Hello',
  },
])
print(response['message']['content'])


# stream = ollama.chat(
#     model='llama3.1',
#     messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
#     stream=True,
# )

# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)