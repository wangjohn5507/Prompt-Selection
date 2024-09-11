import openai
from openai import OpenAI

client = OpenAI(api_key='your_key')
import time

# client = OpenAI(api_key=openai.api_key)

def call_chat_gpt(message, args):
    wait = 1
    while True:
        try:
            ans = client.chat.completions.create(model=args.model,
            max_tokens=args.max_tokens,
            messages=message,
            temperature=args.temperature,
            n=1)
            return ans.choices[0].message.content
        except openai.RateLimitError as e:
            time.sleep(min(wait, 60))
            wait *= 2

def get_embedding(text, model='text-embedding-3-large'):
    response = client.embeddings.create(input=text, model = model)
    return response.data[0].embedding
