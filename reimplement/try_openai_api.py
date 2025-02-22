from openai import OpenAI
import json

client = OpenAI(api_key="sk-uewxmkcqxggjneivwulbemyuqhqvbjrfjvecjdpfawyzkahr", base_url="https://api.siliconflow.cn/v1")
response = client.chat.completions.create(
    model='Pro/deepseek-ai/DeepSeek-R1',
    messages=[
        {'role': 'user', 
        'content': "\"什么你太美?\"这句网络用语的前面是什么?"},
    ],
)

# fmt: off
import IPython; IPython.embed()
# fmt: on

# Print the response to verify its structure
print(response)

# Save the response to a JSON file
with open("misc/response.json", "w", encoding='utf-8') as file:
    json.dump( dict(response.choices[0].message), file, ensure_ascii=False, indent=4)  # `indent=4` makes the JSON readable

