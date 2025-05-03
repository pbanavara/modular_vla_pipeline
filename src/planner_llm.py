import anthropic
import os
from prompt_builder import PromptBuilder

# Load API key from env
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_claude_plan(prompt: str, model="claude-3-opus-20240229") -> str:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.2,
        system="You are a robotic planning assistant. Follow the user’s instructions exactly. Output JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# Use YAML from earlier
with open("aloha.yaml", "r") as f:
    robot_yaml = f.read()

perception_output = [
    {"name": "plate", "labels": ["dirty", "food"]},
    {"name": "cup", "labels": ["clean"]},
]
positions = {
    "plate_0": [0.42, 0.26, 0.1],
    "tap_handle": [0.1, 0.5, 0.2],
    "rack": [0.65, 0.4, 0.1],
}
task = "Clean the dirty plate"

builder = PromptBuilder(robot_yaml)
prompt = builder.build(task, perception_output, positions)

# Call Claude
response = get_claude_plan(prompt)
print(response)
