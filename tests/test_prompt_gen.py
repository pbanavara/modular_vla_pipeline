import pytest
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prompt_builder import PromptBuilder
with open("../src/aloha.yaml", "r") as f:
    ALOHA_YAML = f.read()

def test_prompt_builder():
    perception_output = [
        {"name": "plate", "labels": ["dirty", "food"]},
        {"name": "cup", "labels": ["clean"]},
    ]

    known_positions = {
        "plate_0": [0.42, 0.26, 0.1],
        "tap_handle": [0.1, 0.5, 0.2],
        "rack": [0.65, 0.4, 0.1],
    }

    task = "Clean the dirty plate"

    builder = PromptBuilder(ALOHA_YAML)
    prompt = builder.build(
        task_instruction=task,
        perception_output=perception_output,
        positions=known_positions,
    )
    print(prompt)

    # Basic content checks
    assert "Aloha Dual Arm" in prompt
    assert "Clean the dirty plate" in prompt
    assert "- plate_0: labels = dirty, food" in prompt
    assert "plate_0 is located at [0.42, 0.26, 0.1]" in prompt
    assert (
        '"position": [x, y, z]' in prompt or '"position":' in prompt
    )  # Loose check for format scaffold
    assert "Return a JSON array" in prompt
