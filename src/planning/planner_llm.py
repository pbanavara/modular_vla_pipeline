import anthropic
import os
from planning.prompt_builder import PromptBuilder

class PlannerLLM:
    def __init__(self, 
                 robot_yaml_path: str, 
                 model: str = "claude-3-opus-20240229"):
        self.model = model
        self.prompt_builder = PromptBuilder(robot_yaml_path)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _get_claude_plan(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.2,
            system="You are a robotic planning assistant. Follow the user’s instructions exactly. Output JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def build_action_plan(
        self, task: str, perception_output: list, positions: dict
    ) -> str:
        prompt = self.prompt_builder.build(task, perception_output, positions)
        return self._get_claude_plan(prompt)

    def save_plan(self, plan: str, filename: str) -> None:
        with open(filename, "w") as f:
            f.write(plan)

def main():
    task = "Clean the dirty plate"
    perception_output = [
        {"name": "plate", "labels": ["dirty", "food"]},
        {"name": "cup", "labels": ["clean"]},
    ]
    known_positions = {
        "plate_0": [0.42, 0.26, 0.1],
        "tap_handle": [0.1, 0.5, 0.2],
        "rack": [0.65, 0.4, 0.1],
    }

    planner = PlannerLLM(robot_yaml_path="ALOHA.yaml")
    plan = planner.build_action_plan(task, perception_output, known_positions)
    planner.save_plan(plan, "plan.json")

if __name__ == "__main__":
    main()