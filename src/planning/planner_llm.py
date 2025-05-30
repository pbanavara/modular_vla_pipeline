import anthropic
import os
from planning.prompt_builder import PromptBuilder

class PlannerLLM:
    def __init__(self,
                 robot_yaml_path: str,
                 model: str = "claude-3-7-sonnet-20250219"):
        self.model = model
        self.prompt_builder = PromptBuilder(robot_yaml_path)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _get_claude_plan(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.7,
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
