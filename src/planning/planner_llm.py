import anthropic
import os
from planning.prompt_builder import PromptBuilder

class PlannerLLM:
    def __init__(self,
                 robot_yaml_path: str,
                 model: str = "claude-3-sonnet-20240229"):
        self.model = model
        self.prompt_builder = PromptBuilder(robot_yaml_path)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _get_claude_plan(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.7,
            system=
            """You are a robotic planning assistant specialized in generating precise action sequences for robotic manipulation tasks.
            Your role is to:
            1. Analyze the provided robot specifications, perception data, and task requirements
            2. Generate a detailed sequence of robotic actions in JSON format
            3. Ensure each action is physically feasible given the robot's joint limits and workspace
            4. Generate smooth trajectories with multiple waypoints for each motion
            5. Include precise position and rotation values for every waypoint

            Critical Requirements:
            - Every trajectory MUST contain multiple waypoints (minimum 2-3 per motion)
            - Approach and grasp motions require at least 3-4 waypoints for smoothness
            - All waypoints must have valid position and rotation values
            - Trajectories must enable smooth, continuous motion through all waypoints
            - Motion planning must respect the robot's physical constraints and limits
            """,
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
