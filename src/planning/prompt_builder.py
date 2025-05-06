class PromptBuilder:
    """
    A class to build prompts for the LLM based on robot profile, task instruction, perception output, and positions.
    """
    def __init__(self, robot_profile: str):
        self.robot_profile = robot_profile.strip()

    def build(self, task_instruction: str, 
              perception_output: list, 
              positions: dict):
        # List all perceived objects
        object_lines = [
            f"- {obj['name']}_{i}: labels = {', '.join(obj['labels'])}"
            for i, obj in enumerate(perception_output)
        ]
        object_text = "\n".join(object_lines)

        # Insert known coordinates for task-relevant items
        location_info = "\n".join(
            [f"{k} is located at {v}" for k, v in positions.items()]
        )

        # Final assembled prompt
        prompt = f"""
            You are a robotic control assistant for the following robot:

            {self.robot_profile}

            You receive a high-level task and a perception report from the kitchen sink camera. Your job is to output exact joint-space-compatible trajectories for the robot arm to perform the task.

            ## Perception Report
            Detected objects:
            {object_text}

            ## Task
            {task_instruction}

            ## Known Positions
            {location_info}

            ## Output Format
            Return a JSON array of task steps. Each step must contain:
            - action : "move_to_pose" | "grasp" | "release"
            - arm : "left" | "right"
            - gripper : "open" | "close"
            - trajectory: list of {{ position: [x, y, z], rotation: [rx, ry, rz] }}

            Only return the JSON. No explanation or commentary.
            """
        return prompt.strip()
