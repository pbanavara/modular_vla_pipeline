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

            You receive a high-level task and a perception report from the kitchen sink camera. Your job is to output exact trajectories for the robot arm to perform the task.

            ## Perception Report
            Detected objects:
            {object_text}

            ## Task
            {task_instruction}

            ## Known Positions
            {location_info}

            ## Coordinate System
            - Origin (0,0,0) is at the robot's base
            - X: Positive forward, negative backward
            - Y: Positive right, negative left 
            - Z: Positive up, negative down
            - All positions are in meters
            - Rotations are in radians (Euler angles)

            ## Output Format Instructions
            You must return a JSON array of task steps that forms a complete action sequence to accomplish the task.
            For a grabbing task, you typically need these steps:
            1. Pre-grasp: Move to a position slightly above the target
            2. Approach: Move down to the actual grasp position
            3. Grasp: Close gripper on object
            4. Lift: Move upward with object
            
            Each step must contain these fields:
            - "action": One of ["move_to_pose", "grasp", "release"]
            - "arm": One of ["left", "right"] specifying which arm to use
            - "gripper": One of ["open", "close"] for gripper state
            - "trajectory": List of waypoints, where each waypoint is:
              {{
                "position": [x, y, z],       # 3D coordinates in meters
                "rotation": [rx, ry, rz]     # Euler angles in radians
              }}

            Example grab sequence:
            [
              {{
                "action": "move_to_pose",
                "arm": "left",
                "gripper": "open",
                "trajectory": [
                  {{
                    "position": [target_x - 0.1, target_y, target_z + 0.2],  # Starting position
                    "rotation": [0, 1.57, 0]
                  }},
                  {{
                    "position": [target_x, target_y, target_z + 0.15],  # Pre-grasp 15cm above
                    "rotation": [0, 1.57, 0]  # Gripper facing down
                  }}
                ]
              }},
              {{
                "action": "move_to_pose",
                "arm": "left", 
                "gripper": "open",
                "trajectory": [
                  {{
                    "position": [target_x, target_y, target_z + 0.1],  # Intermediate approach
                    "rotation": [0, 1.57, 0]
                  }},
                  {{
                    "position": [target_x, target_y, target_z + 0.05],  # Slow approach
                    "rotation": [0, 1.57, 0]
                  }},
                  {{
                    "position": [target_x, target_y, target_z + 0.02],  # Just above object
                    "rotation": [0, 1.57, 0]
                  }}
                ]
              }},
              {{
                "action": "grasp",
                "arm": "left",
                "gripper": "close",
                "trajectory": []
              }},
              {{
                "action": "move_to_pose",
                "arm": "left",
                "gripper": "close",
                "trajectory": [
                  {{
                    "position": [target_x, target_y, target_z + 0.05],  # Initial lift
                    "rotation": [0, 1.57, 0]
                  }},
                  {{
                    "position": [target_x, target_y, target_z + 0.15],  # Higher lift
                    "rotation": [0, 1.57, 0]
                  }},
                  {{
                    "position": [target_x - 0.05, target_y, target_z + 0.2],  # Move back slightly
                    "rotation": [0, 1.57, 0]
                  }}
                ]
              }}
            ]

            Requirements:
            1. EVERY trajectory MUST have multiple waypoints for smooth motion:
               - At least 2 waypoints for simple moves
               - At least 3 waypoints for approach and grasp sequences
            2. Pre-grasp position should be 10-15cm above the target
            3. Keep the gripper facing downward (rotation ~[0, 1.57, 0]) for top-down grasps
            4. Move slowly and smoothly when approaching the grasp using intermediate waypoints
            5. Stay within robot workspace (max reach: 65cm radius, 45cm height)
            6. Open gripper before approaching, close when grasping
            7. Lift object gradually with multiple waypoints after grasping
            8. Include waypoints that move slightly away from the target position for better clearance

            Return only valid JSON with no additional text or explanations.
            """
        return prompt.strip()
