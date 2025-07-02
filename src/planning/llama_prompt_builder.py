import yaml
from typing import Dict, Any, List

class LlamaPromptBuilder:
    """
    A Llama-optimized prompt builder that creates structured, step-by-step instructions
    tailored for Llama model characteristics and capabilities.
    """
    
    def __init__(self, robot_yaml_path: str):
        """
        Initialize the Llama prompt builder with robot configuration.
        
        Args:
            robot_yaml_path: Path to the robot YAML configuration file
        """
        with open(robot_yaml_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.robot_profile = self._extract_robot_profile()
        self.llama_config = self.config.get('llama_config', {})

    def _extract_robot_profile(self) -> str:
        """Extract and format robot profile for Llama models."""
        robot = self.config['robot']
        arms = self.config['arms']
        workspace = self.config['workspace']
        
        profile = f"""
ROBOT SYSTEM: {robot['name']}
TYPE: {robot['type']}
MOUNTING: {robot['mounting']}

CAMERAS: {', '.join(robot['camera'])}

GRIPPER SPECIFICATIONS:
- Type: {robot['grippers']['type']}
- Control: {robot['grippers']['control_type']}
- Range: {robot['grippers']['joint_range_m']} meters
- Max finger spacing: {robot['grippers']['finger_spacing_max_cm']} cm

ARM CONFIGURATIONS:
"""
        
        for arm_name, arm_config in arms.items():
            profile += f"\n{arm_name.upper().replace('_', ' ')}:\n"
            for joint in arm_config['joints']:
                profile += f"  - {joint['name']}: range {joint['range_rad']} rad\n"
        
        profile += f"""
WORKSPACE:
- Type: {workspace['type']}
- Reach radius: {workspace['reach_radius_cm']} cm
- Vertical range: {workspace['vertical_range_cm']} cm
- Operating area: {workspace['operating_area']}
"""
        
        return profile.strip()

    def build(self, task_instruction: str, perception_output: list, positions: dict) -> str:
        """
        Build a Llama-optimized prompt with structured instructions.
        
        Args:
            task_instruction: The high-level task to perform
            perception_output: List of detected objects
            positions: Dictionary of object positions
            
        Returns:
            Formatted prompt optimized for Llama models
        """
        # Format detected objects for Llama
        object_list = self._format_objects_for_llama(perception_output)
        
        # Format positions for Llama
        position_list = self._format_positions_for_llama(positions)
        
        # Build the Llama-optimized prompt
        prompt = f"""
# ROBOT CONTROL INSTRUCTIONS FOR LLAMA

## SYSTEM CONTEXT
You are controlling a dual-arm robot system. Follow these instructions precisely.

## ROBOT SPECIFICATIONS
{self.robot_profile}

## TASK DEFINITION
**Primary Task:** {task_instruction}

## PERCEPTION DATA
**Detected Objects:**
{object_list}

**Object Positions:**
{position_list}

## COORDINATE SYSTEM
- Origin (0,0,0): Robot base center
- X-axis: Forward (+), Backward (-)
- Y-axis: Right (+), Left (-)  
- Z-axis: Up (+), Down (-)
- Units: Meters for positions, Radians for rotations

## ACTION REQUIREMENTS

### Step 1: Plan the Approach
1. Identify the target object from perception data
2. Determine optimal arm (left or right) based on object position
3. Calculate pre-grasp position (10-15cm above target)
4. Plan smooth approach trajectory

### Step 2: Execute Grasp Sequence
1. **Pre-grasp:** Move to position above target with open gripper
2. **Approach:** Move down in stages (15cm → 10cm → 5cm → 2cm)
3. **Grasp:** Close gripper on object
4. **Lift:** Move upward with object in stages

### Step 3: Safety Constraints
- Stay within 65cm radius workspace
- Maintain minimum 2cm clearance from surfaces
- Use smooth, multi-waypoint trajectories
- Check gripper state before each action

## OUTPUT FORMAT
Return ONLY valid JSON array with this exact structure:

```json
[
  {{
    "step": 1,
    "action": "move_to_pose",
    "arm": "left",
    "gripper": "open",
    "description": "Move to pre-grasp position",
    "trajectory": [
      {{
        "position": [x, y, z],
        "rotation": [rx, ry, rz],
        "description": "waypoint description"
      }}
    ]
  }},
  {{
    "step": 2,
    "action": "grasp",
    "arm": "left", 
    "gripper": "close",
    "description": "Close gripper on object",
    "trajectory": []
  }}
]
```

## ACTION TYPES
- `move_to_pose`: Move arm to specific position
- `grasp`: Close gripper to grab object
- `release`: Open gripper to release object

## ARM SELECTION
- `left`: Use left arm
- `right`: Use right arm

## GRIPPER STATES
- `open`: Gripper is open
- `close`: Gripper is closed

## TRAJECTORY REQUIREMENTS
- Every trajectory must have at least 2 waypoints
- Grasp sequences need 3-5 waypoints for smooth motion
- Include intermediate waypoints for safety
- Rotations should be [0, 1.57, 0] for downward-facing gripper

## EXAMPLE GRASP SEQUENCE
For grasping an object at position [0.2, -0.1, -0.3]:

```json
[
  {{
    "step": 1,
    "action": "move_to_pose",
    "arm": "left",
    "gripper": "open", 
    "description": "Move to pre-grasp position",
    "trajectory": [
      {{
        "position": [0.2, -0.1, -0.15],
        "rotation": [0, 1.57, 0],
        "description": "Pre-grasp position 15cm above target"
      }}
    ]
  }},
  {{
    "step": 2,
    "action": "move_to_pose",
    "arm": "left",
    "gripper": "open",
    "description": "Approach target in stages",
    "trajectory": [
      {{
        "position": [0.2, -0.1, -0.2],
        "rotation": [0, 1.57, 0],
        "description": "Intermediate approach 10cm above"
      }},
      {{
        "position": [0.2, -0.1, -0.25],
        "rotation": [0, 1.57, 0],
        "description": "Close approach 5cm above"
      }},
      {{
        "position": [0.2, -0.1, -0.28],
        "rotation": [0, 1.57, 0],
        "description": "Final approach 2cm above"
      }}
    ]
  }},
  {{
    "step": 3,
    "action": "grasp",
    "arm": "left",
    "gripper": "close",
    "description": "Close gripper on object",
    "trajectory": []
  }},
  {{
    "step": 4,
    "action": "move_to_pose",
    "arm": "left",
    "gripper": "close",
    "description": "Lift object safely",
    "trajectory": [
      {{
        "position": [0.2, -0.1, -0.25],
        "rotation": [0, 1.57, 0],
        "description": "Initial lift 5cm above grasp"
      }},
      {{
        "position": [0.2, -0.1, -0.15],
        "rotation": [0, 1.57, 0],
        "description": "Higher lift 15cm above grasp"
      }},
      {{
        "position": [0.15, -0.1, -0.1],
        "rotation": [0, 1.57, 0],
        "description": "Move back and up for clearance"
      }}
    ]
  }}
]
```

## CRITICAL REQUIREMENTS
1. Return ONLY valid JSON - no markdown, no explanations
2. Include step numbers for clarity
3. Add descriptions for each action and waypoint
4. Use smooth, multi-waypoint trajectories
5. Stay within workspace limits
6. Follow safety constraints
7. Use appropriate gripper states

Now generate the action plan for: {task_instruction}
"""
        
        return prompt.strip()

    def _format_objects_for_llama(self, perception_output: list) -> str:
        """Format object list for Llama models with clear structure."""
        if not perception_output:
            return "No objects detected"
        
        formatted = []
        for i, obj in enumerate(perception_output, 1):
            labels = ', '.join(obj['labels'])
            formatted.append(f"{i}. {obj['name']} (Labels: {labels})")
        
        return '\n'.join(formatted)

    def _format_positions_for_llama(self, positions: dict) -> str:
        """Format position data for Llama models with clear structure."""
        if not positions:
            return "No position data available"
        
        formatted = []
        for obj_name, pos in positions.items():
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                formatted.append(f"- {obj_name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] meters")
            else:
                formatted.append(f"- {obj_name}: {pos}")
        
        return '\n'.join(formatted) 