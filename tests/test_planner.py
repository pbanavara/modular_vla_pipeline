import os
from planner_llm import PlannerLLM
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize planner with robot yaml
    robot_yaml_path = "planning/aloha.yaml"
    planner = PlannerLLM(robot_yaml_path)
    
    # Test inputs with actual world coordinates
    mapped_object_name = "plate"
    world_coords = [-0.02535211, -0.15968701, 0.18134938]  # Real coordinates in meters
    
    # Create test perception and position data
    perception_output = [
        {"name": mapped_object_name, "labels": ["plate"]},
    ]
    
    known_positions = {
        mapped_object_name: world_coords
    }
    
    # Task description
    task = "Move the left arm to grab the plate in the sink"
    
    try:
        # Log input data
        logger.info(f"Task: {task}")
        logger.info(f"Perception Output: {perception_output}")
        logger.info(f"Known Positions: {known_positions}")
        
        # Generate plan
        plan = planner.build_action_plan(task, perception_output, known_positions)
        logger.info("\nGenerated Plan:")
        logger.info(plan)
        
        # Save plan to file
        planner.save_plan(plan, "test_plan.json")
        logger.info("Plan saved to test_plan.json")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()