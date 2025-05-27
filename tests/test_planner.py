import unittest
import os
import sys
import json
import numpy as np
import mujoco

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.async_sim.async_simulation import MujocoRealtimeExecutor
from src.utils.utilities import get_resolved_path

class TestMujocoPlanner(unittest.TestCase):
    
    def setUp(self):
        # Setup the MuJoCo model path
        model_path = get_resolved_path("src/simulated_sink/aloha/aloha.xml")
        self.executor = MujocoRealtimeExecutor(str(model_path))
        
        # Mock viewer for testing
        self.mock_viewer = MockViewer()
        
        # Create a sample trajectory
        self.sample_trajectory = [
            {"position": [0.4, 0.1, 0.3], "rotation": [0, 0, 0]},
            {"position": [0.45, 0.15, 0.32], "rotation": [0, 0, 0]},
            {"position": [0.5, 0.2, 0.35], "rotation": [0, 0, 0]},
        ]
    
    def test_trajectory_planning(self):
        """Test that the planner properly enqueues actions"""
        # Create a test plan
        test_plan = [
            {
                "action": "move_to_pose",
                "arm": "left",
                "trajectory": [
                    {"position": [0.4, 0.1, 0.3], "rotation": [0, 0, 0]}
                ]
            },
            {
                "action": "move_to_pose",
                "arm": "right",
                "trajectory": [
                    {"position": [0.5, 0.2, 0.35], "rotation": [0, 0, 0]}
                ]
            }
        ]
        
        # Enqueue the plan
        self.executor.enqueue_plan(test_plan)
        
        # Check queue size
        self.assertEqual(self.executor.queue.qsize(), 2, 
                         "Queue should contain exactly 2 actions")
    
    def test_ik_trajectory_integration(self):
        """Test integration between IK solver and trajectory execution"""
        # Store initial state
        left_arm_joints = self.executor.left_arm_joints
        joint_ids = [mujoco.mj_name2id(self.executor.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                     for name in left_arm_joints]
        
        initial_joint_values = np.array([
            self.executor.data.qpos[self.executor.model.jnt_qposadr[jid]]
            for jid in joint_ids
        ])
        
        # Execute just one step of the trajectory without stepping simulation
        # This tests the IK solving part of move_through_trajectory
        target_pos = np.array(self.sample_trajectory[0]["position"])
        
        # Call solve_ik directly to get solution
        q_solution = self.executor.solve_ik(
            "left/gripper",
            left_arm_joints,
            target_pos,
            None,
            q_init=initial_joint_values
        )
        
        # Verify solution is valid
        self.assertEqual(len(q_solution), len(left_arm_joints))
        
        # Apply solution to joints and check end effector position
        temp_data = mujoco.MjData(self.executor.model)
        temp_data.qpos[:] = self.executor.data.qpos[:]
        
        for i, joint_name in enumerate(left_arm_joints):
            joint_id = mujoco.mj_name2id(self.executor.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_index = self.executor.model.jnt_qposadr[joint_id]
            temp_data.qpos[qpos_index] = q_solution[i]
        
        mujoco.mj_forward(self.executor.model, temp_data)
        
        # Get end effector position
        site_id = mujoco.mj_name2id(self.executor.model, mujoco.mjtObj.mjOBJ_SITE, "left/gripper")
        actual_pos = temp_data.site_xpos[site_id].copy()
        
        # Verify position is close to target
        error = np.linalg.norm(actual_pos - target_pos)
        self.assertLessEqual(error, 0.01, 
                            f"Position error {error} exceeds tolerance. "
                            f"Target: {target_pos}, Actual: {actual_pos}")


class MockViewer:
    """Mock viewer class for testing without visualization"""
    def sync(self):
        pass


if __name__ == "__main__":
    unittest.main()