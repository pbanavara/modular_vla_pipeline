import unittest
import os
import sys
import json
import numpy as np
import mujoco
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.async_sim.async_simulation import MujocoRealtimeExecutor
from src.utils.utilities import get_resolved_path

class TestPromptGeneration(unittest.TestCase):
    
    def setUp(self):
        # Setup the MuJoCo model path
        model_path = get_resolved_path("src/simulated_sink/aloha/aloha.xml")
        self.executor = MujocoRealtimeExecutor(str(model_path))
    
    def test_text_prompts_initialization(self):
        """Test that text prompts are properly initialized"""
        # Check that text_prompts is a list
        self.assertIsInstance(self.executor.text_prompts, list)
        
        # Check that text_prompts contains expected items
        self.assertTrue(any("glass" in prompt for prompt in self.executor.text_prompts),
                       "Text prompts should include glass-related prompts")
        
        self.assertTrue(any("plate" in prompt for prompt in self.executor.text_prompts),
                       "Text prompts should include plate-related prompts")
        
        self.assertTrue(any("sink" in prompt for prompt in self.executor.text_prompts),
                       "Text prompts should include sink-related prompts")
    
    @patch('src.pipeline.async_sim.async_simulation.camera_capture.CameraCapture')
    def test_capture_image(self, mock_camera_capture):
        """Test image capture functionality"""
        # Create a mock image
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Configure the mock
        mock_instance = mock_camera_capture.return_value
        mock_instance.capture_image.return_value = mock_image
        
        # Call the capture_image method
        result = self.executor.capture_image()
        
        # Check that camera_capture was called with correct parameters
        mock_camera_capture.assert_called_once_with(model_path=self.executor.model_path)
        mock_instance.capture_image.assert_called_once_with("teleoperator_pov")
        
        # Check that result is a PIL Image
        from PIL import Image
        self.assertIsInstance(result, Image.Image)
        
        # Check image dimensions match expected resize values
        self.assertEqual(result.size, (495, 374))
    
    def test_map_model_detections(self):
        """Test object mapping functionality"""
        # Test known mappings
        self.assertEqual(self.executor.map_model_detections("a plate"), "plate_geom")
        self.assertEqual(self.executor.map_model_detections("a bowl"), "bowl_geom")
        self.assertEqual(self.executor.map_model_detections("a cup"), "cup_geom")
        
        # Test unknown mapping raises ValueError
        with self.assertRaises(ValueError):
            self.executor.map_model_detections("an unknown object")
    
    @patch('src.pipeline.async_sim.async_simulation.SAMSegmentation')
    def test_build_sam_segmentation(self, mock_sam):
        """Test SAM segmentation initialization"""
        # Configure the mock
        mock_instance = mock_sam.return_value
        
        # Call the method
        result = self.executor.build_sam_segmentation()
        
        # Check SAM was initialized with correct parameters
        mock_sam.assert_called_once()
        self.assertEqual(result, mock_instance)


if __name__ == "__main__":
    unittest.main()