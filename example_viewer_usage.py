#!/usr/bin/env python3
"""
Example script demonstrating how to use the ViewerManager to handle existing MuJoCo viewers.
This script shows how to safely launch viewers without crashing when one is already open.
"""

import mujoco
import mujoco.viewer
import time
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.viewer_manager import safe_launch_viewer, close_all_viewers, get_viewer_manager

def example_basic_usage():
    """Basic example of using the viewer manager."""
    print("=== Basic Viewer Manager Usage ===")
    
    # Load a simple MuJoCo model (you'll need to provide your own model path)
    model_path = "src/simulated_sink/aloha/aloha.xml"  # Update this path
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please update the model_path variable to point to your MuJoCo XML file.")
        return
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        print("1. Launching first viewer...")
        viewer1 = safe_launch_viewer(model, data)
        print("   ✓ First viewer launched successfully")
        
        # Simulate some work
        for i in range(50):
            mujoco.mj_step(model, data)
            viewer1.sync()
            time.sleep(0.01)
        
        print("2. Trying to launch second viewer (should reuse existing)...")
        viewer2 = safe_launch_viewer(model, data)
        print("   ✓ Second viewer request handled (reused existing)")
        
        # Simulate more work
        for i in range(50):
            mujoco.mj_step(model, data)
            viewer2.sync()
            time.sleep(0.01)
        
        print("3. Force restarting viewer...")
        viewer3 = safe_launch_viewer(model, data, force_restart=True)
        print("   ✓ Viewer force restarted successfully")
        
        # Final simulation
        for i in range(50):
            mujoco.mj_step(model, data)
            viewer3.sync()
            time.sleep(0.01)
        
        print("4. Closing all viewers...")
        close_all_viewers()
        print("   ✓ All viewers closed")
        
    except Exception as e:
        print(f"Error: {e}")
        close_all_viewers()

def example_error_handling():
    """Example showing error handling with the viewer manager."""
    print("\n=== Error Handling Example ===")
    
    model_path = "src/simulated_sink/aloha/aloha.xml"  # Update this path
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        print("1. Simulating viewer crash scenario...")
        
        # Launch initial viewer
        viewer = safe_launch_viewer(model, data)
        print("   ✓ Initial viewer launched")
        
        # Simulate some work
        for i in range(30):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
        
        print("2. Simulating viewer becoming unresponsive...")
        # In a real scenario, the viewer might become unresponsive
        # The viewer manager will detect this and handle it gracefully
        
        print("3. Trying to get a new viewer (should handle gracefully)...")
        new_viewer = safe_launch_viewer(model, data)
        print("   ✓ New viewer request handled successfully")
        
        # Continue simulation
        for i in range(30):
            mujoco.mj_step(model, data)
            new_viewer.sync()
            time.sleep(0.01)
        
        print("4. Cleanup...")
        close_all_viewers()
        print("   ✓ Cleanup completed")
        
    except Exception as e:
        print(f"Error: {e}")
        close_all_viewers()

def example_multiple_launches():
    """Example showing multiple rapid launches."""
    print("\n=== Multiple Rapid Launches Example ===")
    
    model_path = "src/simulated_sink/aloha/aloha.xml"  # Update this path
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        print("Launching viewers rapidly (should handle gracefully)...")
        
        for i in range(5):
            print(f"  Launch attempt {i+1}/5...")
            viewer = safe_launch_viewer(model, data)
            
            # Quick simulation
            for j in range(10):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)
            
            print(f"    ✓ Launch {i+1} successful")
        
        print("All launches completed successfully!")
        close_all_viewers()
        
    except Exception as e:
        print(f"Error: {e}")
        close_all_viewers()

if __name__ == "__main__":
    print("MuJoCo Viewer Manager Examples")
    print("=" * 40)
    
    # Check if we have a model file
    model_path = "src/simulated_sink/aloha/aloha.xml"
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Please update the model_path in the script to point to your MuJoCo XML file.")
        print("You can still run the examples, but they will exit early.")
    
    try:
        example_basic_usage()
        example_error_handling()
        example_multiple_launches()
        
        print("\n" + "=" * 40)
        print("All examples completed successfully!")
        print("\nKey features demonstrated:")
        print("- Safe viewer launching without crashes")
        print("- Automatic reuse of existing viewers")
        print("- Force restart capability")
        print("- Graceful error handling")
        print("- Process cleanup")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
        close_all_viewers()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        close_all_viewers() 