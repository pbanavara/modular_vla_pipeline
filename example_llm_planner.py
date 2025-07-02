#!/usr/bin/env python3
"""
Example script demonstrating how to use the new LLM planner with Lambda Labs.
This script shows how to switch between different LLM providers and generate robotic action plans.
"""

import sys
import os
import json

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from planning.planner_llm import PlannerLLM
from planning.llm_config import get_llm_config, validate_config

def example_lambda_labs_usage():
    """Example using Lambda Labs Llama 4 Maverick 17B."""
    print("=== Lambda Labs Llama 4 Maverick 17B Example ===")
    
    # Check if API key is available
    if not validate_config("lambda_labs"):
        print("❌ Lambda Labs API key not found. Please set LAMBDA_API_KEY environment variable.")
        return False
    
    try:
        # Initialize planner with Lambda Labs
        planner = PlannerLLM(
            robot_yaml_path="src/planning/aloha.yaml",
            provider="lambda_labs"
        )
        
        print("✅ Lambda Labs planner initialized successfully")
        print(f"🔗 Using API endpoint: {planner.api_base}")
        print(f"🤖 Using model: {planner.model}")
        
        # Example task and perception data
        task = "wash the plate"
        perception_output = [
            {"name": "plate_geom", "labels": ["plate", "ceramic"]}
        ]
        positions = {
            "plate_geom": [0.25, -0.27, -0.42]
        }
        
        print(f"📋 Task: {task}")
        print(f"🔍 Detected objects: {perception_output}")
        print(f"📍 Object positions: {positions}")
        
        # Generate plan
        print("\n🤖 Generating action plan...")
        plan = planner.build_action_plan(task, perception_output, positions)
        
        print("✅ Plan generated successfully!")
        print("\n📄 Generated Plan:")
        print(plan)
        
        # Save plan
        planner.save_plan(plan, "example_plan_lambda.json")
        print("\n💾 Plan saved to example_plan_lambda.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def example_openai_usage():
    """Example using OpenAI GPT-4 (if available)."""
    print("\n=== OpenAI GPT-4 Example ===")
    
    # Check if API key is available
    if not validate_config("openai"):
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return False
    
    try:
        # Initialize planner with OpenAI
        planner = PlannerLLM(
            robot_yaml_path="src/planning/aloha.yaml",
            provider="openai"
        )
        
        print("✅ OpenAI planner initialized successfully")
        print(f"🔗 Using API endpoint: {planner.api_base}")
        print(f"🤖 Using model: {planner.model}")
        
        # Example task and perception data
        task = "pick up the bowl and place it in the sink"
        perception_output = [
            {"name": "bowl_geom", "labels": ["bowl", "ceramic"]},
            {"name": "sink_geom", "labels": ["sink", "stainless steel"]}
        ]
        positions = {
            "bowl_geom": [0.3, 0.1, -0.35],
            "sink_geom": [0.0, 0.0, -0.4]
        }
        
        print(f"📋 Task: {task}")
        print(f"🔍 Detected objects: {perception_output}")
        print(f"📍 Object positions: {positions}")
        
        # Generate plan
        print("\n🤖 Generating action plan...")
        plan = planner.build_action_plan(task, perception_output, positions)
        
        print("✅ Plan generated successfully!")
        print("\n📄 Generated Plan:")
        print(plan)
        
        # Save plan
        planner.save_plan(plan, "example_plan_openai.json")
        print("\n💾 Plan saved to example_plan_openai.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def example_provider_comparison():
    """Compare different providers."""
    print("\n=== Provider Comparison ===")
    
    providers = ["lambda_labs", "openai"]
    results = {}
    
    for provider in providers:
        print(f"\n🔍 Testing {provider}...")
        if validate_config(provider):
            config = get_llm_config(provider)
            print(f"✅ {provider} configuration is valid")
            print(f"   API Base: {config['api_base']}")
            print(f"   Model: {config['model']}")
            results[provider] = True
        else:
            print(f"❌ {provider} configuration is invalid")
            results[provider] = False
    
    print(f"\n📊 Results: {results}")
    return results

def example_custom_configuration():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    try:
        # Initialize planner with custom settings
        planner = PlannerLLM(
            robot_yaml_path="src/planning/aloha.yaml",
            provider="lambda_labs",
            model="llama-4-maverick-17b-128e-instruct-fp8",  # Explicit model
            api_key="your_custom_api_key"  # Custom API key
        )
        
        print("✅ Custom planner initialized successfully")
        print(f"🔗 Using API endpoint: {planner.api_base}")
        print(f"🤖 Using model: {planner.model}")
        
        # Test with simple task
        task = "move the cup to the left"
        perception_output = [
            {"name": "cup_geom", "labels": ["cup", "glass"]}
        ]
        positions = {
            "cup_geom": [0.2, 0.0, -0.3]
        }
        
        print(f"📋 Task: {task}")
        
        # Generate plan
        plan = planner.build_action_plan(task, perception_output, positions)
        print("✅ Custom plan generated successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def example_direct_lambda_labs_test():
    """Direct test using Lambda Labs API like their example."""
    print("\n=== Direct Lambda Labs API Test ===")
    
    try:
        import base64
        from openai import OpenAI
        
        # Check if API key is available
        api_key = os.getenv("LAMBDA_API_KEY")
        if not api_key:
            print("❌ LAMBDA_API_KEY not found in environment variables")
            return False
        
        # Initialize client like in the Lambda Labs example
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.lambda.ai/v1",
        )
        
        model = "llama-4-maverick-17b-128e-instruct-fp8"
        
        # Test with a simple text prompt (no image for now)
        message = {
            "role": "user",
            "content": "Hello! Can you help me with robotic planning? Just say 'Yes, I can help with robotic planning!'"
        }
        
        print("🔗 Testing direct Lambda Labs API connection...")
        chat_response = client.chat.completions.create(
            model=model,
            messages=[message],
            max_tokens=100,
            temperature=0.1
        )
        
        response_content = chat_response.choices[0].message.content
        print(f"✅ Lambda Labs API test successful!")
        print(f"🤖 Response: {response_content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct API test failed: {e}")
        return False

def main():
    """Main function to run all examples."""
    print("LLM Planner Examples")
    print("=" * 50)
    
    # Check if robot YAML file exists
    yaml_path = "src/planning/aloha.yaml"
    if not os.path.exists(yaml_path):
        print(f"⚠️  Warning: Robot YAML file not found at {yaml_path}")
        print("Please ensure the aloha.yaml file exists for full functionality.")
    
    # Run examples
    results = {}
    
    # Test direct Lambda Labs API first
    results["direct_api_test"] = example_direct_lambda_labs_test()
    
    # Test Lambda Labs planner
    results["lambda_labs"] = example_lambda_labs_usage()
    
    # Test OpenAI (if available)
    results["openai"] = example_openai_usage()
    
    # Compare providers
    results["comparison"] = example_provider_comparison()
    
    # Test custom configuration
    results["custom"] = example_custom_configuration()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print("\nKey Features Demonstrated:")
    print("- Multi-provider LLM support (Lambda Labs, OpenAI)")
    print("- Automatic configuration management")
    print("- Error handling and validation")
    print("- Plan generation and saving")
    print("- Custom configuration options")
    print("- Direct API testing")
    
    print("\nTo use in your pipeline:")
    print("1. Set your API key: export LAMBDA_API_KEY='your_key_here'")
    print("2. Initialize planner: planner = PlannerLLM(robot_yaml_path='path/to/robot.yaml')")
    print("3. Generate plans: plan = planner.build_action_plan(task, perception, positions)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}") 