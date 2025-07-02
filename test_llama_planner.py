#!/usr/bin/env python3
"""
Simple test script for Llama planner functionality.
This script tests the basic structure and imports without requiring API keys.
"""

import sys
import os
import json

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("=== Testing Imports ===")
    
    try:
        from planning.llama_prompt_builder import LlamaPromptBuilder
        print("✅ LlamaPromptBuilder imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LlamaPromptBuilder: {e}")
        return False
    
    try:
        from planning.llama_planner import LlamaPlanner
        print("✅ LlamaPlanner imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LlamaPlanner: {e}")
        return False
    
    try:
        from planning.llm_config import validate_config
        print("✅ llm_config imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import llm_config: {e}")
        return False
    
    return True

def test_yaml_config():
    """Test that the Llama YAML configuration can be loaded."""
    print("\n=== Testing YAML Configuration ===")
    
    yaml_path = "src/planning/llama.yaml"
    if not os.path.exists(yaml_path):
        print(f"❌ Llama YAML file not found at {yaml_path}")
        return False
    
    try:
        import yaml
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Check required sections
        required_sections = ['robot', 'arms', 'workspace', 'llama_config']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required section: {section}")
                return False
        
        print("✅ YAML configuration loaded successfully")
        print(f"   Robot: {config['robot']['name']}")
        print(f"   Arms: {list(config['arms'].keys())}")
        print(f"   Llama config sections: {list(config['llama_config'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load YAML configuration: {e}")
        return False

def test_prompt_builder():
    """Test the Llama prompt builder functionality."""
    print("\n=== Testing Prompt Builder ===")
    
    yaml_path = "src/planning/llama.yaml"
    if not os.path.exists(yaml_path):
        print(f"❌ Llama YAML file not found at {yaml_path}")
        return False
    
    try:
        from planning.llama_prompt_builder import LlamaPromptBuilder
        
        # Initialize prompt builder
        prompt_builder = LlamaPromptBuilder(yaml_path)
        print("✅ LlamaPromptBuilder initialized successfully")
        
        # Test prompt generation
        task = "pick up the red cup"
        perception_output = [
            {"name": "red_cup", "labels": ["cup", "red", "container"]}
        ]
        positions = {
            "red_cup": [0.3, -0.2, -0.35]
        }
        
        prompt = prompt_builder.build(task, perception_output, positions)
        
        # Check that prompt contains expected elements
        expected_elements = [
            "ROBOT CONTROL INSTRUCTIONS FOR LLAMA",
            "TASK DEFINITION",
            "PERCEPTION DATA",
            "COORDINATE SYSTEM",
            "ACTION REQUIREMENTS",
            "OUTPUT FORMAT"
        ]
        
        for element in expected_elements:
            if element in prompt:
                print(f"✅ Prompt contains: {element}")
            else:
                print(f"❌ Prompt missing: {element}")
                return False
        
        print(f"✅ Generated prompt length: {len(prompt)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test prompt builder: {e}")
        return False

def test_planner_initialization():
    """Test Llama planner initialization (without API calls)."""
    print("\n=== Testing Planner Initialization ===")
    
    yaml_path = "src/planning/llama.yaml"
    if not os.path.exists(yaml_path):
        print(f"❌ Llama YAML file not found at {yaml_path}")
        return False
    
    try:
        from planning.llama_planner import LlamaPlanner
        
        # Test initialization without API key (should work but warn)
        planner = LlamaPlanner(
            robot_yaml_path=yaml_path,
            model="llama-3.2-70b-instruct",
            enable_caching=True,
            cache_size=100,
            max_workers=2
        )
        
        print("✅ LlamaPlanner initialized successfully")
        print(f"   Model: {planner.model}")
        print(f"   Caching enabled: {planner.enable_caching}")
        print(f"   Cache size: {planner._cache_size}")
        print(f"   Max workers: {planner._max_workers}")
        
        # Test performance stats (should work without API calls)
        stats = planner.get_performance_stats()
        print(f"   Initial stats: {stats['total_requests']} requests")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize planner: {e}")
        return False

def test_cache_functionality():
    """Test the caching functionality."""
    print("\n=== Testing Cache Functionality ===")
    
    try:
        from planning.llama_planner import LlamaPlanner
        
        # Initialize planner with caching
        planner = LlamaPlanner(
            enable_caching=True,
            cache_size=10
        )
        
        # Test cache key generation
        task = "test task"
        perception_output = [{"name": "test_obj", "labels": ["test"]}]
        positions = {"test_obj": [0.1, 0.2, 0.3]}
        
        cache_key = planner._generate_cache_key(task, perception_output, positions)
        print(f"✅ Generated cache key: {cache_key[:16]}...")
        
        # Test cache operations
        test_response = '{"test": "response"}'
        planner._cache_response(cache_key, test_response)
        
        cached_response = planner._get_cached_response(cache_key)
        if cached_response == test_response:
            print("✅ Cache storage and retrieval working")
        else:
            print("❌ Cache storage and retrieval failed")
            return False
        
        # Test cache stats
        stats = planner.get_performance_stats()
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache misses: {stats['cache_misses']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test cache functionality: {e}")
        return False

def test_json_validation():
    """Test JSON validation and fixing functionality."""
    print("\n=== Testing JSON Validation ===")
    
    try:
        from planning.llama_planner import LlamaPlanner
        
        planner = LlamaPlanner()
        
        # Test valid JSON
        valid_json = '{"test": "valid"}'
        result = planner._validate_and_fix_json(valid_json)
        if result == valid_json:
            print("✅ Valid JSON passed through unchanged")
        else:
            print("❌ Valid JSON was modified")
            return False
        
        # Test JSON with markdown
        json_with_markdown = '```json\n{"test": "with markdown"}\n```'
        result = planner._validate_and_fix_json(json_with_markdown)
        if '{"test": "with markdown"}' in result:
            print("✅ JSON with markdown fixed successfully")
        else:
            print("❌ Failed to fix JSON with markdown")
            return False
        
        # Test malformed JSON
        malformed_json = 'Some text before [{"test": "malformed"} and after'
        result = planner._validate_and_fix_json(malformed_json)
        if '{"test": "malformed"}' in result:
            print("✅ Malformed JSON fixed successfully")
        else:
            print("❌ Failed to fix malformed JSON")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test JSON validation: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Llama Planner Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_yaml_config,
        test_prompt_builder,
        test_planner_initialization,
        test_cache_functionality,
        test_json_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Llama planner is ready to use.")
        print("\nTo use with real API calls, set your LAMBDA_API_KEY:")
        print("export LAMBDA_API_KEY='your_api_key_here'")
        print("\nThen run: python example_llama_planner.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 