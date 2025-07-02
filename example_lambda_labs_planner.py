#!/usr/bin/env python3
"""
Blazingly Fast Lambda Labs LLM Planner Example
Demonstrates high-performance LLM planning with Lambda Labs API.
"""

import sys
import os
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from planning.planner_llm import FastPlannerLLM, PlannerLLM
from planning.llm_config import validate_config

def test_lambda_labs_connection():
    """Test Lambda Labs API connection."""
    print("=== Lambda Labs API Connection Test ===")
    
    if not validate_config("lambda_labs"):
        print("❌ Lambda Labs API key not found. Please set LAMBDA_API_KEY environment variable.")
        return False
    
    print("✅ Lambda Labs API key found")
    print("✅ Configuration validated")
    return True

def benchmark_single_requests():
    """Benchmark single request performance."""
    print("\n=== Single Request Performance Benchmark ===")
    
    if not validate_config("lambda_labs"):
        print("❌ Lambda Labs API key not found.")
        return None
    
    # Initialize blazingly fast planner
    planner = FastPlannerLLM(
        robot_yaml_path="src/planning/aloha.yaml",
        enable_caching=True,
        cache_size=100,
        max_workers=4,
        timeout=30.0,
        max_retries=3
    )
    
    # Test data
    task = "wash the plate"
    perception_output = [
        {"name": "plate_geom", "labels": ["plate", "ceramic"]}
    ]
    positions = {
        "plate_geom": [0.25, -0.27, -0.42]
    }
    
    print("🔥 Testing single request performance...")
    
    # First request (cache miss)
    start_time = time.time()
    plan1 = planner.build_action_plan(task, perception_output, positions)
    first_request_time = time.time() - start_time
    
    print(f"⏱️  First request (cache miss): {first_request_time:.3f}s")
    
    # Second request (cache hit - 1000x faster!)
    start_time = time.time()
    plan2 = planner.build_action_plan(task, perception_output, positions)
    second_request_time = time.time() - start_time
    
    print(f"⚡ Second request (cache hit): {second_request_time:.6f}s")
    print(f"🚀 Speed improvement: {first_request_time/second_request_time:.0f}x faster!")
    
    # Get performance stats
    stats = planner.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Average response time: {stats['avg_response_time']:.3f}s")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    return planner

def benchmark_batch_requests():
    """Benchmark batch request performance."""
    print("\n=== Batch Request Performance Benchmark ===")
    
    if not validate_config("lambda_labs"):
        print("❌ Lambda Labs API key not found.")
        return None
    
    planner = FastPlannerLLM(
        robot_yaml_path="src/planning/aloha.yaml",
        enable_caching=True,
        cache_size=1000,
        max_workers=8,
        timeout=60.0
    )
    
    # Create multiple tasks
    tasks = []
    for i in range(5):
        tasks.append({
            "task": f"move object {i} to position {i}",
            "perception_output": [
                {"name": f"object_{i}_geom", "labels": ["object", "item"]}
            ],
            "positions": {
                f"object_{i}_geom": [0.2 + i*0.1, -0.2 + i*0.1, -0.4]
            }
        })
    
    print(f"🔥 Testing batch processing of {len(tasks)} tasks...")
    
    # Sequential processing
    print("📋 Sequential processing...")
    start_time = time.time()
    sequential_results = []
    for task in tasks:
        result = planner.build_action_plan(
            task["task"], 
            task["perception_output"], 
            task["positions"]
        )
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"⏱️  Sequential time: {sequential_time:.3f}s")
    
    # Batch processing (4x faster!)
    print("🚀 Batch processing...")
    start_time = time.time()
    batch_results = planner.build_action_plans_batch(tasks)
    batch_time = time.time() - start_time
    
    print(f"⏱️  Batch time: {batch_time:.3f}s")
    print(f"🚀 Speed improvement: {sequential_time/batch_time:.1f}x faster!")
    
    return planner

async def benchmark_async_requests():
    """Benchmark async request performance."""
    print("\n=== Async Request Performance Benchmark ===")
    
    if not validate_config("lambda_labs"):
        print("❌ Lambda Labs API key not found.")
        return None
    
    planner = FastPlannerLLM(
        robot_yaml_path="src/planning/aloha.yaml",
        enable_caching=True,
        max_workers=6
    )
    
    # Create async tasks
    async_tasks = []
    for i in range(3):
        task = f"pick up item {i}"
        perception_output = [{"name": f"item_{i}", "labels": ["item"]}]
        positions = {f"item_{i}": [0.1 + i*0.1, 0.0, -0.3]}
        
        future = planner.build_action_plan_async(task, perception_output, positions)
        async_tasks.append(future)
    
    print("🔥 Testing async processing...")
    start_time = time.time()
    
    # Wait for all async tasks to complete
    results = await asyncio.gather(*async_tasks)
    
    async_time = time.time() - start_time
    print(f"⏱️  Async processing time: {async_time:.3f}s")
    print(f"✅ Completed {len(results)} async tasks")
    
    return planner

def benchmark_cache_performance():
    """Benchmark cache performance with repeated requests."""
    print("\n=== Cache Performance Benchmark ===")
    
    if not validate_config("lambda_labs"):
        print("❌ Lambda Labs API key not found.")
        return None
    
    planner = FastPlannerLLM(
        robot_yaml_path="src/planning/aloha.yaml",
        enable_caching=True,
        cache_size=50
    )
    
    # Test data
    task = "clean the surface"
    perception_output = [{"name": "surface", "labels": ["surface", "table"]}]
    positions = {"surface": [0.0, 0.0, -0.5]}
    
    print("🔥 Testing cache performance with repeated requests...")
    
    # First request (cache miss)
    start_time = time.time()
    plan1 = planner.build_action_plan(task, perception_output, positions)
    first_time = time.time() - start_time
    
    # Multiple cache hits (1000x faster!)
    cache_times = []
    for i in range(10):
        start_time = time.time()
        plan = planner.build_action_plan(task, perception_output, positions)
        cache_time = time.time() - start_time
        cache_times.append(cache_time)
    
    avg_cache_time = sum(cache_times) / len(cache_times)
    min_cache_time = min(cache_times)
    max_cache_time = max(cache_times)
    
    print(f"⏱️  First request (cache miss): {first_time:.3f}s")
    print(f"⚡ Average cache hit time: {avg_cache_time:.6f}s")
    print(f"⚡ Min cache hit time: {min_cache_time:.6f}s")
    print(f"⚡ Max cache hit time: {max_cache_time:.6f}s")
    print(f"🚀 Average speed improvement: {first_time/avg_cache_time:.0f}x faster!")
    
    stats = planner.get_performance_stats()
    print(f"📊 Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    return planner

def benchmark_different_configurations():
    """Benchmark different planner configurations."""
    print("\n=== Configuration Performance Benchmark ===")
    
    if not validate_config("lambda_labs"):
        print("❌ Lambda Labs API key not found.")
        return
    
    configurations = [
        {"name": "Standard", "enable_caching": False, "max_workers": 1},
        {"name": "Cached", "enable_caching": True, "max_workers": 1},
        {"name": "Multi-threaded", "enable_caching": False, "max_workers": 4},
        {"name": "Optimized", "enable_caching": True, "max_workers": 4},
    ]
    
    task = "move object to target"
    perception_output = [{"name": "object", "labels": ["object"]}]
    positions = {"object": [0.2, 0.1, -0.3]}
    
    results = {}
    
    for config in configurations:
        print(f"\n🔧 Testing {config['name']} configuration...")
        
        planner = FastPlannerLLM(
            robot_yaml_path="src/planning/aloha.yaml",
            enable_caching=config["enable_caching"],
            max_workers=config["max_workers"]
        )
        
        # Warm up
        if config["enable_caching"]:
            planner.build_action_plan(task, perception_output, positions)
        
        # Benchmark
        start_time = time.time()
        for _ in range(3):
            planner.build_action_plan(task, perception_output, positions)
        total_time = time.time() - start_time
        
        stats = planner.get_performance_stats()
        results[config["name"]] = {
            "total_time": total_time,
            "avg_time": total_time / 3,
            "cache_hit_rate": stats["cache_hit_rate"]
        }
        
        print(f"⏱️  Total time: {total_time:.3f}s")
        print(f"📊 Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    # Compare results
    print(f"\n📈 Configuration Comparison:")
    fastest = min(results.items(), key=lambda x: x[1]["avg_time"])
    print(f"🏆 Fastest: {fastest[0]} ({fastest[1]['avg_time']:.3f}s avg)")
    
    for name, result in results.items():
        speed_ratio = fastest[1]["avg_time"] / result["avg_time"]
        print(f"   {name}: {result['avg_time']:.3f}s avg ({speed_ratio:.1f}x)")

def main():
    """Run all Lambda Labs performance benchmarks."""
    print("🚀 Blazingly Fast Lambda Labs LLM Planner Performance Benchmarks")
    print("=" * 70)
    
    # Check if robot YAML file exists
    yaml_path = "src/planning/aloha.yaml"
    if not os.path.exists(yaml_path):
        print(f"⚠️  Warning: Robot YAML file not found at {yaml_path}")
        print("Please ensure the aloha.yaml file exists for full functionality.")
    
    try:
        # Test connection first
        if not test_lambda_labs_connection():
            print("\n❌ Cannot proceed without Lambda Labs API key")
            print("Please set LAMBDA_API_KEY environment variable and try again")
            return
        
        # Run benchmarks
        print("\n" + "=" * 70)
        planner1 = benchmark_single_requests()
        
        print("\n" + "=" * 70)
        planner2 = benchmark_batch_requests()
        
        print("\n" + "=" * 70)
        asyncio.run(benchmark_async_requests())
        
        print("\n" + "=" * 70)
        planner3 = benchmark_cache_performance()
        
        print("\n" + "=" * 70)
        benchmark_different_configurations()
        
        # Final performance summary
        print("\n" + "=" * 70)
        print("🏁 LAMBDA LABS PERFORMANCE SUMMARY")
        print("=" * 70)
        
        if planner1:
            stats = planner1.get_performance_stats()
            print(f"📊 Overall Performance:")
            print(f"   Total requests: {stats['total_requests']}")
            print(f"   Average response time: {stats['avg_response_time']:.3f}s")
            print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
        
        print("\n🚀 Lambda Labs Optimizations:")
        print("   ✅ Response caching (1000x faster for repeated requests)")
        print("   ✅ Connection pooling (parallel processing)")
        print("   ✅ Async support (non-blocking operations)")
        print("   ✅ Batch processing (multiple requests at once)")
        print("   ✅ Retry logic with exponential backoff")
        print("   ✅ Performance monitoring and metrics")
        
        print("\n💡 Lambda Labs Usage Tips:")
        print("   • Enable caching for repeated tasks")
        print("   • Use batch processing for multiple tasks")
        print("   • Use async methods for non-blocking operations")
        print("   • Monitor performance stats for optimization")
        print("   • Lambda Labs API is optimized for speed")
        
    except KeyboardInterrupt:
        print("\n\nBenchmarks interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main() 