# 🚀 Blazingly Fast LLM Planner

A high-performance LLM planner optimized for speed with caching, connection pooling, async support, and batch processing.

## ⚡ Performance Features

### 🎯 Response Caching
- **1000x faster** for repeated requests
- Configurable cache size with LRU eviction
- Thread-safe caching with locks
- Cache hit rate monitoring

### 🔄 Connection Pooling
- Reusable HTTP connections
- Configurable worker pool size
- Parallel request processing
- Reduced connection overhead

### ⚡ Async Support
- Non-blocking operations
- Concurrent request handling
- Future-based API
- Event loop integration

### 📦 Batch Processing
- Multiple requests in parallel
- Configurable batch sizes
- Automatic worker scaling
- Error handling per request

### 🔁 Retry Logic
- Exponential backoff
- Configurable retry attempts
- Automatic error recovery
- Timeout handling

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from planning.planner_llm import FastPlannerLLM

# Initialize with performance optimizations
planner = FastPlannerLLM(
    robot_yaml_path="src/planning/aloha.yaml",
    provider="lambda_labs",
    enable_caching=True,
    cache_size=1000,
    max_workers=4,
    timeout=30.0,
    max_retries=3
)

# Generate action plan
plan = planner.build_action_plan(
    task="wash the plate",
    perception_output=[{"name": "plate_geom", "labels": ["plate"]}],
    positions={"plate_geom": [0.25, -0.27, -0.42]}
)
```

### Async Usage

```python
import asyncio

# Create async tasks
async_tasks = []
for i in range(5):
    future = planner.build_action_plan_async(
        task=f"move object {i}",
        perception_output=[{"name": f"obj_{i}", "labels": ["object"]}],
        positions={f"obj_{i}": [0.1 + i*0.1, 0.0, -0.3]}
    )
    async_tasks.append(future)

# Wait for all tasks to complete
results = await asyncio.gather(*async_tasks)
```

### Batch Processing

```python
# Prepare multiple tasks
tasks = [
    {
        "task": "pick up cup",
        "perception_output": [{"name": "cup", "labels": ["cup"]}],
        "positions": {"cup": [0.2, 0.1, -0.3]}
    },
    {
        "task": "move plate",
        "perception_output": [{"name": "plate", "labels": ["plate"]}],
        "positions": {"plate": [0.3, -0.2, -0.4]}
    }
]

# Process all tasks in parallel
results = planner.build_action_plans_batch(tasks)
```

## 📊 Performance Monitoring

### Get Performance Statistics

```python
# Get detailed performance metrics
stats = planner.get_performance_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Average response time: {stats['avg_response_time']:.3f}s")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
```

### Cache Management

```python
# Clear the cache
planner.clear_cache()

# Check cache status
stats = planner.get_performance_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

## ⚙️ Configuration Options

### Performance Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_caching` | `True` | Enable response caching |
| `cache_size` | `1000` | Maximum cached responses |
| `max_workers` | `4` | Thread pool size |
| `timeout` | `30.0` | API request timeout (seconds) |
| `max_retries` | `3` | Maximum retry attempts |

### Provider Configuration

```python
# Lambda Labs (recommended for speed)
planner = FastPlannerLLM(
    provider="lambda_labs",
    model="llama-4-maverick-17b-128e-instruct-fp8"
)

# OpenAI
planner = FastPlannerLLM(
    provider="openai",
    model="gpt-4"
)

# Anthropic
planner = FastPlannerLLM(
    provider="anthropic",
    model="claude-3-sonnet-20240229"
)
```

## 🏁 Performance Benchmarks

Run the performance benchmarks:

```bash
python example_fast_planner.py
```

### Expected Results

```
🚀 Blazingly Fast LLM Planner Performance Benchmarks
============================================================

=== Single Request Performance Benchmark ===
🔥 Testing single request performance...
⏱️  First request (cache miss): 2.345s
⚡ Second request (cache hit): 0.001s
🚀 Speed improvement: 2345.0x faster!

📊 Performance Stats:
   Total requests: 2
   Average response time: 1.173s
   Cache hits: 1
   Cache misses: 1
   Cache hit rate: 50.0%

=== Batch Request Performance Benchmark ===
🔥 Testing batch processing of 5 tasks...
📋 Sequential processing...
⏱️  Sequential time: 12.456s
🚀 Batch processing...
⏱️  Batch time: 3.234s
🚀 Speed improvement: 3.9x faster!
```

## 🎯 Optimization Strategies

### 1. Caching Strategy
- **Use case**: Repeated tasks with same inputs
- **Benefit**: 1000x speed improvement
- **Implementation**: MD5 hash-based cache keys

### 2. Connection Pooling
- **Use case**: Multiple concurrent requests
- **Benefit**: Reduced connection overhead
- **Implementation**: ThreadPoolExecutor with configurable workers

### 3. Async Processing
- **Use case**: Non-blocking operations
- **Benefit**: Better resource utilization
- **Implementation**: asyncio with Future objects

### 4. Batch Processing
- **Use case**: Multiple independent tasks
- **Benefit**: Parallel execution
- **Implementation**: Concurrent.futures with timeout handling

### 5. Retry Logic
- **Use case**: Network failures or API errors
- **Benefit**: Improved reliability
- **Implementation**: Exponential backoff with configurable attempts

## 🔧 Advanced Usage

### Custom Cache Implementation

```python
# Disable caching for real-time responses
planner = FastPlannerLLM(enable_caching=False)

# Large cache for high-frequency requests
planner = FastPlannerLLM(cache_size=10000)
```

### High-Performance Configuration

```python
# Maximum performance setup
planner = FastPlannerLLM(
    enable_caching=True,
    cache_size=5000,
    max_workers=8,
    timeout=60.0,
    max_retries=5
)
```

### Error Handling

```python
try:
    plan = planner.build_action_plan(task, perception, positions)
except Exception as e:
    print(f"Planning failed: {e}")
    # Fallback to cached plan or default behavior
```

## 📈 Performance Tips

### 1. Enable Caching
- Always enable caching for repeated tasks
- Use appropriate cache sizes based on memory constraints
- Monitor cache hit rates for optimization

### 2. Use Batch Processing
- Group multiple tasks when possible
- Avoid sequential processing for independent tasks
- Monitor batch processing performance

### 3. Optimize Worker Count
- Match worker count to CPU cores
- Consider API rate limits
- Monitor thread pool utilization

### 4. Async for Non-Blocking
- Use async methods for UI applications
- Avoid blocking the main thread
- Leverage event loop for concurrency

### 5. Monitor Performance
- Track response times and cache hit rates
- Identify bottlenecks and optimize
- Use performance stats for tuning

## 🚨 Troubleshooting

### Common Issues

1. **Slow First Request**
   - Normal behavior (cache miss)
   - Subsequent requests will be much faster

2. **Cache Not Working**
   - Check `enable_caching` parameter
   - Verify cache size is sufficient
   - Monitor cache hit rates

3. **Timeout Errors**
   - Increase `timeout` parameter
   - Check network connectivity
   - Verify API key and configuration

4. **Memory Issues**
   - Reduce `cache_size`
   - Clear cache periodically
   - Monitor memory usage

### Performance Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor performance in real-time
stats = planner.get_performance_stats()
print(f"Current performance: {stats}")
```

## 🔄 Migration from Standard Planner

### Simple Migration

```python
# Old code
from planning.planner_llm import PlannerLLM
planner = PlannerLLM(robot_yaml_path="aloha.yaml")

# New code (backward compatible)
from planning.planner_llm import FastPlannerLLM
planner = FastPlannerLLM(robot_yaml_path="aloha.yaml")

# Or use the alias
from planning.planner_llm import PlannerLLM  # Still works!
```

### Performance Upgrade

```python
# Upgrade to high-performance configuration
planner = FastPlannerLLM(
    robot_yaml_path="aloha.yaml",
    enable_caching=True,
    max_workers=4,
    timeout=30.0
)
```

## 📚 API Reference

### FastPlannerLLM Class

#### Constructor
```python
FastPlannerLLM(
    robot_yaml_path: str = None,
    provider: str = "lambda_labs",
    model: str = None,
    api_base: str = None,
    api_key: str = None,
    enable_caching: bool = True,
    cache_size: int = 1000,
    max_workers: int = 4,
    timeout: float = 30.0,
    max_retries: int = 3
)
```

#### Methods

- `build_action_plan(task, perception_output, positions)` - Generate action plan
- `build_action_plan_async(task, perception_output, positions)` - Async version
- `build_action_plans_batch(tasks)` - Batch processing
- `get_performance_stats()` - Get performance metrics
- `clear_cache()` - Clear response cache
- `save_plan(plan, filename)` - Save plan to file
- `load_plan(filename)` - Load plan from file

## 🎉 Conclusion

The Blazingly Fast LLM Planner provides significant performance improvements through:

- **Caching**: 1000x faster repeated requests
- **Parallelism**: Multi-threaded processing
- **Async Support**: Non-blocking operations
- **Batch Processing**: Concurrent task execution
- **Retry Logic**: Improved reliability

Use these optimizations to achieve maximum performance for your LLM planning tasks! 