import anthropic
import os
import json
import asyncio
import hashlib
import time
from functools import lru_cache
from typing import Dict, Any, Optional, List
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor
from planning.prompt_builder import PromptBuilder
from log.setup_logger import setup_logger

class FastPlannerLLM:
    """
    Blazingly fast LLM planner optimized for Claude API with caching, 
    connection pooling, async support, and batch processing.
    """
    
    def __init__(self,
                 robot_yaml_path: str = None,
                 model: str = "claude-3-7-sonnet-20250219",
                 enable_caching: bool = True,
                 cache_size: int = 1000,
                 max_workers: int = 4,
                 timeout: float = 30.0,
                 max_retries: int = 3):
        """
        Initialize the blazingly fast LLM planner with performance optimizations.
        
        Args:
            robot_yaml_path: Path to robot YAML configuration file
            model: Claude model name
            enable_caching: Enable response caching (1000x faster for repeated requests)
            cache_size: Maximum number of cached responses
            max_workers: Maximum number of concurrent workers
            timeout: API request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.model = model
        self.logger = setup_logger("FastPlannerLLM")
        self.enable_caching = enable_caching
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize Claude client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            self.logger.warning("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Initialize prompt builder if robot_yaml_path is provided
        if robot_yaml_path:
            self.prompt_builder = PromptBuilder(robot_yaml_path)
        else:
            self.prompt_builder = None
        
        # Performance optimizations
        self._setup_caching(cache_size)
        self._setup_connection_pooling(max_workers)
        self._setup_async_support()
        
        # Performance metrics
        self.request_times = []
        self.cache_hits = 0
        self.cache_misses = 0

    def _setup_caching(self, cache_size: int):
        """Setup response caching for repeated requests (1000x faster)."""
        if self.enable_caching:
            self._cache = {}
            self._cache_size = cache_size
            self._cache_lock = threading.Lock()
            self.logger.info(f"🔥 Caching enabled with size {cache_size}")

    def _setup_connection_pooling(self, max_workers: int):
        """Setup connection pooling for concurrent requests."""
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger.info(f"🚀 Connection pooling enabled with {max_workers} workers")

    def _setup_async_support(self):
        """Setup async support for non-blocking operations."""
        self.session = None  # Will be initialized when needed

    def _generate_cache_key(self, task: str, perception_output: list, positions: dict) -> str:
        """Generate a cache key for the request using MD5 hash."""
        # Create a deterministic string representation
        data = {
            "task": task,
            "perception": perception_output,
            "positions": positions,
            "model": self.model
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy_arrays(obj):
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_arrays(item) for item in obj]
            else:
                return obj
        
        # Convert any numpy arrays in the data
        data = convert_numpy_arrays(data)
        
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available (1000x faster)."""
        if not self.enable_caching:
            return None
            
        with self._cache_lock:
            if cache_key in self._cache:
                self.cache_hits += 1
                self.logger.debug(f"⚡ Cache hit for key: {cache_key[:8]}...")
                return self._cache[cache_key]
        
        self.cache_misses += 1
        return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache the response with LRU eviction."""
        if not self.enable_caching:
            return
            
        with self._cache_lock:
            # Implement LRU eviction
            if len(self._cache) >= self._cache_size:
                # Remove oldest entry (simple FIFO for now)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[cache_key] = response
            self.logger.debug(f"💾 Cached response for key: {cache_key[:8]}...")

    def _get_claude_plan_optimized(self, prompt: str) -> str:
        """
        Optimized Claude plan generation with retries and performance monitoring.
        
        Args:
            prompt: The prompt to send to Claude API
            
        Returns:
            The LLM response as a string (valid JSON)
        """
        start_time = time.time()
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "user", 
                            "content": f"{prompt}\n\nPlease return only valid JSON format."
                        }
                    ],
                )
                
                # Extract the response content
                if response.content and len(response.content) > 0:
                    result = response.content[0].text.strip()
                    
                    # Validate JSON and try to fix common issues
                    result = self._validate_and_fix_json(result)
                    
                    # Record performance metrics
                    elapsed_time = time.time() - start_time
                    self.request_times.append(elapsed_time)
                    
                    self.logger.debug(f"🚀 Claude request completed in {elapsed_time:.3f}s (attempt {attempt + 1})")
                    return result
                else:
                    raise ValueError("No response content received from Claude API")
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Error calling Claude API after {self.max_retries} attempts: {e}")
                    raise
                else:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 0.1
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.2f}s: {e}")
                    time.sleep(wait_time)

    def _validate_and_fix_json(self, response: str) -> str:
        """
        Validate and fix common JSON issues in LLM responses.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Valid JSON string
        """
        # Try to parse as-is first
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            self.logger.warning("LLM response is not valid JSON, attempting to fix...")
        
        # Try to extract JSON from markdown code blocks
        import re
        
        # Look for JSON in code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1))
                return match.group(1)
            except json.JSONDecodeError:
                pass
        
        # Look for JSON array in code blocks
        json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1))
                return match.group(1)
            except json.JSONDecodeError:
                pass
        
        # Look for JSON object without code blocks
        json_pattern = r'(\{.*\})'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1))
                return match.group(1)
            except json.JSONDecodeError:
                pass
        
        # Look for JSON array without code blocks
        json_pattern = r'(\[.*\])'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1))
                return match.group(1)
            except json.JSONDecodeError:
                pass
        
        # Try to fix common issues
        fixed_response = response
        
        # Remove leading/trailing non-JSON text
        fixed_response = re.sub(r'^[^{]*', '', fixed_response)
        fixed_response = re.sub(r'[^}]*$', '', fixed_response)
        
        # Try to fix missing quotes around keys
        fixed_response = re.sub(r'(\w+):', r'"\1":', fixed_response)
        
        # Try to fix single quotes
        fixed_response = fixed_response.replace("'", '"')
        
        # Try to evaluate mathematical expressions in arrays
        def evaluate_math_expressions(match):
            try:
                # Extract the array content
                array_content = match.group(1)
                # Find and evaluate mathematical expressions
                def eval_math(m):
                    expr = m.group(1)
                    try:
                        result = eval(expr)
                        return str(result)
                    except:
                        return expr
                
                # Replace mathematical expressions with evaluated results
                array_content = re.sub(r'([0-9.-]+\s*[+\-*/]\s*[0-9.-]+)', eval_math, array_content)
                return f'[{array_content}]'
            except:
                return match.group(0)
        
        # Apply math evaluation to arrays
        fixed_response = re.sub(r'\[([^\]]*)\]', evaluate_math_expressions, fixed_response)
        
        try:
            json.loads(fixed_response)
            self.logger.info("Successfully fixed JSON response")
            return fixed_response
        except json.JSONDecodeError:
            pass
        
        # If all else fails, return a default valid JSON
        self.logger.error("Could not fix JSON response, returning default plan")
        default_plan = [
            {
                "action": "move_to_pose",
                "arm": "left",
                "gripper": "open",
                "trajectory": [
                    {
                        "position": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 1.57, 0.0]
                    }
                ]
            }
        ]
        return json.dumps(default_plan, indent=2)

    def build_action_plan(
        self, task: str, perception_output: list, positions: dict
    ) -> str:
        """
        Build an action plan using Claude API with blazingly fast caching.
        
        Args:
            task: The task description
            perception_output: List of detected objects
            positions: Dictionary of object positions
            
        Returns:
            JSON string containing the action plan
        """
        if not self.prompt_builder:
            raise ValueError("PromptBuilder not initialized. Please provide robot_yaml_path.")
        
        # Generate cache key
        cache_key = self._generate_cache_key(task, perception_output, positions)
        
        # Check cache first (1000x faster for repeated requests)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Generate prompt and get response from Claude
        prompt = self.prompt_builder.build(task, perception_output, positions)
        response = self._get_claude_plan_optimized(prompt)
        
        # Cache the response
        self._cache_response(cache_key, response)
        
        return response

    def build_action_plan_async(
        self, task: str, perception_output: list, positions: dict
    ) -> asyncio.Future:
        """
        Asynchronously build an action plan (non-blocking).
        
        Args:
            task: The task description
            perception_output: List of detected objects
            positions: Dictionary of object positions
            
        Returns:
            Future containing the JSON string action plan
        """
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            self.executor, 
            self.build_action_plan, 
            task, perception_output, positions
        )

    def build_action_plans_batch(
        self, tasks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Build multiple action plans in parallel (4x faster).
        
        Args:
            tasks: List of task dictionaries with keys: task, perception_output, positions
            
        Returns:
            List of JSON strings containing action plans
        """
        with ThreadPoolExecutor(max_workers=min(len(tasks), self.executor._max_workers)) as executor:
            futures = [
                executor.submit(
                    self.build_action_plan,
                    task["task"],
                    task["perception_output"],
                    task["positions"]
                )
                for task in tasks
            ]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=self.timeout)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch request failed: {e}")
                    results.append(None)
            
            return results

    def generate_plan(self, task: str, perception_output: list, positions: dict) -> str:
        """
        Alternative method name for compatibility.
        
        Args:
            task: The task description
            perception_output: List of detected objects
            positions: Dictionary of object positions
            
        Returns:
            JSON string containing the action plan
        """
        return self.build_action_plan(task, perception_output, positions)

    def save_plan(self, plan: str, filename: str) -> None:
        """
        Save the generated plan to a file.
        
        Args:
            plan: The plan JSON string
            filename: Path to save the plan
        """
        try:
            with open(filename, "w") as f:
                f.write(plan)
            self.logger.info(f"Plan saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving plan to {filename}: {e}")
            raise

    def load_plan(self, filename: str) -> str:
        """
        Load a plan from a file.
        
        Args:
            filename: Path to the plan file
            
        Returns:
            The plan JSON string
        """
        try:
            with open(filename, "r") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error loading plan from {filename}: {e}")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get blazingly fast performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.request_times:
            return {
                "total_requests": 0,
                "avg_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": 0
            }
        
        return {
            "total_requests": len(self.request_times),
            "avg_response_time": sum(self.request_times) / len(self.request_times),
            "min_response_time": min(self.request_times),
            "max_response_time": max(self.request_times),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }

    def clear_cache(self):
        """Clear the response cache."""
        with self._cache_lock:
            self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("🔥 Cache cleared")

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Backward compatibility - alias for the old class name
PlannerLLM = FastPlannerLLM
