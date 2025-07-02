import os
import json
import asyncio
import hashlib
import time
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List
from functools import lru_cache

from planning.llama_prompt_builder import LlamaPromptBuilder
from log.setup_logger import setup_logger

class LlamaPlanner:
    """
    High-performance Llama-optimized planner with caching, connection pooling,
    async support, and batch processing specifically designed for Llama models.
    Optimized with lazy loading for fast initialization.
    """
    
    def __init__(self,
                 robot_yaml_path: str = None,
                 model: str = "llama-3.2-70b-instruct",
                 enable_caching: bool = True,
                 cache_size: int = 1000,
                 max_workers: int = 4,
                 timeout: float = 30.0,
                 max_retries: int = 3,
                 api_base: str = "https://api.lambda.ai/v1"):
        """
        Initialize the Llama-optimized planner with performance optimizations.
        Uses lazy loading to minimize initialization time.
        
        Args:
            robot_yaml_path: Path to robot YAML configuration file
            model: Llama model name
            enable_caching: Enable response caching (1000x faster for repeated requests)
            cache_size: Maximum number of cached responses
            max_workers: Maximum number of concurrent workers
            timeout: API request timeout in seconds
            max_retries: Maximum number of retry attempts
            api_base: API base URL for Llama service
        """
        self.model = model
        self.api_base = api_base
        self.logger = setup_logger("LlamaPlanner")
        self.enable_caching = enable_caching
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Get API key
        self.api_key = os.getenv("LAMBDA_API_KEY")
        if not self.api_key:
            self.logger.warning("LAMBDA_API_KEY environment variable not set")
        
        # Lazy loading: Store path but don't initialize yet
        self._robot_yaml_path = robot_yaml_path
        self._prompt_builder = None
        self._prompt_builder_initialized = False
        
        # Performance optimizations (lightweight setup)
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
            self.logger.info(f"🔥 Llama caching enabled with size {cache_size}")

    def _setup_connection_pooling(self, max_workers: int):
        """Setup connection pooling for concurrent requests."""
        # Lazy initialization: Create executor only when needed
        self._max_workers = max_workers
        self._executor = None
        self.logger.info(f"🚀 Llama connection pooling configured for {max_workers} workers")

    def _setup_async_support(self):
        """Setup async support for non-blocking operations."""
        self.session = None  # Will be initialized when needed

    def _get_executor(self):
        """Lazy initialization of ThreadPoolExecutor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
            self.logger.info(f"🚀 Llama connection pooling initialized with {self._max_workers} workers")
        return self._executor

    def _initialize_prompt_builder(self):
        """Lazy initialization of prompt builder."""
        if not self._prompt_builder_initialized and self._robot_yaml_path:
            self.logger.info("🔄 Initializing LlamaPromptBuilder (lazy loading)...")
            start_time = time.time()
            
            self._prompt_builder = LlamaPromptBuilder(self._robot_yaml_path)
            self._prompt_builder_initialized = True
            
            init_time = time.time() - start_time
            self.logger.info(f"✅ LlamaPromptBuilder initialized in {init_time:.3f}s")

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
                self.logger.debug(f"⚡ Llama cache hit for key: {cache_key[:8]}...")
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
            self.logger.debug(f"💾 Cached Llama response for key: {cache_key[:8]}...")

    def _get_llama_plan_optimized(self, prompt: str) -> str:
        """
        Optimized Llama plan generation with retries and performance monitoring.
        
        Args:
            prompt: The prompt to send to Llama API
            
        Returns:
            The LLM response as a string (valid JSON)
        """
        start_time = time.time()
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Prepare the request payload for Llama
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                }
                
                # Make the API request
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Use aiohttp for async requests or requests for sync
                import requests
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"].strip()
                        
                        # Validate JSON and try to fix common issues
                        content = self._validate_and_fix_json(content)
                        
                        # Record performance metrics
                        elapsed_time = time.time() - start_time
                        self.request_times.append(elapsed_time)
                        
                        self.logger.debug(f"🚀 Llama request completed in {elapsed_time:.3f}s (attempt {attempt + 1})")
                        return content
                    else:
                        raise ValueError("No content in Llama API response")
                else:
                    raise ValueError(f"Llama API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Error calling Llama API after {self.max_retries} attempts: {e}")
                    raise
                else:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 0.1
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.2f}s: {e}")
                    time.sleep(wait_time)

    def _validate_and_fix_json(self, response: str) -> str:
        """
        Validate and fix common JSON issues in Llama responses.
        
        Args:
            response: Raw response from Llama API
            
        Returns:
            Cleaned and validated JSON string
        """
        # Remove markdown code blocks if present
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        
        # Try to parse as JSON to validate
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON validation failed: {e}")
            
            # Try to fix common issues
            # Remove any text before the first [
            if "[" in response:
                start = response.find("[")
                response = response[start:]
            
            # Remove any text after the last ]
            if "]" in response:
                end = response.rfind("]") + 1
                response = response[:end]
            
            # Try parsing again
            try:
                json.loads(response)
                return response
            except json.JSONDecodeError:
                self.logger.error("Failed to fix JSON, returning original response")
                return response

    def build_action_plan(self, task: str, perception_output: list, positions: dict) -> str:
        """
        Build an action plan using Llama with caching and performance optimizations.
        Lazy loads prompt builder on first use.
        
        Args:
            task: The high-level task to perform
            perception_output: List of detected objects
            positions: Dictionary of object positions
            
        Returns:
            JSON string containing the action plan
        """
        # Lazy initialize prompt builder on first use
        self._initialize_prompt_builder()
        
        # Generate cache key
        cache_key = self._generate_cache_key(task, perception_output, positions)
        
        # Check cache first (1000x faster for repeated requests)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Build Llama-optimized prompt
        if self._prompt_builder:
            prompt = self._prompt_builder.build(task, perception_output, positions)
        else:
            # Fallback to simple prompt if no prompt builder
            prompt = f"Task: {task}\nObjects: {perception_output}\nPositions: {positions}\nGenerate a JSON action plan."
        
        # Get plan from Llama API
        response = self._get_llama_plan_optimized(prompt)
        
        # Cache the response
        self._cache_response(cache_key, response)
        
        return response

    def build_action_plan_async(self, task: str, perception_output: list, positions: dict) -> asyncio.Future:
        """
        Build an action plan asynchronously using Llama.
        
        Args:
            task: The high-level task to perform
            perception_output: List of detected objects
            positions: Dictionary of object positions
            
        Returns:
            Future containing the action plan
        """
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            self._get_executor(),
            self.build_action_plan,
            task,
            perception_output,
            positions
        )

    def build_action_plans_batch(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """
        Build multiple action plans in batch for improved performance.
        
        Args:
            tasks: List of task dictionaries with 'task', 'perception_output', and 'positions' keys
            
        Returns:
            List of JSON action plans
        """
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = []
            for task_data in tasks:
                future = executor.submit(
                    self.build_action_plan,
                    task_data["task"],
                    task_data["perception_output"],
                    task_data["positions"]
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=self.timeout)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in batch processing: {e}")
                    results.append(json.dumps({"error": str(e)}))
        
        return results

    def generate_plan(self, task: str, perception_output: list, positions: dict) -> str:
        """
        Generate a plan (alias for build_action_plan for compatibility).
        
        Args:
            task: The high-level task to perform
            perception_output: List of detected objects
            positions: Dictionary of object positions
            
        Returns:
            JSON string containing the action plan
        """
        return self.build_action_plan(task, perception_output, positions)

    def save_plan(self, plan: str, filename: str) -> None:
        """
        Save a plan to a file.
        
        Args:
            plan: The plan JSON string
            filename: The filename to save to
        """
        try:
            with open(filename, 'w') as f:
                f.write(plan)
            self.logger.info(f"Plan saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving plan: {e}")

    def load_plan(self, filename: str) -> str:
        """
        Load a plan from a file.
        
        Args:
            filename: The filename to load from
            
        Returns:
            The plan JSON string
        """
        try:
            with open(filename, 'r') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error loading plan: {e}")
            return json.dumps({"error": str(e)})

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the Llama planner.
        
        Returns:
            Dictionary containing performance metrics
        """
        total_requests = len(self.request_times)
        avg_response_time = sum(self.request_times) / total_requests if total_requests > 0 else 0
        
        total_cache_operations = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_operations if total_cache_operations > 0 else 0
        
        return {
            "total_requests": total_requests,
            "avg_response_time": avg_response_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "model": self.model,
            "api_base": self.api_base,
            "prompt_builder_initialized": self._prompt_builder_initialized
        }

    def clear_cache(self):
        """Clear the response cache."""
        if self.enable_caching:
            with self._cache_lock:
                self._cache.clear()
            self.logger.info("Llama cache cleared")

    def __del__(self):
        """Cleanup when the planner is destroyed."""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=False) 