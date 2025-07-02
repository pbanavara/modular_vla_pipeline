# LLM Planner with Lambda Labs Support

This document describes the updated LLM planner that now supports multiple LLM providers, including Lambda Labs' Llama 4 Maverick 17B model, using OpenAI-compatible chat completions API.

## Overview

The LLM planner has been completely refactored to:
1. **Support multiple LLM providers** (Lambda Labs, OpenAI, Anthropic)
2. **Use OpenAI-compatible API** for consistent interface
3. **Provide flexible configuration** for different models and settings
4. **Maintain backward compatibility** with existing code

## Key Changes

### 1. Provider Support
- **Lambda Labs**: Llama 4 Maverick 17B 128E Instruct FP8 (default)
- **OpenAI**: GPT-4 Turbo Preview
- **Anthropic**: Claude 3.7 Sonnet (legacy support)

### 2. API Compatibility
- Uses OpenAI client library for all providers
- Consistent chat completions interface
- Automatic configuration management

### 3. Enhanced Features
- Configuration validation
- Error handling and logging
- Plan saving and loading
- Custom parameter overrides

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
# For Lambda Labs (recommended)
export LAMBDA_API_KEY="your_lambda_labs_api_key"

# For OpenAI (alternative)
export OPENAI_API_KEY="your_openai_api_key"

# For Anthropic (legacy)
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

## Usage

### Basic Usage with Lambda Labs

```python
from planning.planner_llm import PlannerLLM

# Initialize with Lambda Labs (default)
planner = PlannerLLM(
    robot_yaml_path="src/planning/aloha.yaml",
    provider="lambda_labs"
)

# Generate a plan
task = "wash the plate"
perception_output = [
    {"name": "plate_geom", "labels": ["plate", "ceramic"]}
]
positions = {
    "plate_geom": [0.25, -0.27, -0.42]
}

plan = planner.build_action_plan(task, perception_output, positions)
print(plan)
```

### Using Different Providers

```python
# OpenAI
planner_openai = PlannerLLM(
    robot_yaml_path="src/planning/aloha.yaml",
    provider="openai"
)

# Anthropic (legacy)
planner_anthropic = PlannerLLM(
    robot_yaml_path="src/planning/aloha.yaml",
    provider="anthropic"
)
```

### Custom Configuration

```python
# Custom model and settings
planner = PlannerLLM(
    robot_yaml_path="src/planning/aloha.yaml",
    provider="lambda_labs",
    model="llama-4-maverick-17b-128e-instruct-fp8",
    api_key="your_custom_key"
)
```

## Configuration

### Lambda Labs Configuration
```python
LAMBDA_LABS_CONFIG = {
    "api_base": "https://api.lambda.ai/v1",
    "model": "llama-4-maverick-17b-128e-instruct-fp8",
    "api_key_env": "LAMBDA_API_KEY",
    "max_tokens": 4096,
    "temperature": 0.1,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1
}
```

### OpenAI Configuration
```python
OPENAI_CONFIG = {
    "api_base": "https://api.openai.com/v1",
    "model": "gpt-4-turbo-preview",
    "api_key_env": "OPENAI_API_KEY",
    "max_tokens": 4096,
    "temperature": 0.1,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1
}
```

## API Reference

### PlannerLLM Class

#### Constructor
```python
PlannerLLM(
    robot_yaml_path: str = None,
    provider: str = "lambda_labs",
    model: str = None,
    api_base: str = None,
    api_key: str = None
)
```

#### Methods

- `build_action_plan(task, perception_output, positions)` - Generate action plan
- `generate_plan(task, perception_output, positions)` - Alias for build_action_plan
- `save_plan(plan, filename)` - Save plan to file
- `load_plan(filename)` - Load plan from file

### Configuration Functions

- `get_llm_config(provider)` - Get provider configuration
- `get_api_key(provider)` - Get API key for provider
- `validate_config(provider)` - Validate provider configuration

## Migration Guide

### From Old Anthropic Implementation

#### Before
```python
from planning.planner_llm import PlannerLLM

planner = PlannerLLM(robot_yaml_path="aloha.yaml")
plan = planner.build_action_plan(task, perception, positions)
```

#### After
```python
from planning.planner_llm import PlannerLLM

# Still works the same way (defaults to Lambda Labs)
planner = PlannerLLM(robot_yaml_path="aloha.yaml")
plan = planner.build_action_plan(task, perception, positions)

# Or explicitly specify provider
planner = PlannerLLM(
    robot_yaml_path="aloha.yaml",
    provider="lambda_labs"  # or "openai", "anthropic"
)
```

### Environment Variables

#### Before
```bash
export ANTHROPIC_API_KEY="your_key"
```

#### After
```bash
# For Lambda Labs (recommended)
export LAMBDA_API_KEY="your_key"

# For OpenAI
export OPENAI_API_KEY="your_key"

# For Anthropic (legacy)
export ANTHROPIC_API_KEY="your_key"
```

## Integration with Pipeline

The updated planner integrates seamlessly with your existing pipeline:

```python
# In your pipeline code
from planning.planner_llm import PlannerLLM

# Initialize planner
planner = PlannerLLM(robot_yaml_path=aloha_yaml_path)

# Generate plan (same as before)
plan = planner.build_action_plan(task, perception_output, known_positions)

# Save plan
planner.save_plan(plan, plan_json_path)
```

## Error Handling

The new implementation includes comprehensive error handling:

```python
try:
    planner = PlannerLLM(robot_yaml_path="aloha.yaml")
    plan = planner.build_action_plan(task, perception, positions)
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## Performance Considerations

### Lambda Labs vs OpenAI
- **Lambda Labs**: Generally faster, more cost-effective, optimized for instruction following
- **OpenAI**: More consistent, better reasoning capabilities
- **Anthropic**: Legacy support, may be deprecated

### Temperature Settings
- **0.1**: More deterministic, consistent outputs (recommended for robotics)
- **0.7**: More creative, varied outputs (original setting)

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   ValueError: API key not found. Please set LAMBDA_API_KEY environment variable.
   ```
   **Solution**: Set the appropriate environment variable

2. **Configuration Validation Failed**
   ```
   Warning: Configuration validation failed for provider: lambda_labs
   ```
   **Solution**: Check API key and network connectivity

3. **Model Not Available**
   ```
   Error: Model llama-4-maverick-17b-128e-instruct-fp8 not found
   ```
   **Solution**: Verify model name and API access

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Example Scripts

### Basic Example
```bash
python example_llm_planner.py
```

### Provider Comparison
```python
from planning.llm_config import validate_config

providers = ["lambda_labs", "openai", "anthropic"]
for provider in providers:
    if validate_config(provider):
        print(f"{provider}: ✅ Available")
    else:
        print(f"{provider}: ❌ Not available")
```

### Direct API Test
```python
import base64
from openai import OpenAI

client = OpenAI(
    api_key="your_lambda_api_key",
    base_url="https://api.lambda.ai/v1",
)

response = client.chat.completions.create(
    model="llama-4-maverick-17b-128e-instruct-fp8",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## Future Enhancements

1. **Local Models**: Support for local LLM inference
2. **Streaming**: Real-time plan generation
3. **Caching**: Plan caching for repeated tasks
4. **Validation**: JSON schema validation for generated plans
5. **Metrics**: Performance and quality metrics

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the example scripts
3. Validate your configuration
4. Check API provider status pages 