# MuJoCo Viewer Manager

This module provides a robust solution for managing MuJoCo viewer instances to prevent crashes when multiple viewers are launched or when a viewer is already running.

## Problem

When a MuJoCo viewer is already open and you try to launch another one using `mujoco.viewer.launch_passive()`, the script crashes. This is a common issue in development and testing scenarios.

## Solution

The `ViewerManager` class provides several features to handle this:

1. **Automatic viewer reuse**: If a viewer is already running, it will be reused instead of creating a new one
2. **Process detection and cleanup**: Automatically finds and terminates existing MuJoCo viewer processes
3. **Force restart capability**: Option to kill existing viewers and start fresh
4. **Error handling**: Graceful handling of viewer launch failures with retry logic
5. **Thread safety**: Safe concurrent access to viewer management functions

## Usage

### Basic Usage

```python
from utils.viewer_manager import safe_launch_viewer, close_all_viewers

# Load your MuJoCo model
model = mujoco.MjModel.from_xml_path("your_model.xml")
data = mujoco.MjData(model)

# Safely launch a viewer (will reuse existing if available)
viewer = safe_launch_viewer(model, data)

# Use the viewer normally
while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.01)

# Clean up when done
close_all_viewers()
```

### Force Restart

If you want to ensure a fresh viewer instance:

```python
# This will kill any existing viewers and start fresh
viewer = safe_launch_viewer(model, data, force_restart=True)
```

### Advanced Usage with ViewerManager

```python
from utils.viewer_manager import get_viewer_manager

# Get the global viewer manager instance
manager = get_viewer_manager()

# Get a viewer with custom options
viewer = manager.get_viewer(model, data, force_restart=False)

# Close the viewer when done
manager.close_viewer()
```

## Integration with Existing Code

### Before (crashes if viewer already open):
```python
# This will crash if a viewer is already open
with mujoco.viewer.launch_passive(model, data) as viewer:
    # ... your code ...
```

### After (safe):
```python
# This is safe even if a viewer is already open
try:
    viewer = safe_launch_viewer(model, data)
    # ... your code ...
    while viewer.is_running():
        viewer.sync()
except Exception as e:
    # Fallback with force restart
    viewer = safe_launch_viewer(model, data, force_restart=True)
    # ... continue with your code ...
```

## Features

### 1. Process Detection
The viewer manager can detect running MuJoCo viewer processes by looking for:
- Process names containing "mujoco", "viewer", or "glfw"
- Command lines containing "mujoco" or "viewer"

### 2. Graceful Termination
When terminating existing viewers:
1. First attempts graceful termination (SIGTERM)
2. Waits up to 5 seconds for graceful shutdown
3. Falls back to force kill (SIGKILL) if necessary

### 3. Viewer Health Checks
The manager checks if the current viewer is still responsive by:
- Testing if the viewer object is still valid
- Checking if `viewer.is_running()` can be called without errors

### 4. Retry Logic
If viewer launch fails:
1. Logs the error
2. Waits 0.5 seconds
3. Attempts one more launch
4. Raises the exception if both attempts fail

## Error Handling

The viewer manager provides comprehensive error handling:

```python
try:
    viewer = safe_launch_viewer(model, data)
except Exception as e:
    print(f"Failed to launch viewer: {e}")
    # Try force restart as fallback
    try:
        viewer = safe_launch_viewer(model, data, force_restart=True)
    except Exception as e2:
        print(f"Failed even after force restart: {e2}")
        raise
```

## Thread Safety

The viewer manager is thread-safe and can be used in multi-threaded applications:

```python
import threading

def worker_thread():
    viewer = safe_launch_viewer(model, data)
    # ... use viewer ...

# Multiple threads can safely request viewers
threads = [threading.Thread(target=worker_thread) for _ in range(3)]
for t in threads:
    t.start()
```

## Dependencies

The viewer manager requires:
- `mujoco` - MuJoCo Python bindings
- `psutil` - For process management
- `threading` - For thread safety (built-in)

## Example Script

See `example_viewer_usage.py` for complete examples demonstrating:
- Basic usage
- Error handling
- Multiple rapid launches
- Force restart scenarios

## Migration Guide

To update existing code:

1. **Replace direct viewer launches**:
   ```python
   # Old
   with mujoco.viewer.launch_passive(model, data) as viewer:
   
   # New
   viewer = safe_launch_viewer(model, data)
   ```

2. **Add error handling**:
   ```python
   try:
       viewer = safe_launch_viewer(model, data)
   except Exception as e:
       viewer = safe_launch_viewer(model, data, force_restart=True)
   ```

3. **Add cleanup**:
   ```python
   # At the end of your script
   close_all_viewers()
   ```

## Troubleshooting

### Viewer still crashes
- Make sure you're using `safe_launch_viewer()` instead of `mujoco.viewer.launch_passive()`
- Try using `force_restart=True` to ensure clean state
- Check if there are any other MuJoCo processes running

### Process detection issues
- The manager looks for processes with "mujoco", "viewer", or "glfw" in the name
- If your viewer process has a different name, you may need to modify the detection logic

### Performance concerns
- The viewer manager adds minimal overhead
- Process detection only happens when launching new viewers
- Viewer reuse eliminates the need for multiple viewer instances

## API Reference

### Functions

- `safe_launch_viewer(model, data, force_restart=False)` - Safely launch a viewer
- `close_all_viewers()` - Close all viewers and clean up processes
- `get_viewer_manager()` - Get the global viewer manager instance

### ViewerManager Class

- `get_viewer(model, data, force_restart=False)` - Get a viewer instance
- `close_viewer()` - Close the current viewer
- `_kill_existing_viewers()` - Kill all existing MuJoCo viewer processes
- `_check_viewer_alive()` - Check if current viewer is responsive 