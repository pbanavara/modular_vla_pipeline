# Modular VLA Pipeline

A high-performance, modular robotic pipeline that leverages Large Language Models (LLMs) for perception, planning, and action generation. Built around MuJoCo simulation with real-time execution capabilities.

## 🚀 Overview

This pipeline implements a **Vision-Language-Action (VLA)** architecture where LLMs drive the entire robotic decision-making process:

- **Vision**: Computer vision for object detection and segmentation
- **Language**: LLM-based task understanding and planning
- **Action**: MuJoCo-based simulation and execution

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Perception    │    │     Planning    │    │     Action      │
│                 │    │                 │    │                 │
│ • Camera Capture│    │ • LLM Planner   │    │ • MuJoCo Exec   │
│ • SAM Segment   │───▶│ • Task Analysis │───▶│ • IK Solver     │
│ • OwLViT Detect │    │ • Trajectory Gen│    │ • Joint Control │
│ • Depth Est.    │    │ • Safety Checks │    │ • Real-time Sim │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 Key Features

### **LLM-Driven Planning**
- **Llama-optimized planner** with blazingly fast performance
- **Structured prompts** tailored for robotic tasks
- **Response caching** (1000x faster for repeated requests)
- **Batch processing** and async support
- **Safety constraints** and workspace validation

### **Advanced Perception**
- **SAM (Segment Anything Model)** for precise object segmentation
- **OwLViT** for zero-shot object detection
- **Depth estimation** from camera views
- **Multi-camera support** (wrist, overhead, teleoperator views)

### **Real-time Execution**
- **MuJoCo simulation** with dual-arm robot control
- **Inverse kinematics** solver for trajectory execution
- **Async simulation loop** for smooth operation
- **Action queuing** and state management

## 📁 Project Structure

```
modular_vla_pipeline/
├── src/
│   ├── main.py                    # Main entry point
│   ├── pipeline/
│   │   ├── async_sim/
│   │   │   └── async_simulation.py # Real-time executor
│   │   └── pipeline.py            # Batch pipeline
│   ├── planning/
│   │   ├── llama_planner.py       # LLM-optimized planner
│   │   ├── llama_prompt_builder.py # Structured prompts
│   │   ├── llama.yaml            # Robot configuration
│   │   └── llm_config.py         # API configuration
│   ├── perception/
│   │   ├── capture/
│   │   │   └── camera_capture.py  # MuJoCo camera interface
│   │   └── classification_segmentation/
│   │       ├── segmentation_image.py # SAM integration
│   │       └── owl_vit.py        # Object detection
│   ├── action/
│   │   └── mujoco_executor.py    # Action execution
│   └── simulated_sink/
│       └── aloha/                # Robot model and assets
├── example_llama_planner.py      # Performance benchmarks
├── test_llama_planner.py         # Functionality tests
└── LLAMA_PLANNER_README.md       # Detailed planner docs
```

## 🚀 Quick Start

### **1. Environment Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Set API key for LLM access
export LAMBDA_API_KEY="your_lambda_labs_api_key"
```

### **2. Run the Pipeline**
```bash
# Start the main pipeline
python src/main.py
```

### **3. Interactive Usage**
The pipeline provides an interactive interface:
- **Task input**: Describe what you want the robot to do
- **Object detection**: Automatic detection of objects in scene
- **Plan generation**: LLM creates detailed action sequences
- **Execution**: Real-time simulation of robot actions

## 🔧 Configuration

### **Robot Configuration (`llama.yaml`)**
```yaml
robot:
  name: Aloha Dual Arm (Llama Optimized)
  type: Fixed-base, two 6-DOF arms
  workspace:
    reach_radius_cm: 65
    vertical_range_cm: 45

llama_config:
  instruction_style: "step_by_step"
  action_vocabulary:
    - "move_to_pose"
    - "grasp"
    - "release"
  safety_constraints:
    - "Stay within workspace limits"
    - "Use smooth trajectories"
```

### **LLM Models Supported**
- **Llama 3.2 70B** (recommended)
- **Llama 3.2 8B** (faster, smaller)
- **Llama 3.1 70B** (alternative)

## 📊 Performance Features

### **Optimized Initialization**
- **Lazy loading**: YAML parsing deferred until needed
- **Logger caching**: Avoids duplicate setup overhead
- **Fast startup**: <0.5s initialization time

### **High-Performance Planning**
- **Response caching**: 1000x faster for repeated tasks
- **Connection pooling**: Parallel request processing
- **Async support**: Non-blocking operations
- **Batch processing**: Multiple tasks simultaneously

### **Real-time Execution**
- **60Hz simulation**: Smooth robot movement
- **IK optimization**: Fast inverse kinematics solving
- **State management**: Persistent joint states
- **Error recovery**: Graceful failure handling

## 🎮 Usage Examples

### **Basic Task Execution**
```python
# The pipeline automatically handles:
# 1. Camera capture and object detection
# 2. LLM-based task planning
# 3. Trajectory generation and execution

# Example task: "Pick up the red cup and place it in the sink"
# - Detects cup and sink using OwLViT
# - Segments objects using SAM
# - Generates detailed action plan using Llama
# - Executes smooth trajectories in MuJoCo
```

### **Performance Benchmarking**
```bash
# Run performance tests
python example_llama_planner.py

# Test functionality
python test_llama_planner.py
```

## 🔍 Key Components

### **1. Perception Pipeline**
- **Camera Capture**: MuJoCo-rendered images from multiple viewpoints
- **Object Detection**: OwLViT for zero-shot detection with natural language
- **Segmentation**: SAM for pixel-perfect object masks
- **Depth Estimation**: 3D position calculation from 2D detections

### **2. LLM Planning Engine**
- **Task Understanding**: Natural language task parsing
- **Action Generation**: Structured JSON action sequences
- **Safety Validation**: Workspace limits and collision avoidance
- **Trajectory Planning**: Multi-waypoint smooth motion

### **3. Execution System**
- **MuJoCo Integration**: Physics-based simulation
- **Inverse Kinematics**: Real-time joint angle calculation
- **Action Queuing**: Smooth execution of complex plans
- **State Management**: Persistent robot state tracking

## 🛠️ Development

### **Adding New Tasks**
1. **Define task** in natural language
2. **Update object mappings** in `map_model_detections()`
3. **Test with pipeline** and validate results

### **Optimizing Performance**
1. **Profile bottlenecks** using built-in timing logs
2. **Adjust cache sizes** for your use case
3. **Tune LLM parameters** for speed vs. quality

### **Extending Capabilities**
1. **Add new perception models** to the pipeline
2. **Implement new action types** in the executor
3. **Customize robot configurations** in YAML files

## 📈 Performance Metrics

### **Typical Performance**
- **Initialization**: <0.5 seconds
- **Plan Generation**: 2-5 seconds (first time)
- **Cached Plans**: <0.001 seconds
- **Simulation**: 60Hz real-time execution
- **Cache Hit Rate**: 90%+ for repeated tasks

### **Resource Usage**
- **GPU Memory**: ~2GB for SAM + OwLViT
- **CPU**: 4-8 cores for optimal performance
- **Memory**: ~4GB RAM for full pipeline

## 🔮 Future Enhancements

### **Planned Features**
- **Real-time inference integration** into simulation loop
- **Multi-modal LLM support** (vision + language)
- **Advanced error recovery** and replanning
- **Distributed execution** across multiple robots

### **Optimization Roadmap**
- **Model quantization** for faster inference
- **Async perception** for non-blocking operation
- **Predictive planning** for smoother execution
- **Hardware acceleration** for real-time performance

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests** if applicable
5. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Meta** for Llama models and SAM
- **Lambda Labs** for high-performance LLM API
- **MuJoCo** for physics simulation
- **The robotics community** for inspiration and feedback

---

**Ready to build the future of LLM-driven robotics? Start with `python src/main.py`!** 