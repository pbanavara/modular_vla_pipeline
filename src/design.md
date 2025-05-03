### Overall design for the planning and action module

#### Perception
From a camera (either stereo or momocular) get the bounding boxes and from the bounding boxes get the world co-ordinates (TBD)
#### Planning
Given a world coordinate, use a LLM with the predefined prompt and robot model to generate the action sequence.
```
llm_planner.py
```
#### Action
Run the simulator for the generated action sequence 
```
mujoco_executor.py
```

### Combine all of these into one seamless pipeline
Start with the video feed
For every frame camera outputs the semantic map of all detected objects
Pick one object with the highest confidence, check if it's dirty. 


