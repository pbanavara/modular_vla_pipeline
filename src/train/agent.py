import openai
import json

def generate_washing_plan(scene_objects, user_prompt):
    """
    Given a set of scene objects and a user prompt, generate a washing plan using a LLM backend.
    """
    prompt = f"""
   given a set of scene objects {scene_objects} and user instructins {user_prompt}, generate a structured JSON dish washing plan. The plan should include the following:
   action, object, duration, force required.
   calculate the force based on the textture of the utensil and the dirt accumulated on the utensil.
   """
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a robotics planning assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
   
    try:
        plan = response.choices[0].message.content
        print("Plan returne", plan)
        return json.loads(plan)
    except json.JSONDecodeError:
        print("Error decoding LLM response.")
        return []
    
scene_objects = ["plate", "glass", "pan", "bownl", "wok"]
user_prompt = "wash all dishes"
washing_plan = generate_washing_plan(scene_objects, user_prompt)
print (json.dumps(washing_plan, indent=2))