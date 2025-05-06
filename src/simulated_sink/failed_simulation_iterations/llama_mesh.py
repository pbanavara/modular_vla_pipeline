"""
Generate a 3D mesh of a kitchen sink using LLaMA-Mesh.
Not working out as expected.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "Zhengyi/LLaMA-Mesh"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # use fp16 for memory efficiency
).to(device)

prompt = """
Generate the 3D mesh of a kitchen sink.
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=2048)
tensor_output = output[0]  # shape: (token_count,)
mesh_text = tokenizer.decode(tensor_output, skip_special_tokens=True)
print(mesh_text)
with open("sink.obj", "w") as sink:
    sink.write(mesh_text)
