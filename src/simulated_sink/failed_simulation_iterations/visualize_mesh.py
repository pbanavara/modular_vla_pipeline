import trimesh

mesh = trimesh.load("kitchen_sink.obj")
print(mesh.extents)

