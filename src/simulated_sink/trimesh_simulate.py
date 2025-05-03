import trimesh
from trimesh.boolean import mesh_boolean

# Create outer shell (50 x 40 x 20 cm)
outer = trimesh.creation.box(extents=(0.5, 0.4, 0.2))

# Create inner cavity (2cm walls and bottom)
inner = trimesh.creation.box(extents=(0.48, 0.38, 0.18))
inner.apply_translation((0, 0, 0.01))  # Raise to leave 1cm bottom

# Subtract cavity from outer
sink = mesh_boolean([outer, inner], operation="difference")
assert sink.is_volume, "Sink (outer - inner) is not a valid volume"

# Create a drain hole (cylinder in center bottom)
drain = trimesh.creation.cylinder(radius=0.025, height=0.02, sections=64)
drain.apply_translation((0, 0, 0.01))

# Subtract drain from sink
sink = mesh_boolean([sink, drain], operation="difference")
assert sink.is_volume, "Sink (minus drain) is not a valid volume"

# Add 2cm rim/lip on top
rim = trimesh.creation.box(extents=(0.54, 0.44, 0.01))
rim.apply_translation((0, 0, 0.2))  # Sits on top

# Union rim and sink
final = mesh_boolean([sink, rim], operation="union")
assert final.is_volume, "Final mesh is not a valid volume"

# Export to OBJ
final.export("kitchen_sink.obj")
print("✅ Exported to kitchen_sink.obj")

# Optional: visualize
scene = final.scene()
scene.show()
