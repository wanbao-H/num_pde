import num_pde as np
import matplotlib.pyplot as plt

from fealpy.mesh import IntervalMesh


mesh = IntervalMesh.from_interval_domain([0, 1], nx=3)
mesh.uniform_refine()

node = mesh.entity('node')
cell = mesh.entity('cell')

fig, axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
plt.show()