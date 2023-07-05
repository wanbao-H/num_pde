import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import IntervalMesh


mesh = IntervalMesh.from_interval_domain([0, 1], nx=3)
mesh.uniform_refine()

node = mesh.entity('node')
cell = mesh.entity('cell')
print(node)
print(cell)

NN = mesh.number_of_nodes()

uh0 = np.zeros(NN, dtype=np.float64)
uh0[0] = 1

uh4 = np.zeros(NN, dtype=np.float64)
uh4[4] = 1

uh6 = np.zeros(NN, dtype=np.float64)
uh6[6] = 1

fig, axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, markersize=50, fontsize=30)
mesh.show_function(axes, uh0)
mesh.show_function(axes, uh4)
mesh.show_function(axes, uh6)
plt.show()
