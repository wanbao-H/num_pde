import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import IntervalMesh

node = np.array([
    [0.0], # 0
    [1.0], # 1
    ]) # (2, 1)
cell = np.array([
    [0, 1] # 0
    ]) # (1, 2)

mesh = IntervalMesh(node, cell)
mesh.uniform_refine(n=2)

node = mesh.entity('node')
cell = mesh.entity('cell')

l = mesh.entity_measure('cell')
bc = mesh.entity_barycenter('cell')
print('l:\n', l)
print('bc:\n', bc)

print('node:\n', node)
print('cell:\n', cell)


# 画网格
fig, axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, markersize=40, fontsize=24)
mesh.find_cell(axes, showindex=True, markersize=50, fontsize=30)

# 画插值点
ps = mesh.interpolation_points(3) 
cell2ipoint = mesh.cell_to_ipoint(3) # (NC, 4)
print('cell2ipoint:\n', cell2ipoint)
fig, axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes, node=ps, 
        color='b', fontcolor='b', 
        showindex=True, markersize=40, fontsize=24)
plt.show()

