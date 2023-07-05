import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh import TriangleMesh 


mesh = TriangleMesh.from_box(nx=2, ny=2) 
NN = mesh.number_of_nodes()

ips = mesh.interpolation_points(3)

cell2ipoint = mesh.cell_to_ipoint(3) # (NC, 10)
print(cell2ipoint)

fig, axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_cell(axes, showindex=True, fontcolor='k')

fig, axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes, node=ips, showindex=True, markersize=50, fontsize=20)
plt.show()
