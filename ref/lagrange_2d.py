import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh import TriangleMesh 


mesh = TriangleMesh.from_box(nx=5, ny=5) 

NN = mesh.number_of_nodes()
node = mesh.entity('node') # (NN, 2)
cell = mesh.entity('cell') # (NC, 3)

uh0 = np.zeros(NN, dtype=np.float64) # (NN, )
uh1 = np.zeros(NN, dtype=np.float64)
uh1[15] = 1

fig, axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, markersize=50, fontsize=30)

fig = plt.figure()
axes = fig.add_subplot(projection='3d')
axes.plot_trisurf(node[:, 0], node[:, 1], uh0, triangles=cell, lw=2)
axes.plot_trisurf(node[:, 0], node[:, 1], uh1, triangles=cell, lw=2)
plt.show()
