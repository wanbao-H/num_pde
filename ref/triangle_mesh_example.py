import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh

    
node = np.array([
    [0.0, 0.0], # 0
    [1.0, 0.0], # 1
    [1.0, 1.0], # 2
    [0.0, 1.0], # 3
    ])

cell = np.array([
    [1, 2, 0], # 0
    [3, 0, 2], # 1
    ])

mesh = TriangleMesh(node, cell) 
mesh.uniform_refine(n=1)

area = mesh.entity_measure('cell')
length = mesh.entity_measure('edge')

bc = mesh.entity_barycenter('cell')
md = mesh.entity_barycenter('edge')

ps = mesh.interpolation_points(3)
cell2ipoint = mesh.cell_to_ipoint(3)

NN = mesh.number_of_nodes()
NE = mesh.number_of_edges()
NF = mesh.number_of_faces()
NC = mesh.number_of_cells()

node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')

edge2cell = mesh.ds.edge_to_cell() # (NE, 4)
cell2cell = mesh.ds.cell_to_cell() # (NC, 3)
cell2edge = mesh.ds.cell_to_edge() # (NC, 3)
cell2node = mesh.ds.cell_to_node() # (NC, NN)
node2cell = mesh.ds.node_to_cell() # (NN, NC)

isBdNode = mesh.ds.boundary_node_flag() # bool 数组 (NN, )
bdNodeIdx = mesh.ds.boundary_node_index() # 
print('cell:\n', cell)
print('edge:\n', edge)
print('node:\n', node)
print('cell2cell:\n', cell2cell)

fig, axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
mesh.find_cell(axes, showindex=True, color='k', fontcolor='k')
plt.show()
