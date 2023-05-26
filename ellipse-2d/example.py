import numpy as np
from  fealpy.mesh import UniformMesh2d
from fealpy.pde.elliptic_2d import SinSinPDEData
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


pde = SinSinPDEData()

nx = 500
ny = 500
domain = pde.domain()

hx = (domain[1]-domain[0])/nx
hy = (domain[3]-domain[2])/nx

mesh = UniformMesh2d((0,nx,0,ny),h=(hx,hy),origin=(domain[0],domain[2]))
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

A = mesh.laplace_operator()
f = mesh.interpolate(pde.source,intertype='node')#.flat[:]
uh = mesh.function('node')
A,f = mesh.apply_dirichlet_bc(gD=pde.dirichlet,A=A,f=f)
uh.flat[:] = spsolve(A,f)

fig = plt.figure(4)
axes = fig.add_subplot(111,projection = '3d')
mesh.show_function(axes,uh)

#误差计算
em = np.zeros((3,1),dtype=np.float64)
em[0,0], em[1,0], em[2,0] = mesh.error(pde.solution,uh)
print(em)
#plt.show()