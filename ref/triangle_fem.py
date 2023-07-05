import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.elliptic_2d import SinSinPDEData
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace

pde = SinSinPDEData()
domain = pde.domain()

mesh = TriangleMesh.from_box(domain, nx=2, ny=2)
space = LagrangeFESpace(mesh, p=1)

qf = mesh.integrator(3)
bcs, ws = qf.get_quadrature_points_and_weights()

phi = space.basis(bcs) # (NQ, 1, ldof) 
gphi = space.grad_basis(bcs) #(NQ, NC, ldof, 2)
cell2dof = space.cell_to_dof() # (NC, ldof)

cm = mesh.entity_measure('cell')
# (NC, ldof, ldof)
S = np.einsum('q, qcid, qcjd, c->cij', ws, gphi, gphi, cm)
# (NC, ldof, ldof)
M = np.einsum('q, qci, qcj, c->cij', ws, phi, phi, cm)

ps = mesh.bc_to_point(bcs)  
print('ps.shape:', ps.shape)

print(bcs) # (NQ, 3)
print(ws)

fig, axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes, node=ps.reshape(-1, 2), showindex=True, markersize=50, fontsize=20)
plt.show()
