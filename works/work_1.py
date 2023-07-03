import numpy as np
from scipy.sparse.linalg import spsolve
# import ipdb
import matplotlib.pyplot as plt
from fealpy.pde.elliptic_1d import ExpPDEData
from fealpy.mesh import UniformMesh1d
from scipy.sparse import diags

class model_Data:
    def domain(self):

        return [0, 1]
    
    def solution(self, p):

        return np.cos(np.pi*p)
    
    def source(self, p):

        pi = np.pi

        return np.pi**2*np.cos(np.pi*p) + 3*np.cos(np.pi*p) + np.cos(np.pi*p)**2
    
    def dirichlet(self, p):

        return self.solution(p)
    
    def reaction(self, p):
        
        return np.cos(np.pi*p) + 3

nx = 2 
maxit = 4
#ipdb.set_trace()
pde = model_Data()
domain = pde.domain()
em = np.zeros((3, maxit), dtype=np.float64)
fig, axes = plt.subplots()
lines = []  # 存储每个曲线
for i in range(maxit):
    nx *= 2
    hx = (domain[1] - domain[0])/nx
    mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])
    uh = mesh.function('node') 
    f = mesh.interpolate(pde.source, intertype='node')
    q = mesh.interpolate(pde.reaction, intertype='node')
    #two_e = 2* np.eye(mesh.number_of_nodes())
    ##############################
    NN = mesh.number_of_nodes()
    A = mesh.laplace_operator() + diags(q, shape=(NN, NN), format='csr')
    A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f, uh=uh)

    uh[:] = spsolve(A, f)
    line = mesh.show_function(axes, uh)
    line[0].set_label('Numerical solution with step size 1/{}'.format((i+1)*4))


    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)

print(em[:, 0:-1]/em[:, 1:])
print("1/4,1/8,1/16,1/32对应的最大模误差：",em[0,:])
line = mesh.show_function(axes, mesh.interpolate(pde.solution, intertype='node'))
line[0].set_label('Exact solution')
axes.legend()
plt.show()