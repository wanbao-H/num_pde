import numpy as np
import matplotlib.pyplot as plt
from fealpy.pde.elliptic_1d import SinPDEData
from fealpy.mesh import IntervalMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DiffusionIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from scipy.sparse.linalg import spsolve
from fealpy.fem import ScalarMassIntegrator


class model_Data:

    def domain(self):
        
        return [0,1]
    
    def solution(self, p):

        return np.sin(np.pi*p)*np.exp(p)
    
    def source(self, p):

        return 2*np.pi*(np.pi*np.sin(np.pi*p)-np.cos(np.pi*p))*np.exp(p)
    
    def gradient(self, p):

        return np.exp(p)*np.sin(np.pi*p) + np.pi*np.exp(p)*np.cos(np.pi*p)
    
    def dirichlet(self, p):
        
        return self.solution(p)
    
    def reaction(self, p):
        
        return np.pi**2+1



pde = model_Data()
nx  = 5
maxit = 4
domain = pde.domain() 
em = np.zeros((2, maxit), dtype=np.float64)
fig, axes = plt.subplots()
for i in range(maxit):
    nx *= 2
    mesh = IntervalMesh.from_interval(domain, nx=nx)
    space = LagrangeFESpace(mesh, p=1) 

    bform = BilinearForm(space)
    bform.add_domain_integrator(DiffusionIntegrator(q=3))
    bform.add_domain_integrator(ScalarMassIntegrator(pde.reaction, q=3))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_domain_integrator(ScalarSourceIntegrator(pde.source, q=3))
    F = lform.assembly()

    bc = DirichletBC(space, pde.dirichlet) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh)
    uh[:] = spsolve(A, F)
    node = mesh.entity('node')
    axes.plot(node, np.array(uh),label = "Numerical solution with step size 1/{}".format(10*2**i))
    H1Error = mesh.error(pde.gradient, uh.grad_value, q=3)
    L2Error = mesh.error(pde.solution, uh, q=3)
    print('H1Error:', H1Error)
    print('L2Error:', L2Error)
    em[0,i], em[1, i] = L2Error, H1Error


print(em[:, 0:-1]/em[:, 1:])
axes.plot(node, pde.solution(node), label = 'Exact solution')
axes.legend()
axes.set_xlabel("x")
axes.set_ylabel("u")
plt.show()
