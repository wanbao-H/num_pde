import numpy as np
from typing import Union, Tuple, List
from fealpy.mesh import UniformMesh1d
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse import diags, csr_matrix
import scipy.sparse as sp

class PdeData:
    def __init__(self, D:Union[Tuple[int,int],List[int]]= (0,2), T:Union[Tuple[int, int],List[int]]= (0,1)) -> None:
        self._domain =  D
        self._duration = T
    
    def domain(self):

        return self._domain
    
    def duration(self):

        return self._duration
    
    def solution(self, p: np.ndarray, t: np.float64):

        return 1 + np.sin(2*np.pi*(p + 2*t))
    
    def init_solution(self, p: np.ndarray):

        return 1 + np.sin(2*np.pi*p)
    
    def source(self, p: np.ndarray, t: np.float64):

        return 0.0
    
    def dirichlet(self, p: np.ndarray, t: np.float64):

        return 1 + np.sin(4*np.pi*t)
    
    def a(self):
        
        return -2 
    
def hyperbolic_lax_wendroff_1(mesh,a, tau):
    """
    @brief Lax-Wendroff 格式
    """
    r = a*tau/mesh.h

    if r > 1.0:
        raise ValueError(f"The r: {r} should be smaller than 1.0")

    NN = mesh.number_of_nodes()
    k = np.arange(NN)

    A = diags([1 - r**2], [0], shape=(NN, NN), format='csr')
    val0 = np.broadcast_to(-r/2 + r**2/2, (NN-1, ))
    val1 = np.broadcast_to(r/2 + r**2/2, (NN-1, ))
    I = k[1:]
    J = k[0:-1]
    A += sp.csr_matrix((val0, (I, J)), shape=(NN, NN), dtype=mesh.ftype)
    A += sp.csr_matrix((val1, (J, I)), shape=(NN, NN), dtype=mesh.ftype)

    return A
    
def hyperbolic_lax_wendroff_2(mesh,a, tau):
    """
    @brief Lax-Wendroff 格式
    """
    r = a*tau/mesh.h

    if r > 1.0:
        raise ValueError(f"The r: {r} should be smaller than 1.0")

    NN = mesh.number_of_nodes()
    k = np.arange(NN)

    A = diags([1 - r**2], [0], shape=(NN, NN), format='csr')
    val0 = np.broadcast_to(-r/2 + r**2/2, (NN-1, ))
    val1 = np.broadcast_to(r/2 + r**2/2, (NN-1, ))
    I = k[1:]
    J = k[0:-1]
    A += sp.csr_matrix((val0, (J, I)), shape=(NN, NN), dtype=mesh.ftype)
    A += sp.csr_matrix((val1, (I, J)), shape=(NN, NN), dtype=mesh.ftype)

    return A

pde = PdeData()
domain = pde.domain()
duration = pde.duration()

nx = 800
hx = (domain[1]-domain[0])/nx

nt = 6400
tau = (duration[1]-duration[0])/nt

mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')
error = np.zeros(nt+1)
solution = lambda p: pde.solution(p,0)
error[0] = mesh.error(solution,uh0,errortype = 'max')
T = np.zeros(nt+1)

def hyperbolic_lax_wendroff(n):
    t = duration[0] + n*tau
    if n == 0:
        return uh0,t
    else:
        r = pde.a() * tau / hx
        # A = hyperbolic_lax_wendroff_1(mesh, pde.a(), tau)
        A = hyperbolic_lax_wendroff_2(mesh, pde.a(), tau)


        uh_0 = uh0[0] + 2*tau/hx*(uh0[1] - uh0[0])
        # uh0[0] = uh0[1] # b
        uh0[:] = A@uh0
       

        gD = lambda p: pde.dirichlet(p,t)
        mesh.update_dirichlet_bc(gD, uh0)
        uh0[0] = uh_0 # a

        # uh0[0] = 2*uh0[1] - uh0[2] # c


        solution = lambda p: pde.solution(p,t)
        e = mesh.error(solution,uh0,errortype = 'max')
        error[n] = e
        T[n] = t
        print(f"the max error is {e}")
        return uh0, t
    
box = [0,2,0,2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, hyperbolic_lax_wendroff, frames=nt+1)

ax2 = plt.figure()
plt.plot(T,error)
plt.xlim()
plt.ylim()
plt.xlabel('t')
plt.ylabel('max_error')
plt.show()