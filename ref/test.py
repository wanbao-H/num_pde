
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import UniformMesh1d

import ipdb

class PDE():
    def __init__(self, D=[0, 1], T=[0, 2]):
        self._domain = D 
        self._duration = T

    def domain(self):
        return self._domain

    def duration(self):
        return self._duration

    def solution(self, p, t):
        """
        @brief PDE 模型的真解
        """
        pi = np.pi
        val0 = np.sin(pi*(p-t))
        val1 = np.sin(pi*(p+t))
        val = (val0 + val1)/2.0
        val -= (val0 - val1)/2.0/pi
        return val

    def init_solution(self, p):
        return np.sin(np.pi*p)

    def init_solution_diff_t(self, p):
        return np.cos(np.pi*p)

    def dirichlet(self, p, t):
        pi = np.pi
        x0 = self._domain[0]
        x1 = self._domain[1]
        val = np.zeros_like(p)
        isLeft = np.abs(p-x0) < 1e-12
        isRight = np.abs(p-x1) < 1e-12
        val[isLeft] = np.sin(pi*t)/pi 
        val[isRight] = -np.sin(pi*t)/pi
        return val


#ipdb.set_trace()
pde = PDE()

# 空间离散
domain = pde.domain()
nx = 10
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 20 
tau = (duration[1] - duration[0])/nt

uh0 = mesh.interpolate(pde.init_solution, intertype='node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, intertype='node')
uh1 = uh0 + tau*vh0

def advance(n):
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        return uh1, t
    else:
        A0, A1, A2 = mesh.wave_operator(tau, theta=0.0) 
        f = A1@uh1 + A2@uh0
        uh0[:] = uh1
        uh1[:] = f

        gD = lambda p : pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh1)

        return uh1, t


box = [0, 1, -1.1, 1.1]
for n in range(nt+1):
    uh, t = advance(n)

    u = lambda p : pde.solution(p, t)
    e = mesh.error(u, uh, errortype='max')
    if n in {5, 10, 15, 20}:
        print('current time:', t, e)
        fig, axes = plt.subplots()
        mesh.show_function(axes, uh, box=box)

        u = mesh.interpolate(u, intertype='node')
        mesh.show_function(axes, u, box=box)
        title = f"test{n:06d}.png" 
        print(title)
        plt.savefig(title)
        plt.close(fig)
