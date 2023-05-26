
from fealpy.mesh import UniformMesh1d
from fealpy.pde.parabolic_1d import SinExpPDEData
import matplotlib.pyplot as plt

pde = SinExpPDEData()
nx = 40
nt = 3200
domain = pde.domain()
duration = pde.duration()
hx = (domain[1]-domain[0])/nx
tau = (duration[1]-duration[0])/nt

mesh = UniformMesh1d((0,nx),hx,origin=domain[0])
# 初值

uh0 = mesh.interpolate(pde.init_solution,intertype='node')
# print(uh0)

def advance_forward(n):
    t = duration[0] + n*tau
    if n == 0:
        return uh0,t
    else:
        A = mesh.parabolic_operator_forward(tau)
        source = lambda p: pde.source(p,t)
        f = mesh.interpolate(source, intertype='node')
        uh0[:] = A@uh0 + tau*f

        gD = lambda p: pde.dirichlet(p,t)
        mesh.update_dirichlet_bc(gD, uh0)

        solution = lambda p: pde.solution(p,t)
        e = mesh.error(solution,uh0,errortype = 'max')
        print(f"the max error is {e}")
        return uh0, t

fig, axes = plt.subplots()
box = [0,1,-1.5,1.5]
mesh.show_animation(fig,axes,box,advance_forward,frames = nt+1)
plt.show()