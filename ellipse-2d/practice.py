import numpy as np
from fealpy.mesh import UniformMesh2d
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

#模型数据
class PdeData:
    """
        -v''=4pi^2(cos(pi y)cos(pi x)-sin(pi x)sin(pi y)-sin(pi x)sin(pi y))
            v(x,0)=v(x,1)=v(0,y)=v(1,y)=0
        exact solution:
            v(x,y)=sin(pi x)sin(pi y)-cos(pi x)cos(pi y)+1
    """
    def domain(self):

        return np.array([0,1,0,1])

    def solution(self,p):
        x = p[...,0]
        y = p[...,1]
        pi = np.pi
        return np.sin(pi*x)*np.sin(pi*y)-np.cos(pi*x)*np.cos(pi*y)+1
    
    def source(self,p):
        x = p[...,0]
        y = p[...,1]
        pi = np.pi
        return 4*pi*pi*(np.cos(pi*x)*np.cos(pi*y)-np.sin(pi*x)*np.sin(pi*y))
    
    def dirichlet(self,p):

        return self.solution(p)
    
pde = PdeData()
# 创建网格
nx = 700
ny = 700
domain = pde.domain()
hx = (domain[1]-domain[0])/nx
hy = (domain[3]-domain[2])/ny

mesh = UniformMesh2d((0,nx,0,ny),(hx,hy),(domain[0],domain[2]))


# 初始矩阵组装
A = mesh.laplace_operator()
f = mesh.interpolate(pde.source,'node')

# 处理边界
uh = mesh.function()
A, f = mesh.apply_dirichlet_bc(pde.dirichlet,A,f,uh)
uh.flat[:] = spsolve(A,f)

fig = plt.figure()
axes = fig.add_subplot(111,projection='3d')
mesh.show_function(axes,uh)

em = np.zeros((3,1),dtype=np.float64)
em[0,0], em[1,0], em[2,0] = mesh.error(pde.solution,uh)
print(em)
plt.show()