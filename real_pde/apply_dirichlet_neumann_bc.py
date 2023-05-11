from scipy.sparse import diags
from scipy.sparse import spdiags
import scipy.sparse as sp
import numpy as np
def redefian_apply_dirichlet_bc(self, gD, A, f, uh=None):
    """
    @brief 组装 u_xx 对应的有限差分矩阵，考虑了 Dirichlet边界
    """
    if uh is None:
        uh = self.function('node')

    node = self.node
    isBdNode = self.ds.boundary_node_flag()
    isBdNode[-1] = False
    uh[isBdNode]  = gD(node[isBdNode])
    A_uh = A@uh.reshape(A.shape[0],1)
    f = f - A_uh
    f[isBdNode] = uh[isBdNode]

    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[isBdNode] = 1
    D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    A = D0@A@D0 + D1
    return A, f 

def apply_neumann_bc(self, gD, A, f, uh=None):
    """
    @brief 组装 u_xx 对应的有限差分矩阵，考虑了 Neumann边界
    """
    if uh is None:
        uh = self.function('node')

    node = self.node
    isBdNode = self.ds.boundary_node_flag()
    isBdNode[0] = False
    uh[isBdNode]  = gD(node[isBdNode])
    #A_uh = A@uh.reshape(A.shape[0],1)
    f -= A@uh.reshape(A.shape[0],1)
    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[isBdNode] = 1
    
    isBdNode[0] = True
    f.reshape(A.shape[0],)[isBdNode] = uh[isBdNode]


    D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])

    NN = A.shape[0]
    data = [-1,1]
    index = [NN-2,NN-1]
    count = np.zeros(NN+1,dtype = np.int_)
    count[-1] = 2
    D1 = (1/self.h)*sp.csr_matrix((data,index,count),shape = (NN,NN))
    A = D0@A@D0 + D1
    return A, f 

