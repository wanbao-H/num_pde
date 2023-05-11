from scipy.sparse import diags
import scipy.sparse as sp
from scipy.sparse import spdiags
import numpy as np
def redefian_apply_dirichlet_bc(self, gD, A, f, uh=None):
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
    f[isBdNode] = uh[isBdNode]

    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[isBdNode] = 1
    D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    A = D0@A@D0 + D1
    return A, f 

