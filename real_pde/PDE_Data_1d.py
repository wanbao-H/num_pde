import numpy as np
from fealpy.decorator import cartesian
from typing import List 


class PDE_Data:
    def domain(self):
        """
        @brief 得到 PDE 模型的区域

        @return: 表示 PDE 模型的区域的列表
        """
        return [0, 1000]

    @cartesian
    def solution(self, p):
        """
        @brief 计算 PDE 模型的精确解
        
        @param p: 自标量 x 的数组

        @return: PDE 模型在给定点的精确解
        """
        return None

    @cartesian
    def source(self, p,flag):
        """
        @brief: 计算 PDE 模型的原项 

        @param p: 自标量 x 的数组

        @return: PDE 模型在给定点处的源项
        """
        if flag == 1 or flag == 0:
            return 10
        return 0

    @cartesian    
    def gradient(self, p):
        """
        @brief: 计算 PDE 模型的真解的梯度

        @param p: 自标量 x 的数组

        @return: PDE 模型在给定点处真解的梯度
        """
        if p == 1000:
            return 0
        return None

    @cartesian    
    def dirichlet(self, p):
        """
        @brief: 模型的 Dirichlet 边界条件
        """
        # dirichlet
        if p == 0:
            return 100
        None
    
    def neumann(self, p):
        """
        @brief: 模型的 neumann 边界条件
        """
        # neumann       
        return self.gradient(p)


# pde = PDE_Data()
# print(pde.source(0,1))
# print(pde.domain())
# print(pde.dirichlet(0))
# print(pde.neumann(1000))