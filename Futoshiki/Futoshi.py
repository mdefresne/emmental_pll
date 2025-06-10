# -*- coding: utf-8 -*-
"""
Futoshi Class
"""

import numpy as np
import torch
from Futoshiki.Solver import *

class Futoshi :
    def __init__(self, grid_size, ine = None, grid = None):
        self.grid_size = grid_size
        if ine is None:
            self.ine = np.zeros((grid_size, grid_size , grid_size, grid_size), dtype=np.int8);
        else:
            self.ine = ine.reshape((grid_size,grid_size,grid_size,grid_size)).astype(np.int8);
        self.grid = grid
        if grid is None:
            self.grid = np.zeros((grid_size, grid_size), dtype = np.int8)

    def get_info(self):
        info = self.ine.reshape(self.grid_size**2,self.grid_size**2)
        return info

    def get_cost(self, W, unary=None):
        cost = 0
        if unary is None:
            unary = np.zeros((self.grid_size**2,self.grid_size))
        for x1 in range(self.grid_size):
            for y1 in range(self.grid_size):
                i=self.grid_size*y1+x1
                cost+=unary[i,self.grid[y1,x1]-1]
                for x2 in range(self.grid_size):
                    for y2 in range(self.grid_size):
                        j=self.grid_size*y2+x2
                        cost+=W[i,j,self.grid[y1,x1]-1,self.grid[y2,x2]-1]
        return cost
                        

    def solve(self, W, unary = None, debug = False):
        if unary is None:
            unary = np.zeros((self.grid_size**2,self.grid_size))
        for x1 in range(self.grid_size):
            for y1 in range(self.grid_size):
                i=y1*self.grid_size+x1
                if(self.grid[y1,x1]!=0): # add negative costs to hints
                    unary[i,self.grid[y1,x1]-1]=-10
        d=-1
        if debug == True:
            d = 0 
        sols = solver(W,unary, debug = d)[0]
        self.grid=np.array(sols[0],dtype=np.int8).reshape((self.grid_size,self.grid_size))

    def check_validity(self):
        for i in range(self.grid_size):
            if(np.unique(self.grid[i]).shape[0]!=self.grid_size):return False; 
            if(np.unique(self.grid[:,i].flatten()).shape[0]!=self.grid_size):return False; 
            for j in range(self.grid_size):
                for u in [i+1,i,i-1]:
                    for v in [j+1,j,j-1]:
                        if(u<0 or j<0 or u>=self.grid_size or v>=self.grid_size or (u!=i and v!=j)):continue
                        if(self.ine[i,j,u,v]==1 and self.grid[i,j]<self.grid[u,v]):return False
                        if(self.ine[i,j,u,v]==-1 and self.grid[i,j]>self.grid[u,v]):return False

        return True


    def __str__(self):
        ret = ""
        for y in range(self.grid_size):
            if y>0:
                for x in range(self.grid_size):
                    if self.ine[y,x,y-1,x]==1:
                        ret += "^"
                    elif self.ine[y,x,y-1,x]==-1:
                        ret += "v"
                    else:
                        ret += " "
                    ret +=" "


            ret +="\n"
            for x in range(self.grid_size):
                ret+=str(self.grid[y,x])
                if x<self.grid_size-1:
                    if self.ine[y,x,y,x+1]==1:
                        ret+=">"
                    elif self.ine[y,x,y,x+1]==-1:
                        ret+="<"
                    else:
                        ret+=" "
            ret+="\n"
        return ret
