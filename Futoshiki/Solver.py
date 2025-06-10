import csv
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import math
from tqdm import tqdm
import pickle

from Futoshiki.CFN import *
#import pytoulbar2 as tb2

def make_CFN(W, unary = None, top=999999, resolution = 1, backtrack = 5000000, allSolutions = 0, verbose = -1):
    
    Problem = CFN(top, resolution, vac=True, backtrack=backtrack, allSolutions = allSolutions, verbose = verbose)
    #Problem = tb2.CFN(top, resolution, vac=True)
    nb_var = W.shape[0]
    nb_val = W.shape[3]
    
    #Create variables
    for i in range(nb_var):
        Problem.AddVariable('x' + str(i), range(1, nb_val+1))
    #costs
    for i in range(nb_var):
        for j in range(i+1, nb_var):
            Problem.AddFunction([i, j], W[i,j].flatten())
            
    #unary costs
    if unary is not None:
        for i in range(nb_var):
            if np.max(unary[i])>0:
                Problem.AddFunction([i], unary[i])#*2*top)
    
    return Problem


def solver(W,unary=None, solution=None, random=False, top=999999, resolution = 1, margin = 1,
          setUB = False, all_solutions = False, onlyDump = False, debug = -1):
    """
    Solve the sudoku with constraints from matrix W and hints in unary
    If solution is given, the solution is penalized for the Hinge Loss
    """
    allSolutions = 1000 if all_solutions else 0
    Problem = make_CFN(W, unary=unary, top = top, resolution = resolution, allSolutions = allSolutions, verbose = debug)
    Problem.Dump("dump.wcsp")
    if onlyDump:
        return;
    
    if setUB:
        Problem.SetUB(10**(-resolution))

    sol = Problem.Solve()
    #print(sol)
    #if (sol == None) or sol[0][0] == 10:
    #    return(random_solver(W, sudoku_in_line))
    
    if all_solutions:
        all_sol = Problem.GetSolutions()
        return all_sol
    else:
        return (sol,)


def genGrid(gridsize = 5, top = 200, pin = 0.07):
    """ generate a futoshi board, top is the cost of an error, and pin the probability that an inequality exists"""
    W = np.zeros((gridsize**2,gridsize**2, gridsize, gridsize), dtype=float);
    futoshi = Futoshi.Futoshi(gridsize);
    unary = np.zeros((gridsize**2, gridsize), dtype=float);
    top = 200;
    pin=0.07
    for x1 in range(gridsize):
        for y1 in range(gridsize):
            i=y1*gridsize+x1
            for x2 in range(gridsize):
                unary[i,x2] = np.random.random()*10;
                for y2 in range(gridsize):
                    j=y2*gridsize+x2
                    if (x1==x2) or (y1==y2):
                        for n1 in range(gridsize):
                            W[i,j, n1,n1] = top;
                    if ((x1==x2+1) and y1==y2) or (x1==x2 and (y1==y2+1)) or ((y1==y2-1) and x1==x2) or (y1==y2 and x1==x2-1):
                        # si les deux cases sont adjacentes, on ajoute une inégalité avec probabilité pin
                        if np.random.random()<pin and futoshi.ine[y1,x1,y2,x2]==0:
                            ineqType = -1 if np.random.random()>0.5 else 1
                            futoshi.ine[y1,x1,y2,x2]=ineqType;
                            futoshi.ine[y2,x2,y1,x1]=-ineqType;
                            for n1 in range(gridsize):
                                for n2 in range(n1+1,gridsize):
                                    if ineqType == 1: # si Ni>Nj
                                        W[i,j,n1,n2]=top
                                        W[j,i,n2,n1]=top
                                    if ineqType == -1: # si Nj>Ni
                                        W[i,j,n2,n1]=top
                                        W[j,i,n1,n2]=top




    sol = solver(W, unary, all_solutions=False)
    futoshi.grid = np.array(sol[0][0]).reshape((gridsize,gridsize))
    return futoshi, sol[0][1]

def __main__():
    grid_size = 5
    n_grids = 20000
    t = tqdm(total=n_grids)
    good_grids=np.zeros((n_grids, grid_size, grid_size))
    nfeatures = np.zeros((n_grids, grid_size**2, grid_size**2))
    i=0
    while (i<n_grids):
        futoshi, vsol = genGrid()
        if(vsol>150):
            continue
        nfeatures[i]=futoshi.get_info()
        good_grids[i]=futoshi.grid
        t.update(1)
        i+=1

    info = (grid_size**2, grid_size,  5) # n_variables, n_values, n_features
    data = (info,nfeatures, good_grids)
    file = open("futoshi.pkl",'wb')
    pickle.dump(data,file)
    file.close()

