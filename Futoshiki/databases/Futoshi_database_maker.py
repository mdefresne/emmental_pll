
import sys
import tqdm
sys.path.insert(0, '..')

import Solver
import numpy as np
import pickle
import Futoshi


def genGrid(gridsize = 5, top = 200, pin = 0.07):
    """ generate a futoshi board, top is the cost of an error, and pin the probability that an inequality exists"""
    W = np.zeros((gridsize**2,gridsize**2, gridsize, gridsize), dtype=float);
    futoshi = Futoshi.Futoshi(gridsize);
    unary = np.zeros((gridsize**2, gridsize), dtype=float);
    top = 200;
    pin=0.07
    futoshi.ine[0,0,0,1]=-1
    futoshi.ine[0,1,0,0]=1
    for n1 in range(gridsize):
        for n2 in range(n1+1,gridsize):
            W[0,1,n2,n1]=top
            W[1,0,n1,n2]=top

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




    sol = Solver.solver(W, unary, all_solutions=False, debug = -1)
    futoshi.grid = np.array(sol[0][0]).reshape((gridsize,gridsize))
    return futoshi, sol[0][1]

def __main__():
    grid_size = 5
    n_grids = 20000
    print("Starting ! ")
    t = tqdm.tqdm(total=n_grids)
    good_grids=np.zeros((n_grids, grid_size, grid_size))
    features = np.zeros((grid_size**2, grid_size**2, 5))
    li = np.linspace(0,1,grid_size)
    for x in range(grid_size):
        for y in range(grid_size):
            i=y*grid_size+x
            for x1 in range(grid_size):
                for y1 in range(grid_size):
                    j=y1*grid_size+x1
                    features[i,j,0]=li[y]
                    features[i,j,1]=li[x]
                    features[i,j,2]=li[y1]
                    features[i,j,3]=li[x1]

    nfeatures = np.array(np.broadcast_to(features[np.newaxis,...],(n_grids,grid_size**2,grid_size**2,5)));
    i=0
    while (i<n_grids):
        futoshi, vsol = genGrid()
        if(vsol>150):
            continue
        if i%1==0:
            print(futoshi)
        nfeatures[i,:,:,4]=futoshi.get_info()
        good_grids[i]=futoshi.grid
        t.update(1)
        i+=1

    info = (grid_size**2, grid_size,  5) # n_variables, n_values, n_features
    data = (info,nfeatures, good_grids)
    file = open("futoshi_complete.pkl",'wb')
    pickle.dump(data,file)
    file.close()

if __name__ == "__main__":
    __main__()
