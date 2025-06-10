import torch
import torch.nn.functional as F
import numpy as np
import pandas

import math
import timeit
from random import sample

def get_indexes(y_mod, nb_val,  masks, rand_y, masks_complementary, masks_d2=None, masks_complementary_d2=None):
    """
    Compute the indexes of W needed to compute conditionnal probabilities of the PLL
    Inputs:
        y_mod the sample from the target distribution with hidden variables for epll, for each mask
        nb_val the max value that a variable can take
        masks the array of randomly generated masks
        masks_complementary the indexes of the variables not in each mask
        masks_d2 : the indexes in each masks, converted to 2nd dimension indexes ( in the case of neighbours )
        masks_complementary_d2 : the neighbouring variables that are not in the corresponding mask, converted to 2nd dimension indexes ( in the case of neighbours )

    Outputs:
        The indexes of W necessary to compute the cost of each tuple, for each mask, for each batch
    """
    device = y_mod.device

    bs, nb_masks, nb_var=y_mod.shape
    bs, nb_masks,nb_rand_y, mask_width = rand_y.shape
    bs, nb_masks, nb_complementary = masks_complementary.shape

    if masks_d2 is None: # if the index of a variable is the same in dim1 and dim2 of W ( this is not 
                    # the case when idx_pairs is not none
        masks_d2 = masks
        masks_complementary_d2 = masks_complementary

    # y_mod_masked contains the values of the masked variables
    y_mod_masked = y_mod[torch.arange(bs, device = device)[:,None, None], torch.arange(nb_masks, device = device)[None,:,None], masks[:,:,:]]
    rand_y = y_mod_masked[:,:,None,:] + rand_y
    rand_y = torch.fmod(rand_y,nb_val) # the modified values are made by adding a random tuple and then taking the modulo 
    # rand_y contains the modified values of the masked variables for each random tuple
    

    # there are two types of indexes that need to be computed : 
    #   the joint indexes, that points to binary costs beetween two masked variables
    #   and the regular indexes, that points to binary costs beetween one masked variable and another one not masked
    nb_joint_indexes = math.comb(mask_width,2)
    nb_nonjoint_indexes = mask_width*(nb_complementary)
    nb_indexes = nb_joint_indexes + nb_nonjoint_indexes
    final_indexes = torch.zeros((bs, nb_masks, nb_rand_y, nb_indexes, 5), dtype = torch.int, device = device)
    
    indexes = torch.zeros((bs,nb_masks,nb_rand_y,mask_width,nb_complementary,4), dtype = torch.int, device = device)
    joint_indexes = torch.zeros((bs,nb_masks,nb_rand_y,mask_width,mask_width,4), dtype = torch.int, device = device)

    # we first compute the regular indexes
    indexes[:,:,:,:,:,0] = masks[:,:,None,:,None]
    indexes[:,:,:,:,:,1] = masks_complementary_d2[:,:,None,None,:]
    indexes[:,:,:,:,:,2] = rand_y[:,:,:,:,None]
    indexes[:,:,:,:,:,3] = y_mod[torch.arange(bs, device = device)[:,None,None,None,None],torch.arange(nb_masks, device = device)[None,:,None,None,None],masks_complementary[:,:,None,None,:]]
    non_joint_indexes = indexes.reshape((bs,nb_masks,nb_rand_y,-1,4))

    # then we compute the joint indexes
    joint_indexes[:,:,:,:,:,0] = masks[:,:,None,:,None]
    joint_indexes[:,:,:,:,:,1] = masks_d2[:,:,None,None,:]
    joint_indexes[:,:,:,:,:,2] = rand_y[:,:,:,:,None]
    joint_indexes[:,:,:,:,:,3] = rand_y[:,:,:,None,:]

    triangular_indices = torch.triu_indices(mask_width, mask_width, 1, device=device)
    joint_indexes = joint_indexes[:,:,:,triangular_indices[0],triangular_indices[1]]

    # finally we add them all
    final_indexes[:,:,:,:,1:] = torch.concatenate((joint_indexes,non_joint_indexes), axis=3)

    final_indexes[:,:,:,:,0] = torch.arange(bs, device = device)[:,None,None,None]

    return final_indexes

def get_random_tuples(nb_val, masks, device, nb_rand_tuples=20): 
    """ 
    Randomly select nb_rand_tuples random tuples without replacement 
    Input:
    nb_val : the max value that a random tuple can hold
    masks :  the mask array ( see PLL_all2 comments ). It is only used to get the number and the width of the random tuples
                array that is going to be returned
    device: the torch device in which to stock tensors
    nb_rand_tuples:  the number of random tuples to be generated for each mask
    """

    bs, nb_masks, mask_width = masks.shape # masks is only used to get these values
    # mask_width is here the length of the tuples that will be generated
    # we start by generating all tuples of size mask_width
    if mask_width == 1:
        all_tuples = torch.zeros((nb_val,mask_width),dtype = torch.int, device = device)
        all_tuples[:,0] = torch.arange(nb_val, device = device)[:]
    if mask_width ==  2:
        all_tuples = torch.zeros((nb_val,nb_val,mask_width),dtype = torch.int, device = device)
        all_tuples[:,:,0] = torch.arange(nb_val, device = device)[:,None]
        all_tuples[:,:,1] = torch.arange(nb_val, device = device)[None,:]
    elif mask_width == 3:
        all_tuples = torch.zeros((nb_val,nb_val,nb_val,mask_width),dtype = torch.int, evice = device)
        all_tuples[:,:,:,0] = torch.arange(nb_val, device = device)[:,None, None]
        all_tuples[:,:,:,1] = torch.arange(nb_val, device = device)[None,:, None]
        all_tuples[:,:,:,2] = torch.arange(nb_val, device = device)[None,None,:]
    elif mask_width == 4:
        all_tuples = torch.zeros((nb_val,nb_val,nb_val,nb_val,mask_width),dtype = torch.int, device = device)
        all_tuples[:,:,:,:,0] = torch.arange(nb_val, device = device)[:,None, None,None]
        all_tuples[:,:,:,:,1] = torch.arange(nb_val, device = device)[None,:, None,None]
        all_tuples[:,:,:,:,2] = torch.arange(nb_val, device = device)[None,None,:,None]
        all_tuples[:,:,:,:,3] = torch.arange(nb_val, device = device)[None,None,None,:]

    all_tuples = all_tuples.reshape((nb_val)**mask_width,mask_width)

    nb_tot_tuple, mask_width = all_tuples.shape

    # for each bs, each mask, we select nb_tot_tuples random tuples without replacement from this array
    rand_tuples_indexes = torch.rand((bs,nb_masks,nb_tot_tuple),device = device)
    rand_tuples_indexes[:,:,0]=-1 # so that we always select the identity ( the null tuple ) first
    rand_tuples_indexes = rand_tuples_indexes.argsort(dim=-1)[...,:nb_rand_tuples]

    rand_tuples = all_tuples[rand_tuples_indexes]
    return rand_tuples
    
def new_PLL(W,idx_pairs,y_true, nb_neigh = 0, T=1,   nb_rand_masks = 500, nb_rand_tuples=100, mask_width = 2 ,unary_costs= None,missing=None, val=False):
    """ function used for compability with the protein code """
    if missing is not None:
        print("missing not implemented !!!")
    nb_val = int(W.shape[2]**0.5)
    W = W.reshape(1, W.shape[0], W.shape[1],nb_val, nb_val)
    y_true = y_true.unsqueeze(0)
    idx_pairs = idx_pairs.unsqueeze(0)
    return PLL_all2(W, y_true, nb_neigh = nb_neigh, T=T,   nb_rand_masks = nb_rand_masks, nb_rand_tuples=nb_rand_tuples, mask_width = mask_width ,hints_logit = unary_costs, idx_pairs =idx_pairs, val =val)

def PLL_all2(W, y_true, nb_neigh = 0, T = 1, nb_rand_masks = 100, nb_rand_tuples=30, mask_width = 2,hints_logit = None, idx_pairs = None, val = False):
    """
    Compute the total stochastic PLL loss over all variables and all batch samples

    Input: the predicted cost tensor W
           the true sequence y_true (tensor)
           the number of neighbours to mask (Gangster PLL parameter), int
           unary costs hints_logit (tensor, optional)
           number of random masks per sample nb_rand_masks, int
           width of the masks mask_width, int
           nb_rand_tuples : number of random tuples over which the conditionnal probability is computed
           idx_pairs : optionnal, array containing at the ith position the indexes of the neighbours of the ith variable ( amino acid )
           val : wether to compute and return the accuracy

    Output: the PLL loss if val is False, else (accuracy value, PLL)
    """

    device = y_true.device
    nb_var = W.shape[1]
    bs = W.shape[0]
    nb_val = W.shape[3]


    if idx_pairs is None:
        #random generation of masks
        rand_indexes = torch.rand((bs, nb_rand_masks, nb_var), device = y_true.device).argsort(dim=-1)
        masks = rand_indexes[...,:mask_width]
        masks_complementary = rand_indexes[...,mask_width:]
        #masks contains the indexes of the masked variable, which are going to recieve nb_rand_tuples random modifications
        #masks_complementary is an array that contains at position i the indexes not in the ith mask
        masks_d2 = masks
        masks_complementary_d2 = masks_complementary
    else:
        #random generation of masks
        nb_pairs = idx_pairs.shape[2]
        rand_indexes = torch.rand((bs, nb_rand_masks, nb_var), device = device).argsort(dim=-1)
        masks = torch.zeros((bs,nb_rand_masks, mask_width), dtype = torch.int, device = device)
        masks[:,:,0] = rand_indexes[...,0]
        rand_indexes_neigh = torch.rand((bs, nb_rand_masks, nb_pairs), device = device).argsort(dim=-1)
        masks[:,:,1:] = idx_pairs[torch.arange(bs, device = device)[:,None,None], masks[:,:,:1],rand_indexes_neigh[...,:mask_width-1]]
        masks_complementary = idx_pairs[torch.arange(bs, device = device)[:,None,None],masks[:,:,:1] ,rand_indexes_neigh[...,mask_width-1:]]
        # masks_d2 and masks_complementary_d2 are the indexes of the masks in the second dimension of W ( these are only relevant when idx_pairs is not None, in the other
        # case they are the same as masks and masks_complementary) 
        masks_complementary_d2 = rand_indexes_neigh[...,mask_width-1:]
        masks_d2 = torch.nn.functional.pad(rand_indexes_neigh[...,:mask_width-1],(1,0), value = 0)

    # y_mod is used to save the random masked values, that are different for each mask
    y_mod = y_true[:,None,:].expand(bs,nb_rand_masks,nb_var).clone() 
    nb_complementary = masks_complementary.shape[2]

    # implementation of unary costs
    if hints_logit is not None:
        nb_var = nb_var+1 # add 1 additionnal variable that represents hints
        Wpad = torch.nn.functional.pad(W,(0,0,0,0,0,1))
        masks_complementary_d2 = torch.nn.functional.pad(masks_complementary_d2,(0,1),value = nb_complementary)
        masks_complementary = torch.nn.functional.pad(masks_complementary,(0,1),value = nb_var-1)
        y_mod = torch.nn.functional.pad(y_mod,(0,1),value = 0) # this variable is always equal to 0
        Wpad[:,:,nb_complementary,:,0] = hints_logit[:,:] # set the binary costs of this variable to be equal to the unary costs
        W = Wpad

    # implementation of epll
    if nb_neigh != 0:
        # add 1 additionnal value whith null binary costs to represent the  masked variables
        Wpad = torch.nn.functional.pad(W,(0,1))
        randindexes = torch.rand((bs,nb_rand_masks,nb_complementary), device = device).argsort(dim=-1)[...,:nb_neigh]
        randindexes = masks_complementary[torch.arange(bs, device = device)[:,None,None],torch.arange(nb_rand_masks, device = device)[None,:,None],randindexes]
        # for each mask, select randomly nb_neigh indexes not in the mask, and mask the associated variables
        y_mod[torch.arange(bs, device = device)[:,None,None],torch.arange(nb_rand_masks, device = device)[None,:,None],randindexes] = nb_val 
        W = Wpad




    # randomly select the tuples used for each mask to compute the conditionnal probability
    rand_tuples = get_random_tuples(nb_val, masks, device, nb_rand_tuples = nb_rand_tuples)
    # compute the indexes in W needed to compute these probabilities
    ny_indices = get_indexes(y_mod,nb_val,masks,rand_tuples,masks_complementary,masks_d2, masks_complementary_d2)
    tny_indices = ny_indices.int()
    # for each mask, compute the cost for each tuple
    values_for_each_y = W[tny_indices[:,:,:,:,0],tny_indices[:,:,:,:,1],tny_indices[:,:,:,:,2], tny_indices[:,:,:,:,3],tny_indices[:,:,:,:,4]]
    cost_for_each_y = -torch.sum(values_for_each_y, axis = 3)
    log_cost = torch.logsumexp(cost_for_each_y, dim=2)

    # compute the cost PLL
    PLL = torch.sum(cost_for_each_y[:,:,0]-log_cost)

    if val: # compute accuracy
        pred = cost_for_each_y.argmax(dim=2)
        acc = (pred == 0).sum()/nb_rand_masks
        return acc, PLL 

    return(PLL)


def PLL_all(W, y_true, nb_neigh = 0, T = 1, hints_logit = None):

    """
    Compute the total PLL loss over all variables and all batch samples

    Input: the predicted cost tensor W
           the true sequence y_true (tensor)
           the number of neighbours to mask (Gangster PLL parameter), int
           unary costs hints_logit (tensor, optional)

    Output: the PLL loss
    """
    nb_var = W.shape[1]
    bs = W.shape[0]
    nb_val = W.shape[3]
    y_indices = (y_true).unsqueeze(-1).expand(bs,nb_var, nb_val).unsqueeze(1)
    Wr = W.reshape(bs, nb_var, nb_var, nb_val, nb_val)

    L_cost = Wr[torch.arange(bs)[:, None, None, None],
            torch.arange(nb_var)[None, :, None, None],
            torch.arange(nb_var)[None, None, :, None],
            torch.arange(nb_val)[None, None, None, :],
            y_indices]


    if nb_neigh > 0 : #number of neighbours to ignore

        samp = [sample([i for i in range(nb_var)], nb_neigh)for j in range(nb_var)] 
        samp = torch.tensor(samp).reshape(nb_var, -1)

        neigh = torch.ones_like(L_cost)
        neigh[:, torch.arange(nb_var)[:, None], samp] = 0
        L_cost *= neigh

    costs_per_value = torch.sum(L_cost, dim=2) 
    _, pred = torch.min(costs_per_value, -1)

    if hints_logit is not None:
        #breakpoint()
        costs_per_value += hints_logit
        pass
    lsm = F.log_softmax(-costs_per_value/T, dim=2)
    val_lsm = lsm[torch.arange(bs)[:, None],
            torch.arange(nb_var)[None, :], y_true]

    return(torch.sum(val_lsm))


def pred_all_var(W, y):

    """
    Predict the value for each variables given all other variables.
    Input: the cost matrix W
           the true sequence y_true
   Output: the predicted values and their associated cost
    """

    nb_var = W.shape[1]
    nb_val = int(nb_var**0.5)
    bs = W.shape[0] 

    y_indices = (y-1).unsqueeze(-1).expand(bs,nb_var, nb_val).unsqueeze(1)
    Wr = W.reshape(bs, nb_var, nb_var, nb_val, nb_val)
    L_cost = Wr[torch.arange(bs)[:, None, None, None],
            torch.arange(nb_var)[None, :, None, None],
            torch.arange(nb_var)[None, None, :, None],
            torch.arange(nb_val)[None, None, None, :],
            y_indices]

    costs_per_value = torch.sum(L_cost, dim=2)
    _, idx = torch.min(costs_per_value, dim=2)

    # PLL
    lsm = F.log_softmax(-costs_per_value, dim=2)
    PLLs = lsm[torch.arange(bs)[:, None],
            torch.arange(nb_var)[None, :], y-1]
    return(torch.stack([(idx+1), PLLs]))


def val_metrics(W, targets):

    """return the number of correctly predicted values and the value of -PLL"""

    best_acc = 0
    for target in targets:
        y_pred, PLL = pred_all_var(W, target)
        target_acc = torch.sum((target-y_pred) == 0)
        target_nPLL = -torch.sum(PLL)

        if target_acc > best_acc:
            best_acc = target_acc
            nPLL = target_nPLL

    return(best_acc, nPLL)
