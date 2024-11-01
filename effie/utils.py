# -*- coding: utf-8 -*-
"""

"""
import torch
import torch.nn.functional as F
import numpy as np
from os import listdir
import sys
import pickle
import random
from random import sample


def PLL_1term(W, y_true, var_to_pred, val=False):

    nb_var = y_true.shape[1]
    nb_aa = int(W.shape[-1] ** 0.5)
    bs = W.shape[0]
    fixed_var = var_to_pred * torch.ones((bs,), dtype=torch.long)

    # ensure no data leakage: remove info on var_to_pred
    y = torch.clone(y_true)
    y[torch.arange(bs), var_to_pred] = 20  # unknown aa

    # 2 equivalent computations, from slow to fast (several 0oM)
    # Except for the order of vector
    # Note: y = y[:, np.arange(nb_var)]

    ### 1 ###
    # L_cost = torch.stack([W[np.arange(bs), fixed_var, j].
    #                 reshape(bs, nb_aa, nb_aa)
    #    [np.arange(bs),:,y[:,j]-1] for j in range(nb_var)])

    ### 2 ###
    L_cost = W[torch.arange(bs), fixed_var, torch.arange(nb_var)].reshape(bs, nb_var, nb_aa, nb_aa)[torch.arange(bs), torch.arange(nb_var), :, y]

    # accuracy
    cost_per_value = torch.sum(L_cost, dim=1)
    _, idx = torch.min(cost_per_value, dim=1)  # idx are the predicted values

    # PLL
    true_value = y_true[torch.arange(bs), fixed_var]
    # -cost because a low cost is good
    PLL = F.log_softmax(-cost_per_value, dim=1)[torch.arange(bs), true_value]

    if val:
        return torch.stack([idx, PLL])

    else:
        return PLL


def PLL_all(W, y_true, val=False, missing=None, extra_metrics=False, balance = None, 
            nb_neigh = 0, blosum = None, return_vector = False, unary_costs = None, CE = None):
    """
    Compute the total PLL loss over all variables and all batch samples.
    If validation, also compute the per-residue accuracy.
    If balance is not None, compute balanced PLL.
    """
    nb_var = W.shape[1]
    nb_aa = int(W.shape[-1] ** 0.5)
    bs = 1 #W.shape[0]

    y_indices = (y_true).unsqueeze(-1).expand(bs, nb_var, nb_aa).unsqueeze(1)
    Wr = W.reshape(bs, nb_var, nb_var, nb_aa, nb_aa)
    L_cost = Wr[
        torch.arange(bs)[:, None, None, None],
        torch.arange(nb_var)[None, :, None, None],
        torch.arange(nb_var)[None, None, :, None],
        torch.arange(nb_aa)[None, None, None, :],
        y_indices,
    ]

    if nb_neigh > 0 : #number of neighbours to ignore (for masked PLL)
        samp = sample([i for i in range(nb_var)]*nb_var, nb_neigh*nb_var) #random choice
        samp = torch.tensor(samp).reshape(nb_var, -1)
        neigh = torch.ones_like(L_cost)
        neigh[:, torch.arange(nb_var)[:, None], samp] = 0
        L_cost *= neigh    
        
        neigh = torch.ones_like(L_cost)
        neigh[:, torch.arange(nb_var)[:, None], samp] = 0
        L_cost *= neigh
    
    costs_per_value = torch.sum(L_cost, dim=2)

    if unary_costs is not None:
        costs_per_value += unary_costs.reshape(bs, -1, nb_aa)
    lsm = F.log_softmax(-costs_per_value, dim=2)
    val_lsm = lsm[torch.arange(bs)[:, None], torch.arange(nb_var)[None, :], y_true]
    
    if balance is not None:
        val_lsm = balance[y_true.squeeze()]*val_lsm
    
    if blosum is not None:
        _, idx = torch.min(costs_per_value, dim=2)
        val_lsm += blosum[idx.type(torch.LongTensor), y_true.type(torch.LongTensor)]
        
    if CE is not None:
        pba = F.softmax(-costs_per_value, dim=2).squeeze()
        y_1H = torch.zeros_like(pba.squeeze()) #no bs
        y_1H[torch.arange(nb_var), y_true] +=1
        
        return -CE(pba, y_1H)

    if val:
        _, idx = torch.min(costs_per_value, dim=2)

        if missing is not None:
            missing = missing.reshape(1, -1)
            acc = torch.sum((idx[~missing] == y_true[~missing])) / torch.sum(~missing)
        else:
            acc = torch.sum(y_true - idx == 0) / nb_var

        if extra_metrics:  # note: k_acc counts missing residues

            _, indices = torch.sort(costs_per_value, -1)
            rank = (y_true.expand(bs, nb_aa, nb_var).transpose(1, 2) == indices).nonzero(as_tuple=True)[-1]
            k_acc = torch.stack([torch.sum(rank < 3), torch.sum(rank < 5), torch.sum(rank < 10)])
            return (acc, -torch.sum(val_lsm), idx, k_acc)

        else:
            return (acc, -torch.sum(val_lsm))

    else:
        return val_lsm if return_vector else torch.sum(val_lsm)
        
        
def new_PLL(W, idx_pairs, y, val = False, nb_neigh = 0, missing = None, 
            tm = None, tm_weight = 1, unary_costs = None):
    
    nb_var, max_neigh, nb_aa = W.shape
    nb_aa = int(nb_aa**0.5)
    W = W.reshape(nb_var, max_neigh, nb_aa, nb_aa)
    L_cost = W[torch.arange(nb_var)[:, None], #on all the residues
               torch.arange(max_neigh)[None, :], #on all the neighbours
               :, 
               y[idx_pairs[torch.arange(nb_var)]] #true identity of each neighbours of each residue
              ]
    
    if nb_neigh >0:
        samp = sample([i for i in range(max_neigh)]*nb_var, nb_neigh*nb_var) #random choice
        samp = torch.tensor(samp).reshape(nb_var, -1).to(W.device)
        neigh = torch.ones(L_cost.shape[0], L_cost.shape[1]).to(W.device)
        neigh[torch.arange(nb_var)[:, None], samp] = 0
        L_cost *= neigh.unsqueeze(-1).expand(nb_var, max_neigh, nb_aa)
    
    costs_per_value = torch.sum(L_cost, dim=1)
    if unary_costs is not None :#and random.random()>0.7:
        costs_per_value += unary_costs

    lsm = F.log_softmax(-costs_per_value, dim=-1)
    val_lsm = lsm[torch.arange(nb_var), y]
    
    if val:
        _, idx = torch.min(costs_per_value, dim=1)
        #correct = y - idx == 0

        if missing is not None:
            #missing = missing.reshape(1, -1)
            acc = torch.sum((idx[~missing] == y[~missing])) / torch.sum(~missing)
        else:
            acc = torch.sum(y - idx == 0) / nb_var

        return (acc, -torch.sum(val_lsm))
    
    if tm is None:
        return torch.sum(val_lsm)
    else: 
        return torch.sum(val_lsm[tm])*tm_penalty + torch.sum(val_lsm[~np.array(tm)])

def val_metrics(W, y):

    nb_var = y.shape[1]
    p1var = torch.stack([PLL_1term(W, y, i, val=True) for i in range(nb_var)])
    y_pred = p1var[:, 0].T
    acc = torch.sum(y - y_pred == 0) / nb_var

    PLL = -p1var[:, 1].T

    return (acc, torch.sum(PLL))


def min_max_norm(ft):
    """
    Normalize each column of ft to be between 0 and 1.
    Thus, normalization is done by feature and by protein
    Output a tensor same size as input
    """

    mini, _ = torch.min(ft, dim=0)
    return (ft - mini) / (torch.max(ft, dim=0)[0] - mini)


def norm_coef(folder, device):
    """
    Compute the normalization coefficient as max of each feature on the whole dataset
    """

    d_max, nb5_max, nb10_max = 0, 0, 0

    for protein in listdir(folder):

        # loading data
        dico = torch.load(folder + protein, map_location=device)
        x = dico["features"]

        if torch.max(x[:, 7]) > d_max:  # distance
            d_max = torch.max(x[:, 7])

        if torch.max(x[:, 8]) > nb5_max:  # nb contact 5
            nb5_max = torch.max(x[:, 8])

        if torch.max(x[:, 9]) > nb10_max:  # nb contact 10
            nb10_max = torch.max(x[:, 9])

    return d_max, nb5_max, nb10_max


def normalize(ft, d_max=None, nb5_max=None, nb10_max=None):
    """
    Input: feature vector (1 quaternion (4 colums), 1 translation (3 columns), 1 distance, 4 nb of contacts)
           d_max (float): maximum distance in the train set
           nb5_max (float): max number of contact at 5 Angstrum
           nb10_max (float): max number of contact at 10 Angstrum

    Output: normalized vector (all feature between 0 and 1)

    Quaternion and translation are shifted to [0, 1]
    Other features are divided by their max (their min is 0)
    """

    ft_max_norm = ft[:, 8:12]
    ft_max_norm = torch.div(ft_max_norm, torch.stack([nb5_max, nb10_max, nb5_max, nb10_max]))
    # trf = (ft[:,:7]+1)/2 #to put nb of contact between 0 and 1 instead of -1 and 1

    return torch.cat((ft[:, :8], ft_max_norm, ft[:, 12:]), dim=1)

def pearson(L1, L2):
    cov = np.cov(L1, L2)
    if np.sum(cov == 0) == 4:
        return None
    else:
        pearson = cov[0, 1]/(cov[0,0]**0.5*cov[1, 1]**0.5)
        return pearson

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save(obj, filename, folder = ''):
    with open(folder + filename, 'wb') as file:
        pickle.dump(obj, file)
        
def load(filename, folder = ''): 
    with open(folder + filename, 'rb') as file:
        return pickle.load(file) 
        
def load_BLOSUM():
    
    """
    Reads the .txt conaining BLOSSUM
    And returns a numpy array of shape 20*20
    """

    file = open('BLOSUM62.txt', 'r')
    L = file.readlines()
    file.close()

    i = 0
    while i < len(L):
        l = L[i]
        if l[0]=="#":
            L.remove(l)
        else:
            i+=1

    BLOSUM = []
    for l in L[1:21]:
        M = []
        for e in l[1:].strip().split(" "):
            if len(e)>0:
                M.append(int(e))
        BLOSUM.append(M)
    BLOSUM = np.array(BLOSUM)[:, :20]
    
    return BLOSUM
    
def calc_E(W, seq):
    
    nb_aa = int(W.shape[-1]**0.5)
    nb_var = W.shape[1]
    W = W.reshape(nb_var, nb_var, nb_aa, nb_aa) 
    E = torch.sum(W[
        torch.arange(nb_var)[:, None],
        torch.arange(nb_var)[None, :], 
        seq[torch.arange(nb_var)[:, None]], 
        seq[torch.arange(nb_var)[None, :]]])/2
    return E
    
    
### code Romain ###
rng = np.random.default_rng()
masks = None

def get_indexes_torch(y_true, nb_val,  masks, rand_y, masks_complementary):
    device = y_true.device
    bs, nb_masks, nb_var= y_true.shape
    bs, nb_masks,nb_rand_y, mask_width = rand_y.shape

    y_true_masked = y_true[torch.arange(bs, device = device)[:,None, None], torch.arange(nb_masks, device = device)[None,:,None], masks[:,:,:]]
    rand_y = y_true_masked[:,:,None,:] + rand_y
    rand_y = torch.fmod(rand_y,nb_val)
   

    nb_joint_indexes = math.comb(mask_width,2)
    nb_nonjoint_indexes = mask_width*(nb_var-mask_width)
    nb_indexes = nb_joint_indexes + nb_nonjoint_indexes
    final_indexes = torch.zeros((bs, nb_masks, nb_rand_y, nb_indexes, 5), dtype = torch.int8, device = device)

    #rand_y = rng.integers(0,nb_val,(bs,nb_mask, 1+nb_rand_y, mask_width))
    #rand_y[:,:,0] = y_true[np.arange(bs)[:,None,None],masks[None,:,:]]

    indexes = torch.zeros((bs,nb_masks,nb_rand_y,mask_width,nb_var-mask_width,4), dtype = torch.int8, device = device)
    diag_indexes = torch.zeros((bs,nb_masks,nb_rand_y,mask_width,mask_width,4), dtype = torch.int8, device = device)

    indexes[:,:,:,:,:,0] = masks[:,:,None,:,None]
    indexes[:,:,:,:,:,1] = masks_complementary[:,:,None,None,:]
    indexes[:,:,:,:,:,2] = rand_y[:,:,:,:,None]
    indexes[:,:,:,:,:,3] = y_true[torch.arange(bs)[:,None,None,None,None],torch.arange(nb_masks)[None,:,None,None,None],masks_complementary[:,:,None,None,:]]

    diag_indexes[:,:,:,:,:,0] = masks[:,:,None,:,None]
    diag_indexes[:,:,:,:,:,1] = masks[:,:,None,None,:]
    diag_indexes[:,:,:,:,:,2] = rand_y[:,:,:,:,None]
    diag_indexes[:,:,:,:,:,3] = rand_y[:,:,:,None,:]

    triangular_indices = torch.triu_indices(mask_width, mask_width, 1)
    joint_indexes = diag_indexes[:,:,:,triangular_indices[0],triangular_indices[1]]
    non_joint_indexes = indexes.reshape((bs,nb_masks,nb_rand_y,-1,4))

    final_indexes[:,:,:,:,1:] = torch.concatenate((joint_indexes,non_joint_indexes), axis=3)
    final_indexes[:,:,:,:,0] = torch.arange(bs)[:,None,None,None]

    return final_indexes


def init_global_variables(bs, nb_var, nb_val, device):
    global r_rand, masks, er_rand, masks_complementary
    #y_true = torch.randint(0,9,(bs,nb_var))
    triu = np.triu_indices(nb_var,1)
    masks = np.concatenate((triu[0][:,None],triu[1][:,None]),axis=1)
    masks = np.broadcast_to(masks[None,:,:],(bs,masks.shape[0],2))
    r_rand = np.zeros((nb_val,nb_val,2),dtype = np.int8)
    r_rand[:,:,0] = np.arange(nb_val)[:,None]
    r_rand[:,:,1] = np.arange(nb_val)[None,:]
    r_rand = r_rand.reshape((nb_val)**2,2)
    r_rand = np.broadcast_to(r_rand[None,None,:,:], (bs, masks.shape[1], r_rand.shape[0],r_rand.shape[1]))
    nb_val+=1
    er_rand = np.zeros((nb_val,nb_val,2),dtype = np.int8)
    er_rand[:,:,0] = np.arange(nb_val)[:,None]
    er_rand[:,:,1] = np.arange(nb_val)[None,:]
    er_rand = er_rand.reshape((nb_val)**2,2)
    er_rand = np.broadcast_to(er_rand[None,None,:,:], (bs, masks.shape[1], er_rand.shape[0],er_rand.shape[1]))

    masks = torch.from_numpy(np.array(masks)).to(device)
    bs, nb_masks, mask_width = masks.shape
    r_rand = torch.from_numpy(np.array(r_rand)).to(device)
    er_rand = torch.from_numpy(np.array(er_rand)).to(device)
    masks_complementary  = torch.where((masks[:,:,None,:]==torch.arange(nb_var, device = device)[None,None,:,None]).sum(axis=3)==0)[2].reshape(bs,nb_masks,-1) # si mask = [1,2], mask_complementary = [3,4,5,6,...], cad tous les indices qui ne sont pas modifiés

   
#r_ind = get_indexes_torch(y_true, nb_val, masks, r_rand)
def PLL_all2(W, y_true, nb_neigh = 0, T = 1, nb_rand_masks = 10,hints_logit = None):
    global r_rand, masks, er_rand, masks_complementary
    """
    Compute the total PLL loss over all variables and all batch samples

    Input: the predicted cost tensor W
           the true sequence y_true (tensor)
           the number of neighbours to mask (Gangster PLL parameter), int
           unary costs hints_logit (tensor, optional)

    Output: the PLL loss
    """
    device = y_true.device
    nb_var = W.shape[1]
    bs = W.shape[0]
    nb_val = W.shape[3]

    if masks is None:
        init_global_variables(bs,nb_var,nb_val, device)


    y_mod = ((y_true-1))[:,None,:].expand(bs,nb_rand_masks,nb_var).clone()
    rand_masks = torch.randint(0,masks.shape[1],(bs,nb_rand_masks), device = y_true.device)

    if nb_neigh != 0:
        # ajoût de la variable regulatrice
        nb_val = nb_val+1
        Wpad = torch.nn.functional.pad(W,(0,1,0,1))
        randindexes = torch.rand((bs,nb_rand_masks,nb_var-2), device = device).argsort(dim=-1)[...,:nb_neigh]
        randindexes = masks_complementary[torch.arange(bs)[:,None,None],rand_masks[:,:,None],randindexes]
        y_mod[torch.arange(bs)[:,None,None],torch.arange(nb_rand_masks)[None,:,None],randindexes] = nb_val-1
        W = Wpad


    if nb_neigh==0:
        ny_indices = get_indexes_torch(y_mod,nb_val,masks[np.arange(bs)[:,None],rand_masks],r_rand[np.arange(bs)[:,None],rand_masks], masks_complementary[np.arange(bs)[:,None],rand_masks])
    else:
        ny_indices = get_indexes_torch(y_mod,nb_val,masks[np.arange(bs)[:,None],rand_masks],er_rand[np.arange(bs)[:,None],rand_masks],masks_complementary[np.arange(bs)[:,None],rand_masks])
    tny_indices = ny_indices.int()
    Wr = W.reshape(bs, nb_var, nb_var, nb_val, nb_val)
    values_for_each_y = Wr[tny_indices[:,:,:,:,0],tny_indices[:,:,:,:,1],tny_indices[:,:,:,:,2], tny_indices[:,:,:,:,3],tny_indices[:,:,:,:,4]]
    cost_for_each_y = -torch.sum(values_for_each_y, axis = 3)
    log_cost = torch.logsumexp(cost_for_each_y, dim=2)

    PLL = torch.sum(cost_for_each_y[:,:,0])-torch.sum(log_cost)

    return(PLL)
