# -*- coding: utf-8 -*-
"""

"""
import torch
import torch.nn.functional as F
import numpy as np
from os import listdir
import sys
import pickle

def read_resfile(filename, path):
    
    resfile = []
    start = False
    f = open(path + filename)

    for line in f.readlines():
        if start and line !="\n":
            resfile.append(int(line.split(" ")[0]))
        if 'start' in line:
            start = True
    f.close()
    
    return np.array(resfile)

def get_chain_dict(chains):
    chain_dict = dict()
    i = 0
    for chain in chains:
        if chain not in chain_dict:
            chain_dict[chain] = i
            i += 1
    return chain_dict
    
def symmetrize_CFN(W, chains):
    
    """
    Input: W tensor of shape n*n*20*20
           list of str indicating chain names
           
    Output: W symmetrized
    
    Example: W of shape (2n*2n*20, 20) and chains = ['A', 'B', 'A', 'B']
    the output has shape n*n*20*20
    """

    W = W.squeeze()
    nb_var = W.shape[1]
    W = W.reshape(nb_var, nb_var, -1)
    nb_aa = int(W.shape[-1]**0.5)
    nb_chain = len(chains)
    if W.shape[0]%nb_chain != 0:
        print("Cautious ! The number of chains does not match the size of matrix W.")

    nb_var_bloc = W.shape[0]//nb_chain
    nb_var = nb_var_bloc*len(set(chains))
    W_total = torch.zeros(nb_var, nb_var, nb_aa**2).to(W.device)
    
    chain_dict = get_chain_dict(chains)

    for i in range(nb_chain):
        for j in range(i, nb_chain):

            W_bloc = torch.clone(W[i*nb_var_bloc:(i+1)*nb_var_bloc, j*nb_var_bloc:(j+1)*nb_var_bloc])
            i_bloc, j_bloc = chain_dict[chains[i]], chain_dict[chains[j]]            

            if i_bloc == j_bloc:
                if i == j:
                    for v in range(nb_var_bloc):
                        W_bloc[v,v] *= torch.zeros(nb_aa, nb_aa).to(W.device).reshape(-1)
            
                else:
                    # unary terms inside the chain (keep only diagonal):
                    for v in range(nb_var_bloc):
                        W_bloc[v,v] *= 1/2*torch.eye(nb_aa, nb_aa).to(W.device).reshape(-1) 
                    #copy triu part (otherwise will not be in CFN):
                    W_bloc += W_bloc.transpose(0, 1).view(
                        nb_var_bloc, nb_var_bloc, nb_aa, nb_aa).transpose(2, 3).reshape(
                        nb_var_bloc, nb_var_bloc, -1)

                #put trilow to 0
                for k, l in torch.tril_indices(nb_var_bloc, nb_var_bloc, -1).T:
                    W_bloc[k,l] = 0
            
            
            if i_bloc>j_bloc:
                W_bloc = W_bloc.transpose(0, 1).view(
                nb_var_bloc, nb_var_bloc, nb_aa, nb_aa).transpose(2, 3).reshape(
                nb_var_bloc, nb_var_bloc, -1)
                i_bloc, j_bloc = j_bloc, i_bloc
            
            W_total[i_bloc*nb_var_bloc:(i_bloc+1)*nb_var_bloc, 
                    j_bloc*nb_var_bloc:(j_bloc+1)*nb_var_bloc] += W_bloc

            
    return(W_total)


def calc_E(W, seq):
    
    nb_var = W.shape[1]
    nb_aa = int(W.shape[-1]**0.5)
    W = W.reshape(nb_var, nb_var, nb_aa, nb_aa) 
    E = torch.sum(W[
        torch.arange(nb_var)[:, None],
        torch.arange(nb_var)[None, :], 
        seq[torch.arange(nb_var)[:, None]], 
        seq[torch.arange(nb_var)[None, :]]])/2
    
    return E
    

def write_energy(W, seq, path, name, filename = "WT_energies"):
    
    E = calc_E(W, seq)
    
    try:
        file = open(path + filename, "a")
        change_line = True
    except:
        file = open(path + filename, "x")
        change_line = False
        
    char = "\n" if change_line else ""
    file.write(char + name + " : " + str(np.round(E.item(), 2)))
    file.close()
    return
    
def write_fasta(fasta_line, str_seq, path, filename = "natives"):
    
    try:
        file = open(path + filename + '.fasta', "a")
        change_line = True
    except:
        file = open(path + filename + '.fasta', "x")
        change_line = False
        
    file.write(fasta_line + "\n")
    file.write(str_seq + '\n')
    
    file.close()
    return
    
   
def seq_to_pred(seq):
    
    letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
    
    pred = []
    for aa in seq:
        pred.append(letter_to_num[aa])

    return np.array(pred) 
    
    
def read_resfile(filename, path):
    
    resfile = []
    start = False
    f = open(path + filename)

    for line in f.readlines():
        if start and line !="\n":
            resfile.append(int(line.split(" ")[0]))
        if 'start' in line:
            start = True
    f.close()
    
    return np.array(resfile)
