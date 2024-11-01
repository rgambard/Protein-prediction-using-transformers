import torch
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
from os import listdir
from torch.utils.data import Dataset
import json, time, copy
import tqdm, random
import os
import pandas as pd
import sys
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from effie.CbNet_optiR import *
#from CbNet_env import *
from effie.utils import *
from effie.utils_assemblies import *
from effie.Features import *

torch.manual_seed(0) #for reproducibility
if torch.cuda.is_available():  
    dev = "cuda:0" 
    print("GPU connected")
else:  
    dev = "cpu"
device = torch.device(dev)

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20}
                       
letter3to1 = {'CYS' : 'C', 'ASP' : 'D', 'SER' : 'S', 'GLN' : 'Q', 'LYS' : 'K',
             'ILE' : 'I', 'PRO' : 'P', 'THR' : 'T', 'PHE' : 'F', 'ALA' : 'A',
             'GLY' : 'G', 'HIS' : 'H', 'GLU' : 'E', 'LEU' : 'L', 'ARG' : 'R',
             'TRP' : 'W', 'VAL' : 'V', 'ASN' : 'N', 'TYR' : 'Y', 'MET' : 'M'}


lr = 0.0001*5
weight_decay = 0.001
use_amp = False #Automated mixed precision
#with AMP, choose dim layer that are *8

nb_aa = 20
# Parameters for gMLP
embed_dim = 256
depth_gMLP = 3
ff_mult = 2
seq_len = 48
gMLP_output_dim = 32*2*2
# Parameters for ResMLP
hidden = 128*2
nblocks = 3
block_size = 2
unary = False
multichain = False

reg_term = 0.0001/10
L1_as_fd = False
adapt_LR = 0 #1/2  #1 to adapt LR with bs, 1/D to adapt with sqrt(bs), 0 to do nothing
nb_it=6
thresh = 15
max_neigh = seq_len
noise_std = 0
perc_mask = 0.3 #for gangster PLL

if L1_as_fd:
    reg_term /= 10
lr /= 100**adapt_LR


#replace CbNet by CbNet_edge
model = CbNet(embed_dim=embed_dim,
            depth_gMLP=depth_gMLP,
            ff_mult=ff_mult,
            seq_len=seq_len,
            gMLP_output_dim = gMLP_output_dim,
            # Options for ResNet
            hidden=hidden,
            nblocks=nblocks,
            nb_it=nb_it,
            block_size=block_size,
            nb_aa=nb_aa,
            multichain=multichain,
            unary=unary)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, threshold=0.0001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp) #Gradient scaling (for amp)

if unary:
    CE_loss = nn.CrossEntropyLoss()
    alpha = 1/2

resume_training =False
if resume_training:
    filename = "PLL_unary_rand_50"  # TO CHANGE ###
    checkpoint = torch.load("../Results/model/" + filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict= False)
    #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #scaler.load_state_dict(checkpoint["scaler_state_dict"])
    #scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch0 = 1
    lr = checkpoint["lr"]
    lr = 0.0002/(3*3*3*3)
    print("Resuming training")
    print(lr)
    #print(optimizer.param_groups[0]['lr'])
    #optimizer.param_groups[0]['lr'] /= 10

else:
    epoch0 = 1

filename = 'PLL_unary'
print(filename)

file = open("../Results/" + filename + ".txt", "a")
file.write(
    "\n Training with the following parameters: "
    + "\n Embedding dim: " + str(embed_dim)
    + "\n depth of gMLP: " + str(depth_gMLP)
    + "\n ff_mult: " + str(ff_mult)
    + "\n gMLP output dim: " + str(gMLP_output_dim)
    + "\n nblock: " + str(nblocks)
    + "\n size of block: " + str(block_size)
    + "\n hidden size: " + str(hidden)
    + "\n lr: " + str(lr)
    + "\n nbit: " + str(nb_it)
    + "\n weight decay: " + str(weight_decay)
    + "\n L1: " + str(reg_term)
    + "\n thresh: " + str(thresh)
    + "\n Gangster PLL: " + str(perc_mask)
    + "\n adapt_lr: " + str(adapt_LR)
)
file.close()


if multichain:
    T_path = '/netapp/tbi/ead1/EAD1_Remaud/Thematic_Groups/Modelo/mdefresn/'
    path = '/gpfsscratch/rech/hro/unk56ix/'
    data_path =  T_path + "pdb_2021aug02/"
    sample_path = "../" + "pdb_2021aug02_sample/"

    q, p = load_assemblies(sample_path, num_examples_per_epoch = 10)
    train_assemblies, val_assemblies = dataset(q), dataset(p)
    #t = load_test_assemblies(sample_path, num_examples_per_epoch = 100)
    #test_assemblies = dataset(t)
    print(len(train_assemblies), len(val_assemblies))
else:
    cath = CATHDataset(path="../Ingraham2019/chain_set.jsonl",
                                splits_path="../Ingraham2019/chain_set_splits.json")
                                    

for epoch in range(epoch0, 300):

    ### Training ###
    PLL_epoch = 0.0
    model.train()

    if resume_training and epoch<5:
        lr = lr * 3

    #reload every few epochs:
    if epoch % 1 == 0 and multichain:
        train_assemblies, val_assemblies = dataset(q), dataset(p)
    train_set = train_assemblies if multichain else cath.train
    random.shuffle(train_set)
    train_set = train_set[:2000]
    
    for protein in tqdm.tqdm(train_set):
        optimizer.zero_grad(set_to_none=True)  # arg to go faster

        if multichain:
            crd, seq, chain_idx = data_from_protein(protein)
        else:
            crd, seq, chain_idx = torch.as_tensor(protein['coords']), protein['seq'], None

        crd, seq, chain_idx,_ = remove_N_C_ter(crd, seq, chain_idx, None)
        missing = torch.isnan(torch.sum(crd.reshape(-1, 4*3), dim =1))
        seq = torch.as_tensor([letter_to_num[a] for a in seq], dtype=torch.long)
        # If a residue is unknown('X'), it will not be predicted
        missing = torch.logical_or(missing, (seq == 20)).to(device)
        seq[seq==20]=0

        crd = crd.to(device)
        y = seq.type(torch.LongTensor).to(device)
        nb_var = crd.shape[0]
        
        if noise_std > 0 and random.random()>0.5:
            noise = torch.randn(*crd.shape).to(device)
            crd += noise*noise_std
            noise = noise.reshape(-1, 12)

        if nb_var <= 3000 and nb_var > 50:

            #print(nb_var)
            optimizer.param_groups[0]['lr'] = lr*(nb_var**adapt_LR) #adapt LR to the size of the protein
            
            #with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = use_amp):
            #Mixed precision training (some operation will ve casted to float 16 (instead of default 32)
            #without reducing the accuracy)
                    
            if unary:
                W, idx_pairs, unary_costs = model(crd, thresh = thresh, max_neigh = max_neigh, chain_idx=chain_idx)
            else:
                W, idx_pairs = model(crd, thresh = thresh, max_neigh = max_neigh, chain_idx=chain_idx)
                unary_costs=None
            PLL = -new_PLL(W, idx_pairs, y, nb_neigh = int(perc_mask*min(max_neigh, nb_var)), unary_costs=unary_costs) 
            
            if L1_as_fd:
                L1 = torch.linalg.vector_norm(d*torch.linalg.vector_norm(W, dim = -1, ord=1)
                    , ord=1)
            else:
                L1 = torch.linalg.vector_norm(W*reg_term, ord=1)
            loss = PLL + L1     
                
            if unary:
                L1 = L1 + torch.linalg.vector_norm(unary_costs*reg_term, ord=1)
                loss = (1-alpha)*PLL + alpha*CE_loss(-unary_costs, y) + L1
            
            PLL_epoch += PLL.item()
            #print(PLL)
                
            # Exits autocast before backward().
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            # source: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            #scaler.scale(loss).backward()
            #scaler.step(optimizer) #skipped if gradients are None
            #scaler.update()
            
            loss.backward()
            optimizer.step()    

    file = open("../Results/" + filename + ".txt", "a")
    file.write("\n Epoch " + str(epoch) + " - PLL loss:" + str(PLL_epoch))
    file.close()
    
    
    ### Validation ###
    with torch.no_grad():

        model.eval()
        L_acc, PLL_tot, L_tot = [], 0, 0
        var_tot = 0

        val_set = val_assemblies if multichain else cath.val
        
        for protein in val_set:

            if multichain:
                crd, seq, chain_idx = data_from_protein(protein)
            else:
                crd, seq, chain_idx = torch.as_tensor(protein['coords']), protein['seq'], None

            crd, seq, chain_idx,_ = remove_N_C_ter(crd, seq, chain_idx, None)
            missing = torch.isnan(torch.sum(crd.reshape(-1, 4*3), dim =1))
            seq = torch.as_tensor([letter_to_num[a] for a in seq], dtype=torch.long)
            # If a residue is unknown('X'), it will not be predicted
            missing = torch.logical_or(missing, (seq == 20)).to(device)
            seq[seq==20]=0

            crd = crd.to(device)
            y = seq.type(torch.LongTensor).to(device)
            nb_var = crd.shape[0]

            if nb_var < 50:
                continue
            #with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = use_amp):
            if unary:
                W, idx_pairs, unary_costs = model(crd, thresh = thresh, max_neigh = max_neigh, chain_idx=chain_idx)
                acc, PLL = new_PLL(W, idx_pairs, y, nb_neigh = 0, val = True, unary_costs = unary_costs)
            else:
                W, idx_pairs = model(crd, thresh = thresh, max_neigh = max_neigh, chain_idx=chain_idx)
                acc, PLL = new_PLL(W, idx_pairs, y, nb_neigh = 0, val = True)
            PLL_tot += torch.nan_to_num(PLL, nan=0.0)
            # weight by the size of the protein
            L_acc.append(acc * torch.sum(~missing))
            var_tot += torch.sum(~missing)
            #print(acc)


        file = open("../Results/" + filename + ".txt", "a")
        try:
            L_acc = torch.stack(L_acc).reshape(1, -1) / var_tot
            acc = str(torch.sum(L_acc).item())
            PLL_tot = PLL_tot.item()
        except:
            acc = str(0)
        file.write("\n  Validation acc " +  acc
            + ", PLL " + str(PLL_tot))
        file.close()
        print("val acc", acc)
        
        optimizer.param_groups[0]['lr'] = lr
        scheduler.step(torch.sum(L_acc).item())
        lr = optimizer.param_groups[0]['lr']
        print("lr ", lr)
        if lr<1e-6:
            break

        if epoch%10==0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "lr": lr},
                "../Results/model/" + filename + "_" + str(epoch),
                )
