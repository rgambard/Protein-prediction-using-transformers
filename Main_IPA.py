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
import EPLL_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import profile, record_function, ProfilerActivity

#from CbNet_env import *
from effie.utils import *
from effie.utils_assemblies import *
from effie.Features import *

import effie.IPA as Net
torch.manual_seed(0) #for reproducibility
if torch.cuda.is_available():  
    dev = "cuda:0" 
    print("GPU connected")
else:  
    dev = "cpu"
#dev = "cpu"
device = torch.device(dev)

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20}
                       
letter3to1 = {'CYS' : 'C', 'ASP' : 'D', 'SER' : 'S', 'GLN' : 'Q', 'LYS' : 'K',
             'ILE' : 'I', 'PRO' : 'P', 'THR' : 'T', 'PHE' : 'F', 'ALA' : 'A',
             'GLY' : 'G', 'HIS' : 'H', 'GLU' : 'E', 'LEU' : 'L', 'ARG' : 'R',
             'TRP' : 'W', 'VAL' : 'V', 'ASN' : 'N', 'TYR' : 'Y', 'MET' : 'M'}

filename = 'V5_big1'

lr = 0.0001*5
weight_decay = 0.001
#weight_decay=0
use_amp = False#Automated mixed precision
#with AMP, choose dim layer that are *8

nb_aa = 20
# Parameters for gMLP
embed_dim = 256
depth_gMLP = 3#+3
ff_mult = 2
seq_len = 16
gMLP_output_dim = 32*2*2
# Parameters for ResMLP
hidden = 256#*2
nblocks = 3#+7
block_size = 2
unary = False
multichain = False

reg_term = 0#.0001#/10
#reg_term=0
L1_as_fd = False
adapt_LR = 0 #1/2  #1 to adapt LR with bs, 1/D to adapt with sqrt(bs), 0 to do nothing
thresh = 15
max_neigh = 16
noise_std = 0
perc_mask = 0.3 #for gangster PLL
nb_it = 5 # nb it max

if L1_as_fd:
    reg_term /= 10
lr /= 100**adapt_LR

n_feat = 20

#replace CbNet by CbNet_edge
model = Net.CbNet()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=0.0001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp) #Gradient scaling (for amp)

if unary:
    CE_loss = nn.CrossEntropyLoss()
    alpha = 0

resume_training = False
if resume_training:
    filename = "PLL"  # TO CHANGE ###
    checkpoint = torch.load("../Results/model/" + filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #scaler.load_state_dict(checkpoint["scaler_state_dict"])
#    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch0 = checkpoint["epoch"] + 1
#    lr = checkpoint["lr"]
    print("Resuming training")
    print(lr)
    breakpoint()
    #print(optimizer.param_groups[0]['lr'])
    #optimizer.param_groups[0]['lr'] /= 10

else:
    epoch0 = 1


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
                                    
just_test = False
for epoch in range(epoch0, 500):

    ### Training ###
    PLL_epoch = 0.0
    model.train()

    #reload every few epochs:
    if epoch % 1 == 0 and multichain:
        train_assemblies, val_assemblies = dataset(q), dataset(p)
    train_set = train_assemblies if multichain else cath.train
    random.shuffle(train_set)
    train_set = train_set[:1000]
    if just_test:
        train_set = train_set[:0]
    
    train_metrics = []
    train_metrics_by_i = [[] for i in range(nb_it)]
    for protein in tqdm.tqdm(train_set):
        optimizer.zero_grad()  # arg to go faster

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

        crd = crd[~missing]
        y = y[~missing]
        nb_var = crd.shape[0]

        
        if noise_std > 0:
            noise = torch.randn(*crd.shape).to(device)
            crd += noise*noise_std
            noise = noise.reshape(-1, 12)

        if nb_var <= 3000 and nb_var > 50:

            #print(nb_var)
            optimizer.param_groups[0]['lr'] = lr*(nb_var**adapt_LR) #adapt LR to the size of the protein
            
            #with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = use_amp):
            #Mixed precision training (some operation will ve casted to float 16 (instead of default 32)
            #without reducing the accuracy)
            #with torch.amp.autocast(device_type="cuda", dtype=torch.float16):       
            hint_perc = 0.5*torch.rand(1).item()
            t_probs = torch.nn.functional.one_hot(y, num_classes = 20)
            noise = torch.rand((nb_var), device = device)
            mask = torch.where(noise>hint_perc)
            nb_hints = (noise<=hint_perc).sum().item()
            probs = t_probs.clone()
            probs[mask] = 0

            loss = 0
            perc_add = 0.2
            for i in range(2):
                probs,confi = model(crd, probs.detach())
                
                #acc, PLL = EPLL_utils.new_PLL(W, idx_pairs, y, nb_neigh = int(perc_mask*min(max_neigh, nb_var)), unary_costs=unary_costs,nb_rand_masks = 100,nb_rand_tuples=400, mask_width=2, val=True, missing = None) 
                #acc1, PLL1 = EPLL_utils.new_PLL(W, idx_pairs, y, nb_neigh = int(perc_mask*min(max_neigh, nb_var)), unary_costs=unary_costs,nb_rand_tuples=20, mask_width=1, val=True, missing = None) 
                #rand_var = int(0.2*nb_var)
                #variables = torch.randint(0,nb_var,(rand_var,), device = device)
                lprobs = torch.log(probs[0])
                #probs = torch.nn.functional.softmax(-unaries, dim = 1)
                acc = (lprobs.argmax(dim=-1)==y).sum()/nb_var
                acc_mean = (probs[0,torch.arange(nb_var, device = device),y]).detach().sum()/nb_var
                PLL = lprobs[torch.arange(nb_var, device = device),y]
                PLL = -PLL.sum()
                
                #L1 = torch.linalg.vector_norm(unaries*reg_term, ord=1)
                
                loss = PLL+loss
                PLL_epoch += PLL.item()/nb_var

                nprobs = probs[0].detach().clone()
                prop_y = nprobs.argmax(dim=-1)
                best_probs = nprobs[torch.arange(nb_var, device = device),prop_y]
                best_probs = best_probs.sort()[1][-int((perc_add*(i+1)*nb_var)+nb_hints):]
                nprobs[:] = 0
                nprobs[best_probs, prop_y[best_probs]] = 1

                found = nprobs.sum().item()
                errors = torch.logical_and(nprobs==1,t_probs ==0).sum().item()

                train_metrics.append([PLL.item()/nb_var, acc.item(), acc_mean.item()])
                train_metrics_by_i[i].append([PLL.item()/nb_var, acc.item(), found/nb_var, errors/nb_var])
                probs = None
                probs = nprobs

            #print(PLL)
                
            # Exits autocast before backward().
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            # source: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            #scaler.scale(loss).backward()
            #scaler.step(optimizer) #skipped if gradients are None
            #scaler.update()
            loss.backward()
            optimizer.step()
            
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)   
            #scaler.update()

            
    file = open("../Results/" + filename + ".txt", "a")
    file.write("\n Epoch " + str(epoch) + " - PLL loss:" + str(PLL_epoch))
    file.close()
    
    if False:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],record_shapes=True) as prof:
            unaries, i = model(crd, p_it = p_it, thresh = thresh, max_neigh = max_neigh, chain_idx=chain_idx)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    
    thresh = 0.5
    ### Validation ###
    with torch.no_grad():

        model.eval()
        L_acc, PLL_tot, L_tot = [], 0, 0
        var_tot = 0

        val_set = val_assemblies if multichain else cath.val
        metrics = []
        metrics_by_i = [[] for i in range(nb_it)]
        for protein in tqdm.tqdm(val_set):

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

            crd = crd[~missing]
            y = y[~missing]
            nb_var = crd.shape[0]
            
            if nb_var > 3000 or nb_var < 50:
                continue
            #with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = use_amp):
            probs = torch.ones((nb_var, nb_aa), device = device)*0
            t_probs = torch.nn.functional.one_hot(y, num_classes = 20)
            loss = 0
            perc_add = 0.15
            for i in range(nb_it):
                probs,confi = model(crd, probs.detach())

                #acc, PLL = EPLL_utils.new_PLL(W, idx_pairs, y, nb_neigh = int(perc_mask*min(max_neigh, nb_var)), unary_costs=unary_costs,nb_rand_masks = 100,nb_rand_tuples=400, mask_width=2, val=True, missing = None) 
                #acc1, PLL1 = EPLL_utils.new_PLL(W, idx_pairs, y, nb_neigh = int(perc_mask*min(max_neigh, nb_var)), unary_costs=unary_costs,nb_rand_tuples=20, mask_width=1, val=True, missing = None) 
                #rand_var = int(0.5*nb_var)
                #variables = torch.randint(0,20,(rand_var,), device = device)
                lprobs = torch.log(probs[0])
                acc = (lprobs.argmax(dim=-1)==y).sum()/nb_var
                PLL = lprobs[torch.arange(nb_var, device = device),y]
                PLL = -PLL.sum()
                PLL_tot += torch.nan_to_num(PLL, nan=0.0)/nb_var
                nprobs = probs[0].detach()
                prop_y = nprobs.argmax(dim=-1)
                best_probs = nprobs[torch.arange(nb_var, device = device),prop_y]
                best_probs = best_probs.sort()[1][-int(perc_add*nb_var*(i+1)):]
                nprobs[:] = 0
                nprobs[best_probs, prop_y[best_probs]] = 1
                found = nprobs.sum().item()
                errors = torch.logical_and(nprobs==1,t_probs ==0).sum().item()

                metrics.append([PLL.item()/nb_var, acc.item()])
                metrics_by_i[i].append([PLL.item()/nb_var, acc.item(), found/nb_var, errors/nb_var])
                # weight by the size of the protein
                L_acc.append(acc * torch.sum(~missing))
                var_tot += torch.sum(~missing)
                probs = nprobs


        file = open("../Results/" + filename + ".txt", "a")
        L_acc = torch.stack(L_acc).reshape(1, -1) / var_tot
        acc = str(torch.sum(L_acc).item())
        PLL_tot = PLL_tot.item()
        file.write("\n  Validation acc " +  acc
            + ", PLL " + str(PLL_tot) + " lr "+str(lr))
        file.write("\n metrics"+ str(torch.mean(torch.Tensor(metrics),axis=0)))
        file.write("\n train metrics" +str(torch.mean(torch.Tensor(train_metrics),axis=0)))
        file.close()
        
        print("val acc", acc, "lr", lr, " PLL_tot: ",PLL_tot)
        print("metrics", torch.mean(torch.Tensor(metrics),axis=0))
        print("metricsi", [(torch.mean(torch.Tensor(metrics_by_i[i]),axis=0), len(metrics_by_i[i])) for i in range(nb_it)])
        print("train metrics" ,torch.mean(torch.Tensor(train_metrics),axis=0))
        print("trainmetricsi", [(torch.mean(torch.Tensor(train_metrics_by_i[i]),axis=0), len(train_metrics_by_i[i])) for i in range(nb_it)])
        
        optimizer.param_groups[0]['lr'] = lr
        scheduler.step(PLL_tot)
        lr = optimizer.param_groups[0]['lr']
        if lr<1e-5:
            print('lr too small')
            break
        if not just_test and epoch%10==0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "lr": lr},
                "../Results/model/" + filename + "_" + str(epoch),
                )
