import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from os import listdir
from effie.Features import *
import random
import os
from collections import Counter

from effie.utils import *
from effie.ResNet import *
from effie.gMLP import *
from effie.Solver import *
#from .MLP import *
from effie.CbNet_optiR import *
from effie.utils_assemblies import *

import tqdm
import concurrent.futures
import EPLL_utils
import math

torch.set_printoptions(precision=2)
torch.manual_seed(0) #for reproducibility

if torch.cuda.is_available():  
    dev = "cuda:1" 
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
             

#BLOSUM = load_BLOSUM()
nb_aa = 20
# Parameters for gMLP
embed_dim = 256
depth_gMLP = 3#+3
ff_mult = 2

gMLP_output_dim = 32*2*2
# Parameters for ResMLP
hidden = 256#*2
nblocks = 3#+7
block_size = 2
multichain = False
seq_len = 48 #if multichain else 45
transmembrane = False
nb_it = 6

max_neigh = 48
unary = True
#filename = "PLL_1_B_110" ### TO CHANGE ###
#filename = "PLL_2_nrnr_100" ### TO CHANGE ###
filename = "with_unaries" ### TO CHANGE ###
#filename = "data"
#filename = "PLL_1_225"
model = CbNet(embed_dim=embed_dim,
            depth_gMLP=depth_gMLP,
            ff_mult=ff_mult,
            seq_len = seq_len,
            gMLP_output_dim = gMLP_output_dim,
            # Options for ResNet
            hidden=hidden,
            nblocks=nblocks,
            block_size=block_size,
            nb_aa=nb_aa,
            nb_it = nb_it,
            multichain=multichain,
            transmembrane=transmembrane,
            unary=unary)
model.to(device)

thresh = 15 #threshold on pairs to comput costs


#filename = "data" ### TO CHANGE ###
checkpoint = torch.load("../Results/model/"+ filename, map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])

print("Model loaded")
print("Number of trainable  parameters: ", count_parameters(model))


keating = False
if keating:
    test_set = listdir("../features/Keating/myft_test/")
    val_set = test_set
    
elif multichain:
    T_path = '/netapp/tbi/ead1/EAD1_Remaud/Thematic_Groups/Modelo/mdefresn/'
    path = '/gpfsscratch/rech/hro/unk56ix/'
    data_path =  T_path + "pdb_2021aug02/"
    sample_path = "../" + "pdb_2021aug02_sample/"

    #test_assemblies = load_test_assemblies(data_path, num_examples_per_epoch = 200)
    val_assemblies = listdir("../features/multi_val/")
    val_set = val_assemblies[:100]

else:
    cath = CATHDataset(path="../Ingraham2019/chain_set.jsonl",
                                splits_path="../Ingraham2019/chain_set_splits.json")
    val_set = cath.test


#f = open("../features/pdb_exclude_list.txt", 'r')
#L = f.readlines()
#f.close()
#L_tm = eval(L[0].split("=")[1])
L_tm = []
#tm_path = "../features/PDBTM/fasta_multi/"

import subprocess
import os
import time
path = '../LR-BCD/code/'
os.chdir(path)
cmd = './mixing test.wcsp 2 -1 1 -f'

toulbar =True
if toulbar:
    nb_pred_seq = 1
else:
    nb_pred_seq = 50
seq_min, seq_mean = [], []


nb_aa = 20
L_pears = []
L_mean_nsr, L_best_nsr, L_soft_nsr, L_nssr = [], [], [], []
L_nsr_tm = []

d_max, nb5_max, nb10_max = torch.Tensor([259.45, 11, 45]).to(device)
class Protein:
    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, W, y, idx_pairs=None):
        self.nb_var = W.shape[0]
        self.nb_neigh = W.shape[1]
        self.nb_val = int(W.shape[2]**0.5)
        if state == None:
            state = torch.randint(0,self.nb_val, (self.nb_var,))
        self.state = state
        self.y = y
        self.idx_pairs = idx_pairs
        W = W.reshape(self.nb_var,self.nb_var,self.nb_val,self.nb_val)
        self.W=W
        self.Tinit = 1
        self.T = 0.1


    def move(self):
        W = self.W
        var = torch.randint(0,self.nb_var,(1,))
        #unary_costs = W[var,torch.arange(self.nb_neigh),:,self.state[idx_pairs[torch.arange(self.nb_var)]]]
        unary_costs = W[var][0,torch.arange(self.nb_var)[:,None],torch.arange(self.nb_val)[None,:],self.state[torch.arange(self.nb_var)][:,None]]
        unary_costs = unary_costs.sum(axis=0)
        probs = torch.softmax(-unary_costs/self.T, dim=0)
        m = torch.distributions.categorical.Categorical(probs)
        nval = m.sample()
        self.state[var] = nval

    def get_energy(self,state):
        #unary_costs = self.W[torch.arange(self.nb_var)[:,None],torch.arange(self.nb_neigh)[None,:],:,self.state[idx_pairs[torch.arange(self.nb_neigh)][None,:]]
        energy = self.W[torch.arange(self.nb_var)[:,None],torch.arange(self.nb_var)[None,:],self.state[:,None],self.state[None,:]]
        return energy.sum()

    def anneal(self):
        for i in range(10000):
            self.T = self.Tinit/math.log(2+i)
            self.move()
            if i%500==0:
                print(self.get_energy(self.state))
                print((self.y==self.state).sum()/self.nb_var)
        acc = ((self.y==self.state).sum()/self.nb_var)
        return self.state,acc

def predict1(args):
    if args is None:
        return None
    W_full,unary_costs, y, nb_var, name, missing, infos = args
    prot = Protein(None,torch.Tensor(W_full).to("cpu"),torch.Tensor(y).to("cpu"))
    state, acc = prot.anneal()
    return acc

def predict(args):
    #everything should be on cpu 
    if args is None:
        return None
    try:
        W_full,unary_costs, y, nb_var, name, missing, infos = args
        global filename, toulbar
        instance = 'dump/NSR_Cb'+filename+name+'.wcsp'
        sol_file = 'dump/sol_NSR_Cb'+filename+name+'.txt'
        if toulbar :
            cmd = f"./toulbar2 {instance} -w=" + sol_file+" -timer=80"
        else:
            cmd = f"./mixing {instance} 2 -it=3 -k=-2 -nbR={nb_pred_seq} -f=" + sol_file
        Problem = make_CFN(W_full, idx = None, resolution=3, 
                           unary_costs=unary_costs)
        Problem.Dump(instance)

        
        ### Run convex relaxation ###
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                   shell=True, preexec_fn=os.setsid) 
        try:
            p.communicate(timeout = 90)#0 if multichain else 90)
        except subprocess.TimeoutExpired as e:
            print("interrupt ! ")
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            #return None, y, name, nb_var, missing, None


        #.communicate to wait until the file is written before keep going

        file = open(sol_file, 'r')
        L = file.readlines()
        file.close()

        predictions = []
        if toulbar:
            vals = np.zeros((nb_var))
            i = 0
            count_val = 0
            line = L[0]
            for c in line.split(" "):
                vals[i]=int(c)
                i+=1
            predictions.append(vals)
        else:
            for line in L[:-1]:
                line = line.strip().split(' ')
                line = [int(l) for l in line]
                line = np.array(line).reshape(nb_var, nb_aa)
                predictions.append(np.argmax(line, axis = 1)) 

        return predictions, y, name, nb_var, missing, L, infos
    except Exception as e:
        print(e)
        return None
 
class ProteinIterator:
    def __init__(self, val_set):
        self.val_set = val_set
        return
    def __iter__(self):
        return self
    def __next__(self):
        with torch.no_grad():
            model.eval()
            protein = next(self.val_set)
            name = protein if keating else protein["name"].split("_")[0]
            if keating: 
                dico = torch.load("../../features/Keating/myft_test/" + protein)
                crd, seq, chain_idx = dico["coordinates"], dico["int_seq"], dico["chain_idx"]
                missing = dico["missing"]
                tm = None
            
            elif multichain:
                protein = torch.load("../../features/multi_val/" + protein)
                is_tm = protein["name"].split("_")[0] in L_tm
                crd, seq, chain_idx, tm = data_from_protein(protein, is_tm, tm_path=tm_path)

            else:
                crd, seq, chain_idx, tm = torch.as_tensor(protein['coords']), protein['seq'], None, None
        
                # uncomment to run multi model on Ingraham
                #crd, seq = torch.as_tensor(protein['coords']), protein['seq']
                #chain_idx = [1]*len(seq)
            
            crd, seq, chain_idx, tm = remove_N_C_ter(crd, seq, chain_idx, tm)
            if not keating:
                missing = torch.isnan(torch.sum(crd.reshape(-1, 4*3), dim =1))
                seq = torch.as_tensor([letter_to_num[a] for a in seq], dtype=torch.long)
            # If a residue is unknown('X'), it will not be predicted
            missing = torch.logical_or(missing, (seq == 20)).to(device)
            seq[seq==20]=0

            #crd = torch.nan_to_num(crd, 0.0).to(device)
            y = seq.type(torch.LongTensor).to(device)
            #crd = crd[~missing]
            #y = y[~missing]
            nb_var = crd.shape[0]
            #print(nb_var, name)
            
            if (nb_var > 1500 or nb_var < 50): return None # and (protein["name"].split("_")[0] not in L_tm):
        
            try:
                if unary:
                    W, idx_pairs, unary_costs = model(crd, thresh = thresh, max_neigh = max_neigh, chain_idx=chain_idx, tm=tm)
                    unary_costs = unary_costs-unary_costs.min(axis=-1)[0][:,None]
                    unary_costs_n = unary_costs.cpu().detach().numpy()
                else:
                    W, idx_pairs = model(crd.to(device), thresh = thresh, max_neigh = max_neigh, chain_idx=chain_idx, tm=tm)
                    unary_costs = None
                    unary_costs_n = None
            except Exception as e:
                        print(e)
                        print("problem with ", name)
                        return None
             

            #unary_costs*=0
            ########unary_costs[:]= 0
            W = W-W.min(axis=-1)[0][:,:,None]
            #W[:]= 0
            acc, PLL = new_PLL(W, idx_pairs, y, nb_neigh = 36, val = True, unary_costs=unary_costs)
            #acc2, PLL2 = EPLL_utils.new_PLL(W, idx_pairs, y, nb_neigh = 0, val = True, unary_costs=unary_costs)
                

            #val = torch.max(W, dim = -1)[0] - torch.min(W, dim = -1)[0]
            #idx = (val>0.1).squeeze().detach().cpu().numpy()

            W_full = torch.zeros(nb_var, nb_var, 400).to(W.device)
            #W_full[torch.arange(nb_var)[:, None], idx_pairs[None, :]]=W[torch.arange(nb_var)]
            
            y=y.flatten().detach().cpu().numpy()
            infos = (PLL.item(),acc.item()) 
            return (W_full.cpu().detach().numpy(), unary_costs_n,y, nb_var, name, missing.cpu(), infos)

        

executor = concurrent.futures.ProcessPoolExecutor() 
id_prot = 0

    
print(" Collecting results ( be patient )")
def collect(result):
    global NSR, L_pears, L_best_nsr, L_soft_nsr, infos, id_prot
    #result = future.result()
    if result is None:
        return
    predictions, y, name, nb_var, missing, L, info = result

    NSR = []
    NSSR = []
    infos = []
    for i in range(nb_pred_seq):
        NSR.append(np.sum((y-predictions[i] == 0)[~missing.cpu()])/torch.sum(~missing).item())
        #NSR.append(np.sum((y-predictions[i] == 0))/torch.sum(~missing).item())
        #NSSR.append(np.sum((BLOSUM[y,predictions[i]] >= 0)[~missing.cpu()])/torch.sum(~missing).item())
    if not toulbar:
        E = np.array([float(l) for l in L[-1].strip().split(' ')])
        E0 = min(E)
    
    if (name not in L_tm):
        if not toulbar:
            L_pears.append(pearson(E, NSR))
            m = nb_var*(nb_var-1)/2 #normalise with the number of pairs
            softmax = np.exp((E0-E)/m)
            softmax /= np.sum(softmax)
            L_best_nsr.append(NSR[np.argmin(E)]) #NSR with lowest E
            #L_nssr.append(NSSR[np.argmin(E)])
            L_soft_nsr.append(np.sum(np.array(NSR)*softmax)) #weighted avg NSR
            #print(pearson(E, NSR), np.mean(NSR), NSR[np.argmin(E)], np.sum(np.array(NSR)*softmax))
            #print(nb_var, name,NSR[np.argmin(E)])#, NSSR[np.argmin(E)])
        L_mean_nsr.append(np.mean(NSR)) #mean NSR
        infos.append(info)
        
    else:
        L_nsr_tm.append(NSR[np.argmin(E)])


    id_prot +=1
    if id_prot%1==0:
        print(L_mean_nsr[-1])
        print(np.mean(L_mean_nsr), np.mean(L_best_nsr), np.mean(L_soft_nsr))


  
for query in ProteinIterator(iter(tqdm.tqdm(val_set))):
    future = executor.submit(predict, query)
    future.add_done_callback(lambda res : collect(res.result()))
    #predict1(query)
    #collect(predict(query))

          
                
print(filename )
#print("Soluble only")
print(np.mean(np.array(infos), axis=0))
print(np.mean(L_mean_nsr), np.mean(L_best_nsr), np.mean(L_soft_nsr))
#, np.mean(L_best_nsr), np.mean(L_soft_nsr))
#print(np.median(L_mean_nsr), np.median(L_best_nsr), np.median(L_soft_nsr))
#print(np.mean(L_nssr), np.median(L_nssr))
#print("With tm")
#print(np.mean(L_best_nsr+L_nsr_tm), np.median(L_best_nsr+L_nsr_tm))
        
