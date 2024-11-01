# -*- coding: utf-8 -*-
"""

"""
import torch
import torch.nn.functional as F
import numpy as np
from os import listdir
import sys

#sys.path.insert(1, "../../Sudoku_git/nn_tb2/")
from .CFN import CFN
import subprocess
import os

def add_hints(problem, nb_var, hints, solution=None, margin=0, top=999999):

    for i in range(nb_var):
        sol = int(solution[i]) if solution is not None else 0
        hint = int(hints[i])
        costs = np.zeros(20)

        if sol:
            costs[sol] = margin  # hyperparameter to tune

        if hint < 20:
            extra_costs = 2 * top * np.ones(20)
            extra_costs[hint] = margin
            costs += extra_costs

        problem.AddFunction([i], costs)


def make_CFN(W, idx = None, var_names = None, domains=None, unary_costs = None, top=999999, resolution=3, backtrack = 9999999999):
    
    """
    Create a CFN object described by the W function.
    Input: - the matrix (numpy array W) of size (nb_var, nb_var, nb_aa)
           - a Boolean matrix idx (shape nb_var, nb_var) whose value is True for the constraint to consider 
           (default is None: all constraints are written)
           - int top (default 999999)
           - int resolution (default 3)
           - int backtrack (default 20 000)
    """

    Problem = CFN(top, resolution, vac=True, backtrack = backtrack)
    nb_var = W.shape[1]
    nb_aa = int(W.shape[-1] ** 0.5)
    if idx is None: #if idx is not None:
        idx = np.ones((nb_var, nb_var))
        
    #Defining variables & domains
    for i in range(nb_var):
        var_name = ("x" + str(i+1) if var_names is None else var_names[i]) 
        Problem.AddVariable(var_name, range(0, nb_aa) if domains is None else domains)
        
    #Defining cost functions
    for i in range(nb_var):
    
        # unary costs
        if unary_costs is not None:
            Problem.AddFunction([i], unary_costs[i])
        else:
            Problem.AddFunction([i], np.diag(W[i, i].reshape(nb_aa, nb_aa)) * idx[i, i])

        for j in range(i + 1, nb_var):
            #binary costs
            Problem.AddFunction([i, j], W[i, j] * idx[i, j])
                
    return Problem


def symmetrize_CFN(W, nb_chain):
    
    W = W.squeeze()
    if W.shape[0]%nb_chain != 0:
        print("Cautious ! The number of chains does not match the size of matrix W.")
    
    nb_aa = int(W.shape[-1]**0.5)
    nb_var = W.shape[0]//nb_chain
    W_unary = torch.zeros(nb_var, nb_var, nb_aa**2)
    for i in range(nb_chain):
        for j in range(i+1, nb_chain):
            W_unary += W[i*nb_var:(i+1)*nb_var, j*nb_var:(j+1)*nb_var]
        
    W_unary += W_unary.transpose(0, 1).view(nb_var, nb_var, nb_aa, nb_aa).transpose(
        2, 3).reshape( nb_var, nb_var, -1)

    for i in range(nb_chain):
        W_same_chain = W[i*nb_var:(i+1)*nb_var, i*nb_var:(i+1)*nb_var]
        #unary terms inside the chain (ie keep only diagonal)
        for j in range(nb_var):
            W_unary[j, j] *= torch.eye(nb_aa, nb_aa).reshape(-1)
        #removing tri lower from matrices of intercation inside same chain
        for k, l in torch.tril_indices(nb_var, nb_var, -1).T:
            W_same_chain[k,l] = 0    

        W_unary += W_same_chain
        
    return W_unary

                       
def LR_BCD(W, y, missing = None, hint=None, nb_pred_seq = 20, filename = 'NSR_Cb'):
    
    try:
        path = '../LR-BCD-main/code/'
        os.chdir(path)
        cmd = './mixing test.wcsp 2 -1 1 -f'
    except:
        pass
    
    sol_file = 'sol_' + filename + '.txt'
    instance = filename + '.wcsp'
    cmd = f"./mixing {instance} 2 -it=3 -k=-2 -nbR={nb_pred_seq} -f=" + sol_file
    nb_aa = 20
    
    try:
        y=y.flatten().detach().cpu().numpy()
        W=W.detach().cpu().numpy()
    except:
        pass
    nb_var = len(y)
    W = W.reshape(nb_var,nb_var,-1)
    Problem = make_CFN(W, idx = None, resolution=3)
    if hint is not None:
        add_hints(Problem, nb_var, hint, solution = None)
    Problem.Dump(instance)


    ### Run convex relaxation ###
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
               shell=True, preexec_fn=os.setsid) 
    p.communicate(timeout = 900)
    if p.poll() is None: # p.subprocess is alive
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    #.communicate to wait until the file is written before keep going

    file = open(sol_file, 'r')
    L = file.readlines()
    file.close()

    predictions = []
    for line in L[:-1]:
        line = line.strip().split(' ')
        line = [int(l) for l in line]
        line = np.array(line).reshape(nb_var, nb_aa)
        predictions.append(np.argmax(line, axis = 1)) 

    NSR = []
    for i in range(nb_pred_seq):
        if missing is not None:
            NSR.append(np.sum((y-predictions[i] == 0)[~missing.cpu()])/torch.sum(~missing).item())
        else:
            NSR.append(np.sum((y-predictions[i] == 0))/nb_var)
    E = np.array([float(l) for l in L[-1].strip().split(' ')])
    
    os.chdir('../../nn_tb2')
    
    return NSR[np.argmin(E)], predictions[np.argmin(E)]
    
    
