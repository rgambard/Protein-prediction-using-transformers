import csv
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from .Transformer import *
from .invariant_point_attention import *
from .ResNet import *

class CbNet(nn.Module):
    def __init__(
        self,
        seq_len = 48,
        embed_dim = 256,
        depth_trans = 1,
        pairwise_dim = 32,
        hidden = 256,
        nblocks = 1,
        block_size = 2,
        nb_aa = 20,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.nb_aa = nb_aa
        self.input_dim = 4*3+nb_aa
        self.pairwise_dim = pairwise_dim

        self.input_resnet = ResMLP(self.embed_dim,
                        self.input_dim,
                        hidden_size=2*hidden,
                        nblocks=2*nblocks,
                        block_size=block_size,
                        BatchNorm=False)

        self.pairwise_resnet = ResMLP(self.pairwise_dim,
                        2*4*3+2*nb_aa,
                        hidden_size=2*hidden,
                        nblocks=2*nblocks,
                        block_size=block_size,
                        BatchNorm=False)

        self.trans= nn.ModuleList([])
        for _ in range(depth_trans):
            self.trans.append(
                IPABlock(dim = self.embed_dim, heads=8, require_pairwise_repr=True, pairwise_repr_dim = self.pairwise_dim)
            )


        self.pairwise_linear_it = torch.nn.Linear(self.embed_dim, 32, bias=False)
        self.pairwise_self_linear_it = torch.nn.Linear(self.embed_dim, 32, bias=False)
        self.pairwise_resnet_it = ResMLP(self.pairwise_dim,
                        2*4*3+2*32+self.pairwise_dim,
                        hidden_size=hidden,
                        nblocks=nblocks,
                        block_size=block_size,
                        BatchNorm=False)

        self.trans_it= nn.ModuleList([])
        for _ in range(depth_trans):
            self.trans_it.append(
                IPABlock(dim = self.embed_dim, heads=16, require_pairwise_repr=True, pairwise_repr_dim = self.pairwise_dim)
            )


        self.output_resnet = ResMLP(
                        self.nb_aa+1,#confidence
                        self.embed_dim,
                        hidden_size=hidden,
                        nblocks=nblocks,
                        block_size=block_size,
                        BatchNorm=False)

        self.input_resnet.apply(weights_init)
        self.output_resnet.apply(weights_init)

    
    #@profile
    def forward(self, crd, probs, nb_it = 0):
        nb_var = crd.shape[0]
        device = crd.device
        coords = (crd[:,:]-crd[:,1][:,None]).reshape(1,nb_var,4*3)
        #translations = (crd).reshape(1,nb_var,3)
        translations = (crd[:,1]).reshape(1,nb_var,3)
        basis1 =  crd[:,2]-crd[:,1]
        basis1 = basis1/torch.linalg.vector_norm(basis1, dim=1)[:,None]
        basis2 = crd[:,0]-crd[:,1]
        basis2 = torch.cross(basis1,basis2,dim=1)
        basis2 = basis2/torch.linalg.vector_norm(basis2, dim=1)[:,None]
        basis3 = torch.cross(basis1,basis2,dim=1)
        rotations = torch.cat((basis1[:,None,:],basis2[:,None,:],basis3[:,None,:]),dim=1)
        crd_trans = crd - crd[:,1][:,None]
        crd_rot = torch.einsum("bji, bni -> bnj", rotations,crd_trans)
        #rotations = repeat(torch.eye(3, device = device), 'r1 r2 -> b n r1 r2', b = 1, n = nb_var)
        basic_coords = crd_rot.reshape((1,-1,4*3))
        pairwise_dist_ca =  torch.sqrt(torch.square(crd[:,1][:,None]-crd[:,1][None,:]).sum(axis=-1))
        idx_pairs = pairwise_dist_ca.argsort(dim=-1)[:,:self.seq_len]
        coord_pairwise = crd[idx_pairs]-crd[:,1][:,None,None]
        coord_pairwise_rot = torch.einsum("bji, bkni -> bknj", rotations,coord_pairwise)
        coord_pairwise_rot_f = coord_pairwise_rot.reshape(nb_var,self.seq_len,12)
        coord_pairwise_rot_fa = torch.cat((coord_pairwise_rot_f,crd_rot[:,None].expand(nb_var,self.seq_len,4,3).clone().reshape(nb_var, self.seq_len,12)),dim=-1)
        rotations = rotations.unsqueeze(0)


        # initial processing
        inp = torch.cat((basic_coords,probs.unsqueeze(0)), dim=-1)
        ft = self.input_resnet(inp)
        proj_neigh = probs[idx_pairs]
        proj_self = probs[:,None].expand(nb_var,self.seq_len,self.nb_aa)
        inp_pairwise = torch.cat((proj_self, proj_neigh,coord_pairwise_rot_fa), dim=-1)
        pairwise_neigh = self.pairwise_resnet(inp_pairwise)

        pairwise_repr = torch.zeros(1, nb_var, nb_var, self.pairwise_dim, device = device)
        pairwise_repr[0,torch.arange(nb_var,device = device)[:,None],idx_pairs] = pairwise_neigh
        
        for layer in self.trans:
            ft = layer(ft, pairwise_repr=pairwise_repr,rotations = rotations, translations = translations)

        for it in range(nb_it):
            nft = ft
            proj_neigh = self.pairwise_linear_it(nft[0,idx_pairs])
            proj_self = self.pairwise_self_linear_it(nft[0,:])[:,None].expand(nb_var,self.seq_len,self.pairwise_dim).clone()
            inp_pairwise = torch.cat((proj_self,proj_neigh,coord_pairwise_rot_fa,pairwise_neigh), dim=-1)
            pairwise_neigh = pairwise_neigh + self.pairwise_resnet_it(inp_pairwise)

            pairwise_repr = torch.zeros(1, nb_var, nb_var, self.pairwise_dim, device = device)
            pairwise_repr[0,torch.arange(nb_var,device = device)[:,None],idx_pairs] = pairwise_neigh
            
            for layer in self.trans_it:
                nft = layer(nft, pairwise_repr=pairwise_repr,rotations = rotations, translations = translations)
            ft = nft#+ft



        out = self.output_resnet(ft)
        #confidence = self.confidence_resnet(ft)
        probs = torch.softmax(out[:,:,:self.nb_aa], dim=-1)
        uniform = torch.ones_like(probs)*1/self.nb_aa
        confi = torch.sigmoid(out[:,:,self.nb_aa])
        probs = confi[:,:,None]*probs+(1-confi)[:,:,None]*uniform

        return probs, confi
