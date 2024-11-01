
import torch
import torch.nn as nn
import numpy as np
from .gMLP import *


def weights_init(m):
    """
    For initializing weights of linear layers (bias are put to 0).
    """

    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        # torch.nn.init.uniform_(m.weight, a=0.09*gain, b=0.11*gain)
        # torch.nn.init.constant_(m.weight, 0)
        # torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)


class MLP(nn.Module):

    """
    Define a MLP with layer norm & ReLU 
    Init: size of the input (int)
          size of the output (int)
          list of the hidden layer dimensions (there will be as many hidden layers as elements in the list)
    """

    def __init__(self, input_size, output_size, hidden_sizes):
        super(MLP, self).__init__()

        self.MLP = torch.nn.Sequential()
        layer_sizes = [input_size]+hidden_sizes
        
        for k in range (1, len(layer_sizes)):
            self.MLP.add_module("Linear layer " + str(k), nn.Linear(layer_sizes[k-1], layer_sizes[k])),
            self.MLP.add_module("LN " + str(k), nn.LayerNorm(layer_sizes[k])),
            self.MLP.add_module("ReLU " + str(k), nn.ReLU())
        self.MLP.add_module("Output layer", nn.Linear(layer_sizes[-1], output_size))
            
        self.MLP.apply(weights_init)

    def forward(self, x):

        return self.MLP(x)
        
        
class MLP_3layers(nn.Module):

    """
    MLP with 3 hidden layer, dropout and batch norm (uncomment).
    Init: size of the output output size (int)
          size of the input (int)
          size of the hidden layers (suggestion: 128 or 256)
    """

    def __init__(self, output_size, input_size, hidden_size):
        super(MLP_3layers, self).__init__()

        dropout_rate = 0.1
        self.MLP = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            # nn.Dropout(dropout_rate), #remove dropout to simplify (no pb of generalization, large dataset)
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            # nn.ReLU() #added after traing 2 epochs with pred10 var
            # nn.Dropout(dropout_rate) #NO dropout before BN [see Li et al. 2018]!
        )
        self.MLP.apply(weights_init)

    def forward(self, x):

        return self.MLP(x)



class ResBlock(nn.Module):

    """
    Residual block of 2 hidden layer for resMLP
    Init: size of the input (the output layer as the same dimension for the sum)
          size of the 1st hidden layer
          size of residual blocks (int, min and default: 2)
    """

    # In ResNet v2: BN, relu, weight, BN, relu, weight, sum
    # no dropout and kaiming init

    def __init__(self, input_size, hidden_size, block_size=2, BatchNorm=True):
        super(ResBlock, self).__init__()

        activation = torch.nn.GELU() 
        ### Input block ###
        self.block = torch.nn.Sequential()
        self.block.add_module("In_BN", nn.BatchNorm1d(num_features=hidden_size, 
                                                          track_running_stats=False)
                              if BatchNorm else nn.LayerNorm(hidden_size))
        #self.block.add_module("In_BN", nn.LayerNorm(input_size))
        self.block.add_module("relu_in", activation)
        self.block.add_module("In_layer", nn.Linear(input_size, hidden_size))

        ### Intermediate blocks (optionnal) ###
        for k in range(block_size - 2):
            self.block.add_module("BN_" + str(k), nn.BatchNorm1d(num_features=hidden_size, 
                                                          track_running_stats=False)
                              if BatchNorm else nn.LayerNorm(hidden_size))
            #self.block.add_module("BN_" + str(k), nn.LayerNorm(hidden_size))
            self.block.add_module("relu_" + str(k), activation)
            self.block.add_module("layer" + str(k), nn.Linear(hidden_size, hidden_size))

        ### Output block ###
        self.block.add_module("Out_BN", nn.BatchNorm1d(num_features=hidden_size, 
                                                          track_running_stats=False)
                              if BatchNorm else nn.LayerNorm(hidden_size))
        #self.block.add_module("Out_BN", nn.LayerNorm(hidden_size))
        self.block.add_module("relu_out", activation)
        self.block.add_module("Out_layer", nn.Linear(hidden_size, input_size))

    def forward(self, x):

        x_out = self.block(x)
        x = x_out + x

        return x


class ResMLP(nn.Module):

    """
    ResMLP with 5 residual blocks of 2 hidden layers.
    Init: size of the output output size (int)
          size of the input (int)
          size of the hidden layers
    """

    def __init__(self, output_size, input_size, hidden_size, nblocks=2, block_size=2, BatchNorm=True):
        super(ResMLP, self).__init__()

        self.ResNet = torch.nn.Sequential()
        self.ResNet.add_module("In_layer", nn.Linear(input_size, hidden_size))
        # self.ResNet.add_module("relu_1", torch.nn.ReLU())
        for k in range(nblocks):
            self.ResNet.add_module("ResBlock" + str(k), ResBlock(hidden_size, hidden_size, block_size, 
                                                                 BatchNorm=BatchNorm))
        self.ResNet.add_module("Final_BN", nn.BatchNorm1d(num_features=hidden_size, 
                                                          track_running_stats=False)
                              if BatchNorm else nn.LayerNorm(hidden_size))
        #self.ResNet.add_module("Final_BN", nn.LayerNorm(hidden_size))
        self.ResNet.add_module("relu_n", torch.nn.GELU())
        self.ResNet.add_module("Out_layer", nn.Linear(hidden_size, output_size))

    def forward(self, x):

        x = self.ResNet(x)

        return x


class ResNet(nn.Module):

    """
    Network composed of embedding + MLP
    Init: grid_size (int)
          hiddensize (int): number of neurons in hidden layer (suggestion: 128 or 256)
          resNet (bool). If False (default), use a regular MLP. Else, use a ResMLP
          nblocks (int): number of residual blocks. Default is 2
    """

    def __init__(self, input_size, nb_aa, hidden_size=128, resnet=True, nblocks=2, block_size=2,
                 BatchNorm=True):
        super(ResNet, self).__init__()

        # dropout_rate = 0.1
        self.nb_aa = nb_aa

        # ResMLP
        if resnet:
            self.MLP = ResMLP(self.nb_aa ** 2, input_size, hidden_size, nblocks, block_size, BatchNorm)

        # else, regular MLP
        else:
            self.MLP = MLP(self.nb_aa ** 2, input_size, [hidden_size])

        self.MLP.apply(weights_init)

    def forward(self, x, device, thresh=None, d=None):

        bs = 1
        nb_pairs, _ = x.shape
        nb_var = int(nb_pairs ** 0.5)  # len of the protein
        x = x.reshape(nb_pairs * bs, -1)  # bs*nb_pairs, nb_ft

        W = torch.zeros((nb_pairs, self.nb_aa ** 2), device=device, dtype=x.dtype)
        idx = torch.triu(torch.ones((nb_var, nb_var)), 1)
        idx = torch.flatten(idx.type(torch.BoolTensor)).to(device)

        if isinstance(thresh, int):
            idx *= torch.all((d < thresh).view(-1, 1), axis=1)  # line where d<thresh

        idx *= torch.all((d>0).view(-1, 1), axis = 1) #to not take missing residues into account

        pred = self.MLP(x[idx])
        W[idx] = pred
        W = W.reshape(bs, nb_var, nb_var, -1)
        W += W.transpose(1, 2).view(bs, nb_var, nb_var, self.nb_aa, self.nb_aa).transpose(3, 4).reshape(bs, nb_var, nb_var, -1)

        return W


def _positional_embeddings(d, device, num_embeddings=16, period_range=[2, 1000]):  # default value of GVP
    # Adapted from https://github.com/jingraham/neurips19-graph-protein-design

    frequency = torch.exp(torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device) * -(np.log(10000.0) / num_embeddings))
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E
    
def grbf_encod(d, device, dmax = 20, nb_kernel = 16):
    
    """distance encoding with Gaussian rdf"""
    dmin = 0
    dd = d.reshape(-1, 1).expand(-1, 16)
    dd = dd - torch.arange(dmin, dmax * (1 + 1 / (nb_kernel - 1)), (dmax - dmin) / (nb_kernel - 1)).reshape(1, -1).to(device)
    dd = torch.exp(-torch.pow(dd, 2))
    
    return dd


class Net(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        depth_gMLP=3,
        ff_mult=2,
        seq_len=45,
        gMLP_output_dim = 32,
        hidden=256,  # for resnet
        nblocks=10,
        block_size=2,
        nb_aa=20,
    ):

        super().__init__()

        self.g_rdf = True  # whether to use Gaussian rbf on distance
        self.nb_kernel = 16
        # ft used to encode info between pair of neighbours
        self.n_ft = 8 + self.g_rdf * (self.nb_kernel - 1)
        self.keep_extra_ft = True  # contact nb & dihedral
        self.central_residue = False
        self.embed_dim = embed_dim
        self.in_layer_per_ft = False #if True, 1 different input layer per type of ft
        self.gMLP_input_dim = ((2 + self.central_residue) *self.embed_dim) if self.in_layer_per_ft else self.embed_dim
        self.gMLP_output_dim = gMLP_output_dim
        self.seq_len = seq_len
        self.nb_aa = nb_aa + 1*self.central_residue #+1 for masked residue
        self.num_pos_embedding = 16
        self.grbf_contact_nb = True
        self.input_size = (
            2 * self.gMLP_output_dim
            + 2 * self.n_ft - 1 
            + 2 * (2*(self.nb_kernel if self.grbf_contact_nb else 1) + 4) * self.keep_extra_ft 
            + self.num_pos_embedding 
            - self.g_rdf * (self.nb_kernel - 1))
        self.BatchNorm = False #whether to take BatchNorm or LayerNorm in ResNet

        
        if self.in_layer_per_ft:
            self.in_layer_crd = nn.Linear(self.n_ft, self.embed_dim)
            self.in_layer_seq = nn.Linear(self.num_pos_embedding, self.embed_dim, bias = False)
            if self.central_residue:     
                self.in_layer_bb = nn.Linear(20, self.embed_dim)
             
        else:
            self.input_layer = nn.Linear(self.n_ft + 20 * self.central_residue + self.num_pos_embedding, self.embed_dim)

        self.gMLP = gMLP(
            output_dim = self.gMLP_output_dim,
            input_dim = self.gMLP_input_dim,  # input dim
            depth=depth_gMLP,
            ff_mult=ff_mult,  # the hidden size in GMLP block is ff_mult*output_dim
            seq_len=self.seq_len,  # nb10_max is 45
            circulant_matrix=True,
        )

        self.resnet = ResNet(self.input_size, self.nb_aa, hidden, nblocks=nblocks, block_size=block_size,
                            BatchNorm=self.BatchNorm) 
        
        self.MLP = MLP_3layers(self.nb_aa - 1*self.central_residue, self.gMLP_output_dim * (1 + (1+self.central_residue)*self.in_layer_per_ft) + 20, hidden) #+20 for dihedral angles
        self.softmax = torch.nn.Softmax(dim = 1)
        

    def forward(self, x, device, d, thresh_nb=10, thresh=None):

        nb_var = int(x.shape[0] ** 0.5)
        #Quaternion convention: q and - q represent the same rotation
        #we restrict to a half sphere (so that first component>0)
        changing_quat_sign = ((x[:,0]>0)*2-1).reshape(-1, 1)
        x[:, :4] *= changing_quat_sign
        
        if self.g_rdf:  # distance encoding with Gaussian rdf
            dd = grbf_encod(d, device, dmax = 20, nb_kernel = self.nb_kernel)
            x = torch.cat((x[:, :7], dd, x[:, 8:]), dim=1)

        # positionnal embedding to encode distance in sequence
        tsr_i = torch.arange(nb_var, device=device).expand(nb_var, nb_var)
        pos = _positional_embeddings(tsr_i - tsr_i.T, device, self.num_pos_embedding)
        x = x.reshape(nb_var, nb_var, -1)
        
        dihedrals = x[0, :, -4:]
        dihedrals[0] = x[0, 1][12 : 12 + 4]
        # For a residue i, gives the dihedral angles i-2, i-1, i, i+1, i+2
        dihedrals = torch.cat(
            (
                torch.cat((torch.zeros((1, dihedrals.shape[-1]), device=device), dihedrals[:-1])),
                torch.cat((torch.zeros((2, dihedrals.shape[-1]), device=device), dihedrals[:-2])),
                dihedrals,
                torch.cat((dihedrals[1:], torch.zeros((1, dihedrals.shape[-1]), device=device))),
                torch.cat((dihedrals[2:], torch.zeros((2, dihedrals.shape[-1]), device=device))),
            ),
            dim=1,
        )

        if self.central_residue:

            central_ft = torch.zeros((nb_var, self.n_ft + self.num_pos_embedding), device=device)
            central_ft[:, 0] = torch.ones(nb_var, device=device)
            central_ft = torch.cat((central_ft, dihedrals), dim=1)

        idx = torch.all((d < thresh_nb).view(-1, 1), axis=1) * torch.all((d > 0).view(-1, 1), axis=1)
        idx = idx.reshape(nb_var, nb_var)

        x_nn = torch.stack(
            [
                padding(
                    torch.cat((x[i, :, : self.n_ft], pos[i]), dim=1)[idx[i]],
                    self.seq_len,
                    device,
                    central_ft[i] if self.central_residue else None,
                    dist=d.reshape(nb_var, nb_var)[i, idx[i]],
                )
                for i in range(nb_var)
            ]
        )
        x_nn = torch.squeeze(x_nn)

        # Vectorized version (2 times slower)
        # _, ids = torch.sort(x, dim = 1)
        # x_nn = x[torch.arange(nb_var)[:, None], ids[:,:,-1]][:, :seq_len]
        # cutoff = x_nn[:, :,-1] < thresh_nb
        # x_nn = (cutoff.unsqueeze(2).expand(nb_var,seq_len, 8)*x_nn)

        if self.in_layer_per_ft:  
            if self.central_residue:
                x_nn = torch.cat((self.in_layer_crd(x_nn[:,:,:self.n_ft]),
                                  self.in_layer_seq(x_nn[:,:,self.n_ft:self.n_ft+self.num_pos_embedding]),
                                  self.in_layer_bb(x_nn[:,:,self.n_ft+self.num_pos_embedding:]))
                                 , dim = -1)
                
            else:
                x_nn = torch.cat((self.in_layer_crd(x_nn[:,:,:self.n_ft]),
                               self.in_layer_seq(x_nn[:,:,self.n_ft:self.n_ft+self.num_pos_embedding]))
                               , dim = -1)
                               
        else:
            x_nn = self.input_layer(x_nn)
        
        x_nn = self.gMLP(x_nn)

        if self.central_residue:
            x_nn = x_nn[:, 0, :]
        else:
            # for global ft: avg or max pooling ?
            x_nn = torch.mean(x_nn, dim=1)
            
        
        ### Predicting residue identity ###
        #add dihedral to ft vector describing the environment 
        #pba = self.MLP(torch.cat((x_nn, dihedrals), dim = 1))
        #pba = self.softmax(pba)
        
        ### Predicting binary matrices ###
        t = torch.triu_indices(nb_var, nb_var, 1)
        l, c = t
        out = torch.zeros(nb_var, nb_var, 2 * self.gMLP_output_dim, device=device, dtype=x_nn.dtype)
        # out contains [ft_i, ft_j] for all i<j
        out[l, c] = torch.swapaxes(x_nn[t], 0, 1).reshape(-1, 2 * self.gMLP_output_dim)
        #grbf encodding of contact number
        if self.grbf_contact_nb:
            nb5_max, nb10_max = 11, 45 #computed on whole trainset

            out = torch.cat((out, 
                             x[:, :, :7], #crd i/j
                             torch.transpose(x, 0, 1)[:, :, :7], #crd j/i
                             x[:, :, 7:7+(self.nb_kernel if self.g_rdf else 1)], #distances
                             grbf_encod(x[:, :, 6+ (self.nb_kernel if self.g_rdf else 1)+1].flatten(), device, dmax = nb5_max, nb_kernel=self.nb_kernel).reshape(nb_var, nb_var, -1), #contact nb
                             grbf_encod(x[:, :, 6+ (self.nb_kernel if self.g_rdf else 1)+2].flatten(), device, dmax = nb10_max, nb_kernel=self.nb_kernel).reshape(nb_var, nb_var, -1) ,
                             grbf_encod(x[:, :, 6+ (self.nb_kernel if self.g_rdf else 1)+3].flatten(), device, dmax = nb5_max, nb_kernel=self.nb_kernel).reshape(nb_var, nb_var, -1) ,
                             grbf_encod(x[:, :, 6+ (self.nb_kernel if self.g_rdf else 1)+4].flatten(), device, dmax = nb10_max, nb_kernel=self.nb_kernel).reshape(nb_var, nb_var, -1) ,

                             x[:, :, -8:]),
                              dim=-1)
            
        else:
            out = torch.cat((out, x[:, :, :7], torch.transpose(x, 0, 1)[:, :, : x.shape[-1] if self.keep_extra_ft else self.n_ft]), dim=-1) #order i,j reversed between ft ?
            # cut x at 7 ft to avoid putting the distance twice

        # add positionnal embedding
        out = torch.cat((out, pos), dim=-1)
        out = out.reshape(nb_var * nb_var, -1)
        out = out.type(x_nn.dtype)
        out = self.resnet(out, device, thresh, d)

        return out
