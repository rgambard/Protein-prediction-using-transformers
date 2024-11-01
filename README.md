
- pour le modele sans termes binaires ( precision validation 50.5, NSR 45.6% ( avec toulbar 80 s) )
```
 le modele est dans ../Results/model/no_unaries
 qui va avec le script pytorch effie/CbNet_optiR.py
 entrainé avec le script Main_Cb_modif.py
```
 les parametres sont :
```

from effie.CbNet_optiR import *
# replace CbNet_opti by CbNet_optiR
lr = 0.0001*5
weight_decay = 0.001

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
unary = True
multichain = False

reg_term = 0.0001/10
L1_as_fd = False
adapt_LR = 0 #1/2  #1 to adapt LR with bs, 1/D to adapt with sqrt(bs), 0 to do nothing

nb_it=6 # for the modified net
thresh = 15
max_neigh = 48
noise_std = 0
perc_mask = 0.3 #for gangster PLL

if L1_as_fd:
    reg_term /= 10
lr /= 100**adapt_LR


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

```

- le modele avec les termes unaires : resulats unaires + binaires : precision (50.5%) NSR : 46.2 (mixing ) 46.6 ( toulbar avec 80 s )
 ``` 
`le modele est dans ../Results/model/with_unaries
 qui va avec le script pytorch effie/CbNet_optiR.py
 entrainé avec le script Main_Cb_modif.py
```
 Les parametres sont les memes que plus haut, mais avec unary = True

```

from effie.CbNet_optiR import *
# replace CbNet_opti by CbNet_optiR
lr = 0.0001*5
weight_decay = 0.001

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
unary = True
multichain = False

reg_term = 0.0001/10
L1_as_fd = False
adapt_LR = 0 #1/2  #1 to adapt LR with bs, 1/D to adapt with sqrt(bs), 0 to do nothing

nb_it=6 # for the modified net
thresh = 15
max_neigh = 48
noise_std = 0
perc_mask = 0.3 #for gangster PLL

if L1_as_fd:
    reg_term /= 10
lr /= 100**adapt_LR


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
```
```
