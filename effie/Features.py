# -*- coding: utf-8 -*-
"""

"""
import torch
import pickle
import math
from torch.utils.data import Dataset
import numpy as np
import json
import time
import copy
import tqdm
import random
import torch.nn.functional as F
import pandas as pd
#from quaternion import *

### Data loaders ###


class CATHDataset:
    """
    Loader and container class for the CATH 4.2 dataset downloaded
    from http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/.

    Has attributes `self.train`, `self.val`, `self.test`, each of which are
    JSON/dictionary-type datasets as described in README.md.

    :param path: path to chain_set.jsonl
    :param splits_path: path to chain_set_splits.json or equivalent.
    """

    def __init__(self, path, splits_path):
        with open(splits_path) as f:
            dataset_splits = json.load(f)
        train_list, val_list, test_list = dataset_splits["train"], dataset_splits["validation"], dataset_splits["test"]

        self.train, self.val, self.test = [], [], []

        with open(path) as f:
            lines = f.readlines()

        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            name = entry["name"]
            coords = entry["coords"]

            entry["coords"] = list(zip(coords["N"], coords["CA"], coords["C"], coords["O"]))

            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)
            elif name in test_list:
                self.test.append(entry)


def save_obj(obj, name, folder):
    with open(folder + "/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(name, folder):
    with open(folder + "/" + name, "rb") as f:
        return pickle.load(f)


### To handle missing residues ####


def missing_residues(crd):

    # missing = torch.sum(torch.tensor([torch.zeros(3) in crd[i, :4] for i in range(crd.shape[0])])).item()
    nb_var = crd.shape[0]
    missing = torch.stack([torch.sum(torch.isnan(crd[i])) > 0 for i in range(nb_var)])

    return missing


def remove_N_C_ter(crd, seq, chain_idx=None, tm=None):
    """
    Remove missing residues at N- and C-ter
    """

    zero = torch.zeros((4, 3))
    while torch.sum(torch.abs(crd[0, :4])) == 0 or torch.isnan(torch.sum(torch.abs(crd[0, :4]))):  # N-ter
        crd = crd[1:]
        seq = seq[1:]
        if chain_idx is not None:
            chain_idx = chain_idx[1:]
        if tm is not None:
            tm = tm[1:]

    while torch.sum(torch.abs(crd[-1, :4])) == 0 or torch.isnan(torch.sum(torch.abs(crd[-1, :4]))):  # C-ter
        crd = crd[:-1]
        seq = seq[:-1]
        if chain_idx is not None:
            chain_idx = chain_idx[:-1]
        if tm is not None:
            tm = tm[:-1]

    return (crd, seq, chain_idx, tm)
    
    
def list_TM():
    
    L_TM = []
    f = open("../features/PDBTM/pdbtm_all.list")
    for l in f.readlines():
        L_TM.append(l[:4]+'.'+l.strip()[-1])
    f.close()
    
    return L_TM


### KORP features ###


def get_features_KORP(crd):
    """
    Not used anymore
    Return the features (6 KORP + contact numbers) for each residue of the protein
    """

    L_ft = []
    thresh = [5, 10]
    contact_nb = contact_number(crd, thresh)

    l = crd.shape[0]  # length of the protein
    for i in range(l):
        for j in range(l):

            if i == j:
                L_ft.append(torch.zeros(10))
            else:
                ft = torch.cat((get_KORP_ft(crd[i], crd[j]), contact_nb[:, i], contact_nb[:, j]))
                if math.isnan(torch.max(ft)):  # due to missing residues
                    L_ft.append(torch.zeros(10))
                else:
                    L_ft.append(ft)
    return torch.stack(L_ft)


def get_KORP_ft(crd_i, crd_j):
    """
    Returns KORP features from a pair of aa coordinates
    Input: coordinates of residue i (tensor of size 14*3.
           The 1st 3 lines are the crs of N, Calpha and C bb atoms)
           coordinates of residue j

    Outputs: The 6 features of KORP
    """

    r_ij = crd_j[1] - crd_i[1]
    Vx_i, Vy_i, Vz_i = get_frame(crd_i)
    Vx_j, Vy_j, Vz_j = get_frame(crd_j)
    theta_i, phi_i = get_angles(r_ij, Vz_i, Vx_i, Vy_i)
    theta_j, phi_j = get_angles(r_ij, Vz_j, Vx_j, Vy_j)
    w = get_torsion_angle(Vz_i, r_ij, Vz_j)

    return torch.Tensor([torch.norm(r_ij), theta_i, phi_i, theta_j, phi_j, w])


def contact_number(crd, thresh=[5, 10]):
    """
    Compute the contact number (number of neighbours within a distance) for each residue.
    Input: the coordinates of the protein, with the missing residues removed
           a thresold for the distance (int/float or list of int/float). Default is [5, 10]

    Output: a Tensor of shape l (or (len(thresh)xl) if thresh is a list)
    """

    Ca_crd = crd[:, 1]  # crd of all Ca
    l = crd.shape[0]
    D = torch.zeros((l, l))
    for i in range(l):
        for j in range(i + 1, l):
            D[i, j] = torch.norm(Ca_crd[i] - Ca_crd[j])
    D += torch.clone(D).t()

    L_contact = []
    if isinstance(thresh, float) or isinstance(thresh, int):
        L_contact = [torch.sum(D[i] < thresh) for i in range(l)]
    elif isinstance(thresh, list):
        for t in thresh:
            L_contact.append(torch.stack([torch.sum(D[i] < t) for i in range(l)]))
    else:
        print("Please input a int/float or list of int/float threshold")
        return

    return torch.stack(L_contact)


# Auxiliary function


def get_angles(r_ij, Vz_i, Vx_i, Vy_i):

    # projection of r_ij in local frame i (spheric crd)
    theta_i = torch.acos(torch.dot(r_ij, Vz_i) / torch.norm(r_ij))  # acos(z/r)
    phi_i = torch.atan2(torch.dot(r_ij, Vy_i), torch.dot(r_ij, Vx_i))  # atan(y/x)
    return (theta_i, phi_i)


# from dihedral angle wikipedia page


def get_torsion_angle(u1, u2, u3):

    w = torch.atan2((torch.norm(u2) * u1).dot(torch.cross(u2, u3)), torch.cross(u1, u2).dot(torch.cross(u2, u3)))
    return w


### Quaternion features ####


def get_quaternion(crd_i, crd_j):
    """
    Returns quaternion representing the orientation of residue j wrt residue i.
    Input: coordinates of i and j wrt reference frame (Tensor)
    Output: quaternion (Tensor of shape 4)
    """

    frame_i = get_frame(crd_i)
    crd_j_in_i = torch.matmul(frame_i, crd_j.T)  # each atom crd are in COLUMN
    frame_j = get_frame(crd_j_in_i.T)  # frame j/i (rotation matrix)

    # quaternion (w, x, y, z)
    # w is the angle, between -1 and 1 (angle = 2arcsin(w))
    # (x, y, z) is the axis crd, each between -1 and 1
    quat = matrix_to_quaternion(frame_j)

    return quat


def get_frame(crd_i):
    """ Returns the local frame given the amino acid coordinates """

    r_CCa = crd_i[2] - crd_i[1]
    r_NCa = crd_i[0] - crd_i[1]

    Vz = (r_CCa + r_NCa) / torch.norm(r_CCa + r_NCa)
    Vy = torch.cross(Vz, r_NCa)
    Vy /= torch.norm(Vy)
    Vx = torch.cross(Vy, Vz)

    return torch.cat((Vx, Vy, Vz)).reshape(-1, 3)


def get_translation(crd_i, crd_j):
    """
    Returns the unit translation vector from i to j, expressed in the frame i.
    """

    frame_i = get_frame(crd_i)
    new_crdi = torch.matmul(frame_i, crd_i.T).T
    new_crdj = torch.matmul(frame_i, crd_j.T).T

    return (new_crdj[1] - new_crdi[1]) / torch.norm(new_crdj[1] - new_crdi[1])


def pdb_plouf(line):

    atom = line[0:6]
    atom_nb = line[6:11]
    atom_type = line[12:16]
    alt_location = line[16:17]
    res_name = line[17:20]
    chain = line[21:22]
    res_seq_nb = line[22:26]
    insert_res = line[26:27]
    x = line[30:38]
    y = line[38:46]
    z = line[46:54]
    occupancy = line[54:60]
    T_factor = line[60:66]
    element_symbol = line[76:78]
    atom_charge = line[78:80]

    return [atom_nb, atom_type, res_name, res_seq_nb, x, y, z, occupancy, chain]


def new_dihedral(p):
    """Praxeolitic formula (given by Marc)
    angle in radian"""

    b0 = -1.0 * (p[1] - p[0])
    b1 = p[2] - p[1]
    b2 = p[3] - p[2]

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    # = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    # = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


def dihedral_old(crd):
    """
    Input backbone coordinates (size nb_var,3,4)
    Output Tensor of the cos and sin of phi and psi angle for eash residue (size nb_var*4)
    """

    nb_var = crd.shape[0]
    dihedral_angles = torch.zeros((nb_var, 2))
    for i in range(1, nb_var - 1):

        phi_plane = np.concatenate((crd[i - 1, 2].reshape(1, -1), crd[i, :3]), axis=0)
        psi_plane = np.concatenate((crd[i, :3], crd[i + 1, 0].reshape(1, -1)), axis=0)
        phi, psi = new_dihedral(phi_plane), new_dihedral(psi_plane)
        dihedral_angles[i] = torch.tensor([phi, psi])

    dihedral_angles = torch.cat((torch.cos(dihedral_angles), torch.sin(dihedral_angles)), dim=1)

    dihedral_angles = torch.nan_to_num(dihedral_angles, nan=0.0)

    return dihedral_angles


def get_features(crd):
    """
    Returns the pairwise features (20-vector for each pair of protein):
        - 4-vector (quaternion) representing relative orientation
        - translation vector (in local coordinates)
        - distance
        - contact numbers at 5 and 10 A for each residue of the protein
        - Dihedral angles of each protein (cos, sin of phi and psi, 8 in total)
        
    """

    L_ft = []
    nb_ft = 12 + 4 * 2
    thresh = [5, 10]
    contact_nb = contact_number(crd, thresh)
    dihedrals = dihedral(crd)

    # identity ft when i = j (1 due tu quaternion and cosine of 0 angles)
    id_ft = torch.zeros(nb_ft)
    id_ft[0], id_ft[-3], id_ft[-4] = 1, 1, 1

    l = crd.shape[0]  # length of the protein
    for i in range(l):
        for j in range(l):

            if i == j:
                L_ft.append(id_ft)
            else:
                r_ij = torch.norm(crd[j][1] - crd[i][1]).view(1)
                ft = torch.cat(
                    (
                        get_quaternion(crd[i], crd[j]),
                        get_translation(crd[i], crd[j]),
                        r_ij,
                        contact_nb[:, i],
                        contact_nb[:, j],
                        dihedrals[i],
                        dihedrals[j],
                    )
                )

                if torch.sum(torch.isnan(ft)) > 0:  # due to missing residues
                    L_ft.append(torch.zeros(nb_ft))
                else:
                    L_ft.append(ft)

    return torch.stack(L_ft)
    
    
    
    
def extract_ft_from_PDB(path, filename, quat_ft = False):

    """
    Given a PDB file, the function creates a dataframe, extract backbone coordinates and computes pairwise features
    Input : path of the file, filename
    Output: dict of features
    """

    letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                           'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                           'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                           'N': 2, 'Y': 18, 'M': 12}
                           
    letter3to1 = {'CYS' : 'C', 'ASP' : 'D', 'SER' : 'S', 'GLN' : 'Q', 'LYS' : 'K',
                 'ILE' : 'I', 'PRO' : 'P', 'THR' : 'T', 'PHE' : 'F', 'ALA' : 'A',
                 'GLY' : 'G', 'HIS' : 'H', 'GLU' : 'E', 'LEU' : 'L', 'ARG' : 'R',
                 'TRP' : 'W', 'VAL' : 'V', 'ASN' : 'N', 'TYR' : 'Y', 'MET' : 'M'}
                          

    file = open(path + filename, 'r')
    L = file.readlines()
    file.close()

    ### Put pdb data in a dataframe to extract seq and crd ###
    df = pd.DataFrame(columns=['id_atom', 'atom_type', 'AA', 'id_aa', 'x', 'y', 'z', 'occupancy'])
    for i in range(len(L)-1):
        if 'TER' not in  L[i] and L[i][:4] == 'ATOM':
            df = df.append(pd.Series(pdb_plouf(L[i]), index = df.columns), ignore_index=True)
        
    df["id_atom"] = pd.to_numeric(df["id_atom"])
    df["id_aa"] = pd.to_numeric(df["id_aa"])
    df["x"] = pd.to_numeric(df["x"])
    df["y"] = pd.to_numeric(df["y"])
    df["z"] = pd.to_numeric(df["z"])
    df["occupancy"] = pd.to_numeric(df["occupancy"])

    atom = df['atom_type']
    atom = atom.apply(lambda x: x.strip())
    df = df[(atom == 'N') | (atom =='CA') | (atom =='C') | (atom == 'O')]

    nb_var = df["id_aa"].iat[-1] - df["id_aa"].iat[1] + 1
    start_num = df["id_aa"].iat[1] #numerotation in the PDB file
        
    #In case of occupancy <1 (thus several crd for the same atom)
    for i in range(start_num,  df["id_aa"].iat[-1] +1):
        if len(df[df["id_aa"] == i]) > 4:

            atoms = df[df["id_aa"]==i]
            atoms["atom_type"] = atoms["atom_type"].apply(lambda x: x.strip())
            for a in ['N', 'CA', 'C', 'O']:
                idx = atoms[atoms["atom_type"]==a].sort_values('occupancy', ascending=False).index[1:]
                df = df.drop(idx)

                
    crd = torch.cat((torch.Tensor(df['x'].values).reshape(-1, 1),
       torch.Tensor(df['y'].values).reshape(-1, 1),
       torch.Tensor(df['z'].values).reshape(-1, 1)), dim = 1)
    
    missing = []
        
    #finding which residu does not have good number of atom
    for i in range(start_num,  df["id_aa"].iat[-1] +1):
        if len(df[df["id_aa"] == i]) < 4:

            #Finding which atom is missing
            missing_atom = []
            atom = df[df["id_aa"] == i+start_num]["atom_type"]
            atom = atom.apply(lambda x: x.strip())
            if 'N' not in atom.unique():
                missing_atom.append(0)
            if 'CA' not in atom.unique():
                missing_atom.append(1)
            if 'C' not in atom.unique():
                missing_atom.append(2)
            if 'O' not in atom.unique():
                missing_atom.append(3)

            missing.append([i-start_num, missing_atom])
    #inserting unknown atoms at the right place
    for res, atoms in missing:
        for a in atoms:
            insert = res*4 + a
            crd = torch.cat((crd[:insert], torch.zeros((1, 3)), crd[insert:]), dim = 0)
            
            
    print("Correct dimension:", crd.shape[0] == nb_var*4)
    crd = crd.reshape(-1, 4, 3)

    seq = df['AA'].values[0:-1:4]
    seq = torch.tensor([letter_to_num[letter3to1[a]] for a in seq], dtype=torch.long)
    missing = torch.zeros_like(seq).type(torch.BoolTensor)

    # Extract features
    ft = get_features(crd) if quat_ft else None

    #save features
    dico = dict()
    dico["features"] = ft
    dico["int_seq"] = seq.type(torch.LongTensor)
    dico["distance"] = torch.norm(crd[:, 1].unsqueeze(1).expand(nb_var, nb_var, 3) 
                        - crd[:, 1].unsqueeze(1).expand(nb_var, nb_var, 3).transpose(0,1), dim = -1)
    dico["missing"] = missing
    dico["coordinates"] = crd
    torch.save(dico, path + filename[:-4]) #remove .pdb from filename
    
    return dico
    
    
### Computing angles in a parallzelized fashion ###
def normalize_batch_vector(b):
    
    nb_var = b.shape[0]
    return b/torch.linalg.norm(b, dim = 1).unsqueeze(-1).expand(nb_var, 3)

def dihedral_parallel(plane):
    
    """Praxeolitic formula (from wikipedia)
    Input: n*4*3"""

    nb_var = plane.shape[0]
    b0 = -(plane[:, 1] - plane[:, 0])
    b1 = plane[:, 2] - plane[:, 1]
    b2 = plane[:, 3] - plane[:, 2]
    b1 = normalize_batch_vector(b1)

    v = b0 - torch.matmul(b0, b1.T).diag().unsqueeze(-1).expand(nb_var, 3)*b1
    w = b2 - torch.matmul(b2, b1.T).diag().unsqueeze(-1).expand(nb_var, 3)*b1

    x = torch.matmul(v, w.T).diag()
    y = (torch.cross(b1, v)@w.T).diag()

    return torch.atan2(y, x)

def dihedral(crd):
    """
    Input backbone coordinates (torch tensor of size nb_var,3,4)
    Output Tensor of the cos and sin of phi, psi, omega angles for eash residue (size nb_var*6)
    """

    nb_var = crd.shape[0]
    crd = torch.cat((crd[:, :4], torch.zeros((1, 4, 3)).to(crd.device)))
    N, Ca, C, O = crd[:, 0].unsqueeze(1), crd[:, 1].unsqueeze(1), crd[:, 2].unsqueeze(1), crd[:, 3].unsqueeze(1)
    
    phi_plane = torch.cat((C[0:nb_var], N[1:nb_var+1], Ca[1:nb_var+1], C[1:nb_var+1]), 
                          dim = 1) #plane (C, N, Cα, C)
    psi_plane = torch.cat((N[0:nb_var], Ca[0:nb_var], C[0:nb_var], N[1:nb_var+1]), 
                          dim = 1) #plane (N, Cα, C, N)
    omega_plane = torch.cat((Ca[0:nb_var], C[0:nb_var], N[1:nb_var+1], Ca[1:nb_var+1]), 
                            dim = 1) #plane (Cα, C, N, Cα)
    
    dihedral_angles = torch.stack([dihedral_parallel(phi_plane), 
                                   dihedral_parallel(psi_plane), 
                                   dihedral_parallel(omega_plane)]).T
    dihedral_angles = torch.cat((torch.cos(dihedral_angles), torch.sin(dihedral_angles)), dim=1)
    dihedral_angles = torch.nan_to_num(dihedral_angles, nan=0.0)
    
    return dihedral_angles

def calc_angle(vector_a, vector_b):
    
    angle = torch.acos(torch.matmul(normalize_batch_vector(vector_a),
                        normalize_batch_vector(vector_b).T).diag())
    
    return torch.nan_to_num(angle, nan=0.0)

def bond_angles(crd):
    
    """
    Input coordinates Tensor (shape n*4*3) 
    Output cos, sin of bond angles (shape n*6)
    """
    
    nb_var = crd.shape[0]
    crd = torch.cat((crd[:, :4], torch.zeros((1, 4, 3)).to(crd.device)))
    
    N_Ca = crd[:nb_var, 1] - crd[:nb_var, 0]
    Ca_C = crd[:nb_var, 2] - crd[:nb_var, 1]
    C_N = crd[1:nb_var+1, 0] - crd[:nb_var, 2]

    alpha = calc_angle(-N_Ca, Ca_C) #angle N, Ca, C
    beta = calc_angle(C_N, -N_Ca) #angle C, N, Ca
    gamma = calc_angle(Ca_C, C_N) #angle Ca, C, N
    
    bond_ang = torch.stack([alpha, beta, gamma]).T
    bond_ang = torch.cat((torch.cos(bond_ang), torch.sin(bond_ang)), dim=1)
    bond_ang = torch.nan_to_num(bond_ang, nan=0.0)
    
    return bond_ang
