import atom_features as af
from inspect import signature
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
import torch
from torch_geometric.utils import add_self_loops


def get_bond_types(smile):
    
    mol = Chem.MolFromSmiles(smile)
    bond_map = {BondType.SINGLE : 0,
                BondType.DOUBLE : 1,
                BondType.TRIPLE : 2,
                BondType.AROMATIC : 3}
    return torch.tensor([bond_map[bond.GetBondType()] for bond in mol.GetBonds()]).repeat_interleave(2)


def get_bond_index(smile):
    
    mol = Chem.MolFromSmiles(smile)
    bond_index = []
    for bond in mol.GetBonds():
        indices = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        bond_index.append(indices)
        bond_index.append(indices[::-1])
    bond_index = torch.tensor(bond_index, dtype=torch.long).t().contiguous()
    return bond_index
    
    
def get_atom_coords(smile):
    
    drug = Chem.MolFromSmiles(smile)
    AllChem.Compute2DCoords(drug)
    coords = []
    
    for i, atom in enumerate(drug.GetAtoms()):
        
        pos = drug.GetConformer().GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
        
    return torch.tensor(coords)


def get_atom_features(smile):

    """TODO: Choose initial atomic features to pass to model"""
    pass
