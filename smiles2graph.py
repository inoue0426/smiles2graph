import re
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

def sanitize_mol(mol):
    try:
        Chem.SanitizeMol(mol, 
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
                                     Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
                                     Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                                     Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
                                     Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                         catchErrors=True)
    except:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    return mol

def process_atom(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetFormalCharge(),
    ]

def preprocess_smiles(smiles):
    smiles = smiles.replace('[i+]', '[I+]')
    smiles = smiles.replace('\\', '')
    smiles = smiles.replace('/', '')
    smiles = smiles.replace('[ZH]', '[H]')
    smiles = smiles.replace('[Z]', '[C]')
    smiles = smiles.replace('[M+]', '[C+]')  # Replace [M+] with [C+] instead of removing it
    
    # Fix common syntax errors
    smiles = re.sub(r'\[NH2\)', '[NH2]', smiles)  # Fix [NH2) to [NH2]
    smiles = re.sub(r'\(([^()]*)\)', r'(\1)', smiles)  # Ensure all opening brackets have a closing bracket
    
    # Balance brackets
    open_count = smiles.count('(')
    close_count = smiles.count(')')
    if open_count > close_count:
        smiles += ')' * (open_count - close_count)
    elif close_count > open_count:
        smiles = '(' * (close_count - open_count) + smiles
    
    return smiles

def getDrugGraph(smiles):
    try:
        original_smiles = smiles
        smiles = preprocess_smiles(smiles)
        
        # Split the SMILES string if it contains multiple parts
        parts = smiles.split('.')
        all_nodes = []
        all_edges = []
        total_atoms = 0

        for part in parts:
            mol = Chem.MolFromSmiles(part, sanitize=False)
            if mol is None:
                print(f"Warning: Could not create molecule from SMILES part: {part}")
                continue

            mol = sanitize_mol(mol)

            for atom in mol.GetAtoms():
                atom.SetIntProp('orig_idx', atom.GetIdx() + total_atoms)
                all_nodes.append(process_atom(atom))

            for bond in mol.GetBonds():
                all_edges.append([
                    bond.GetBeginAtomIdx() + total_atoms,
                    bond.GetEndAtomIdx() + total_atoms,
                    int(bond.GetBondType()),
                    int(bond.GetStereo()),
                ])

            total_atoms += mol.GetNumAtoms()

        if not all_nodes:
            raise ValueError("No valid atoms found")

        node_features = torch.Tensor(all_nodes)
        node_features[:, 0] = node_features[:, 0] / 100

        atom_type = torch.Tensor([n[0] for n in all_nodes])

        if all_edges:
            edges = np.array(all_edges)
            edge_index = torch.tensor(edges[:, :2].T, dtype=torch.long)
            edges_attr = torch.tensor(edges[:, 2:], dtype=torch.float)
            edges_attr[:, 0] = edges_attr[:, 0] / 10
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edges_attr = torch.empty((0, 2), dtype=torch.float)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edges_attr,
            atom_type=atom_type,
        )

    except Exception as e:
        print(f"Error creating graph for SMILES {original_smiles}: {e}")
        return None
