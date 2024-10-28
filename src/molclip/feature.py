import numpy as np
import torch
from deepchem.feat import graph_features
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data


def get_mol_graph(smiles: str, input_ids: torch.Tensor | None = None) -> Data:
    mol = Chem.MolFromSmiles(smiles)

    atom_features: list[np.ndarray] = []
    for atom in mol.GetAtoms():  # type: ignore
        features = graph_features.atom_features(atom, use_chirality=True)
        atom_features.append(features)
    atom_features_tensor = torch.tensor(np.array(atom_features), dtype=torch.float)

    edge_features = []
    for bond in mol.GetBonds():  # type: ignore
        features = graph_features.bond_features(bond, use_chirality=True)
        edge_features.append(features)
    adj_matrix: np.ndarray = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
    edge_features_tensor = torch.tensor(np.array(edge_features), dtype=torch.float)

    return Data(
        x=atom_features_tensor,
        edge_index=edge_index,
        edge_attr=edge_features_tensor,
        y=input_ids,
    )
