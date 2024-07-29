# smiles2graph

## How to use

```python
from smiles2graph import getDrugGraph

PyG_graph = getDrugGraph('C1=CC=C2C(=C1)N=C(S2)SSC3=NC4=CC=CC=C4S3')
print(PyG_graph)
Data(x=[20, 5], edge_index=[2, 23], edge_attr=[23, 2], atom_type=[20])
```

## Requirement

```sh
numpy
rdkit
torch
torch_geometric
```
