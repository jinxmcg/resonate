import torch
_o = torch.load
torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
from ogb.linkproppred import LinkPropPredDataset
d = LinkPropPredDataset("ogbl-wikikg2", root="data_ogb")
g = d[0]
print("num_nodes", g["num_nodes"], "edges", g["edge_index"].shape, "max reltype", g["edge_reltype"].max())
s = d.get_edge_split()
for k in s: print(k, {kk: getattr(v, "shape", None) for kk, v in s[k].items()})
print("DATA_DONE")
