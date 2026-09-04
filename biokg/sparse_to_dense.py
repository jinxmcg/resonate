"""Convert a sparse-shell checkpoint (resonate_wiki.SparseTableResonatE:
E_real (N, 2M) real view) into the dense resonate.ResonatE checkpoint
format the public scripts load (E complex (N, M)). Exact: the sparse
class's own selftest asserts identical scores for this reinterpretation."""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
d = torch.load(src, map_location="cpu", weights_only=False)
sd = d["model"]; er = sd.pop("E_real").float(); n = er.shape[0]
sd_new = {"E": torch.view_as_complex(er.view(n, -1, 2).contiguous()).contiguous()}
sd_new.update(sd)                                   # H, log_tau (+ b if ent_bias)
args = dict(d["args"]); args["shell"] = "dense"; args["converted_from"] = "sparse"
torch.save({"model": sd_new, "offset": d["offset"], "n_rel": d["n_rel"], "args": args}, dst)
print(dst, {k: tuple(v.shape) for k, v in sd_new.items()})
