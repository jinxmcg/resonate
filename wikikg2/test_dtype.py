import sys, torch
from resonate_wiki import SparseTableResonatE, score_batch
from rowadagrad import RowAdagrad
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
for dt in (torch.float16, torch.bfloat16):
    m = SparseTableResonatE(200, 4, k=4, block_size=2, table_dtype=dt, device=dev)
    opt = RowAdagrad(m.table_params(), lr=0.3)
    R = lambda *s: torch.randint(0, 200, s, device=dev)
    src, rel, dst, negs = R(8), torch.randint(0, 4, (8,), device=dev), R(8), R(30)
    lg, z, e = score_batch(m, src, rel, dst, negs)
    assert lg.dtype == torch.float32
    torch.nn.functional.cross_entropy(lg, torch.zeros(8, dtype=torch.long, device=dev)).backward()
    before = m.E_real.data.clone(); opt.step()
    assert m.E_real.dtype == dt and not torch.equal(before, m.E_real.data)
    print(dt, "table ok on", dev)
