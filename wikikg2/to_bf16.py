"""Re-save checkpoints with a bf16 table (half the file) for use as distillation teachers."""
import sys, torch
for path in sys.argv[1:]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    d["model"]["E_real"] = d["model"]["E_real"].to(torch.bfloat16); d["args"]["table_dtype"] = "bf16"
    out = path.replace(".pt", ".bf16.pt"); torch.save(d, out); print("->", out, flush=True)
