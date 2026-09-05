"""Read-only BioKG validation diagnosis; never trains or loads test edges.

Run from repo root with python -m biokg.analyze_errors --help.
Ranks use fp32 scores and the official OGB tie convention. Output records
every directed validation query, not only selected mistakes. Graph features
are computed from unique TRAIN triples only; they never affect predictions.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ogb.linkproppred import Evaluator

from resonate import ResonatE
from resonate_wiki import SparseTableResonatE
from biokg.train_biokg_comp import globalize, load


def restore(path, n_ent, n_rel, offset, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ca = ck["args"]
    if ca.get("comp", 0):
        raise ValueError("This diagnostic supports plain entity-table checkpoints only")
    if ck["n_rel"] != n_rel or ck["offset"] != offset:
        raise ValueError("Checkpoint and dataset indexing differ")
    kwargs = dict(k=ca["k"], block_size=ca.get("block_size", 2),
                  ent_bias=ca.get("ent_bias", False), rel_gain=ca.get("rel_gain", False),
                  low_rank=ca.get("low_rank", 0), low_rank_local=ca.get("low_rank_local", False))
    if "E_real" in ck["model"]:
        model = SparseTableResonatE(n_ent, n_rel, device=device,
                                   table_dtype=ck["model"]["E_real"].dtype, **kwargs)
    else:
        model = ResonatE(n_ent, n_rel, block=True, tied_reverse=ca.get("tied_reverse", False),
                         real=ca.get("real", False), **kwargs).to(device)
    model.load_state_dict(ck["model"])
    model.eval().requires_grad_(False)
    return model, ca


def rank_statistics(sp, sn, candidates):
    """Keep exact tie-aware ranks and the highest-scoring negative candidate."""
    if not (torch.isfinite(sp).all() and torch.isfinite(sn).all()):
        raise ValueError("Non-finite scores")
    gt = (sn > sp[:, None]).sum(1)
    ge = (sn >= sp[:, None]).sum(1)
    rank = 1 + (gt + ge).float() / 2
    official = Evaluator("ogbl-biokg").eval({"y_pred_pos": sp, "y_pred_neg": sn})
    torch.testing.assert_close(official["mrr_list"], rank.reciprocal(), rtol=0, atol=0)
    top_score, top_idx = sn.max(1)
    return dict(rank=rank, rr=official["mrr_list"], ties=ge - gt,
                positive_score=sp, top_negative_score=top_score,
                top_negative=candidates.gather(1, top_idx[:, None]).squeeze(1))


@torch.inference_mode()
def score_validation(model, part, offset, n_rel, device, chunk=512):
    h, r, t = globalize(part, offset)
    records = {}
    for direction in (0, 1):  # tail, then head; aligns with eval_split
        src, dst = (h, t) if direction == 0 else (t, h)
        target_type = part["tail_type"] if direction == 0 else part["head_type"]
        negatives = part["tail_neg"] if direction == 0 else part["head_neg"]
        for start in range(0, len(r), chunk):
            sl = slice(start, start + chunk)
            rel = torch.as_tensor(r[sl] + direction * (n_rel // 2), device=device)
            source = torch.as_tensor(src[sl], device=device)
            target = torch.as_tensor(dst[sl], device=device)
            target_offset = np.array([offset[x] for x in target_type[sl]])
            cand = torch.as_tensor(negatives[sl] + target_offset[:, None], device=device)
            z = model.out(model.hop(model.embed(source), rel), rel)
            tau = model.log_tau.exp()
            sp = (z * model.rows(target).conj()).sum(-1).real * tau
            sn = torch.einsum("bm,bcm->bc", z, model.rows(cand).conj()).real * tau
            if model.b is not None:
                sp = sp + model.b[target]
                sn = sn + model.b[cand]
            for key, value in rank_statistics(sp, sn, cand).items():
                records.setdefault(key, []).append(value.cpu().numpy())
        print(f"Scored {'tail' if direction == 0 else 'head'}: {len(r):,} queries", flush=True)
    out = {key: np.concatenate(value) for key, value in records.items()}
    out.update(valid_index=np.tile(np.arange(len(r)), 2), relation=np.tile(r, 2),
               direction=np.repeat(np.array([0, 1], dtype=np.int8), len(r)),
               source=np.concatenate([h, t]), target=np.concatenate([t, h]))
    return out


def membership(sorted_keys, query):
    if not len(sorted_keys):
        return np.zeros(len(query), dtype=bool)
    idx = np.searchsorted(sorted_keys, query)
    return (idx < len(sorted_keys)) & (sorted_keys[np.minimum(idx, len(sorted_keys) - 1)] == query)


def train_features(q, train, offset, n_ent, num_nodes):
    h, r, t = globalize(train, offset)
    degree = np.zeros(n_ent, dtype=np.int64)
    int_keys = ("source_relation_degree", "target_relation_degree", "top_negative_relation_degree")
    bool_keys = ("positive_in_train", "reverse_in_train", "top_negative_in_train")
    for key in int_keys:
        q[key] = np.zeros(len(q["rank"]), dtype=np.int64)
    for key in bool_keys:
        q[key] = np.zeros(len(q["rank"]), dtype=bool)
    collision = []
    for ri in np.unique(r):
        train_mask = r == ri
        pairs = np.unique(h[train_mask] * n_ent + t[train_mask])
        hh, tt = pairs // n_ent, pairs % n_ent
        hc = np.bincount(hh, minlength=n_ent)
        tc = np.bincount(tt, minlength=n_ent)
        degree += hc + tc
        i0 = np.flatnonzero(train_mask)[0]
        ht, tt_name = train["head_type"][i0], train["tail_type"][i0]
        reverse_pairs = np.sort(tt * n_ent + hh)
        for direction in (0, 1):
            mask = (q["relation"] == ri) & (q["direction"] == direction)
            src_count, dst_count, known = (hc, tc, pairs) if direction == 0 else (tc, hc, reverse_pairs)
            s, d, neg = q["source"][mask], q["target"][mask], q["top_negative"][mask]
            q["source_relation_degree"][mask] = src_count[s]
            q["target_relation_degree"][mask] = dst_count[d]
            q["top_negative_relation_degree"][mask] = dst_count[neg]
            q["positive_in_train"][mask] = membership(known, s * n_ent + d)
            q["reverse_in_train"][mask] = membership(known, d * n_ent + s)
            q["top_negative_in_train"][mask] = membership(known, s * n_ent + neg)
            n_target = int(num_nodes[tt_name if direction == 0 else ht])
            # Expected known-positive sampled entries per query, assuming the
            # trainer's uniform-with-replacement sampling over the target type.
            collision.append(dict(relation=int(ri), direction="tail" if direction == 0 else "head",
                                  train_triples=int(len(pairs)), target_type_size=n_target,
                                  expected_known_positives_per_4096=float(
                                      4096 * np.square(src_count.astype(float)).sum() / len(pairs) / n_target)))
    for role in ("source", "target", "top_negative"):
        q[role + "_degree"] = degree[q[role]]
    return collision


def summary(q, mask=None):
    if mask is None:
        mask = np.ones(len(q["rank"]), dtype=bool)
    rank, rr = q["rank"][mask], q["rr"][mask].astype(np.float64)
    deficit = float((1 - rr).sum())
    all_deficit = float((1 - q["rr"].astype(np.float64)).sum())
    return dict(queries=len(rank), mrr=float(rr.mean()), hits1=float((rank <= 1).mean()),
                hits3=float((rank <= 3).mean()), hits10=float((rank <= 10).mean()),
                not_top1=int((rank > 1).sum()), beyond_top10=int((rank > 10).sum()),
                mrr_deficit_contribution=deficit / len(q["rank"]),
                share_of_mrr_deficit=deficit / all_deficit if all_deficit else 0.,
                median_source_degree=float(np.median(q["source_degree"][mask])),
                median_target_degree=float(np.median(q["target_degree"][mask])))


def groups(q, labels):
    rows = [dict(group=str(label), **summary(q, labels == label)) for label in np.unique(labels)]
    return sorted(rows, key=lambda row: -row["mrr_deficit_contribution"])


def mapping(path):
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return {int(row[0]): row[1] for row in reader}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True, help="Fresh directory; existing outputs are never overwritten")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=512)
    args = p.parse_args()
    if args.chunk < 1:
        p.error("--chunk must be positive")
    out = Path(args.out)
    if out.exists():
        p.error("Output already exists; choose a fresh directory")
    split, offset, n_ent, types, num_nodes = load(args.data_root, include_test=False)
    n_rel = 2 * (int(split["train"]["relation"].max()) + 1)
    model, ca = restore(args.checkpoint, n_ent, n_rel, offset, torch.device(args.device))
    q = score_validation(model, split["valid"], offset, n_rel, torch.device(args.device), args.chunk)
    collision = train_features(q, split["train"], offset, n_ent, num_nodes)
    map_dir = Path(args.data_root) / "ogbl_biokg" / "mapping"
    rel_names = mapping(map_dir / "relidx2relname.csv.gz")
    names = {t: mapping(map_dir / f"{t}_entidx2name.csv.gz") for t in types}
    rel_label = np.array([rel_names[int(r)] for r in q["relation"]])
    dir_label = np.where(q["direction"] == 0, "tail", "head")
    family = np.array(["drug-drug (all)" if x.startswith("drug-drug_") else
                       "protein-protein (all)" if x.startswith("protein-protein_") else x for x in rel_label])
    rank = q["rank"]
    bands = np.select([rank <= 1, rank <= 3, rank <= 10, rank <= 100],
                      ["rank 1", "rank (1,3]", "rank (3,10]", "rank (10,100]"], default="rank >100")
    tables = dict(rank_bands=groups(q, bands), directions=groups(q, dir_label),
                  families=groups(q, family), relations=groups(q, rel_label),
                  relation_directions=groups(q, np.char.add(np.char.add(rel_label, ":"), dir_label)))
    for feature in ("source_degree", "target_degree", "source_relation_degree", "target_relation_degree"):
        x = q[feature]
        bucket = np.select([x == 0, x <= 5, x <= 20, x <= 100, x <= 1000],
                           ["0", "1-5", "6-20", "21-100", "101-1000"], default="1001+")
        tables[feature] = groups(q, bucket)
    coverage = np.where((q["source_relation_degree"] == 0) | (q["target_relation_degree"] == 0),
                        "at least one endpoint unseen in this relation", "both endpoints seen in this relation")
    tables["relation_coverage"] = groups(q, coverage)
    tables["reverse_in_train"] = groups(q, np.where(q["reverse_in_train"], "yes", "no"))
    for row in collision:
        row["relation_name"] = rel_names[row["relation"]]
    with open(args.checkpoint, "rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=checkpoint_hash,
                  checkpoint_args=ca, protocol="validation only; official OGB fp32 tie-aware ranks; no training",
                  negative_candidates_per_query=int(split["valid"]["head_neg"].shape[1]),
                  overall=summary(q), groups=tables,
                  checks=dict(positive_in_train=int(q["positive_in_train"].sum()),
                              top_negative_in_train=int(q["top_negative_in_train"].sum()),
                              queries_with_score_ties=int((q["ties"] > 0).sum())),
                  train_negative_collision_expectations=collision)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "queries.npz", **q)
    with (out / "summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with gzip.open(out / "queries.csv.gz", "wt", newline="") as handle:
        columns = ["valid_index", "direction", "relation", "source_type", "source_id",
                   "target_type", "target_id", "top_negative_id", "rank", "rr", "score_margin",
                   "source_degree", "target_degree", "source_relation_degree", "target_relation_degree",
                   "top_negative_relation_degree", "reverse_in_train"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i in range(len(rank)):
            vi, di = int(q["valid_index"][i]), int(q["direction"][i])
            st = split["valid"]["head_type" if di == 0 else "tail_type"][vi]
            dt = split["valid"]["tail_type" if di == 0 else "head_type"][vi]
            row = {key: q[key][i].item() for key in columns if key in q}
            row.update(direction="tail" if di == 0 else "head", relation=rel_label[i], source_type=st,
                       source_id=names[st][int(q["source"][i] - offset[st])], target_type=dt,
                       target_id=names[dt][int(q["target"][i] - offset[dt])],
                       top_negative_id=names[dt][int(q["top_negative"][i] - offset[dt])],
                       score_margin=float(q["positive_score"][i] - q["top_negative_score"][i]))
            writer.writerow(row)
    print(json.dumps(dict(overall=report["overall"], checks=report["checks"],
                          largest_deficits=tables["relation_directions"][:12]), indent=2), flush=True)
    print(f"Saved all query records and grouped summaries to {out}", flush=True)


if __name__ == "__main__":
    main()
