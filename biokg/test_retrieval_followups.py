"""RF1 synthetic feature, support and selection checks; no real data reads."""

import unittest

import numpy as np
import torch
from scipy import sparse

from resonate import ResonatE
from biokg.compare_feature_pipeline import build_features as baseline_features, fit_recipe, paired_masks
from biokg.mixed_operator import from_student
from biokg.relation_analogy import rank_rows
from biokg.retrieval_followups import (
    HalfFeature, build_features, candidate_followups, feature_views, graph_similarities,
    pool_followups, rarity_weights, support_quality, supported_feature,
)
from biokg.train_joint_operator import parameter_hashes


class RetrievalFollowupTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3620)
        torch.set_num_threads(2)
        self.adj = np.array([[1,1,0,0,0], [1,0,1,0,0], [1,0,0,1,0], [0,0,0,0,0]], np.float32)
        self.graph = sparse.csr_matrix(self.adj)

    def test_rarity_definition_and_uniform_equivalence(self):
        weights = rarity_weights(self.graph)
        np.testing.assert_allclose(weights, 1 + np.log(4 / (self.adj.sum(0) + 1)), atol=1e-7)
        self.assertGreater(weights[1], weights[0])
        np.testing.assert_array_equal(graph_similarities(self.graph, [0,1,3]),
                                       graph_similarities(self.graph, [0,1,3], np.ones(5, np.float32)))

    def test_weighted_jaccard_bruteforce(self):
        weights = rarity_weights(self.graph)
        actual = graph_similarities(self.graph, np.arange(4), weights)
        for a in range(4):
            for b in range(4):
                intersection = (self.adj[a] > 0) & (self.adj[b] > 0)
                union = (self.adj[a] > 0) | (self.adj[b] > 0)
                expected = weights[intersection].sum()/weights[union].sum() if union.any() else 0.
                self.assertAlmostEqual(float(actual[a,b]), float(expected), places=6)

    def test_rare_shared_neighbour_is_more_informative(self):
        adj = np.array([[1,1,0,0], [1,0,1,0], [0,1,0,1], [1,0,0,0], [1,0,0,0]], np.float32)
        graph = sparse.csr_matrix(adj)
        plain = graph_similarities(graph, [0])
        weighted = graph_similarities(graph, [0], rarity_weights(graph))
        self.assertEqual(plain[0,1], plain[0,2])
        self.assertGreater(weighted[0,2], weighted[0,1])

    def test_empty_query_rows_do_not_change_train_population(self):
        extended = sparse.vstack([self.graph, sparse.csr_matrix((3,5), dtype=np.float32)]).tocsr()
        np.testing.assert_array_equal(rarity_weights(self.graph), rarity_weights(extended))
        sims = np.array([[[.2,.4,.7,.9]]], np.float32)
        original = support_quality(sims, np.diff(self.graph.indptr)>0, np.array([0]))
        extended_sims = np.concatenate([sims, np.full((1,1,3), 100, np.float32)], axis=2)
        changed = support_quality(extended_sims, np.diff(extended.indptr)>0, np.array([0]))
        np.testing.assert_array_equal(original[0], changed[0][:,:,:4])
        for a,b in zip(original[1:], changed[1:]):
            np.testing.assert_array_equal(a,b)

    def test_support_background_against_bruteforce(self):
        rng = np.random.default_rng(10)
        sims = rng.normal(size=(2,3,7)).astype(np.float32)
        active = np.array([True,True,False,True,True,False,True])
        local = np.array([0,2,6])
        quality, means, stds, counts = support_quality(sims, active, local)
        for metric in range(2):
            for j, source in enumerate(local):
                eligible = active.copy()
                eligible[source] = False
                values = sims[metric,j,eligible].astype(np.float64)
                mu, sd = np.float32(values.mean()), np.float32(values.std())
                excess = np.maximum(sims[metric,j]-mu, 0)
                expected = excess/(excess+sd+np.float32(1e-8))
                np.testing.assert_allclose(quality[metric,j], expected, atol=1e-7)
                self.assertAlmostEqual(float(means[metric,j]), float(mu), places=6)
                self.assertAlmostEqual(float(stds[metric,j]), float(sd), places=6)
                self.assertEqual(counts[j], eligible.sum())

    def test_constant_or_insufficient_background_has_zero_support(self):
        sims = np.ones((2,2,4), np.float32)
        for active in (np.ones(4,bool), np.zeros(4,bool), np.array([1,0,0,0],bool)):
            quality, _, _, _ = support_quality(sims, active, np.array([0,3]))
            np.testing.assert_array_equal(quality, np.zeros_like(quality))

    def test_holder_gate_bruteforce_and_candidate_permutation(self):
        graph = sparse.csr_matrix(np.array([[1,0,0,1,0], [1,1,0,0,0], [1,0,1,0,0], [0,0,0,0,0]], np.float32))
        local = np.array([0,3])
        candidates = np.array([[0,1,2,3,4,1], [0,1,2,3,4,1]])
        rng = np.random.default_rng(3)
        prepared = rng.random((3,2,4), dtype=np.float32)
        got = candidate_followups(prepared, graph, graph.tocsc(), local, candidates)
        for j, source in enumerate(local):
            for c, target in enumerate(candidates[j]):
                hs = np.flatnonzero(graph.toarray()[:,target])
                hs = hs[hs != source]
                for metric in (1,2):
                    expected = np.sort(prepared[metric,j,hs])[-3:].sum()/3
                    if metric == 2 and graph[source].nnz == 0:
                        expected = 0
                    self.assertAlmostEqual(float(got[metric+1,j,c]), float(expected), places=6)
        order = np.array([5,3,1,4,2,0])
        other = candidate_followups(prepared, graph, graph.tocsc(), local, candidates[:,order])
        np.testing.assert_array_equal(other, got[:,:,order])
        np.testing.assert_array_equal(got[:,:,1], got[:,:,5])
        np.testing.assert_array_equal(got[2:,0,3:5], np.zeros((2,2)))

    def test_one_vs_several_equal_strong_holders(self):
        graph = sparse.csr_matrix(np.array([[1,1], [0,1], [0,1], [0,0]], np.float32))
        prepared = np.full((3,1,4), .9, np.float32)
        out = candidate_followups(prepared, graph, graph.tocsc(), np.array([3]), np.array([[0,1]]))
        np.testing.assert_allclose(out[2,0], [.3,.9], atol=1e-7)

    def test_supported_feature_endpoints_and_half_control(self):
        model = np.array([[0,1,2],[2,1,0]], np.float32)
        retr = model[:,::-1].copy()
        np.testing.assert_array_equal(supported_feature(model,retr,0), model)
        np.testing.assert_array_equal(supported_feature(model,retr,1), retr)
        np.testing.assert_array_equal(HalfFeature(model,retr)[[1,0]], ((model+retr)*.5)[[1,0]])
        with self.assertRaises(ValueError):
            supported_feature(model,retr,1.1)
        with self.assertRaises(ValueError):
            supported_feature(model,retr,np.nan)

    def test_fit_selection_cannot_see_report_labels(self):
        rng = np.random.default_rng(5)
        base = rng.normal(size=(8,80,9)).astype(np.float32)
        weighted = rng.normal(size=(2,80,9)).astype(np.float32)
        gates = rng.random((2,80,9), dtype=np.float32)
        supported = np.stack([supported_feature(base[3],base[c],gates[j//2]) for j,c in enumerate((4,5,6,7))])
        fit, relation, direction = paired_masks(40), np.tile(np.repeat([0,1],20),2), np.repeat([0,1],40)
        family = np.array(["family"]*80)
        for arm in ("rarity","support","half_strength"):
            view = feature_views(base,weighted,supported)[arm]
            recipe = fit_recipe(view,fit,relation,family,direction,min_rows=8)
            poisoned = np.stack([feature[np.arange(80)] for feature in view])
            poisoned[:,~fit,0] += 100
            self.assertEqual(recipe,fit_recipe(poisoned,fit,relation,family,direction,min_rows=8))

    def test_end_to_end_frozen_state_list_metadata_and_duplicate_edges(self):
        base = ResonatE(8,2,k=2,block=True,block_size=2)
        model = from_student(dict(model=base.state_dict(),args=dict(k=2,block_size=2)),"single").eval().requires_grad_(False)
        train = dict(head=np.array([0,0,1,2,2,3]),tail=np.array([4,5,4,4,6,7]),relation=np.zeros(6,np.int64),
                     head_type=["entity"]*6,tail_type=["entity"]*6)
        valid = dict(head=np.array([0,1,7]),tail=np.array([6,5,6]),relation=np.zeros(3,np.int64),
                     head_type=["entity"]*3,tail_type=["entity"]*3,
                     head_neg=np.tile(np.arange(500)%8,(3,1)),tail_neg=np.tile(np.arange(500)%8,(3,1)))
        reference = np.empty((8,6,501),np.float32)
        baseline_features([model,model],train,valid,{"entity":0},8,2,reference,lambda **x:None,chunk=2)
        output = np.empty((4,6,501),np.float32)
        before = parameter_hashes(model)
        audit = build_features(model,train,valid,{"entity":0},8,2,output,reference,lambda **x:None,chunk=2)
        again = np.empty_like(output)
        repeated = {key:np.concatenate([np.asarray(value),np.asarray(value)[:1]]) for key,value in train.items()}
        repeated_audit = build_features(model,repeated,valid,{"entity":0},8,2,again,reference,lambda **x:None,chunk=2)
        np.testing.assert_array_equal(output,again)
        self.assertEqual(sum(row["duplicate_edges_removed"] for row in repeated_audit),2)
        self.assertEqual(parameter_hashes(model),before)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        self.assertTrue(all(row["candidate_permutation_passed"] for row in audit))
        # Independently check weighted holder pooling for every toy candidate.
        for d in (0,1):
            a,b = ("head","tail") if d == 0 else ("tail","head")
            graph = sparse.csr_matrix((np.ones(6,np.float32),(train[a],train[b])),shape=(8,8))
            sims = graph_similarities(graph,np.arange(8),rarity_weights(graph))
            for j,source in enumerate(valid[a]):
                candidates = np.r_[valid[b][j],valid[b+"_neg"][j]]
                for k,target in enumerate(candidates):
                    hs = np.flatnonzero(graph.toarray()[:,target])
                    hs = hs[hs != source]
                    scores = sims[source,hs] if graph[source].nnz else np.array([])
                    expected = [scores.max(),np.sort(scores)[-3:].mean()] if len(scores) else [-1,-1]
                    np.testing.assert_allclose(output[:2,d*3+j,k],expected,atol=1e-6)
        self.assertTrue(np.isfinite(rank_rows(output[0])).all())


if __name__ == "__main__":
    unittest.main()
