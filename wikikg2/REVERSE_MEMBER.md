# RV1: the opposite-direction operator as a member (registered 2026-09-08 01:30, user approved)

Idea (user): a head question "?, r, t" is answered with the reverse operator;
ask the forward operator instead, for every candidate c, how strongly
"c, r → t" holds, i.e. treat each candidate as the source of a novel link
to the known entity. The two directions have separate learned operators
(535 forward, 535 reverse), so they are different scorers. On biokg the
same idea (RS1, 2026-09-07, drug–drug only) gained +0.0003, under its
+0.0005 gate; there the two directions score alike. On wikikg2 the head
direction is 0.48 and the forward direction 0.96, which is the case for it.

## Members (`reverse_wiki.py`, one frozen released model, no training)

For a question with known entity q, operator op and candidates x_0 (the
answer) … x_500 (OGB's decoys), let op' be the opposite operator (reverse
for tail questions, forward for head questions), z_x = out(hop(embed(x), op'), op'),
and S[x, y] = Re⟨z_x, row(y)⟩·τ over targets y ∈ {q} ∪ {x_1 … x_500}.
* `rev_raw`: S[x, q], the opposite operator's score of the link x → q.
* `rev_nov`: S[x, q] − logsumexp_y S[x, y], the same score normalised per
  candidate against the 500 random alternative targets: how much x prefers
  q over other links ("novel link" reading).
Both members are written in the standard cache layout and z-scored per row
like every other member. Self-check: for the head block, rev_raw at the true
head equals the model's own tail-direction score of the same triple.

## Test (validation only; no test cache is written)

Selection blend (`blend_wiki.py search`, seed 0 halves, guard 250, the
allowed tuning): the seven HC members vs. the same plus rev_raw, plus
rev_nov, plus both; student s1 and teacher s0. Bar, fixed now: held-out MRR
must rise by ≥ +0.002 with a reverse member in, and the head direction must
not fall. The validation-fit learned combiner with and without the members
is reported for information only (it is not filed). If the bar is met, the
member joins the selection-blend filing candidates (one test read per
released seed, under a separate registration). If not, it is recorded and
dropped.

## RV1 RESULT (2026-09-08 02:00, box 50209059): PASS on both models, both members

Members alone on validation (student s1 / teacher s0): rev_raw 0.6088 / 0.5566
(tail 0.775 / 0.666, head 0.443 / 0.448), rev_nov 0.3979 / 0.3827. Self-check
exact to 1e-5 on both models (`results/rv1/reverse_*.log`).

Selection blend, held-out half (seed 0, guard 250), MRR and head direction:

| members | student | student head | teacher | teacher head |
|---|---|---|---|---|
| seven | 0.7542 | 0.542 | 0.7450 | 0.529 |
| + rev_raw | 0.7665 | 0.566 | 0.7615 | 0.561 |
| + rev_nov | 0.7589 | 0.547 | 0.7496 | 0.534 |
| + both | **0.7711** | 0.571 | **0.7656** | 0.565 |

Bar (+0.002 held-out, head not falling): met by every set on both models;
the tail direction is unchanged (0.966 → 0.967). For information only, the
validation-fit learned combiner moves 0.7826 → 0.7903 (student) and
0.7740 → 0.7840 (teacher) with both members; not filed.
(`learned_blend.py` also prints a test-relation-mix-weighted column; it is
the disclosed development signal and played no part here.)

Decision: rev_raw and rev_nov join the member set. The filable row is
"T=2 student + nine members + selection blend" (allowed tuning), to be read
on test once per released student under a separate registration; a
validation-to-test gap of ~0.06 on blended rows puts it near 0.71 on test.
Receipts: `results/rv1/`.
