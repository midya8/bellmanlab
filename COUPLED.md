# Two simultaneous expected-cost budgets

BellmanLab0.5 adds a separate `coupled` workflow. Ordinary planning, rectangular robust planning and the one-budget occupancy API remain separate and unchanged. This is exact finite supplied-model optimization, not a learned controller or a hard safety guarantee.

## Input and scaling

```json
{"version":1,"model":{"version":1,"discount":[1,2],"states":[{"actions":[{"reward":0,"transitions":[{"next":0,"prob":1}]},{"reward":3,"transitions":[{"next":0,"prob":1}]},{"reward":2,"transitions":[{"next":0,"prob":1}]}]}]},"initial":[1],"costs":[[[0,1,0]],[[0,0,1]]],"budgets":[[1,3],[1,4]]}
```

Exactly two cost tables, each with one cost per action, and two upper budgets. The existing ordinary model schema supplies a common discount `0<=gamma<1`, all rewards and all transitions. Initial probabilities are nonnegative and sum exactly to1. Limits:1..4states, at most8actions total, signed cost/budget magnitude<=100; rational integers or `[numerator,positive denominator]` input only. Numerators/denominators are bounded at1,000,000 and the combined normalized denominator LCM across the entire input is at most1,000,000. Zero-probability transitions remain canonical. Booleans, decimal/exponent/nonfinite tokens, duplicate/unknown keys, malformed dimensions and larger models fail. CLI run input is limited to65,536bytes; verification reports to2,097,152bytes. Failure prints no partial JSON to stdout. No output file is opened or overwritten.

The normalized state/action occupancy is

`x(s,a)=(1-gamma) sum[t>=0] gamma^t Pr(S_t=s,A_t=a)`.

Let `A[t,(s,a)]=1{t=s}-gamma P(t|s,a)` and `b=(1-gamma)mu`. Feasible occupancies satisfy

`Ax=b, x>=0, C0 x<=B0, C1 x<=B1`.

Summing the flow equations yields `sum x=1`. Reward is `r·x`; costs are `Ck·x`. Unnormalized expected discounted reward, each cost and each budget are divided by `1-gamma`, exactly once. A budget is a normalized expectation under the supplied initial distribution, not a per-step, per-trajectory, chance, almost-sure, worst-case or physical-safety cap. A signed cost need not be a probability. All three criteria use the same discount. No robust-uncertainty combination is implemented.

## Three original distinctions

### A pair of policies can be insufficient

The one-state input above has deterministic `(reward,cost0,cost1)` rows `(0,0,0),(3,1,0),(2,0,1)`. At budgets`1/3,1/4`, the unique optimum has occupancies`5/12,1/3,1/4` and reward`3/2`. Every action has positive mass. A mixture of only two deterministic policies cannot attain it. The only feasible deterministic policy has reward0. Both dual prices are positive, `lambda=(3,2)`, and make all three adjusted rewards tie at0. A tie-breaking deterministic action is not a recovered feasible optimum.

### Separately feasible can be jointly impossible

Two actions have costs`(1,0)`and`(0,1)`. Each budget`1/3` is attainable alone. No mixture meets both. Weights`(1/2,1/2)` give every policy a normalized weighted cost of`1/2`, but the two budgets would imply a weighted cost at most`1/3`. The exact separating gap is`1/6`. A failed deterministic search is not used as evidence of randomized infeasibility.

### Episode mixing is not local-row averaging

The three-state wait/route control has deterministic episode-mixture weights`4/15,2/5,1/3`. Their full occupancy has state masses`19/30,1/5,1/6`. The equivalent local probabilities at the decision state are`8/19,6/19,5/19`, not those episode weights. Substituting episode weights as local probabilities and re-solving flow violates both budgets. This alternative flow is actually solved in the tests.

Other supplied controls cover stochastic routes, a never-visited fourth state, discount0, redundant budgets, one/both slack constraints, signed costs/negative ceilings, a whole tied face and identical return projections with distinct full occupancies. The duplicate-return control has6feasible full vertices but only4distinct `(cost0,cost1,reward)` projections and2optimal full vertices. Teaching inputs are original, not real-world tasks or selected benchmark scores.

## Complete feasible occupancy vertices

The generator adds two nonnegative slack variables and enumerates every square basis of the augmented equalities `[A 0; C I]`, using exact rational Gauss-Jordan. It checks nonnegativity, deduplicates the entire action occupancy vector, and sorts lexicographically. `A` has full row rank: a left null vector would satisfy `v=gamma P_policy v` under any deterministic policy, impossible except0 when gamma<1. The two independent slack columns then give augmented row rank`n+2`. Discounted occupancies sum to1, so the feasible action polytope is bounded. Every feasible vertex has a basis representation, including degenerate vertices with zero basic entries.

These are all vertices of the **budget-cut full action-occupancy polytope**, not merely its two- or three-dimensional projection. Different full occupancies with the same return projection remain separate. `optimal_vertices` lists all vertices attaining maximum reward. Their convex hull is the full optimal face; the list does not enumerate its continuum of occupancies or all policies. Select maximum reward, then lexicographically minimum `(cost0,cost1,index)`. Sorting affects presentation/ties, not the objective.

A selected vertex has a complete full-occupancy witness using at most3deterministic policies. The generator chooses the first smallest-size lexicographic policy-index subset with positive weights that reproduces **every action occupancy** and sums to1, not merely its three returns. Local policy extraction uses `pi(a|s)=x(s,a)/sum_a x(s,a)`. Zero-mass states use an explicitly labeled arbitrary first action. That action is not claimed optimal from an unvisited state. A separate stationary flow solve replays the extracted policy and returns every initial/incoming/outgoing/residual row.

## Exact upper and lower bounds

For a feasible plan choose `lambda0,lambda1>=0` and state potentials`v` satisfying every action inequality

`v(s)-r(s,a)+lambda·c(s,a)-gamma P(s,a)·v>=0`.

Multiplying by a feasible occupancy gives

`reward<=b·v+lambda·cost<=b·v+lambda·budgets=D`.

The report returns all action slacks, occupied-action slack products, both budget slacks and multiplier/slack products, D and exact gap D-reward. All products and gap are0 at the selected optimum. Feasibility and an equal upper bound certify optimality without a floating tolerance. The generator enumerates tight dual bases, restricted by the chosen primal complementarity equalities, chooses minimum `(sum lambda,lambda0,lambda1)` among those multiplier pairs, then independently evaluates every adjusted deterministic policy to obtain the componentwise maximum value vector. This is a finite witness selection rule, not a general sensitivity interval or uniqueness claim.

For infeasibility, enumerate switch points of `min_policy lambda·cost_policy` on `lambda0+lambda1=1,lambda>=0`, including the axes. A positive separating gap gives a pair of weights. Independently evaluated minimum weighted-cost potentials satisfy

`u(s)<=lambda·c(s,a)+gamma P(s,a)·u`.

Thus every policy's weighted cost is at least `b·u`. If that lower bound exceeds `lambda·budgets`, the joint feasible set is empty. The report returns the weights, all lower-bound action slacks, the lower bound, budget ceiling and positive gap. `selected=null`, no fake plan. The generator chooses maximum gap, then lexicographic weights, among these switch/axis candidates. This is a proof witness, not a per-budget minimum or trajectory-risk estimate.

## Separate checker and numerical reference

`coupled_check.verify_coupled` imports only the new input compiler/canonicalizer and rational packing. It imports no generator flow, basis, dot product, policy enumeration, solve, value, mixture, dual, separator or report-assembly helper. Generator-helper patch tests raise if those helpers are called. It enumerates every deterministic policy independently, solves its flow with determinant permutation expansions/Cramer's rule, and intersects every1-,2-,3-policy simplex with both budgets. It keeps only full vertices using nonzero active-restricted column minors. Intersecting a polytope with two halfspaces can create a vertex only on an original face of dimension at most2; each such point belongs to a triangle of original deterministic occupancy vertices. Repeated/unreachable policy rows and degenerate constraints are checked, not assumed distinct.

The checker independently reconstructs every full vertex, all return/optimal rows, selected full mixture, extracted policy, replay, balance and scaling field. It recomputes every dual/separator dependent field and checks nonnegative multipliers/slacks and an equal reward upper bound or strictly positive infeasibility gap. **A valid alternative dual or separator multiplier is accepted.** Those witnesses are not canonical facts; we do not claim the checker rejects a different valid certificate. Fractions, primitive types, keys, lengths and array order are strict. Shared schema, theory and canonical presentation rules remain common assumptions. This is not a machine-checked proof, external audit or independent model validation.

Checks actually run on the private tree:

-103Python tests, including15new tests,80seed20957models,729exhaustive one-state two-cost tables,12controls, bridge comparisons with the existing one-budget solver, and958separately changed noninput primitive leaves rejected across4controls.
-8Node tests across old/new engines and actual Chrome.822complete native/BigInt reports agree and pass the separate exact checker:12controls+80seeded+729exhaustive+1four-state near-unit-discount full-bound input;2,801deterministic rows and1,926full feasible vertices. The last case contains exact output integers larger than2^53; transfer uses raw integer JSON, not rounded JavaScript Numbers.
-Optional SciPy1.15.3 actually invokes`linprog(method="highs")` with every full state/action variable, both cost inequalities, all normalized flow equalities and nonnegative bounds.822LPsolves:692feasible,130infeasible. Explicit absolute1e-8objective/flow/cost/nonnegativity checks; observed worst reward-objective difference`4.407954001806047e-10`. Retrieved API docs say1.18.0, not the installed version. Numerical agreement on these inputs is not exact proof for all supported rationals, a scaling benchmark, or a solver ranking.
-Actual Chrome at1280/390/320 checks36control reports,108deterministic selections,117full vertex selections,54flow rows,123action-proof rows,66budget-proof rows, plus3dense bound-scale reports.12trusted completed certificate downloads replay through the native checker, editable input invalidates stale output/download, trusted valid/invalid Run, contained horizontal tables/plot and44pxcontrols. Long fractions wrap in proof panels. Screenshot preparation scrolls each page to paint offscreen layers before unmasked full-page capture; an earlier capture's blank offscreen compositor tiles were caught by pixel inspection and corrected in capture readiness, not hidden by masks or ignored assertions.
-Fresh isolated wheel outside checkout:11module origins under site-packages;12controls via module+console entrypoints,24direct/48CLIforgery rejections. The separate older installed-wheel check also reproduces3previous demo families and all12one-budget controls through both old entrypoints, with24direct/48CLIforgery rejections.

These are pre-publication results, not deployment verification. Main CI, Pages, exact public-source/asset readbacks and deployed behavior/pixels are separate release gates. Source and showcase publication must not claim those gates before they pass.

## Reproduce

```sh
python -m unittest discover -s tests -v
node --test tests/test_occupancy_engine.cjs tests/test_occupancy_browser.cjs tests/test_coupled_engine.cjs tests/test_coupled_browser.cjs
python tools/check_coupled_installed.py
python tools/check_occupancy_installed.py
python -m bellmanlab.coupled_cli demo --control three_way
python -m bellmanlab.coupled_cli run input.json > report.json
python -m bellmanlab.coupled_cli verify report.json
bellmanlab-coupled demo --control jointly_impossible
python -m pip install scipy==1.15.3
python tools/check_coupled_reference.py
```

Default package is dependency-free Python3.10+. SciPy is optional reference-only. Node22+ and actual Chrome support browser tests, not a DOM emulator. Exact output integers may be much larger than input integers; the browser writes raw BigInt JSON. Native verification is required before treating browser JSON as an accepted certificate elsewhere. Proof rational integers are capped at4,096bits in the checker.

## Inspected source ledger

Read November10 2026. Original implementation, controls, schema and UI; no external implementation code or dataset copied. Classical theory is not claimed new.

1. [Altman, Constrained Markov Decision Processes, author-hosted1999monograph preview](https://www-sop.inria.fr/members/Eitan.Altman/PAPERS/h.pdf). Primary theory; inspected introduction1.3/1.7 and normalized occupation equations1.1-1.2, not every chapter. Multiple expected costs, occupation polytope, initial policy mixing, local extraction and Lagrangian duality. Sample-path constraints are distinguished, not borrowed as a guarantee.
2. [Mahajan, Linear programming formulation](https://adityam.github.io/stochastic-control/mdps/linear-programming.html). University notes, returned June23 2026. Flow/extraction/basic solutions and constrained LP. Its normalized definition and adjacent unnormalized RHS differ. This workflow consistently derives `(1-gamma)mu`, and converts cost minimization to reward maximization. No incomplete displayed sum is copied literally.
3. [Roughgarden, Stanford Lecture9, February2 2016](http://theory.stanford.edu/~tim/w16/l/l9.pdf). Educational primary lecture; inspected weak duality, complementary slackness and equal feasible objectives. Used for exact witness inequalities, not a claim that the source verifies our code.
4. [Stanford CME307, LP geometry/modeling/solution techniques, Fall2025](https://stanford-cme-307.github.io/web/lectures/2025fall/lp.pdf). Educational lecture; inspected vertex enumeration and its explicit practical limits. The implementation is bounded enumeration, not an efficient large-model optimizer.
5. [SciPy official linprog HiGHS docs](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html). Software contract; retrieved1.18.0docs versus actually installed1.15.3reference. Continuous bounds, statuses0/2, equality/cost residuals and opposite multiplier sign convention inspected. No solver implementation copied.
6. [Achiam et al., Constrained Policy Optimization,2017](https://publications.ri.cmu.edu/storage/publications/2017/11/1705_10528.pdf). Primary ML paper; inspected introduction and conclusion. CMDP constraints are expectations; high-dimensional approximate learned policy search is a different problem. This project does not implement CPO or transfer its training guarantees.
7. [Python fractions documentation](https://docs.python.org/3/library/fractions.html). Official language documentation; returned3.15.2. Normalization and binary-float mismatch inspected; runtime uses Python3.10+ integer/pair Fraction operations, no newer-only API.

## Non-goals

No learned model, data fitting, model API, samples, partial observability, unequal discounts, more than two budgets, risk-sensitive/sample-path control, robust-uncertainty coupling, average reward, online learning, real-world safety, new algorithm/theory, scalable implementation or unique-policy guarantee. Enumeration is exponential and exact arithmetic has variable bit cost. Correct finite optimization does not validate the supplied world model.
