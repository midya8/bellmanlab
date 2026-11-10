# Exact discounted occupancy with one expected-cost constraint

BellmanLab0.4 adds a separate workflow. All original ordinary/rectangular robust code, CLI and all six original test source files (70tests) remain unchanged. A budget does not turn the rectangular robust APIs into a constrained robust solver.

## What the model says

Input schema:

```json
{"version":1,"model":{"version":1,"discount":[1,2],"states":[{"actions":[{"reward":0,"transitions":[{"next":0,"prob":1}]},{"reward":2,"transitions":[{"next":0,"prob":1}]}]}]},"initial":[1],"costs":[[0,1]],"budget":[1,2]}
```

`model` is an existing ordinary version1model. `initial` is a nonnegative exact state distribution summing to one. Each cost row has exactly one signed rational cost per action. `budget` limits **normalized expected discounted cost**, with the same discount as reward. Input rationals are integers or `[integer,positive integer]` pairs, never floats/bools/strings. Existing reward/probability/transition restrictions still apply. New limits:1..4states, at most8actions total, absolute cost/budget≤100, all input numerators/denominators≤1,000,000, combined normalized denominator LCM≤1,000,000, rawJSON≤65,536bytes. Duplicate keys, decimal/exponent/nonfinite numbers and unknown keys fail. Stored zero-probability transitions remain in the canonical model. Output exact fraction integers can be much larger than input limits; BigInt serialization writes them as raw JSON integers, never rounded JS Numbers.

For state/action occupancy

`x(s,a) = (1 - γ) Σ[t≥0] γ^t Pr(S_t=s,A_t=a)`,

flow conservation is

`Σ_a x(s,a) - γ Σ_(z,a) P(s|z,a)x(z,a) = (1 - γ)μ(s)`.

Summing all state equations gives `(1-γ)Σx=1-γ`, hence `Σx=1`. Normalized reward is `R=Σrx`, cost is `C=Σcx`, and the constraint is `C≤B`. Unnormalized expected discounted reward/cost/budget are `R/(1-γ)`, `C/(1-γ)`, `B/(1-γ)`. Do not insert the normalized budget into an unnormalized LP. This is an expectation under the supplied initial distribution, not a per-step, per-trajectory, chance, worst-case or physical-safety constraint. A signed cost is not automatically a probability.

## Two original controls

One state, self loops, discount1/2, actions(reward,cost)=(0,0),(2,1), budget1/2. The best feasible deterministic reward is0. A half/half stationary policy has normalized cost1/2 and reward1. A correct Lagrange multiplierλ2 makes both actions tie, but picking either deterministic tie does not produce the optimal budget-feasible policy: one loses reward, the other exceeds the budget. This is not a failure of LP/Lagrange duality. It is a failure to recover a feasible primal mixture from ties.

The default has initial state0 and discount1/2. At state0, wait stays, exit goes to state1; both pay/cost0. State1 pays reward2/cost1 and absorbs. Deterministic wait and exit have normalized(cost,reward)=(0,0),(1/2,1). Their equal initial mixture gives `(1/4,1/2)` with occupancies `[1/2,1/4,1/4]`. State0 mass is3/4, so equivalent stationary exit probability is `(1/4)/(3/4)=1/3`, not1/2. A directly averaged half-exit local policy has state1 mass1/3, cost1/3 and violates budget1/4. The test solves those alternative flow equations too. We do not equate initial policy mixing with per-state probability averaging.

Other controls: unreachable states, discount0, truly infeasible budget, signed costs/negative reward, slack constraint, a whole optimal face, duplicate projected policy points, a stochastic polygon, a four-state model and singleton boundary. These are explicitly supplied teaching models, not selected benchmark scores, learned environments or real-world tasks.

## Generator and exact certificates

`bellmanlab.occupancy.occupancy(problem)` returns schema`bellmanlab-occupancy-v1`. Enumerate every deterministic stationary policy, at most18under the stated action/state limits. Solve its adjoint flow by exact rational Gauss-Jordan. Keep all full occupancies and projected(cost,reward) points, including duplicates/interior points. Monotone-chain convex hull constructs the complete attainable **two-dimensional projection**, not all full occupancy-polytope vertices. Clip its boundary against`cost≤budget` using exact comparisons. Degenerate point/line cases are real cases, not artificial areas.

Each projected hull/feasible-hull vertex carries a deterministic policy or a canonical two-policy occupancy mixture. Select maximum reward, then minimum cost, then the hull index. Return all optimal feasible **polygon vertices**, not all optimal occupancies or policies. A continuum can join tied vertices. Keep the first lexicographic deterministic representative for identical points; otherwise choose the first policy-index pair whose open segment contains the point. Hull order starts at lexicographically minimum(cost,reward), counterclockwise, excluding collinear interior points. Closed budget intersections and coincident points are deduplicated exactly.

For chosen occupancy, set`d(s)=Σ_a x(s,a)` and`π(a|s)=x(s,a)/d(s)`. If d=0, choose first action deterministically and list the state in`zero_mass_states`; this fallback is arbitrary, not evidence the action is globally optimal from that state. Re-solve the extracted stationary flow and return`replay_occupancy`, plus every initial/incoming/outgoing/residual row. Discount<1 makes the fixed-policy flow solve unique, even when some initial mass is zero.

### Reward upper bound

Choose the smallest nonnegative candidate supporting slopeλ from hull pair slopes and0, with supporting bound equal to selected optimum. For adjusted reward`r-λc`, evaluate every deterministic policy and take the componentwise maximum value`v`. For every state/action:

`slack(s,a)=v(s)-r(s,a)+λc(s,a)-γP(s,a)·v≥0`.

Multiplying by feasible x and using flow gives

`R≤λC+(1-γ)μ·v≤λB+(1-γ)μ·v=D`.

The certificate includes every action slack,`x×slack`,budget slack,`λ×budget slack`,D and exact gap`D-R`. Primal feasibility and zero exact gap are enough to establish optimality. We do not call an unverified multiplier, a small numerical tolerance or reward minus a guessed penalty a certificate.

### True infeasibility

Independently evaluate minimum-cost policies to get`u` and every inequality

`u(s)≤c(s,a)+γP(s,a)·u`.

Their normalized initial value is the attainable minimum cost. If this lower bound exceeds B, no policy is feasible. Return empty feasible hull,`selected=null`,no fake extracted plan, and positive minimum-cost budget gap. Feasible reports retain the minimum-cost certificate as another check.

## Separate checker and references

`bellmanlab.occupancy_check.verify_occupancy(report)` reconstructs the entire report and requires strict keys, primitive types, canonical fractions and array order. It imports only input compilation/canonicalization and fraction packing. It imports no generator policy enumeration, Gaussian solve, hull, clip, mixture, value or assembly helper. It uses determinant permutation expansions and Cramer solves, a supporting-line gift-wrap hull, all deterministic-pair budget crossings and its own full certificate construction. Independence tests patch the generator helpers to raise. Both implementations necessarily share input schema, classic theory and canonical tie rules; this is not a machine-checked theorem proof or a third-party audit.

Additional`flow_vertices` oracle builds the full state/action LP directly. It enumerates supports and square row minors, checks all equality rows, nonnegative variables and budget inequality, with active and inactive budget cases. It never calls deterministic-policy enumeration or a projected hull. Tests compare its optimum over28controls/seededmodels, including four states, degeneracy, unreachable states and infeasibility. This oracle returns full occupancy vertices; it is deliberately not mislabeled as the2Dprojection.

The optional actual SciPy1.15.3 reference calls`scipy.optimize.linprog(method="highs")` on all state/action variables, every normalized balance equality, nonnegative bounds and cost upper bound. It also separately minimizes cost. 861cases=12controls+120seededmodels(seed91843)+729exhaustive one-state parameter combinations. Actual1,722LPsolves:684feasible,177infeasible constrained problems, observed worst reward-objective error`7.993605777301127e-15`, with explicit absolute1e-8objective/flow/cost/nonnegativity checks. This is numerical agreement on these inputs, not exact proof for every supported rational, a scaling benchmark or a solver-version ranking. No deterministic-only or soft-penalty surrogate is substituted. SciPy is optional and not a runtime dependency.

Browser uses a separate BigInt engine with exact fraction operations, policy-flow/value solves and gift-wrap/all-pair geometry.861complete reports equal native Python exactly after raw JSON transfer and pass the determinant checker:2,418deterministic rows,1,913hull vertices,1,323feasible vertices,816selected flow rows and1,630selected dual action slacks. Output integers beyond2^53are explicitly tested. Inputs are checked by the native verifier before browser-supplied JSON is trusted for other uses.

## Reproduction

```sh
python -m unittest discover -s tests -v
node --test tests/test_occupancy_engine.cjs tests/test_occupancy_browser.cjs
python tools/check_occupancy_installed.py
python -m bellmanlab.occupancy_cli demo --control visitation_reweighting
python -m bellmanlab.occupancy_cli run input.json > certificate.json
python -m bellmanlab.occupancy_cli verify certificate.json
bellmanlab-occupancy demo --control randomization_gap
python -m pip install scipy==1.15.3
python tools/check_occupancy_reference.py
```

The existing`bellmanlab`console/module CLI is unchanged. Separate`bellmanlab-occupancy`and`python -m bellmanlab.occupancy_cli`entrypoints share this new workflow. Wheel checks use a fresh isolated install outside checkout, assert seven module origins, reproduce all three prior demo families through both old entrypoints, all12newcontrols through both new entrypoints, and reject48CLI/24direct forgeries. Default Python package remains dependency-free. Browser tests need Node22+and actual Chrome, not a DOM emulator; CDP drives trusted mouse downloads, awaits completed disk files, verifies full bytes and independently checks certificates.

## Source ledger and formulation differences

All sources read November10 2026. No external implementation code or dataset is copied. Code, controls, certificate schema, checker and UI are original. Classical occupancy/LP results are not new.

1. [Altman, Constrained Markov Decision Processes,1999,author-hosted preview](https://www-sop.inria.fr/members/Eitan.Altman/PAPERS/h.pdf). Primary monograph. Actually inspected introduction and equations1.2-1.7, not claimed fullChapter3. Occupation LP, initial mixing, Lagrange formulation, distinction expected/sample-path cost. No assertion that its average/total/infinite-state results all follow here.
2. [Mahajan, Linear programming formulation,universitynotes](https://adityam.github.io/stochastic-control/mdps/linear-programming.html). Returnedpage June23 2026. Basic solutions, stationary extraction and constrained LP. **Literal normalized definition and nearby unnormalized LP RHS differ.** Our derivation above uses consistent `(1-γ)μ`; the cost-minimization orientation is converted to reward maximization. Some displayed sums in that note are incomplete, not copied.
3. [Nan Jiang,CS542Note1,section2.3](http://nanjiang.cs.illinois.edu/files/cs542/note1.pdf). Academic notes. Normalized state/action occupancy and complete flow expansion, transpose orientation. Date not visible. Fully observed discounted setup only.
4. [Matt Landers,Constrained MDPs](https://mattlanders.net/constrained-mdps.html). Independent tutorial, revised August13 2025. Normalized flow proof, cost scaling and zero-mass arbitrary fallback. We use its occupancy derivation, not claims about general policy-parameter nonconvex optimization or soft-penalty safety.
5. [SciPy linprog official docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html). Software API contract. Retrieved documentation says1.18.0; **actual reproducible reference is1.15.3**, stated separately. HiGHS status/constraints/results inspected; no solver implementation copied.
6. [Ying and Zhu,A note on optimization formulations of MDPs,arXiv2012.09417](https://ar5iv.labs.arxiv.org/html/2012.09417). Primary paper, discounted-standard sections2.1/2.2. Bellman inequalities, occupation dual and complementarity. Their positive e and unnormalized formulation differ; this contract permits zero initial mass and proves its bound directly. No borrowing entropy/average-case claims.
7. [Routledge,Altman bookpage](https://www.routledge.com/Constrained-Markov-Decision-Processes/Altman/p/book/9780849303821). Official bibliography/description only, not theorem proof.

A supplementary author TEMP PDF fetch returned no content; not relied on. The ledger deliberately distinguishes fetched preview from unavailable full chapter. No need to claim stronger source access than occurred.

## Limits

Finite fully observed supplied dynamics, single fixed discount<1, one normalized expected-cost upper bound. No learned model, model API, samples, partial observability, rectangular/coupled uncertainty combination, average-reward extension, multiple budgets, risk-sensitive/per-trajectory safety, scalable implementation, uniqueness guarantee or new learning algorithm. Up to18policies makes enumeration inspectable, not efficient for large MDPs. Rational arithmetic has variable bit cost. Exact optimization does not validate a model's connection to reality.
