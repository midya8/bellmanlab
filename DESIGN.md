# Design and certificate contract

## Exact supplied model

The input is a finite discounted MDP: g in [0,1), one expected state/action reward r(s,a), and a successor distribution P(s'|s,a). Values are infinite-horizon discounted returns. Input rationals are explicit integers or numerator/denominator pairs and are normalized with Fraction. No binary-float conversion happens. Every transition row sums exactly to one. A zero-reward absorbing state represents termination without a special stopping flag.

Strict schema/index/row validation precedes planning. Scope caps are 16 states, eight actions per state, 64 actions and 512 stored transition entries, |r|<=100, raw rational integer components <=1e6, model-denominator LCM<=1e6 and JSON file<=256KiB. Initial-value denominators must preserve the combined LCM cap. Normalized rather than raw pair denominators enter the LCM. This bounds representational diversity as well as row counts; it is not a claim about RAM bytes or fixed-time rational arithmetic. Zero-probability entries remain in the canonical model and read accounting. Each state/action index is defined by its array position; successor rows are canonicalized by index. Inputs and outputs do not share mutable containers.

## Bellman contraction and the value-error certificate

For a vector v, Q_v(s,a)=r(s,a)+g*sum P(s'|s,a)*v(s'), and (Tv)(s)=max_a Q_v(s,a). Max is nonexpansive in the sup norm and each probability row is nonnegative with unit sum, so ||Tv-Tw||<=g||v-w||. The unique discounted fixed point v* satisfies Tv*=v*. The triangle inequality gives

    ||v-v*|| <= ||v-Tv|| + ||Tv-Tv*||
               <= residual + g||v-v*||

Rearrange using 1-g>0 to obtain `||v-v*|| <= residual/(1-g)`. BellmanLab evaluates the residual and division exactly. This is a deterministic value-function error bound for the supplied model, not a statistical interval or greedy-policy-loss bound. It may be loose. The one-state self-loop example makes it tight at every checkpoint, so the demo exposes misuse without relying on a loose certificate.

Value iteration uses synchronous vectors: every state backup reads the previous whole vector, never a partially updated sweep. It returns the initial audit, then one audit per completed sweep. The final residual is computed on the final returned vector. Tolerance gates the certified bound, not the raw residual or the last adjacent-vector difference. Zero sweeps still return an initial audit. A finite sweep cap can yield `converged=false`; no optimality is inferred from having used the budget. At most 256 backups plus 257 audits are allowed. In implementation the already-computed Bellman vector from an audit supplies the next synchronous vector, so each audit performs exactly one full model transition read pass; no extra hidden pass is charged for copying the vector.

The transition-read counter counts every stored successor entry of every action once per audit, even at zero discount or zero probability. Planned reads are (requested_sweeps+1)*stored_entries, an upper bound if early stopping occurs. Executed reads are number_of_returned_audits*stored_entries. This does not count validation, set work, arithmetic operations, integer bit widths, comparisons, serialization or any policy-evaluation work. No FLOP, speedup, wall-time or byte-memory comparison follows.

## Policy evaluation and improvement

For a deterministic stationary policy pi, solve `(I-g*P_pi)v = r_pi` exactly by Gauss-Jordan elimination. Discounting makes this matrix invertible: if (I-gP)x=0, then ||x||=||gPx||<=g||x|| forces x=0. Nonzero rational pivots, including row swaps, avoid a floating singularity tolerance. Recompute all policy Bellman equations and assert exact equality after solving.

Policy-equation residual zero proves correct evaluation of this policy, not its optimality. The separate optimality operator audit checks all actions. If its residual is zero, the policy's value is v*, though other optimal policies can exist.

Policy improvement retains the old action if it belongs to the exact greedy set. Otherwise choose the first maximizing action index. Thus any change is a strict Q-value improvement somewhere, not a gratuitous tie switch. The standard improvement argument gives the next evaluated policy a componentwise nondecreasing value; the implementation asserts this for successive returned evaluations. A stable policy has T_pi v = Tv = v and hence is optimal by uniqueness of the contraction fixed point. A cap of 64 evaluations is an operational bound, not a universal number of iterations needed for every supported MDP. If the cap is hit before stability, return the last evaluated policy/values and `optimal=false`. The proposed improved policy remains visible in that history row, but is not passed off as already evaluated.

Audits return every greedy tie and a deterministic first-index representative. Policy improvement instead retains an old greedy action. No unique-policy claim follows from either. The finite deterministic policy class contains an optimal policy for this discounted fully observed problem; arbitrary stochastic policy input is not supported by this version.

## Fixtures and independent references

The residual fixture is directly defined by reward 1/100 and discount 99/100. The true value is one; at v=0 the residual is 1/100 and actual/certified value error is one. Comparing either to a value tolerance 1/10 passes/fails respectively. At sweep k, the true error and certificate are (99/100)^k. The first accepted sweep is 230, while 229 fails. The demo does not search a parameter, seed or checkpoint to manufacture a result; it executes the specified tolerance rule. The SVG's six checkpoints are explicitly selected for inspection rather than an exhaustive plot; the full history remains in the API.

The delayed-return model has start-state action 0 paying 1 and moving to a zero absorbing state, versus action 1 paying 0 and moving to a recurrent state paying 1/5. At discount 9/10, these start values are 1 and(9/10)*(1/5)/(1-9/10)=9/5. This is a planning example under an identical supplied model, not a comparison of learning algorithms.

Tests enumerate all four deterministic policies on each of 243 two-state models (three fixed transition patterns, 3^4 signed reward choices), evaluating each through a separate two-by-two determinant/Cramer formula rather than the production Gaussian solver. Their per-state maximum is an independent reference for the optimal value. Tests check exact policy-iteration values, every returned intermediate policy evaluation, and every value-iteration checkpoint's error bound. Other references include the one-state geometric series and a16-state delayed deterministic chain. Tests reconstruct synchronous backups and all Q-values, retain exact tiny advantages, verify ties, policy-improvement monotonicity, cap returns, read counters, pivots, strict validation and isolation. Four SVG tests check the common linear geometry, full declared exact certificate subset, regeneration, and rejection if even an undisplayed history entry changes. Total 34 tests.

Dynamic programming and policy improvement follow [Sutton and Barto, Chapter 4 lecture slides](https://people.montefiore.uliege.be/lwh/AIA/Dynamic-programming-Sutton.pdf). The implementation, exact schema/certificate contract, fixtures, tests and figure are original. No external code or dataset is copied. Exact arithmetic is not model validation, and none of the results establish online learning performance, real-world control safety, partial-observability guarantees or an undiscounted theorem.
