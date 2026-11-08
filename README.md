# BellmanLab

Exact rational planning for small supplied discounted Markov decision processes. Dependency-free Python 3.10+, strict JSON CLI, full iteration histories and independently tested certificates. This is model-based planning, not learning an environment from samples.

![Bellman residual, certified value error and actual value error on the same scale](assets/bellman-audit.svg)

## A small residual can hide a large value error

```sh
python -m bellmanlab demo
python -m unittest discover -s tests -v
PYTHONPATH=. python examples/render_audit.py --check
```

The directly specified one-state model has reward 1/100, discount 99/100 and a self-transition with probability one. Its exact discounted value is `(1/100)/(1-99/100) = 1`. Starting from value zero:

| Quantity | Exact value | Within value-error tolerance 1/10? |
|---|---:|---|
| Raw Bellman residual | 1/100 | Yes, but this is the wrong quantity to gate |
| Certified value-error upper bound | 1 | No |
| Actual value error, independently known | 1 | No |

For the optimality operator T, the contraction certificate is `||v-v*||_infinity <= ||Tv-v||_infinity/(1-discount)`. A residual is not an interchangeable value-error tolerance. BellmanLab gates the bound. With synchronous backups, 230 sweeps bring that certified error below 1/10; the initial audit and all 230 subsequent audits are returned. The demo asserts the initially passing raw-residual test, failing certified-error test and eventually passing certified-error test. This constructed example is not a convergence-rate benchmark or a statistical confidence interval.

A separate three-state model contrasts immediate reward with delayed return. At state zero, action zero pays 1 then moves to a zero-reward absorbing state. Action one pays 0 then reaches a state paying 1/5 forever. With discount 9/10, the immediate policy has start value 1, while exact policy iteration chooses the delayed action and returns 9/5. Both use the same complete supplied model. The fixture does not measure sample efficiency or generalize to an unknown environment.

## What is implemented

- `audit(model, values)`: exact action values, Bellman backups, all greedy ties, sup residual and certified value-error bound.
- `evaluate_policy(model, policy)`: exact rational solution of `(I-discount*P_policy)v = reward_policy`, full action values and a separate Bellman-optimality audit. Policy-equation residual zero alone does not mean the policy is optimal.
- `value_iteration(model, initial=None, sweeps=100, tolerance=[1,1000])`: synchronous full backups and complete audit history; stops only when the certified value-error bound meets the tolerance or the sweep cap is reached. A cap is not reported as convergence.
- `policy_iteration(model, initial=None, iterations=64)`: exact policy evaluation, strict greedy improvement and complete policy/value/action traces. Old actions are retained on exact ties. A stable policy is optimal; hitting the cap returns the last **evaluated** policy with `optimal=false`, not an unevaluated proposed improvement.

Scores and probabilities are exact fractions of the **explicit input rationals**, not rounded floats. No numerical equality tolerance is used. All ties are returned; the first tied action is the audit's deterministic greedy representative, while policy improvement keeps an already-greedy old action. Neither behavior is a unique-policy claim. Output fractions are `[numerator, denominator]` arrays in lowest terms.

## Input and CLI

A version-1 model has an exact discount in [0,1), and indexed states with one or more actions. An action has one expected immediate reward and a sparse successor-probability row. The reward is state/action based, not a separate transition-dependent reward. A terminal outcome is represented by an ordinary zero-reward absorbing state.

```json
{
  "version": 1,
  "discount": [99, 100],
  "states": [
    {"actions": [
      {"reward": [1, 100], "transitions": [{"next": 0, "prob": 1}]}
    ]}
  ]
}
```

Every input rational is an integer or `[integer numerator, positive integer denominator]`; floats, bools and numeric strings are rejected. Each transition row must sum **exactly** to one. Duplicate successors, invalid indices, unknown keys and duplicate JSON object keys are errors. Zero-probability stored transitions are permitted and counted, not silently removed. State and action order define their indices.

```sh
python -m bellmanlab audit examples/residual.json --values '[0]'
python -m bellmanlab evaluate examples/delayed.json --policy '[0,0,0]'
python -m bellmanlab value examples/residual.json --sweeps 256 --tolerance '[1,10]'
python -m bellmanlab policy examples/delayed.json
python -m pip install .
bellmanlab policy examples/delayed.json
```

Limits: 1..16 states, 1..8 actions per state, at most 64 actions and 512 stored transition entries in total, absolute rewards <=100, integer input numerators and denominators <=1,000,000 in magnitude (denominators positive). The least common multiple of normalized model denominators must be <=1,000,000; initial values must preserve that combined denominator budget. Value iteration allows 0..256 sweeps; policy iteration allows 1..64 evaluations. JSON model files are limited to 256 KiB. These are operational scope limits, not runtime or memory-byte guarantees. Rational arithmetic has variable integer bit cost. Full histories deliberately retain earlier audits.

Value iteration counts model transition-entry reads for every full Bellman audit, including the initial and final audit. Its planned count is an upper bound under the requested sweep cap; the executed count reflects early stopping. These counts exclude validation, Fraction arithmetic/bit cost, comparisons, serialization, policy evaluation and Python-object overhead. They are not FLOPs, wall time or a comparison against an implementation that skips final residual verification.

## Why the checks matter

34 tests include 243 fixed two-state stochastic models, each with all four deterministic policies evaluated by an independent closed-form two-by-two Cramer-rule oracle: 972 exact policy evaluations. Policy iteration matches the per-state optimal envelope over those policies. Every returned value-iteration checkpoint is checked against that independently computed optimum and its certified bound. Other tests reconstruct each synchronous backup, full action values and tie sets; verify closed-form near-one-discount values and a 16-state delayed chain; cover zero discount, negative rewards, exact tiny advantages, retained ties, monotone policy values, cap semantics, final-audit read counts, nonzero linear pivots, strict validation, denominator/action/transition budgets, model/output/RNG isolation, JSON round trips and all CLI commands. Four figure tests inspect every bar and tolerance coordinate, exact selected certificate metadata, byte regeneration and rejection of a changed undisplayed history entry.

The SVG embeds complete exact initial/final certificates, six explicitly selected history checkpoints (0,1,10,100,229,230) and the complete delayed-policy trace. The full 231-audit value history remains in the demo API/CLI. No rational certificate is rounded for a stopping decision. The figure uses one linear scale and a shared zero, not separate scales that make the residual look comparable to the error.

Dynamic programming, policy evaluation/improvement and value iteration follow [Sutton and Barto, Chapter 4 lecture slides](https://people.montefiore.uliege.be/lwh/AIA/Dynamic-programming-Sutton.pdf). Code, rational contracts, fixtures, independent test oracles and figure are original. The residual proof and cap semantics are in [DESIGN.md](DESIGN.md). MIT licensed. No network, API key, paid service, model download, GPU or external dataset is needed.

Only finite, fully observed, discounted, fully specified models are supported. No learned transition/reward model, online interaction, stochastic-policy search, partial observability, undiscounted/average-reward guarantee, statistical confidence or certified greedy-policy-loss claim is made. Exact arithmetic does not validate whether the supplied model represents the world.
