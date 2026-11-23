# Discounted SurvivalRisk for online container placement

## 1. Approach and rationale

For each arrival, minimize the sum of discounted probabilities that residents below it depart first. This balances stack height against uncertain departure order, including the risk that the arrival misses its next vessel. The frozen policy uses one scheduled departure jump, an exponential background exit hazard, a 240-hour exponential discount, and deterministic `(score, height, block, bay, row)` tie-breaking. Empty stacks have zero cost. All non-full stacks are considered; every resident in a stack contributes.

Assign each scheduled container to its vessel's earliest rotation with `load_end >= now` (S3). Interpolate its jump time within that loading window using lexicographic port order and HEAVY, MEDIUM, LIGHT weight order: fraction `(3*port_rank + weight_rank + 0.5)/(3*port_count)`. Unknown vessel ports use the schedule-wide sorted port union; completely unknown ports use the midpoint. Missing schedules and unscheduled vessels receive background-only survival, classified by whether stored ETD has passed.

There are 11 observable cohorts: missed-LOAD, expired without a future rotation, unscheduled active, scheduled overdue, scheduled initial active, and three ETD lead-time buckets each for active DISCHARGE and TRUCK_RECV arrivals (`<24h`, `24-72h` inclusive, `>72h`). Cohort alpha is the LOAD incidence among residents present at the start of a fully observed assigned loading window. Background mu is non-assigned-rotation exits divided by resident exposure hours, with assigned-window LOADs excluded from its numerator. Exposure changes cohort at schedule and ETD boundaries. All numeric estimates were fit strictly on Days 1-12, ending before January 13, 2025; they are embedded in `solution/survival_params.py`.

At placement time, let alpha be jump incidence, mu the hourly background rate, and tau the nonnegative hours until the interpolated jump (infinity with alpha=0 when absent). For resident x and arrival c, the implemented closed form is:

```text
L = mu_x + mu_c + 1/240; E_u = exp(-L*tau_u)
p_bg   = mu_x/L * (1 - alpha_x*E_x - alpha_c*E_c
                   + alpha_x*alpha_c*exp(-L*max(tau_x,tau_c)))
p_jump = alpha_x*E_x * (1 - alpha_c*phi(tau_x-tau_c))
```

Absent-jump terms are omitted. Here phi is 0, 0.5, or 1 for negative, zero, or positive arguments; stack score is the sum of `p_bg + p_jump`. This is an expected discounted blocking indicator, not an undiscounted lifetime probability. Probabilities are memoized by resident `(cohort, absolute_jump_time)` only within the current placement. A LOAD for an ID never initially present or previously arrived records a missed LOAD; a later arrival with that ID uses the low-hazard missed-LOAD cohort. IDs serve only as identity keys. No event file is accessed by the strategy.

## 2. Research and relationship to this implementation

Borgman, B., van Asperen, E., & Dekker, R. (2010). **Online rules for container stacking.** *OR Spectrum*, 32(3), 687-716. [DOI](https://doi.org/10.1007/s00291-010-0205-4); [EUR full text](https://pure.eur.nl/ws/portalfiles/portal/46848049/art-3A10.1007-2Fs00291-010-0205-4.pdf). Its LDT departure-aware stacking motivates our initial reference. Its expected-versus-actual departure experiments also show substantial information loss and larger penalties in taller stacks. Our continuous asymmetric survival score is an original adaptation to this simulator's noisy ETDs and long-stay arrivals, not the paper's algorithm.

Dekker, R., Voogd, P., & van Asperen, E. (2006). **Advanced methods for container stacking.** *OR Spectrum*, 28(4), 563-586. [DOI and publisher abstract](https://doi.org/10.1007/s00291-006-0038-3); [EUR bibliographic record](https://pure.eur.nl/en/publications/advanced-methods-for-container-stacking-4/). The verified abstract describes category stacking with exchangeable containers during loading. Here retrievals name specific IDs, so equal-category containers are not interchangeable: our explicit 50/50 tie risk is an adaptation, not a claim attributed to that paper. No claims rely on its inaccessible full text.

## 3. Training patterns and validation discipline

The training replay has 10,385 arrivals, 10,207 retrieval events, and 8,874 found targets. All 1,333 absent LOAD IDs arrive later; none is subsequently retrieved in training. These observations are diagnostics, never runtime future labels. Utilization rises from 50% to a 74.3% peak. Initial stacks are disordered. Summing initial blockers above each original stack's first retrieved initial target proves a policy-independent bound of 1,325 reshuffles (`r >= .129813`); initial inversion-pair counts alone are not such a bound. The train-only hindsight LDT diagnostic achieves 1,961, illustrating prediction headroom without claiming optimality or admissibility.

Among found LOADs, 24.55% use a different rotation from stored ETD (66.5% later, 33.5% earlier). Every scheduled-vessel truck delivery shares a vessel rotation ETD, so ETD alone cannot identify exit mode. Actual within-rotation LOAD sequences have zero port/weight-order reversals. Scheduled overdue residents often exit on their next rotation; expired residents without one behave as lower-risk bases. Missed-LOAD arrivals have zero observed fitting-period exits, a censored estimate rather than proof of permanent storage. Second-rotation LOAD incidence is 128/587 = 21.8%, near the fitted overdue 23.6%.

Parameters and configuration selection use Days 1-12. Days 13-20 is warm-up continuation from each policy's own yard state and was repeatedly consulted for acceptance gates; it is not an untouched holdout. Search stopped before opening test data. No vessel-ID outcome parameters, ID patterns, or runtime training labels are used.

## 4. Decisions, trade-offs, and rejected alternatives

Train reshuffles progressed from deterministic four-tier LDT variants (8,394-9,575; Greedy: 8,036), to symmetric ExpectedBlocking (7,026), expired-resident correction M2a (6,771), expanded hard-horizon SurvivalRisk (6,595), and frozen discounted SurvivalRisk (6,408). Top-only checks miss deeper disorder; non-strict tie compatibility piles ID-specific ties together. Empty-stack/gap penalties did not justify retention. Initial uncertainty M1, blanket-overdue M2b, and sharp within-rotation M3 failed the isolated acceptance gates. Discounting removes the finite-horizon cliff. Expanded 240h saves 49 early and 92 late reshuffles against its matching Core model. Multi-rotation modeling adds runtime without passing its early-period gate. Static per-vessel alpha risks memorizing early completed vessels. The strategy controls placement only; relocation remains entirely simulator-controlled.

## 5. Complexity and state lifecycle

Let M be stack count, H_max maximum tiers, N resident containers, U distinct initial/arrived IDs so far, A distinct observed absent LOAD IDs, and R schedule storage including vessel and port records. Scoring is `O(M*H_max)`, with `O(K)` transient pair probabilities for K distinct resident descriptors. Refreshing D dirty stacks costs `O(D*H_max + D*log D + B_D)`, where B_D counts all stacks in touched blocks for block-count recomputation. A reshuffle invalidates its block; a conservative capacity guard can invalidate every block. A total-count mismatch triggers a full public-API refresh. This guard does not detect arbitrary unreported count-preserving edits; normal simulator callbacks provide the required invalidations.

Schedule transitions amortize one pass over rotation ends; at a crossed boundary, next-rotation lookup costs `O(V*log R_max)`, followed at the next placement by `O(M*H_max)` descriptor rebuilding. Each registered ETD incurs one heap push/pop of `O(log U)`, amortized over the stream. Thus a placement processing q ETD deadlines and a schedule change costs `O(M*H_max + M*log M + q*log U + V*log R_max)`, including sorted dirty-index refresh. Registration costs `O(P + log U)`, where P is the applicable port-list length, and parses metadata once. N <= M*H_max. Total space is `O(M*H_max + U + A + R)`, with per-placement `O(K)` transient memory included in that bound.

Metadata is deleted on successful retrieval. Cached contents and CID-to-stack entries refresh after callbacks. Initial IDs, seen IDs, origin/arrival history, and missed-LOAD IDs remain until `initialize()` resets the strategy, intentionally preventing a previously seen ID from becoming a false missed LOAD. Retrieved IDs can leave stale ETD heap entries until their deadline; the heap is `O(U)`. Per-call probabilities are discarded; no stack scores persist across timestamps. Measured production cached/live runtimes are 11.42s / 69.41s per full train replay.

## 6. Results and reproducibility

Official r divides reshuffles by all 10,207 retrieval events. Early/late rates divide by 5,042/3,832 found retrievals. Random uses unchanged code with seeds 0-4; +/- denotes population standard deviation. All training runs have zero errors, hard violations, and placement failures.

| Train policy | Reshuffles | Official r | Score /40 | Early found rate | Late found rate |
|---|---:|---:|---:|---:|---:|
| Random, 5 seeds | 8,969.6 +/- 54.0 | .87877 +/- .00529 | 10.00 +/- 0 | .93241 +/- .00514 | 1.11388 +/- .01248 |
| Greedy | 8,036 | .787303 | 10.54 | .82071 | 1.01722 |
| Frozen MyStrategy | 6,408 | .627804 | 17.38 | .64181 | .82777 |

| Train policy | LOAD | TRUCK_DLVR | Initial | DISCHARGE | TRUCK_RECV | Repeated blocker moves |
|---|---:|---:|---:|---:|---:|---:|
| Random mean +/- SD | 6,352.2 +/- 47.9 | 2,617.4 +/- 31.9 | 5,680.0 +/- 33.0 | 2,484.8 +/- 32.6 | 804.8 +/- 19.3 | 2,380.8 +/- 47.7 |
| Greedy | 5,627 | 2,409 | 5,174 | 1,984 | 878 | 1,947 |
| Frozen MyStrategy | 4,081 | 2,327 | 4,607 | 1,299 | 502 | 1,515 |

The frozen policy saves 20.26% versus Greedy, with excess 5,083 above the proven bound. Its full placement sequence matches Phase 3D's SHA-256 `74a6945f533e5cbe55d16fecf93d09fdb0b886da151bd06bfa4fad464fb2f414`. Cached/live controls and repeated required-CLI runs are recorded in `results/phase4/`. CLI `elapsed_sec` naturally varies; other output fields and decisions match exactly.

| Test policy | Reshuffles | Official r | Score /40 | Runtime | Errors / violations / failures |
|---|---|---|---|---|---|
| Random, seeds 0-4 mean +/- SD | 8,101.8 +/- 32.0 | 0.839826 +/- 0.003317 | 10.00 +/- 0 | 0.832s +/- 0.007s | 0 / 0 / 0 |
| Greedy | 7,416 | 0.768736 | 11.34 | 3.88s | 0 / 0 / 0 |
| Frozen MyStrategy | 10,055 | 1.042293 | 10.00 | 1.56s | 9,861 / 0 / 0 |

**The official test run failed the error gate.** Its 9,861 exceptions cover every arrival; reported reshuffles and score reflect simulator fallback placements, not successful SurvivalRisk decisions. The test snapshot has 313 sparse stacks with 352 empty tier slots. The frozen cache assumed every tier through stack height held a container; the public API returns an empty ID for padded gaps and `get_container_info("")` returns None, causing the logged exception. Thus training gains did not produce a valid test submission, and this run cannot establish survival-model generalization. Strategy and parameter hashes matched before and after; there was exactly one official test replay, with no repair, retuning, or second instrumented pass. Raw output remains unchanged in `results/results.json`; fixed-seed baselines and read-only diagnostics are in `results/phase5/`.

## 7. Limitations and next steps

The sparse-input failure exposes a missing production robustness test. Future work must distinguish physical stack height from occupied container entries and validate cache counts accordingly. No such fix was made after test opening. Initial-origin truck deliveries in training still cost 1,693 reshuffles versus Greedy's 1,627. Supported vessel-level alpha ranges roughly 18.6%-86.6%; the available features do not establish an irreducible Bayes ceiling. Future generalizable capacity-versus-queue features could explain this variation without per-vessel memorization. The zero-exit missed-LOAD estimate needs longer observation or smoothing on independent data. Alpha is observed competing-exit incidence, used as a Bernoulli jump approximation; exponential backgrounds, independent container exits, and one future rotation simplify the true process. The objective discounts direct pairwise risk, omitting secondary relocation damage and future space opportunity costs. Relocation-aware block geometry and better survival calibration are future work, not post-test tuning.
