# Findings so far (as of 2026-10-13)

Rates in bit/s/Hz. A run whose QoS shortfall exceeds 1e-2 counts as rate 0. This file records the current framework
and measurements; it does not recommend a direction.

## 1. The system and the framework (fixed in this package)

* **Network** (100 m × 100 m): B BSs at height 15 m with N antennas each (λ = 0.1 m, half-wavelength spacing); in
  the thesis B = 2 at (25, 50, 15) and (75, 50, 15) m. K single-antenna UEs at height 1.5 m (in the thesis:
  independent uniform drops over the area). L = 3 double-sided STAR surfaces (DS-STARs) with M elements; initial
  (prescribed) deployment (33, 50, 7.5), (50, 50, 7.5), (67, 50, 7.5) m with azimuth 0, i.e. on the line between the
  two thesis BSs, where every cascaded channel vanishes (the BSs lie in the surface planes). The test scenario with
  B = 3 is in §6.
* **Channels** (code: deterministic LoS): path gain β0 d^(−α) with β0 = 10^−3 and α = 2.2 on every hop, noise
  σ² = 10^−12 W, element pattern |cos|^q with q = 3, P_max = 1 W per BS. Every cascade is rank one.
* **Streams** (hierarchical rate splitting, HRS): one super-common stream (decoded by every UE), one group-common
  stream per group (decoded by its members), K private streams; each UE decodes super-common, then its group-common,
  then its private stream (SIC), all other served streams are interference. Text: `doc/thesis_streams_and_problem.tex`.
* **Constraints on the discrete decisions** (problem (eq:problem) of the thesis): (c-group) the groups are disjoint,
  each of size 2 … K−1, at most ⌊K/2⌋ groups; (c-front) BS b carries at most S_max streams (a stream "carried" by b
  has a nonzero precoder block w_{j,b}); per-BS power P_max. A stream may be carried by any subset of the BSs
  (precoded jointly over their antennas). `discrete.check` tests these constraints for every run, and the requirement
  of this round (BRIEF.md) that every BS carries the super-common stream.
* **Evaluation of a deployment** (`algo.AOEvaluator`): AO of the precoders and the coupled DS-STAR coefficients
  (QT precoder steps with clarabel, QT coefficient steps with a projection), stopped at the loose tolerance ε_E = 3,
  returning the merit F = sum rate − λ · (total QoS shortfall), λ = 200; for given precoders the rate allocation is
  in closed form (smallest total shortfall). About 0.3–2 CPU-s per evaluation.
* **How a design enters the evaluation:** the served streams and their decoders (rates and interference sets,
  `hrs.Design`); `d.bsmask` (the precoder step constrains w_{j,b} = 0 for the pairs not carried; the start zeroes
  them); `d.core` (the start of the AO, `algo.init_core`: regularized ZF over the served private streams of the core
  users on the stacked BN-antenna channel, the super-common stream along the dominant eigenvector of the normalized channels
  of all UEs, every other served stream at the noise level of its weakest decoder; blocks not carried are zeroed;
  scaled to the per-BS power); the coefficient start (`algo.seed_rule`: the closed-form rule at the direct channels
  with this start).
* **Search:** design R, trimmed (`doc/search_R.md`, `code/physbo.py`; the user's choice for the thesis from now on),
  budget Q = 50, Q0 = 5 initial deployments, D_max = 10 m, as in the main runs. Its rate score uses the served
  streams, their decoders and the BSs that carry them. Every caller of `algo.bayes_opt_fast` uses it in this package
  (`algo.SEARCH = 'R'`).
* **Refinement** at the selected deployment: the AO continued to tolerance 0.1 (`newprop._ao`).

## 2. The current discrete design (thesis §4.1, `doc/thesis_discrete_design.tex`, `hrs.joint_design`)

* **Joint transmission:** every BS carries the same |J+| = min(S_max, 1 + G + K) streams.
* **Core users:** the greedy projection step of semi-orthogonal user selection (SUS) on the stacked direct channels,
  until |C| = min(K, BN). **Groups:** every non-core UE joins the core user with the most correlated direct channel;
  every core user with at least one follower heads a group. `d.core` lists all of C; the AO start uses the core users
  whose private streams are served.
* **Order:** super-common; core privates (SUS order); group-common streams (decreasing group size); non-core privates
  (decreasing direct-channel energy). The first |J+| streams are served.
* Consequences of the order: min(|C|, S_max − 1) core private streams are served; group-common streams are served
  only if S_max ≥ BN + 2 and K > BN, i.e. in the thesis sweep (S_max ∈ {N+1, 2N, 4N}, B = 2) only for S_max = 4N and
  K > 2N. In the grid of `harness/grid.py` (B = 2 and 3): settings a and f in T2, setting f in E3.
* **Stream-set test** (`code/design_select.py`; what the user calls "testing and take best"): whenever the order
  serves group-common streams, the HRS set and the same order without group-common streams (1-layer RSMA) are both
  evaluated at the 5 initial deployments of the search; the set with the larger best F is kept (HRS on ties) and its
  evaluations are the first evaluations of its search. In the harness, J is this framework, JH the design without the
  test, J1 its 1-layer RSMA (`tools/report.py` derives J from JH and J1 exactly; `harness/selftest.py` checks it
  against a direct run of the framework with the test).
* Nothing in the design depends on B = 2 except through BN; with B BSs it serves the same streams from all of them.
* J1 (1-layer RSMA): the same order without group-common streams; the super-common stream and up to S_max − 1 private
  streams, core users first, every BS carrying all of them.

## 3. Group-common streams vs 1-layer RSMA at 100 m (thesis runs, search T, Q = 50)

* **Main sweep** (N ∈ {3, 5}, K ∈ {8, 12, 16}, M ∈ {100, 300, 500}, S_max ∈ {N+1, 2N, 4N}, R_th ∈ {0, 0.5, 1, 1.5,
  2}, 5 drops; 1350 runs): the test was made in 375 runs (exactly those with S_max = 4N and K > 2N) and kept 1-layer
  RSMA in 123 (33%); by K: 73% (K = 8), 30% (12), 15% (16); by R_th: 37, 40, 32, 24, 31% (R_th = 0 … 2).
* **Composition study** (K = 16, N = 3, M = 500, S_max = 12, 48 drops):

  | R_th | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 |
  |---|---|---|---|---|---|---|
  | HRS without the test (JH) | 119.90 | 117.36 | 107.94 | 89.58 | 75.24 | 69.04 |
  | Thesis framework with the test (J) | 120.16 | 117.23 | 107.56 | 89.58 | 75.04 | 69.04 |
  | 1-layer RSMA (J1) | 120.33 | 117.56 | 104.33 | 79.48 | 53.51 | 0.83 |
  | J − J1 (± SE) | −0.17 ± 0.31 | −0.33 ± 0.58 | +3.23 ± 0.59 | +10.11 ± 0.91 | +21.53 ± 1.01 | +68.20 ± 1.97 |
  | Drops where J1 > J | 16 | 19 | 9 | 3 | 0 | 0 |
  | Test kept 1-layer (drops) | 14 | 15 | 7 | 0 | 1 | 0 |
  | QoS met J / J1 (of 48) | 48 / 48 | 48 / 48 | 48 / 48 | 48 / 48 | 48 / 48 | 47 / 1 |

  Group-common streams served by J: 3.1, 3.0, 3.8, 4.4, 4.3, 4.4 on average (R_th = 0 … 2.5); private streams
  7.9 … 6.6 (1-layer RSMA: 11).
* Earlier attempts at a discrete design that decides the group-common streams without a test are not recorded in this
  package (no results available). Code options that exist: `hrs.priority_list` rules 'private_first' (current),
  'group_first', 'cluster'; core rules 'sus' (current) and 'zf' (greedy ZF user selection). Ablation of the thesis
  (setting below): random grouping (the UEs partitioned uniformly at random into groups of the proposed sizes) 69.31
  vs 77.95.

## 4. The current design in the settings of requirements R2–R4 (thesis figures)

* **Benchmark figure (Fig. 5.10)**, T2, K = 16, N = 3, S_max = 12, R_th = 2, M ∈ {100, 300, 500}, drops 801–805,
  trimmed search R, Q = 50; stored runs in `reference/bench_fig510.jsonl` (15 per scheme):

  | Scheme | M = 100 | M = 300 | M = 500 | mean | QoS met |
  |---|---|---|---|---|---|
  | current framework (J) | 73.51 | 81.95 | 88.72 | 81.39 | 15/15 |
  | sparse beamforming | 63.08 | 75.09 | 81.03 | 73.07 | 15/15 |
  | stream clustering (on the current design's streams) | 44.18 | 54.38 | 64.51 | 54.36 | 15/15 |

  * **Sparse beamforming** (`benchmarks.sparse_bf_design`, [Dai2014sparse]): reweighted-ℓ1 per-BS stream count in the
    QT precoder update, from unit weights, with link removal, at the prescribed deployment; candidate streams: the HRS
    order of the current design (independent of the design under test); supports fixed during the search. In these
    runs: 11–13 streams, mostly carried by both BSs, 7–8 private streams. About 85 CPU-s per run.
  * **Stream clustering** (`benchmarks.stream_cluster_design(net, p, d)`, [Ahmad2021cran], Alg. 1): every stream of the
    design d (super-common, the group-common streams of its groups, all private streams) requests the BSs in decreasing
    order of its mean path-loss quality to its decoders; a BS above S_max keeps its S_max strongest streams and leaves.
    On the current design's streams: 19–20 streams, about 330 CPU-s per run.
  * `discrete.run` reproduces the stored runs of both benchmarks bit-identically (checked for drop 801, M = 100).
* **Shares of the common streams (Fig. 5.7)**, N = 3, S_max = 4N = 12, K = 16, M ∈ {100, 300, 500}, five drops
  (thesis runs, earlier search T; ratio of the summed rates over the runs with a feasible solution), R_th = 0, 0.5, 1,
  1.5, 2: super-common 1.8, 4.2, 7.5, 12.0, 16.7%; group-common 0.0, 0.1, 2.2, 5.2, 10.8% of the sum rate.
* **Sum rate versus S_max (Fig. 5.5)**, N = 3, averaged over M ∈ {100, 300, 500}, R_th ∈ {0, 0.5, …, 2} and five drops
  (thesis runs, earlier search T), S_max = 4 / 6 / 12: K = 8: 71.8 / 101.2 / 107.3; K = 12: 64.7 / 92.9 / 107.7;
  K = 16: 50.7 / 77.0 / 99.1. S_max ∈ {8, 10, 14, 16} was not run. In the current design, group-common streams appear
  only for S_max ≥ BN + 2 (= 8 for N = 3, B = 2), where the stream-set test decides.

## 5. Stream-to-BS association at 100 m (records, search T)

* **Thesis text (§4.1):** "in our tests with the network of Chapter 5 (twelve configurations, four deployments
  each), serving every UE from its strongest BS only, with up to 2S^max−1 distinct streams, attains only 51–78% of
  the joint-transmission rate at the same fronthaul limit."
* **Ablation** (K = 16, N = 3, M = 500, S_max = 12, R_th = 2, 48 drops; means over the 12 drops where every variant
  meets the QoS): random fronthaul assignment (each BS carries min(S_max, J) streams drawn uniformly from all J
  streams of the HRS design) 52.25 vs 77.95; QoS met in 40 vs 48 of 48 drops.
* **Existing code:** `ablation._assoc_design` (used by the SDMA and 2-UE NOMA benchmarks of the thesis): every
  stream goes to the BS with the larger direct-channel energy to its weakest decoder (the other BS if that one is
  full); each BS then fills its remaining slots with the other streams in decreasing order of that energy.

## 6. The test scenarios (`code/scenarios.py`)

| Scenario | BSs (m) | UEs |
|---|---|---|
| T2 (the thesis) | (25, 50, 15), (75, 50, 15) | uniform over the area (the bank's drops) |
| E3 | (10, 10, 15), (90, 10, 15), (50, 90, 15): far apart, near the edge of the area | uniform (the drops of T2) |

* The DS-STARs start at the same initial deployment in both scenarios (§1).
* Direct-link SNR of one BS with all its power on one UE, P_max N β0 d^−α / σ² for N = 3, over the UEs of drops
  901–916 (K = 8, 12, 16), and the gap between a UE's strongest and second-strongest BS:

  | Scenario | strongest BS, median | 2nd strongest, median | gap: median (10%, 90% quantiles) | UEs with gap < 3 dB |
  |---|---|---|---|---|
  | T2 | 61.4 dB | 55.6 dB | 5.5 dB (1.4, 10.8) | 23% |
  | E3 | 61.0 dB | 54.9 dB | 6.2 dB (1.1, 14.9) | 26% |

## 7. Cost

Measured CPU per run of the current design (T2, one drop, R_th = 1; trimmed search R; this includes the design, the
search and the refinement):

| Setting | a | b | c | d | e | f |
|---|---|---|---|---|---|---|
| Q = 5 | 4 s | 2 s | 6 s | 3 s | 14 s | 21 s |
| Q = 50 | 52 s | 37 s | 55 s | 53 s | 140 s | about 300 s |

* E3 costs about 15% more than T2 (setting e at Q = 50: 150 against 130 CPU-s). Measured CPU of every run is in its
  record (`cpu`; `cpu_design` for the discrete design alone).
* The cost of an evaluation grows with the number of served streams: N = 5, K = 16, M = 500, S_max = 10 in T2, one
  drop: a BS-specific association serving 20 distinct streams (the existing rule of §5 applied to the HRS order) took
  about 3 times the CPU of the current design (10 streams). In the thesis runs, the random fronthaul ablation took
  about 6 minutes per run.
* Benchmark setting (T2, K = 16, N = 3, S_max = 12, Q = 50): the current design about 90–120 CPU-s per run, sparse
  beamforming about 60–110, stream clustering on the current design's streams about 330.
* Totals per variant: README.md, "Compute plan".
