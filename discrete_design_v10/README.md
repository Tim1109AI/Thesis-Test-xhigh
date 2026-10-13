# Design offload: the discrete design (package v10)

You (a Claude session) have been asked by the user to **design and test** a new discrete design for their thesis
simulations, in which each BS has its own stream set (the groups, the served streams and which BS carries which
stream), settled before the deployment search. This is a research/design task: you come up with ideas, implement them, run
experiments, measure, iterate, and return your best design with evidence.

Read, in this order: **BRIEF.md** (the user's request in their own words, the ground rules, how designs are compared,
what to return), **FINDINGS.md** (the framework, the current design and the facts measured so far), then this README
for the mechanics.

## Contents

```
BRIEF.md, FINDINGS.md, PROMPT.md   the task, the record so far, the prompt the user gives to each session
rsma_scenarios.json                scenario bank (BSs, UE drops, initial DS-STAR deployments of the 100 m area)
requirements.txt                   numpy 2.4.4, scipy 1.17.1, clarabel 0.11.1, cvxpy 1.9.3 (reference versions)
code/discrete.py                   the plug-in point: baseline designs, constraint check, run(cfg, design_fn)
code/scenarios.py                  the test scenarios T2 (B = 2, the thesis) and E3 (B = 3)
code/hrs.py                        system model, hrs.Design (stream bookkeeping), joint_design (the current design),
                                   params(B=...), make_network(BS=...), QT precoder step (honours d.bsmask)
code/algo.py, physbo.py            AO evaluator and its starts; the deployment search (trimmed design R for every caller)
code/design_select.py              the current stream-set test (the "testing and take best" of the current design)
code/benchmarks.py                 sparse beamforming and stream clustering (the benchmarks of requirement R2)
code/composition.py                rate composition; nogroup_design (1-layer RSMA of a design)
code/ablation.py, newprop.py, run.py, turbo.py   thesis code used by the above (ablation schemes, AO settings, bank)
harness/grid.py                    runs a variant on the test grid -> results/<VARIANT>__<scenario>.jsonl
harness/variants.py                variant name -> design function (add yours here)
harness/selftest.py                reproduces reference runs (environment check)
harness/launch.sh, stop.sh         background launch with a PID file / stop by PID file
tools/report.py                    scores a variant against J, JH, J1 (the coordinating session uses the same tool)
reference/selftest.jsonl           reference runs for the self-test
reference/bench_fig510.jsonl       stored runs of the benchmark figure (current framework J, sparse BF, stream clustering)
doc/                               thesis text: streams and problem, the current discrete design; search R (trimmed)
results/                           your outputs go here
```

## Setup and self-test (about 2 minutes)

```bash
pip install -r requirements.txt        # if refused: pip install --break-system-packages -r requirements.txt
python3 harness/selftest.py            # expect SELFTEST PASSED and "identical trace: True" for all cases
```

If a trace is not identical (different CPU rounding), results are still usable, since every comparison of the report
is paired with baseline runs from your own machine; mention it in DESIGN.md.

## How a design plugs in

A discrete design is a function `design(net, p) -> hrs.Design` in a module under `code/` (see the docstring of
`code/discrete.py`):

* `net`: the drop: `net['BS']` (B × 3), `net['UE']` (K × 3), `net['Hd']` (K × BN direct channels, noise-normalized,
  BS b in columns bN … bN+N−1), `net['p0']`, `net['angles0']` (initial DS-STAR deployment). `hrs.channel_terms(pose,
  net, p)` gives the cascades of a deployment (`hrs.pose_from(net['p0'], net['angles0'])` is the initial one).
* `p`: the settings: `B`, `K`, `N`, `M`, `BN`, `Smax` (per BS), `Rth`, `Pmax` (per BS), `L`, `Dmax`, `X`, `Y`, channel
  constants (`PL0`, `alpha`, `noise`, `q`, `lam`). B = 2 in T2 and 3 in E3.
* Return `hrs.Design(groups, served, K)` with `d.core` (UEs whose served private streams start the AO; with core UEs of
  some BSs only, all of them), `d.priority` (a
  stream order; stream clustering on your design takes it when it lists every stream), `d.rule` (a name), and, when
  some stream is carried by some BSs only, `d.bsmask` (B × nS bool, column i = served stream `d.served[i]`) and
  `d.bs_sets` (per BS, the global ids it carries). Global ids: 0 super-common, 1 … G group-common (group g − 1 of
  `groups`), G + 1 + k private stream of UE k.
* The framework (`discrete.run`) checks the constraints (`discrete.check`, including the super-common stream at every
  BS), builds the evaluator, runs the search R (Q = 50) and the refinement, and records the rate (0 if the QoS
  shortfall exceeds 1e-2), the served streams per BS, the composition (`comp`: rates of the super-common,
  group-common and private streams) and the CPU.

Add your design and its 1-layer counterpart to `harness/variants.py`, e.g. `'X': dict(design='mydesign:design',
one_layer='X1')` and `'X1': dict(design='mydesign:design_1layer')`. `X@SC` (no entry needed) is the stream-clustering
benchmark on the streams of X's design. Use a new name instead of editing a variant that already has results.

## Compute plan

The search makes a run expensive (about 1–5 CPU-minutes at Q = 50), so the harness has preset tiers. The user has
allowed more time for screening and confirming than in the earlier rounds.

| Tier | Runs per variant | Search budget | CPU per variant (current design) | Requirement |
|---|---|---|---|---|
| `--tier screen` | 288: T2 and E3 × 6 settings × 6 R_th × 4 drops | Q = 5 | about 0.75 CPU-h | R1 (screening) |
| `--tier confirm` | 216: T2 and E3 × 6 settings × 6 R_th × 3 drops | Q = 50 | about 6.8 CPU-h | R1 |
| `--tier bench` | 15: T2, K = 16, N = 3, S_max = 12, R_th = 2, M ∈ {100, 300, 500}, drops 801–805 | Q = 50 | about 0.4 CPU-h (X@SC: about 1.4) | R2, R3 |
| `--tier smax` | 126: T2, N = 3, M = 300, K ∈ {8, 12, 16}, S_max ∈ {4, 6, …, 16}, R_th ∈ {0, 1, 2}, drops 901–902 | Q = 50 | about 2.5 CPU-h | R4 |

* Screen every idea and its 1-layer counterpart with `--tier screen` (and the baselines JH and J1 once, for
  reference). Q = 5 ranks discrete designs at near-initial deployments, so check the ideas you shortlist at Q = 50 on
  subsets you choose (`--budget 50` with `--scenarios`, `--settings`, `--rth`, `--drops`, the counterpart on the same
  subsets), and run `--tier bench` (with `X@SC`) and `--tier smax` on them.
* Confirm your final design and its 1-layer counterpart with `--tier confirm`; JH and J1 at Q = 50 only if time
  allows (they are for reference). Q = 50 subsets run earlier are part of the same files and are not run again.
* Designs whose BSs carry different streams serve more distinct streams and cost more per run (up to about 3 times
  for twice the streams; FINDINGS.md §7); E3 costs about 15% more than T2.
* A whole round (screening, Q = 50 checks, the final design and its counterpart confirmed, bench and smax) is about
  25–45 CPU-hours, i.e. about 13–23 hours of wall time on 2 cores, less with more cores.

## Running the grid

```bash
python3 harness/grid.py X --smoke                                    # 2 short runs (harness check)
python3 harness/grid.py JH --tier screen                             # reference (J is derived from JH and J1)
python3 harness/grid.py J1 --tier screen
python3 harness/grid.py X  --tier screen
python3 harness/grid.py X1 --tier screen                             # the 1-layer counterpart (R1)
python3 tools/report.py X --budget 5 [--by-setting]                  # R1 at Q = 5 (works on partial results)
python3 harness/grid.py X  --budget 50 --settings a,f --rth 0,1,2.5 --drops 2   # a Q = 50 check (same for X1)
python3 harness/grid.py X  --tier bench                              # R2, R3
python3 harness/grid.py X@SC --tier bench                            # stream clustering on X's streams (R2)
python3 tools/report.py X --bench
python3 harness/grid.py X  --tier smax                               # R4
python3 tools/report.py X --smax
python3 harness/grid.py X  --tier confirm                            # final: R1 at Q = 50
python3 harness/grid.py X1 --tier confirm
python3 tools/report.py X --by-setting --json
./harness/launch.sh job_x python3 harness/grid.py X --tier screen    # background; log job_x.log, PID job_x.pid
./harness/stop.sh job_x
```

Scenarios T2 and E3 (`code/scenarios.py`, FINDINGS.md §6); settings a–f covering N ∈ {3, 5}, K ∈ {8, 12, 16},
M ∈ {300, 500}, S_max ∈ {N+1, 2N, 4N}; R_th ∈ {0, 0.5, 1, 1.5, 2, 2.5}; drops 901–916 (details in `harness/grid.py`).
Results go to `results/<VARIANT>__<scenario>.jsonl` (Q = 50), `results/<VARIANT>__<scenario>__Q5.jsonl`,
`results/<VARIANT>__T2__bench.jsonl` and `results/<VARIANT>__T2__smax.jsonl`. Files are resumable (finished runs are
skipped) and jobs run drop by drop, so a subset run now is completed by a larger run later. Every comparison of the
report is paired on the runs common to the schemes compared, so run the counterpart on the same subsets.

## Returning results

See BRIEF.md, "What to return".
