# Design brief: a discrete design in which each BS has its own stream set

## Context

The thesis ("DS-STAR-assisted Cooperative Multi-BS Downlink RSMA Systems: Joint Deployment and Resource Allocation
Optimization") studies a downlink with B cooperating BSs, a per-BS fronthaul limit of S_max streams, hierarchical rate
splitting (super-common, group-common and private streams) and three movable double-sided STAR surfaces in a
100 m × 100 m area. The framework has three stages: (1) a **discrete design** fixes the groups, the served streams and
which BS carries which stream; (2) a deployment search (design R, a Bayesian optimization) chooses the surface
deployment, every candidate being evaluated by an AO of the precoders and surface coefficients; (3) the AO is refined
at the selected deployment. FINDINGS.md §1–2 describe the framework and the current discrete design, a joint design:
every BS carries the same streams (core users, nearest-core groups and a fixed stream order, plus a stream-set test
that evaluates the design with and without group-common streams by the AO at the first deployments of the search and
keeps the better one).

This round asks for a discrete design in which each BS has its own stream set. Stages (2) and (3) are fixed.

## The user's request (verbatim)

Request (2026-10-13 08:33):

> Offload another discrete design task. The core UE selection logic can be kept if worthwhile. But I want the design
> to let each BS choose stream on its own (Not a forced joint design). So each BS stream selection can be different.
> Doesn't matter if this scheme performs worse than the joint selection design, as long as
>
> 1. It beats its corresponding 1-layer RSMA (i.e. the 1-layer RSMA use the same stream selection logic) at all R_th
>    threshold 0 to 2.5 (Tying on low threshold (0~0.5) is acceptable)
> 2. In the benchmark figure, proposed design still beats sparse beamforming and stream clustering clearly

The coordinating session's questions (abridged) and the user's answers (2026-10-13 08:38):

* Should every BS still carry the super-common stream, or is it also left to each BS's own choice? — "Super-common
  should still be transmitted by all BS"
* "On its own": may each BS's choice use the whole network's channels, i.e. one central design that just allows
  different sets per BS? Or must each BS decide from its own channels only? — "One central design that just allows
  different sets per BS"
* Stream clustering is currently built on the current design's groups. Should the comparison use the benchmark
  figure's results as they stand, or rebuild stream clustering on the new design's groups? — "On the new design
  groups"
* Scenarios: T2 only, or T2 plus E3? — "T2 plus E3"

Clarification (2026-10-13 08:51; "the first point" is the statement that the design need not match the current joint
design):

> Also, just to clarify with the first point: My only performance requirements are those aforementioned 2 points, plus
> super-common AND group common stream still playing a vital role in higher R_th, e.g. both around or over 10% of total
> sum rate at K = 16 and R_th = 2, plus rate still rises monotonically (including flattening out) as S_max rises.
> And I expect to see design that assigns some different streams for different BS (in the group-common and private
> stream part) in some scenarios (You can count this as a design requirement)
> Remember, straight-forward design.

Addition (2026-10-13 09:19):

> On second thought, while core UE selection by SUS is all well and good, we should not fix all core UE to be shared by
> all BS; it would be a ok design if some BS choose core UE of their own, and some core UE are chosen by multiple BSs.

Addition (2026-10-13 09:26):

> Just to be more clear, while simulation only simulate scenarios with BS = 2 (or BS = 3 in our internal testing), the
> discrete algorithm design should be designed with multiple scattered BS in a large scenario in mind (you can thus see
> it does not make sense to associate all core UEs to all BSs). So the design should have the flexibility to associate
> core UE to only some BS; but of course if the scenario desire (all) core UE to be associated to all BSs, then the
> algorithm should naturally do so.

## The requirements, as confirmed by the user

1. **The design.** One central discrete design, settled before the deployment search, decides each BS's stream set
   from the whole network's channels; the sets may differ between BSs. Every BS carries the super-common stream. The
   core-UE selection of the current design (SUS) may be kept if it is worthwhile. The design is meant for many BSs
   scattered over a large area, although the simulations have B = 2 (T2) and B = 3 (E3): it must be able to associate
   a core UE with only some of the BSs (some BSs may choose core UEs of their own, and some core UEs may be chosen by
   several BSs), and it must associate (all) core UEs with all BSs by itself when the scenario calls for it. The
   design need not match the current joint design.
2. **Design requirement.** In some scenarios, the design assigns different streams to different BSs (in the
   group-common and private stream part).
3. **Performance requirements** (the only ones):
   * **R1.** It beats its own 1-layer RSMA (the same stream selection logic without group-common streams) at every
     R_th from 0 to 2.5 in 0.5 steps; a tie at R_th = 0 and 0.5 is acceptable.
   * **R2.** In the setting of the benchmark figure, the proposed framework with the design clearly beats sparse
     beamforming and stream clustering (stream clustering built on the new design's groups).
   * **R3.** The super-common and group-common streams still play a vital role at higher R_th, e.g. both around or
     over 10% of the total sum rate at K = 16 and R_th = 2.
   * **R4.** The sum rate still rises monotonically (flattening out included) as S_max rises.
4. **Method.** A straightforward design. No AO tests. No empirically tuned parameters. One deterministic design for
   every setting. Simplicity and elegance are vital.
5. **Setup.** Scenarios T2 and E3; the trimmed search R with Q = 50 and D_max = 10 m; the area 100 m × 100 m. The
   current joint design (J) and its 1-layer RSMA (J1) are reported for reference only.

"Simplicity and elegance" and the kinds of design the user has rejected refer to the user's statements in earlier
design rounds, quoted from the record:

> "elegance/simplicity is vital"

> "Any algorithm design should be about improving the performance within a setting. The design should not
> specifically optimize towards making a figure look better. If a design is truly good enough, all figures would
> display nice results naturally."

and to the kinds of design the user has rejected in those rounds (from the coordinating session's record, not
verbatim): choosing by running the full evaluation (here: the AO) on alternatives and keeping the better one, as the
current stream-set test does; warm starts or best-of over runs; constants tuned on the test drops or settings, or
parameters tied to the study's grid of settings; rules justified by how a figure looks rather than by the problem; ad
hoc fixes stacked on a design.

## Ground rules

1. **The discrete design is settled before the deployment search** and stays fixed during the search and the
   refinement. It is computed per drop and setting from the drop and the settings (`net`, `p`: positions, channels,
   B, N, K, M, S_max, R_th, …).
2. **Not acceptable: AO tests**, i.e. running the AO evaluator (or the merit F of AO-optimized precoders and
   coefficients) on alternatives to choose among them.
3. **No parameter that needs empirical tuning.**
4. **The super-common stream is served by every BS** (`discrete.check` enforces this in every run). **Any number of
   BSs at any positions:** the design is defined for B BSs scattered over a large area; nothing in it relies on B = 2
   or 3 or on the BS positions of T2 and E3.
5. **Deterministic**; its computation is part of the cost of a run (recorded as `cpu_design`).
6. **Feasible:** disjoint groups of size 2 … K−1, at most ⌊K/2⌋ groups, at most S_max streams per BS, a stream
   counting only at the BSs that carry it (checked by `discrete.check` in every run).
7. **One design for both scenarios and every setting:** nothing in it is tuned to the scenarios, settings, R_th values
   or drops of the test grid.
8. **The search is design R** (trimmed; `doc/search_R.md`) with the main-run settings (Q = 50, Q0 = 5, D_max = 10 m),
   the evaluator and the refinement as they are. If your design needs a change elsewhere (for example in the AO start
   for streams carried by some BSs only), add it as a switch that is off by default (so that JH and J1 stay
   bit-identical), explain why in DESIGN.md, and note that the same evaluation then applies to every scheme of the
   thesis that uses the design (the stream-clustering benchmark built on your design uses it too).
9. **The 1-layer counterpart** of your design (R1) is a variant of its own (`one_layer` in `harness/variants.py`):
   the same stream selection logic without group-common streams.

## How designs will be compared

With `tools/report.py` on the runs of `harness/grid.py` (README.md):

* **R1:** the rate of your design against its 1-layer counterpart at every R_th ∈ {0, 0.5, 1, 1.5, 2, 2.5} (mean over
  settings and drops, paired, with standard errors), in T2 and in E3 (`--tier screen` for every idea, `--tier confirm`
  for the final design); J for reference.
* **R2:** `--tier bench` (the benchmark figure's setting: T2, K = 16, N = 3, S_max = 12, R_th = 2, M ∈ {100, 300,
  500}, drops 801–805): your design against sparse beamforming (stored runs) and against stream clustering on your
  design's streams (`VARIANT@SC`); `tools/report.py VARIANT --bench`.
* **R3:** the shares of the super-common and group-common streams in the sum rate in the same runs (the setting of the
  thesis figure of the shares, Fig. 5.7, at K = 16 and R_th = 2), printed by `--bench`.
* **R4:** `--tier smax` (T2, N = 3, M = 300, K ∈ {8, 12, 16}, S_max ∈ {4, 6, …, 16}, R_th ∈ {0, 1, 2}, 2 drops);
  `tools/report.py VARIANT --smax`.
* **Design requirement:** where the BSs of your design carry different group-common and private streams (printed by
  every report).
* **Show the results** of your design and of the other methods you try.
* **Cost:** CPU per run (discrete design, search, refinement).
* **The argument for the thesis:** what the design does and why (a thesis-ready description), how it behaves with
  many BSs scattered over a large area (when it associates a core UE with some BSs only and when with all of them),
  its simplicity and elegance, and why nothing in it needs empirical tuning.
* The coordinating session will also run the design on drops, settings and BS layouts that are not in the grid.

## What to return

A zip `design_return_<your-label>.zip` with:

* `DESIGN.md`:
  * Your discrete design, as a thesis-ready description (paragraphs, equations if needed, an algorithm listing) with
    the argument why it works and why it needs no empirical tuning; its 1-layer counterpart; its cost.
  * Its results (paste the `tools/report.py` outputs): R1 in both scenarios (with `--by-setting`), R2 and R3
    (`--bench`), R4 (`--smax`), and where its BSs carry different streams.
  * The other methods you tried, with their results.
  * Its weak points.
* Your code (`code/…`), a unified diff against the package (`diff -u`), and `harness/variants.py` with your entries.
* `results/*.jsonl` of every run (harness format, unedited) and `results/*report*.json`.
