# DESIGN.md — a BS-specific discrete design: zero-forcing-minimal cooperation (round v10, label `cloud_xhigh`)

## 0. Summary

**The design.** It keeps the private streams of the current design and changes which BSs carry them. The first
$n=\min\{K,BN,S^{\max}-1\}$ UEs of the SUS order are the core users (the private streams the current design serves).
Each private stream is carried by the $m=\lceil n/N\rceil$ BSs with the strongest direct channels to its user, the
fewest BSs whose antennas can zero-force it at the other core users. Joint transmission ($m=B$) arises by itself when
$n>(B-1)N$. Every BS fills the fronthaul slots its private streams leave free with group-common streams (nearest-core
groups), the groups whose weakest member it reaches best first. Every BS carries the super-common stream. Only core
users have private streams. The 1-layer counterpart keeps the same core users and BSs and drops the group-common
streams. The design has no test, no tuned parameter and takes about 1 ms. One switch outside the design (off by
default): the AO start computes each stream's direction on the antennas of the BSs that carry it.

**Variants** (`harness/variants.py`): the design is `XZnc` (`bsd:design`), its 1-layer counterpart `XZn1c`
(`bsd:design_1layer`), stream clustering on its streams `XZnc@SC`. (The names come from the design study, Section 4;
the code is `code/bsd.py`.)

**Requirements** (rates in bit/s/Hz; R1 = the design minus its 1-layer counterpart, mean ± SE, paired):

| R1 | $R_{th}$ = 0 | 0.5 | 1 | 1.5 | 2 | 2.5 |
|---|---|---|---|---|---|---|
| R1, T2, Q = 50 (confirm, 18 runs per $R_{th}$) | +0.61 ± 1.26 | +0.87 ± 0.65 | +1.31 ± 0.72 | +0.30 ± 0.96 | +3.62 ± 1.28 | +4.61 ± 1.92 |
| R1, E3, Q = 50 (confirm, 18 runs per $R_{th}$) | +1.18 ± 1.24 | +0.06 ± 1.22 | +2.25 ± 0.99 | +7.45 ± 2.10 | +15.66 ± 2.91 | +23.95 ± 5.29 |
| R1, T2, Q = 5 (screen, 24 runs per $R_{th}$) | +0.57 ± 0.47 | +0.65 ± 0.51 | +1.05 ± 0.50 | +1.91 ± 0.76 | +3.64 ± 1.06 | +7.72 ± 2.49 |
| R1, E3, Q = 5 (screen, 24 runs per $R_{th}$) | +2.20 ± 0.71 | +2.97 ± 0.75 | +4.97 ± 1.02 | +9.91 ± 1.74 | +21.07 ± 4.33 | +28.65 ± 4.97 |

| Requirement | Result |
|---|---|
| R1 (above) | at Q = 50 the mean difference is positive at every $R_{th}$ in both scenarios; from $R_{th}=1$ on it is +0.3 to +4.6 (T2) and +2.3 to +24.0 (E3); at 0 and 0.5 within 1.4 SE (ties, as allowed); T2 at 1.5 (+0.30 ± 0.96) is the weakest point |
| R2 (benchmark, Q = 50) | 80.6 vs sparse beamforming 73.1 (+7.5 ± 2.6, paired) and stream clustering on its streams 54.7 (+25.9 ± 1.7); J: 81.4 |
| R3 (K = 16, $R_{th}$ = 2) | super-common 15.6%, group-common 9.2% of the sum rate |
| R4 ($S^{\max}$ = 4…16) | rises for K = 8, 12, 16 and flattens from $S^{\max}$ ≈ 10–12 (largest dip −0.5 ± 0.5) |
| BS-specific streams | in T2 b, d and E3 b–e (all runs): private streams at 1 or 2 BSs, group-common streams differing between BSs |
| Method | no AO test, no tuned constant, deterministic, ≈ 1 ms; one start switch (off by default) |
| For reference: the current framework J (Q = 50) | the design minus J: T2: -2.5 ± 0.9 over all runs (-7.5 at $R_{th}=0$, +3.1 at 2.5); E3: +1.7 ± 2.4 over all runs (-17.7 at $R_{th}=0$, +29.4 at 2.5) (not a requirement; Section 3.1) |


## 1. The design (thesis-ready description)

### 1.1 Idea

Under joint transmission every served stream occupies a fronthaul slot at every BS, $B$ slots per stream, whatever
the stream needs. The current design therefore serves $S^{\max}-1$ private streams and no group-common stream whenever
$S^{\max}-1<BN$. The proposed design keeps the same private streams and gives every stream only the BSs it needs:

* the **super-common stream** is decoded by every UE and carried by every BS (as required);
* a **private stream** is spatially multiplexed and must be zero-forced at the other private streams' users. Its
  precoder is carried by a set $\mathcal A$ of BSs and has $|\mathcal A|N$ coefficients, so nulling it at the $n-1$
  other core users needs $|\mathcal A|N\ge n$, i.e. $m=\lceil n/N\rceil$ BSs. The design gives it exactly the $m$
  BSs with the strongest channels to its user;
* the fronthaul this frees carries the layer of hierarchical rate splitting that the current design has no room for
  at small $S^{\max}$, the **group-common streams**. Each BS fills its free slots with the group-common streams whose
  weakest member, the decoder that limits the stream's rate, it reaches best. A BS thus chooses its own group-common
  streams.

Only the core users have private streams. Every other UE is served by the common streams: the super-common stream and
the group-common stream of its group.

### 1.2 The design

Let $\mathbf h_{d,k}=[\mathbf f_{1,k},\ldots,\mathbf f_{B,k}]$ be the stacked direct channel of UE $k$ and
$E_{k,b}=\|\mathbf f_{b,k}\|_2^2$ its energy from BS $b$; let $M_{b,j}\in\{0,1\}$ indicate that BS $b$ carries stream
$j$, so that \eqref{eq:c-front} reads $\sum_j M_{b,j}\le S^{\max}$.

**Core users.** The UEs are ordered by the greedy projection step of SUS \eqref{eq:core_selection} on the stacked direct
channels, $u_1,u_2,\ldots$ (as in the current design), and the core users are the first
$$
n=\min\{K,\;BN,\;S^{\max}-1\}\qquad\text{(eq:n)}
$$
of them, $\mathcal C=\{u_1,\ldots,u_n\}$: the private streams of the current design, which a BS carries together
with the super-common stream under joint transmission.

**Private streams.** In the SUS order, the private stream of core user $k$ is carried by the
$$
m=\lceil n/N\rceil\qquad\text{(eq:m)}
$$
BSs with the largest $E_{k,b}$ among those with a free slot. $m$ is the smallest number of BSs whose antennas can
null the stream at the other $n-1$ core users, $mN\ge n$; since $n\le S^{\max}-1$, every BS has room for all private
streams, so every private stream gets its $m$ strongest BSs. When $n>(B-1)N$, $m=B$ and every BS carries every
private stream (joint transmission). Otherwise every core user is served by its $m<B$ strongest BSs: different BSs
carry different private streams, a BS serves core users of its own, and core users are shared by several BSs when
$m\ge2$.

**Groups.** Every non-core UE joins the core user with the most correlated direct channel \eqref{eq:nearest_core}, and
every core user with followers heads a group $\mathcal U_g$. This is the current design's grouping applied to the core
users $\mathcal C$; the groups are disjoint, of size $2\ldots K-1$, and there are at most $\lfloor K/2\rfloor$ of them.

**Group-common streams.** Every BS $b$ fills the $S^{\max}-1-\sum_{k\in\mathcal C}M_{b,k}$ slots its private streams
leave free with group-common streams, in decreasing order of
$$
q_{g,b}=\min_{k\in\mathcal U_g}E_{k,b},\qquad\text{(eq:q)}
$$
its direct-channel energy to the weakest member of group $g$. A group-common stream carried by several BSs is precoded
jointly over their antennas.

**Super-common stream:** carried by every BS.

### 1.3 Algorithm listing

```
Input: direct channels h_d,k (stacked, BN), E_kb = ||f_b,k||^2, B, N, K, S^max
1  u_1, u_2, ... <- SUS order on the stacked direct channels (min{K, BN} steps)
2  n <- min{K, BN, S^max - 1};  m <- ceil(n / N);  C <- {u_1, ..., u_n}
3  for every BS b: J_b <- {c}                                            (super-common stream)
4  for k = u_1, ..., u_n:                                                 (private streams, SUS order)
5      add the private stream of k to J_b for the m BSs b with the largest E_kb among those with |J_b| < S^max
6  groups: every UE k not in C joins argmax_{a in C} |h_k h_a^H|^2 / (||h_k||^2 ||h_a||^2); U_a = {a} + followers
7  for every BS b: add group-common streams g to J_b in decreasing order of min_{k in U_g} E_kb while |J_b| < S^max
Output: J_1, ..., J_B (M_bj = 1 iff j in J_b), the groups, the core users C (start of the AO)
```

Code: `code/bsd.py` (`design`, `design_1layer`; about 60 lines with the alternative core size of Section 4).

### 1.4 The 1-layer counterpart (requirement R1)

`bsd.design_1layer`: steps 1–5 unchanged (the same core users, carried by the same BSs), no groups and no
group-common streams. The non-core UEs, which have no private stream in either design, are served by the
super-common stream only. The design is its 1-layer counterpart plus group-common streams in the fronthaul slots the
private streams leave free, so the comparison isolates the group-common layer. Section 3.6 reports a counterpart that
instead fills those slots with private streams of non-core UEs (J1 style) as a sensitivity check.

### 1.5 Why it works

**The group-common layer pays for itself (R1).** With $R_{th}>0$ every UE must receive $R_{th}$. A core user receives
it on its private stream. A non-core UE has no private stream. In the 1-layer counterpart its QoS can only come from
the super-common stream, whose rate is limited by the weakest of all $K$ UEs and which must deliver $(K-n)R_{th}$; the
power this takes is taken from the private streams. The design adds group-common streams that carry the QoS of their
members at the rate of their own weakest member, from the BSs that reach that member best. They sit in fronthaul slots
that zero-forcing does not need. The design contains its counterpart (same core users, same BSs; the group-common
streams start at the noise level and the AO can leave them off), so it gains wherever the super-common stream is
loaded: increasingly with $R_{th}$ and with the number $K-n$ of non-core UEs. At $R_{th}=0$ there is no QoS to carry
and the two coincide (a tie).

**Why these private streams.** In the interference-limited 100 m area (direct-link SNR around 60 dB) the sum rate
comes from the zero-forced private streams. In all 288 screen runs of the current design (JH) the private streams of
non-core users carry no rate (`r_pn` = 0). The design keeps the current design's private layer: its core
users, their number and their SUS order, i.e. the core-UE selection logic, which the user allowed to keep. It removes
only the carriage that zero-forcing does not need and gives that fronthaul to the group-common layer. That carriage
is not free: in the dense 100 m area the extra BSs add beamforming gain (Section 4.3). So the design trades rate at
low $R_{th}$ for the group-common layer, which pays at high $R_{th}$ (Section 3).

**Against the current joint design.** Where joint transmission is needed for zero-forcing ($n>(B-1)N$: T2 settings a,
c, e, f and the benchmark setting; E3 settings a and f), the design serves the same private streams as the current
design from every BS. Its group-common streams fill the slots the current design gives to them and to non-core
private streams. Elsewhere (T2 b, d; E3 b–e) it serves the same private streams from fewer BSs and adds group-common
streams. It is then behind J at low $R_{th}$ (less beamforming gain) and ahead at high $R_{th}$, where the current
design's non-core UEs depend on the super-common stream alone and miss the QoS (J meets the QoS in 20/24 runs at
$R_{th}=2$ and 16/24 at 2.5 in each scenario of the screen).

### 1.6 Many BSs scattered over a large area

Nothing in the design refers to $B=2$ or 3 or to the positions of T2 and E3. With $n\le S^{\max}-1$ private streams,
every private stream is carried by at most $\lceil (S^{\max}-1)/N\rceil$ BSs (4 for $N=3$, $S^{\max}=12$), its
strongest, however large the network. Every core user is served by a small cluster of nearby BSs, the clusters of
nearby core users overlap, and a BS serves the core users near it. Joint transmission by all BSs arises by itself, and
only, when the private streams need all BSs for zero-forcing, $n>(B-1)N$: with two BSs and $S^{\max}\ge2N$, or with all
UEs being core users of a few BSs (T2 a, c, e, f; E3 a, f). `tools/large_area_demo.py` shows the association
(design only; UEs uniform, seed 1; "distance" is the largest distance from a core user to the BSs that carry its
private stream, averaged over the core users):

| core size | layout | $B$ | $K$ | $S^{\max}$ | $n$ | $m$ | distance to carrying BSs vs. farthest BS | group-common streams served (BSs each) |
|---|---|---|---|---|---|---|---|---|
| design | T2 positions, 100 m | 2 | 16 | 12 | 6 | 2 = B | 53 m vs 56 m | 3 (2.0) |
| design | T2 positions, 100 m | 2 | 16 | 4 | 3 | 1 | 13 m vs 56 m | 2 (1.5) |
| design | E3 positions, 100 m | 3 | 16 | 6 | 5 | 2 | 71 m vs 84 m | 4 (1.2) |
| design | 2 x 2 grid, 100 m | 4 | 8 | 12 | 8 | 3 | 56 m vs 67 m | 0 (0.0) |
| design | 4 x 4 grid, 400 m | 16 | 64 | 12 | 11 | 4 | 113 m vs 355 m | 10 (13.2) |
| design | 6 x 6 grid, 1 km | 36 | 120 | 12 | 11 | 4 | 194 m vs 993 m | 11 (32.0) |
| alternative (4.1) | T2 positions, 100 m | 2 | 16 | 12 | 6 | 2 = B | 53 m vs 56 m | 3 (2.0) |
| alternative (4.1) | T2 positions, 100 m | 2 | 16 | 4 | 3 | 1 | 13 m vs 56 m | 2 (1.5) |
| alternative (4.1) | E3 positions, 100 m | 3 | 16 | 6 | 6 | 2 | 68 m vs 84 m | 3 (1.0) |
| alternative (4.1) | 2 x 2 grid, 100 m | 4 | 8 | 12 | 8 | 3 | 56 m vs 67 m | 0 (0.0) |
| alternative (4.1) | 4 x 4 grid, 400 m | 16 | 64 | 12 | 22 | 8 | 220 m vs 355 m | 5 (1.0) |
| alternative (4.1) | 6 x 6 grid, 1 km | 36 | 120 | 12 | 33 | 11 | 439 m vs 993 m | 17 (1.9) |

The private layer of the design is that of the current design, $\min\{K,BN,S^{\max}-1\}$ streams for the whole
network. Over a large area with many BSs this is small, and a large part of the fronthaul goes to group-common
streams, which then fill the free slots of almost every BS. The alternative core size of Section 4.1, the largest
number of private streams that the whole fronthaul can carry with zero-forcing, grows the private layer with the
network ($n\approx\sqrt{BN(S^{\max}-1)}$, $m\approx n/N$; second half of the table). It is the better choice for a
large network but leaves the group-common layer few slots in the test grid (Section 4.1, Section 5).

### 1.7 Nothing to tune

* $n$ is the current design's number of private streams; $m$ follows from the zero-forcing condition $mN\ge n$; no
  threshold, weight or constant enters.
* Which UEs are core users (SUS), which BSs carry a private stream (the $m$ strongest), the groups (nearest core) and
  which group-common streams a BS carries (largest $q_{g,b}$) are ordering/argmax decisions on the direct channels.
* The design is computed once per drop and setting, before the deployment search, in about 1 ms; it is
  deterministic, independent of $R_{th}$, and runs no AO and no evaluation of $F$.

### 1.8 One change outside the design: the AO start on the carried antennas (switch, off by default)

The AO start of the framework (`algo.init_core`) computes the regularized ZF directions of the core users' private
streams on the stacked $BN$-antenna channel and then zeroes the blocks of the BSs that do not carry a stream. With
BS-specific carriage this destroys the zero-forcing that the design is built on: a stream carried by $m$ BSs can only
be nulled by its precoder on those $mN$ antennas. `algo.init_core_carried` computes every direction on the antennas of
the BSs that carry the stream. `SEEDS['rule_carried']` is the coefficient seed computed from it.
* The private stream of core user $k$ carried by $\mathcal A_k$ uses column $k$ of the regularized ZF matrix of the
  core channels restricted to the antennas of $\mathcal A_k$, with the same regularization.
* A common stream uses the dominant eigenvector of its decoders' normalized channels restricted to its carrying
  antennas.
* The power split is that of `init_core`.

Under joint transmission (`d.bsmask` None) it is exactly `init_core`, so it changes nothing for JH, J1 or any joint
design; `harness/selftest.py` still passes with identical traces. It is off by default and set by the variant
(`opt=dict(init='core_carried', seed='rule_carried')` in `harness/variants.py`). Stream clustering on the design's
streams (`XZnc@SC`) inherits it, because harness/grid.py passes the base variant's `opt`. On the identical design it
gives a higher rate at lower cost than the default start (Section 3.7).


## 2. Cost

* **The design itself:** about 1 ms per drop and setting (`cpu_design` 1.0–1.5 ms; SUS dominates). It runs no AO and
  no evaluation of $F$.
* **Per run** (design + search + refinement; `cpu` of the records, mean over the runs):

| Tier | scenario | design | 1-layer | J | JH | J1 |
|---|---|---|---|---|---|---|
| screen, Q = 5 | T2 | 5.9 s | 4.0 s | 7.7 s | 6.1 s | 4.4 s |
| screen, Q = 5 | E3 | 25.2 s | 12.0 s | 13.6 s | 8.8 s | 10.2 s |
| confirm, Q = 50 | T2 | 66.0 s | 44.5 s | 67.8 s | 73.6 s | 49.9 s |
| confirm, Q = 50 | E3 | 210.7 s | 116.4 s | 100.6 s | 89.8 s | 100.2 s |
| bench, Q = 50 | T2 | 103.2 s | stream clustering on it: 354 s | 102.2 s (stored) | | |
| smax, Q = 50 | T2 | 43.1 s | | | | |

  In T2 the design costs about as much as the current design. It serves fewer distinct streams in settings b and d
  and the same or fewer in the others. In E3 it serves more distinct streams than J: the same private streams, each at 1
  or 2 of the 3 BSs, plus group-common streams (up to 14 against 10 streams in setting e). The most expensive settings are E3 c, e and f at Q = 50
  (310–390 CPU-s per run on average, at most 760). With the default AO start the design costs 20–40% more (Section 3.7).

## 3. Results of the design

Rates in bit/s/Hz, 0 when the QoS is missed. Differences are paired (mean ± standard error over the runs). `XZnc` is
the design and `XZn1c` its 1-layer counterpart. J, JH and J1 are the current framework, its design without the
stream-set test and its 1-layer RSMA; they are shown for reference wherever their runs exist at that budget. All
blocks are unedited `tools/report.py` output.

### 3.1 R1 at Q = 50 (`--tier confirm`, 3 drops)

```
===== scenario T2, Q = 50: 108 common runs (XZnc: 108, XZn1c: 108, JH: 108, J1: 108)
-- T2, all settings
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0   18    117.20    116.59    124.68    123.91    123.76    +0.61± 1.26       -7.48± 2.99      18/18/18/18/18
    0.5   18    115.47    114.61    121.41    120.44    120.85    +0.87± 0.65       -5.93± 2.70      18/18/18/18/18
      1   18    110.32    109.01    113.54    113.02    112.95    +1.31± 0.72       -3.21± 1.79      18/18/18/18/18
    1.5   18    101.18    100.88    103.61    103.47    103.59    +0.30± 0.96       -2.43± 1.90      18/18/18/18/18
      2   18     94.28     90.65     93.54     94.06     93.22    +3.62± 1.28       +0.73± 0.81      18/18/18/18/18
    2.5   18     79.97     75.35     76.86     77.34     75.61    +4.61± 1.92       +3.11± 1.47      15/15/15/15/15
    all  108    103.07    101.18    105.61    105.37    105.00    +1.89± 0.51       -2.54± 0.90      105/105/105/105/105
-- XZnc: mean served streams 9.11, group-common served 1.78, private served 6.33; BS-specific in 36/108 runs; per run, carried by a subset of the BSs: group-common 0.39, private 1.33
-- J: stream-set test made in 36/108 runs, 1-layer kept in 12
-- CPU per run (s): XZnc 66.0, XZn1c 44.5, J 67.8, JH 73.6, J1 49.9; XZnc discrete design 0.001
===== scenario E3, Q = 50: 108 common runs (XZnc: 108, XZn1c: 108, JH: 108, J1: 108)
-- E3, all settings
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0   18    121.78    120.60    139.50    139.60    139.09    +1.18± 1.24      -17.72± 4.78      18/18/18/18/18
    0.5   18    119.56    119.50    133.02    133.37    132.32    +0.06± 1.22      -13.46± 3.99      18/18/18/18/18
      1   18    117.86    115.61    124.99    126.26    124.38    +2.25± 0.99       -7.12± 3.67      18/18/18/18/18
    1.5   18    114.77    107.32    112.79    112.47    112.00    +7.45± 2.10       +1.98± 2.85      18/18/18/18/18
      2   18    112.26     96.60     95.46     95.46     95.61   +15.66± 2.91      +16.81± 4.92      18/18/16/16/16
    2.5   18    107.88     83.93     78.43     79.12     78.16   +23.95± 5.29      +29.45± 6.15      18/15/12/12/12
    all  108    115.69    107.26    114.03    114.38    113.59    +8.43± 1.38       +1.65± 2.42      108/105/100/100/100
-- XZnc: mean served streams 11.11, group-common served 2.61, private served 7.50; BS-specific in 72/108 runs; per run, carried by a subset of the BSs: group-common 1.72, private 3.67
-- J: stream-set test made in 18/108 runs, 1-layer kept in 9
-- CPU per run (s): XZnc 210.7, XZn1c 116.4, J 100.6, JH 89.8, J1 100.2; XZnc discrete design 0.002
```


By setting:

```
===== scenario T2, Q = 50: 108 common runs (XZnc: 108, XZn1c: 108, JH: 108, J1: 108)
-- T2, all settings
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0   18    117.20    116.59    124.68    123.91    123.76    +0.61± 1.26       -7.48± 2.99      18/18/18/18/18
    0.5   18    115.47    114.61    121.41    120.44    120.85    +0.87± 0.65       -5.93± 2.70      18/18/18/18/18
      1   18    110.32    109.01    113.54    113.02    112.95    +1.31± 0.72       -3.21± 1.79      18/18/18/18/18
    1.5   18    101.18    100.88    103.61    103.47    103.59    +0.30± 0.96       -2.43± 1.90      18/18/18/18/18
      2   18     94.28     90.65     93.54     94.06     93.22    +3.62± 1.28       +0.73± 0.81      18/18/18/18/18
    2.5   18     79.97     75.35     76.86     77.34     75.61    +4.61± 1.92       +3.11± 1.47      15/15/15/15/15
    all  108    103.07    101.18    105.61    105.37    105.00    +1.89± 0.51       -2.54± 0.90      105/105/105/105/105
-- T2, setting a
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3    107.17    104.63    110.24    106.84    106.33    +2.53± 1.64       -3.08± 2.25      3/3/3/3/3
    0.5    3    107.17    105.07    111.23    107.15    110.08    +2.10± 1.91       -4.07± 2.76      3/3/3/3/3
      1    3    107.27    104.43    111.42    107.38    109.07    +2.84± 1.59       -4.15± 2.59      3/3/3/3/3
    1.5    3    105.44    105.76    108.84    105.70    108.44    -0.32± 1.40       -3.40± 3.11      3/3/3/3/3
      2    3    107.31    104.13    110.08    108.48    108.73    +3.18± 3.01       -2.77± 2.61      3/3/3/3/3
    2.5    3    104.32    107.26    106.47    109.33    104.77    -2.94± 1.12       -2.15± 1.65      3/3/3/3/3
    all   18    106.44    105.21    109.71    107.48    107.90    +1.23± 0.83       -3.27± 0.89      18/18/18/18/18
-- T2, setting b
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3     75.28     77.04     84.75     84.75     84.75    -1.75± 0.55       -9.46± 2.68      3/3/3/3/3
    0.5    3     74.99     75.52     78.06     78.06     78.06    -0.54± 1.31       -3.08± 3.87      3/3/3/3/3
      1    3     71.69     69.55     71.01     71.01     71.01    +2.14± 1.08       +0.68± 1.23      3/3/3/3/3
    1.5    3     64.04     63.18     62.39     62.39     62.39    +0.86± 2.33       +1.65± 1.54      3/3/3/3/3
      2    3     58.08     52.77     53.41     53.41     53.41    +5.31± 2.84       +4.67± 1.85      3/3/3/3/3
    2.5    3     50.25     41.00     42.77     42.77     42.77    +9.25± 2.58       +7.48± 3.32      3/3/3/3/3
    all   18     65.72     63.18     65.40     65.40     65.40    +2.55± 1.13       +0.32± 1.60      18/18/18/18/18
-- T2, setting c
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3    117.51    117.51    117.51    117.51    117.51    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    0.5    3    111.22    111.22    111.22    111.22    111.22    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      1    3     95.29     95.29     95.29     95.29     95.29    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    1.5    3     74.50     74.50     74.50     74.50     74.50    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      2    3     49.93     49.93     49.93     49.93     49.93    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    2.5    3      0.00      0.00      0.00      0.00      0.00    +0.00± 0.00       +0.00± 0.00      0/0/0/0/0
    all   18     74.74     74.74     74.74     74.74     74.74    +0.00± 0.00       +0.00± 0.00      15/15/15/15/15
-- T2, setting d
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3     95.97     97.39    129.06    129.06    129.06    -1.41± 6.77      -33.09± 3.88      3/3/3/3/3
    0.5    3     96.34     97.25    125.63    125.63    125.63    -0.91± 1.14      -29.30± 1.80      3/3/3/3/3
      1    3     99.42     98.14    117.95    117.95    117.95    +1.29± 2.21      -18.53± 0.62      3/3/3/3/3
    1.5    3     91.76     90.56    108.74    108.74    108.74    +1.20± 4.75      -16.98± 3.81      3/3/3/3/3
      2    3     96.29     84.42     94.69     94.69     94.69   +11.87± 3.72       +1.60± 1.49      3/3/3/3/3
    2.5    3     94.41     77.00     81.65     81.65     81.65   +17.41± 1.93      +12.76± 2.68      3/3/3/3/3
    all   18     95.70     90.79    109.62    109.62    109.62    +4.91± 2.19      -13.92± 4.05      18/18/18/18/18
-- T2, setting e
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3    158.62    158.62    158.62    158.62    158.62    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    0.5    3    155.20    155.20    155.20    155.20    155.20    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      1    3    143.46    143.46    143.46    143.46    143.46    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    1.5    3    137.38    137.38    137.38    137.38    137.38    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      2    3    127.31    127.31    127.31    127.31    127.31    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    2.5    3    110.35    110.35    110.35    110.35    110.35    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    all   18    138.72    138.72    138.72    138.72    138.72    +0.00± 0.00       +0.00± 0.00      18/18/18/18/18
-- T2, setting f
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3    148.63    144.33    147.89    146.67    146.28    +4.29± 4.31       +0.74± 0.23      3/3/3/3/3
    0.5    3    147.94    143.40    147.09    145.38    144.94    +4.54± 2.08       +0.85± 3.00      3/3/3/3/3
      1    3    144.80    143.21    142.09    143.02    140.89    +1.59± 3.84       +2.71± 1.13      3/3/3/3/3
    1.5    3    133.96    133.89    129.85    132.11    130.11    +0.07± 4.03       +4.11± 4.04      3/3/3/3/3
      2    3    126.74    125.36    125.83    130.55    125.27    +1.38± 1.10       +0.90± 2.42      3/3/3/3/3
    2.5    3    120.49    116.52    119.92    119.92    114.14    +3.98± 5.88       +0.57± 2.50      3/3/3/3/3
    all   18    137.09    134.45    135.45    136.27    133.60    +2.64± 1.39       +1.65± 0.93      18/18/18/18/18
-- XZnc: mean served streams 9.11, group-common served 1.78, private served 6.33; BS-specific in 36/108 runs; per run, carried by a subset of the BSs: group-common 0.39, private 1.33
-- J: stream-set test made in 36/108 runs, 1-layer kept in 12
-- CPU per run (s): XZnc 66.0, XZn1c 44.5, J 67.8, JH 73.6, J1 49.9; XZnc discrete design 0.001
===== scenario E3, Q = 50: 108 common runs (XZnc: 108, XZn1c: 108, JH: 108, J1: 108)
-- E3, all settings
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0   18    121.78    120.60    139.50    139.60    139.09    +1.18± 1.24      -17.72± 4.78      18/18/18/18/18
    0.5   18    119.56    119.50    133.02    133.37    132.32    +0.06± 1.22      -13.46± 3.99      18/18/18/18/18
      1   18    117.86    115.61    124.99    126.26    124.38    +2.25± 0.99       -7.12± 3.67      18/18/18/18/18
    1.5   18    114.77    107.32    112.79    112.47    112.00    +7.45± 2.10       +1.98± 2.85      18/18/18/18/18
      2   18    112.26     96.60     95.46     95.46     95.61   +15.66± 2.91      +16.81± 4.92      18/18/16/16/16
    2.5   18    107.88     83.93     78.43     79.12     78.16   +23.95± 5.29      +29.45± 6.15      18/15/12/12/12
    all  108    115.69    107.26    114.03    114.38    113.59    +8.43± 1.38       +1.65± 2.42      108/105/100/100/100
-- E3, setting a
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3    137.13    137.13    137.13    137.13    137.13    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    0.5    3    135.09    135.09    135.09    135.09    135.09    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      1    3    143.92    143.92    143.92    143.92    143.92    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    1.5    3    138.21    138.21    138.21    138.21    138.21    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      2    3    143.74    143.74    143.74    143.74    143.74    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    2.5    3    138.79    138.79    138.79    138.79    138.79    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    all   18    139.48    139.48    139.48    139.48    139.48    +0.00± 0.00       +0.00± 0.00      18/18/18/18/18
-- E3, setting b
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3     62.88     60.13     76.67     76.67     76.67    +2.75± 0.69      -13.79± 2.55      3/3/3/3/3
    0.5    3     62.22     60.95     68.77     68.77     68.77    +1.27± 0.94       -6.55± 1.99      3/3/3/3/3
      1    3     61.91     56.76     60.77     60.77     60.77    +5.15± 1.66       +1.14± 2.56      3/3/3/3/3
    1.5    3     60.29     50.76     51.52     51.52     51.52    +9.53± 2.08       +8.77± 2.56      3/3/3/3/3
      2    3     59.30     43.61     42.80     42.80     42.80   +15.69± 1.28      +16.50± 1.87      3/3/3/3/3
    2.5    3     57.73     32.29      0.00      0.00      0.00   +25.44± 2.65      +57.73± 1.88      3/3/0/0/0
    all   18     60.72     50.75     50.09     50.09     50.09    +9.97± 2.12      +10.63± 5.69      18/18/15/15/15
-- E3, setting c
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3    105.54    105.24    122.35    122.35    122.35    +0.30± 3.64      -16.81± 2.42      3/3/3/3/3
    0.5    3     99.45    100.39    105.73    105.73    105.73    -0.95± 1.04       -6.29± 2.12      3/3/3/3/3
      1    3     87.95     83.74     85.46     85.46     85.46    +4.21± 0.34       +2.49± 1.36      3/3/3/3/3
    1.5    3     77.02     64.19     64.42     64.42     64.42   +12.83± 2.22      +12.60± 2.17      3/3/3/3/3
      2    3     69.75     36.60     12.73     12.73     12.73   +33.15± 8.08      +57.02± 8.93      3/3/1/1/1
    2.5    3     58.36      0.00      0.00      0.00      0.00   +58.36± 9.12      +58.36± 9.12      3/0/0/0/0
    all   18     83.01     65.03     65.12     65.12     65.12   +17.98± 5.50      +17.89± 7.40      18/15/13/13/13
-- E3, setting d
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3     98.57     92.55    124.94    124.94    124.94    +6.02± 3.98      -26.36± 6.97      3/3/3/3/3
    0.5    3     98.40     96.01    116.46    116.46    116.46    +2.39± 3.33      -18.07± 4.35      3/3/3/3/3
      1    3     99.75     93.05    107.64    107.64    107.64    +6.70± 2.23       -7.89± 3.75      3/3/3/3/3
    1.5    3     99.13     87.05     95.37     95.37     95.37   +12.08± 1.94       +3.76± 4.16      3/3/3/3/3
      2    3     97.83     75.81     81.89     81.89     81.89   +22.02± 1.12      +15.94± 4.46      3/3/3/3/3
    2.5    3     99.39     66.53     67.18     67.18     67.18   +32.86± 2.87      +32.21± 3.23      3/3/3/3/3
    all   18     98.84     85.17     98.91     98.91     98.91   +13.68± 2.74       -0.07± 5.10      18/18/18/18/18
-- E3, setting e
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3    143.45    144.85    195.79    195.79    195.79    -1.41± 3.19      -52.34± 8.92      3/3/3/3/3
    0.5    3    144.93    139.47    191.44    191.44    191.44    +5.46± 1.23      -46.51± 1.68      3/3/3/3/3
      1    3    135.96    134.72    173.39    173.39    173.39    +1.25± 2.06      -37.43± 8.57      3/3/3/3/3
    1.5    3    135.20    134.05    150.58    150.58    150.58    +1.15±10.72      -15.38±10.88      3/3/3/3/3
      2    3    132.98    117.31    124.31    124.31    124.31   +15.67± 5.54       +8.67± 6.47      3/3/3/3/3
    2.5    3    127.80     99.52     98.11     98.11     98.11   +28.29± 8.37      +29.70± 9.63      3/3/3/3/3
    all   18    136.72    128.32    155.60    155.60    155.60    +8.40± 3.31      -18.88± 7.78      18/18/18/18/18
-- E3, setting f
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    3    183.13    183.69    180.11    180.70    177.67    -0.56± 4.50       +3.02± 3.02      3/3/3/3/3
    0.5    3    177.27    185.09    180.64    182.69    176.42    -7.82± 3.36       -3.37± 5.63      3/3/3/3/3
      1    3    177.69    181.49    178.75    186.37    175.12    -3.79± 0.81       -1.06± 2.85      3/3/3/3/3
    1.5    3    178.75    169.66    176.63    174.75    171.90    +9.09± 4.64       +2.12± 5.79      3/3/3/3/3
      2    3    169.97    162.51    167.27    167.27    168.19    +7.46± 1.50       +2.71± 2.39      3/3/3/3/3
    2.5    3    165.20    166.44    166.51    170.67    164.90    -1.24± 3.73       -1.32± 3.67      3/3/3/3/3
    all   18    175.34    174.81    174.98    177.07    172.37    +0.52± 1.86       +0.35± 1.53      18/18/18/18/18
-- XZnc: mean served streams 11.11, group-common served 2.61, private served 7.50; BS-specific in 72/108 runs; per run, carried by a subset of the BSs: group-common 1.72, private 3.67
-- J: stream-set test made in 18/108 runs, 1-layer kept in 9
-- CPU per run (s): XZnc 210.7, XZn1c 116.4, J 100.6, JH 89.8, J1 100.2; XZnc discrete design 0.002
```


**R1.** At Q = 50 the design's mean is ahead of its 1-layer counterpart at every $R_{th}$ in both scenarios. The gap grows
with $R_{th}$: T2 +1.31 ± 0.72 at $R_{th}=1$ and +4.61 ± 1.92 at 2.5; E3 +2.25 ± 0.99 at 1 and
+23.95 ± 5.29 at 2.5. At $R_{th}\le0.5$ there is little QoS to carry, and the differences are positive but within
1.4 standard errors: ties, as allowed. T2 at $R_{th}=1.5$ (+0.30 ± 0.96) is the weakest point: positive
but not significant at 3 drops; at Q = 5 (4 drops) it is +1.91 ± 0.76.

In the settings where the two coincide except for the group-common streams, the paired runs differ by up to ±15
bit/s/Hz, even when the group-common streams carry no rate. Example: E3 f, one group of two UEs. There the first five
evaluations of the search agree to 0.1 bit/s/Hz, and the search then takes different paths, since its rate score
counts every served stream. This variability, not the design, sets the error bars. The design's advantage shows
where its group-common streams have room (over all $R_{th}$: T2 a +1.23 ± 0.83, T2 b +2.55 ± 1.13, T2 d +4.91 ± 2.19, T2 f +2.64 ± 1.39;
E3 b +9.97 ± 2.12, E3 c +17.98 ± 5.50, E3 d +13.68 ± 2.74, E3 e +8.40 ± 3.31). In settings c and e of T2 and a of E3 the design and its
counterpart are the same (no free slot or no non-core UE), and the differences are exactly zero. In the screen
(Q = 5, Section 3.2) the search barely moves from the initial design, and the design leads at every $R_{th}$ with
smaller error bars.

**Against the current framework (J, for reference; not a requirement).** The design minus J by $R_{th}$ (0, 0.5, …, 2.5): T2 (108 runs): 0 -7.5, 0.5 -5.9, 1 -3.2, 1.5 -2.4, 2 +0.7, 2.5 +3.1; QoS met 105/108 against J 105/108; E3 (108 runs): 0 -17.7, 0.5 -13.5, 1 -7.1, 1.5 +2.0, 2 +16.8, 2.5 +29.4; QoS met 108/108 against J 100/108. The design is behind J at low $R_{th}$ in the settings where it carries private streams by fewer than all BSs (T2 b, d; E3 b–e; Section 4.3) and ahead at high $R_{th}$ (T2 at 2.5, E3 from 1.5 on), where in E3 J misses the QoS more often. Where the design has no free slot or no non-core UE (T2 c, e; E3 a) it is the current design itself, and the runs are identical.

### 3.2 R1 at Q = 5 (`--tier screen`, 4 drops)

```
===== scenario T2, Q = 5: 144 common runs (XZnc: 144, XZn1c: 144, JH: 144, J1: 144)
-- T2, all settings
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0   24     96.48     95.91    104.77    104.76    104.41    +0.57± 0.47       -8.28± 2.69      24/24/24/24/24
    0.5   24     95.87     95.22    102.52    102.51    102.16    +0.65± 0.51       -6.65± 2.34      24/24/24/24/24
      1   24     91.16     90.11     95.64     95.63     95.34    +1.05± 0.50       -4.48± 1.83      24/24/24/24/24
    1.5   24     84.99     83.08     87.23     87.23     86.94    +1.91± 0.76       -2.24± 1.17      24/24/24/24/24
      2   24     70.37     66.73     70.13     70.40     69.41    +3.64± 1.06       +0.24± 0.84      20/20/20/20/20
    2.5   24     62.05     54.33     55.88     55.88     54.42    +7.72± 2.49       +6.17± 2.90      19/17/16/16/16
    all  144     83.49     80.90     86.03     86.07     85.45    +2.59± 0.52       -2.54± 0.93      135/133/132/132/132
-- XZnc: mean served streams 9.12, group-common served 1.79, private served 6.33; BS-specific in 48/144 runs; per run, carried by a subset of the BSs: group-common 0.38, private 1.33
-- J: stream-set test made in 48/144 runs, 1-layer kept in 16
-- CPU per run (s): XZnc 5.9, XZn1c 4.0, J 7.7, JH 6.1, J1 4.4; XZnc discrete design 0.001
===== scenario E3, Q = 5: 144 common runs (XZnc: 144, XZn1c: 144, JH: 144, J1: 144)
-- E3, all settings
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0   24    113.52    111.32    128.91    128.82    128.91    +2.20± 0.71      -15.39± 3.25      24/24/24/24/24
    0.5   24    112.23    109.26    122.90    122.83    122.90    +2.97± 0.75      -10.67± 2.97      24/24/24/24/24
      1   24    109.31    104.34    113.25    113.24    113.25    +4.97± 1.02       -3.94± 2.36      24/24/24/24/24
    1.5   24    106.15     96.23    101.61    101.60    101.71    +9.91± 1.74       +4.54± 2.11      24/24/24/24/24
      2   24    102.36     81.29     83.68     83.68     83.66   +21.07± 4.33      +18.69± 4.45      24/20/20/20/20
    2.5   24     97.80     69.15     70.64     70.64     70.63   +28.65± 4.97      +27.16± 4.95      23/16/16/16/16
    all  144    106.90     95.27    103.50    103.47    103.51   +11.63± 1.41       +3.40± 1.90      143/132/132/132/132
-- XZnc: mean served streams 11.21, group-common served 2.71, private served 7.50; BS-specific in 96/144 runs; per run, carried by a subset of the BSs: group-common 1.92, private 3.67
-- J: stream-set test made in 24/144 runs, 1-layer kept in 13
-- CPU per run (s): XZnc 25.2, XZn1c 12.0, J 13.6, JH 8.8, J1 10.2; XZnc discrete design 0.002
```


<details><summary>By setting (Q = 5)</summary>

```
===== scenario T2, Q = 5: 144 common runs (XZnc: 144, XZn1c: 144, JH: 144, J1: 144)
-- T2, all settings
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0   24     96.48     95.91    104.77    104.76    104.41    +0.57± 0.47       -8.28± 2.69      24/24/24/24/24
    0.5   24     95.87     95.22    102.52    102.51    102.16    +0.65± 0.51       -6.65± 2.34      24/24/24/24/24
      1   24     91.16     90.11     95.64     95.63     95.34    +1.05± 0.50       -4.48± 1.83      24/24/24/24/24
    1.5   24     84.99     83.08     87.23     87.23     86.94    +1.91± 0.76       -2.24± 1.17      24/24/24/24/24
      2   24     70.37     66.73     70.13     70.40     69.41    +3.64± 1.06       +0.24± 0.84      20/20/20/20/20
    2.5   24     62.05     54.33     55.88     55.88     54.42    +7.72± 2.49       +6.17± 2.90      19/17/16/16/16
    all  144     83.49     80.90     86.03     86.07     85.45    +2.59± 0.52       -2.54± 0.93      135/133/132/132/132
-- T2, setting a
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    0.5    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
      1    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    1.5    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
      2    4     93.80     93.80     93.81     93.81     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    2.5    4     93.55     93.55     93.56     93.56     93.56    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    all   24     93.76     93.76     93.77     93.77     93.77    +0.01± 0.00       -0.01± 0.00      24/24/24/24/24
-- T2, setting b
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4     60.56     59.71     73.49     73.49     73.49    +0.85± 0.68      -12.93± 1.34      4/4/4/4/4
    0.5    4     60.13     59.28     67.64     67.64     67.64    +0.85± 0.69       -7.51± 1.33      4/4/4/4/4
      1    4     57.26     55.33     59.67     59.67     59.67    +1.93± 0.82       -2.42± 1.00      4/4/4/4/4
    1.5    4     50.48     47.88     50.97     50.97     50.97    +2.60± 1.66       -0.49± 1.41      4/4/4/4/4
      2    4     43.81     39.69     42.45     42.45     42.45    +4.12± 2.00       +1.35± 1.88      4/4/4/4/4
    2.5    4     30.11      8.08      0.00      0.00      0.00   +22.02± 9.59      +30.11±10.15      3/1/0/0/0
    all   24     50.39     44.99     49.04     49.04     49.04    +5.40± 2.15       +1.35± 3.26      23/21/20/20/20
-- T2, setting c
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4     96.70     96.70     96.70     96.70     96.70    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    0.5    4     93.91     93.91     93.91     93.91     93.91    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      1    4     79.27     79.27     79.27     79.27     79.27    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    1.5    4     58.57     58.57     58.57     58.57     58.57    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      2    4      0.00      0.00      0.00      0.00      0.00    +0.00± 0.00       +0.00± 0.00      0/0/0/0/0
    2.5    4      0.00      0.00      0.00      0.00      0.00    +0.00± 0.00       +0.00± 0.00      0/0/0/0/0
    all   24     54.74     54.74     54.74     54.74     54.74    +0.00± 0.00       +0.00± 0.00      16/16/16/16/16
-- T2, setting d
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4     78.77     74.75    113.38    113.38    113.38    +4.02± 1.40      -34.61± 3.12      4/4/4/4/4
    0.5    4     78.77     74.31    109.01    109.01    109.01    +4.46± 1.57      -30.24± 3.13      4/4/4/4/4
      1    4     77.17     72.53     99.85     99.85     99.85    +4.65± 2.14      -22.67± 3.92      4/4/4/4/4
    1.5    4     76.40     68.05     88.14     88.14     88.14    +8.35± 2.37      -11.75± 4.75      4/4/4/4/4
      2    4     72.82     61.37     74.76     74.76     74.76   +11.45± 2.39       -1.95± 4.68      4/4/4/4/4
    2.5    4     67.52     52.61     60.94     60.94     60.94   +14.91± 4.75       +6.58± 6.56      4/4/4/4/4
    all   24     75.24     67.27     91.01     91.01     91.01    +7.97± 1.29      -15.77± 3.50      24/24/24/24/24
-- T2, setting e
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4    133.91    133.91    133.91    133.91    133.91    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    0.5    4    133.43    133.43    133.43    133.43    133.43    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      1    4    124.60    124.60    124.60    124.60    124.60    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    1.5    4    116.82    116.82    116.82    116.82    116.82    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      2    4    101.53    101.53    101.53    101.53    101.53    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    2.5    4     80.51     80.51     80.51     80.51     80.51    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    all   24    115.13    115.13    115.13    115.13    115.13    +0.00± 0.00       +0.00± 0.00      24/24/24/24/24
-- T2, setting f
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4    115.15    116.60    117.30    117.27    115.16    -1.44± 1.44       -2.14± 1.66      4/4/4/4/4
    0.5    4    115.15    116.60    117.30    117.27    115.16    -1.44± 1.44       -2.14± 1.66      4/4/4/4/4
      1    4    114.84    115.14    116.61    116.58    114.85    -0.30± 0.29       -1.77± 1.74      4/4/4/4/4
    1.5    4    113.89    113.40    115.09    115.08    113.33    +0.49± 0.44       -1.20± 1.25      4/4/4/4/4
      2    4    110.26    104.01    108.19    109.85    103.91    +6.25± 2.51       +2.07± 1.91      4/4/4/4/4
    2.5    4    100.61     91.23    100.26    100.26     91.50    +9.39± 4.87       +0.35± 0.12      4/4/4/4/4
    all   24    111.65    109.50    112.46    112.72    108.99    +2.16± 1.23       -0.80± 0.63      24/24/24/24/24
-- XZnc: mean served streams 9.12, group-common served 1.79, private served 6.33; BS-specific in 48/144 runs; per run, carried by a subset of the BSs: group-common 0.38, private 1.33
-- J: stream-set test made in 48/144 runs, 1-layer kept in 16
-- CPU per run (s): XZnc 5.9, XZn1c 4.0, J 7.7, JH 6.1, J1 4.4; XZnc discrete design 0.001
===== scenario E3, Q = 5: 144 common runs (XZnc: 144, XZn1c: 144, JH: 144, J1: 144)
-- E3, all settings
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0   24    113.52    111.32    128.91    128.82    128.91    +2.20± 0.71      -15.39± 3.25      24/24/24/24/24
    0.5   24    112.23    109.26    122.90    122.83    122.90    +2.97± 0.75      -10.67± 2.97      24/24/24/24/24
      1   24    109.31    104.34    113.25    113.24    113.25    +4.97± 1.02       -3.94± 2.36      24/24/24/24/24
    1.5   24    106.15     96.23    101.61    101.60    101.71    +9.91± 1.74       +4.54± 2.11      24/24/24/24/24
      2   24    102.36     81.29     83.68     83.68     83.66   +21.07± 4.33      +18.69± 4.45      24/20/20/20/20
    2.5   24     97.80     69.15     70.64     70.64     70.63   +28.65± 4.97      +27.16± 4.95      23/16/16/16/16
    all  144    106.90     95.27    103.50    103.47    103.51   +11.63± 1.41       +3.40± 1.90      143/132/132/132/132
-- E3, setting a
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4    125.65    125.65    125.65    125.65    125.65    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    0.5    4    125.56    125.56    125.56    125.56    125.56    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      1    4    125.42    125.42    125.42    125.42    125.42    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    1.5    4    125.25    125.25    125.25    125.25    125.25    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      2    4    125.04    125.04    125.04    125.04    125.04    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    2.5    4    125.14    125.14    125.14    125.14    125.14    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    all   24    125.34    125.34    125.34    125.34    125.34    +0.00± 0.00       +0.00± 0.00      24/24/24/24/24
-- E3, setting b
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4     62.35     58.89     72.05     72.05     72.05    +3.46± 0.82       -9.70± 2.02      4/4/4/4/4
    0.5    4     61.58     57.38     64.42     64.42     64.42    +4.19± 1.00       -2.85± 1.45      4/4/4/4/4
      1    4     60.84     53.97     55.60     55.60     55.60    +6.86± 1.47       +5.24± 1.71      4/4/4/4/4
    1.5    4     59.66     46.46     46.85     46.85     46.85   +13.19± 1.23      +12.81± 1.62      4/4/4/4/4
      2    4     58.49     38.73     38.17     38.17     38.17   +19.77± 1.81      +20.32± 1.99      4/4/4/4/4
    2.5    4     55.58      0.00      0.00      0.00      0.00   +55.58± 2.59      +55.58± 2.59      4/0/0/0/0
    all   24     59.75     42.57     46.18     46.18     46.18   +17.18± 3.81      +13.57± 4.47      24/20/20/20/20
-- E3, setting c
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4     95.28     95.18    113.87    113.87    113.87    +0.10± 0.10      -18.58± 1.35      4/4/4/4/4
    0.5    4     89.21     87.49     97.36     97.36     97.36    +1.72± 0.71       -8.16± 1.16      4/4/4/4/4
      1    4     77.04     70.32     76.29     76.29     76.29    +6.73± 2.22       +0.75± 1.76      4/4/4/4/4
    1.5    4     69.46     50.27     54.52     54.52     54.52   +19.18± 3.75      +14.93± 2.84      4/4/4/4/4
      2    4     59.43      0.00      0.00      0.00      0.00   +59.43± 6.32      +59.43± 6.32      4/0/0/0/0
    2.5    4     40.90      0.00      0.00      0.00      0.00   +40.90±14.00      +40.90±14.00      3/0/0/0/0
    all   24     71.89     50.54     57.01     57.01     57.01   +21.34± 5.15      +14.88± 6.18      23/16/16/16/16
-- E3, setting d
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4     95.90     87.80    115.43    115.43    115.43    +8.10± 1.04      -19.52± 3.63      4/4/4/4/4
    0.5    4     95.54     87.04    109.73    109.73    109.73    +8.50± 0.88      -14.18± 3.80      4/4/4/4/4
      1    4     94.90     82.55     99.18     99.18     99.18   +12.35± 0.56       -4.28± 4.55      4/4/4/4/4
    1.5    4     93.88     77.00     86.12     86.12     86.12   +16.88± 2.26       +7.76± 5.22      4/4/4/4/4
      2    4     91.92     67.47     72.35     72.35     72.35   +24.46± 4.35      +19.57± 6.01      4/4/4/4/4
    2.5    4     91.45     55.37     58.64     58.64     58.64   +36.08± 4.74      +32.81± 5.84      4/4/4/4/4
    all   24     93.93     76.20     90.24     90.24     90.24   +17.73± 2.31       +3.69± 4.24      24/24/24/24/24
-- E3, setting e
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4    135.25    133.18    179.41    179.41    179.41    +2.07± 1.85      -44.16± 4.17      4/4/4/4/4
    0.5    4    134.91    131.12    173.48    173.48    173.48    +3.80± 2.32      -38.57± 5.35      4/4/4/4/4
      1    4    131.98    128.05    157.45    157.45    157.45    +3.93± 1.67      -25.47± 5.44      4/4/4/4/4
    1.5    4    126.18    115.95    135.06    135.06    135.06   +10.24± 2.18       -8.87± 5.69      4/4/4/4/4
      2    4    123.02    100.26    110.21    110.21    110.21   +22.76± 3.03      +12.82± 5.64      4/4/4/4/4
    2.5    4    118.43     79.19     84.76     84.76     84.76   +39.24± 4.51      +33.68± 6.59      4/4/4/4/4
    all   24    128.30    114.62    140.06    140.06    140.06   +13.67± 2.96      -11.76± 6.13      24/24/24/24/24
-- E3, setting f
   R_th    n      XZnc     XZn1c         J        JH        J1        XZnc-XZn1c            XZnc-J   QoS met XZnc/XZn1c/J/JH/J1
      0    4    166.66    167.20    167.03    166.53    167.04    -0.54± 0.50       -0.37± 0.58      4/4/4/4/4
    0.5    4    166.61    166.99    166.86    166.43    166.87    -0.39± 0.34       -0.25± 0.47      4/4/4/4/4
      1    4    165.67    165.70    165.53    165.51    165.54    -0.03± 0.02       +0.14± 0.15      4/4/4/4/4
    1.5    4    162.45    162.47    161.85    161.82    162.46    -0.02± 0.03       +0.61± 0.58      4/4/4/4/4
      2    4    156.27    156.27    156.28    156.28    156.22    -0.00± 0.02       -0.01± 0.01      4/4/4/4/4
    2.5    4    155.31    155.22    155.32    155.32    155.25    +0.09± 0.07       -0.01± 0.00      4/4/4/4/4
    all   24    162.16    162.31    162.15    161.98    162.23    -0.15± 0.10       +0.02± 0.16      24/24/24/24/24
-- XZnc: mean served streams 11.21, group-common served 2.71, private served 7.50; BS-specific in 96/144 runs; per run, carried by a subset of the BSs: group-common 1.92, private 3.67
-- J: stream-set test made in 24/144 runs, 1-layer kept in 13
-- CPU per run (s): XZnc 25.2, XZn1c 12.0, J 13.6, JH 8.8, J1 10.2; XZnc discrete design 0.002
```

</details>

### 3.3 R2 and R3: the benchmark setting (`--tier bench`, Q = 50)

```
===== benchmark setting (T2, K = 16, N = 3, S_max = 12, R_th = 2, Q = 50): 15 common runs (XZnc: 15, sparse_bf: 15, XZnc@SC: 15, J: 15, J@SC: 15)
sparse_bf, J (the current framework) and J@SC (stream clustering on its streams): stored runs of Fig. 5.10
-- by M
      M    n      XZnc sparse_bf   XZnc@SC         J      J@SC    XZnc-sparse_bf      XZnc-XZnc@SC   QoS met XZnc/sparse_bf/XZnc@SC/J/J@SC
    100    5     74.68     63.08     46.34     73.51     44.18   +11.61± 3.43      +28.35± 2.43      5/5/5/5/5
    300    5     81.44     75.09     55.63     81.95     54.38    +6.35± 6.04      +25.81± 2.44      5/5/5/5/5
    500    5     85.68     81.03     62.09     88.72     64.51    +4.65± 4.33      +23.59± 4.15      5/5/5/5/5
    all   15     80.60     73.07     54.69     81.39     54.36    +7.54± 2.65      +25.92± 1.74      15/15/15/15/15
-- XZnc: mean served streams 11.40, group-common served 4.40, private served 6.00; BS-specific in 0/15 runs; per run, carried by a subset of the BSs: group-common 0.00, private 0.00
-- XZnc: shares of the sum rate over its 15 runs meeting the QoS: super-common 15.6%, group-common 9.2%, private 75.1%
```


* **R2:** the design (80.6) beats sparse beamforming (73.1) by +7.5 ± 2.6 over the 15 paired
  runs. Its mean is ahead at every $M$: clearly at $M=100$, by about one standard error at $M=300$ and 500. It beats
  stream clustering on its own streams and groups (`XZnc@SC`, built from `d.groups`, `d.priority` and `d.core`, with
  the design's AO start; 54.7) by +25.9 ± 1.7, and by at least 23.6 at every $M$. Stream
  clustering on J's streams (J@SC, stored) reaches 54.4. In this setting ($n=6=BN$, $m=2=B$) every BS carries every private stream, and the free slots carry all
  group-common streams (4.4 per run). The design is then the current design
  without its non-core private stream, which J serves in drops with 4 groups, and its rate is that of J (81.4).
* **R3:** shares of the sum rate (Fig. 5.7 definition, over the 15 runs, all meeting the QoS): super-common
  15.6%, group-common 9.2%. In every run the two common layers carry the QoS of the 10 non-core UEs
  ($r_c+r_g$ = 20.0–20.3 bit/s/Hz, $10R_{th}=20$), the group-common streams 4.0–9.6 of it.
  The group-common share is just under 10%, the super-common share well over it.

### 3.4 R4: sum rate versus $S^{\max}$ (`--tier smax`, Q = 50)

```
===== sum rate versus S_max (T2, N = 3, M = 300, Q = 50): 126 runs of XZnc
-- K = 8: mean sum rate over the drops (QoS met / runs)
  R_th          S=4          S=6          S=8         S=10         S=12         S=14         S=16
     0   71.76  2/2   101.91  2/2   101.61  2/2   102.22  2/2   102.22  2/2   102.22  2/2   102.22  2/2 
     1   70.11  2/2    98.44  2/2   103.04  2/2   102.34  2/2   102.34  2/2   102.34  2/2   102.34  2/2 
     2   65.43  2/2    99.06  2/2   103.46  2/2   102.07  2/2   102.07  2/2   102.07  2/2   102.07  2/2 
   all   69.10  6/6    99.80  6/6   102.70  6/6   102.21  6/6   102.21  6/6   102.21  6/6   102.21  6/6 
   change to the next S_max (paired over R_th and drops): 4->6 +30.70±2.48, 6->8 +2.90±1.49, 8->10 -0.49±0.46, 10->12 +0.00±0.00, 12->14 +0.00±0.00, 14->16 +0.00±0.00
-- K = 12: mean sum rate over the drops (QoS met / runs)
  R_th          S=4          S=6          S=8         S=10         S=12         S=14         S=16
     0   72.37  2/2   113.44  2/2   121.29  2/2   124.14  2/2   123.78  2/2   123.78  2/2   123.78  2/2 
     1   68.31  2/2   106.67  2/2   120.43  2/2   118.51  2/2   119.36  2/2   119.36  2/2   119.36  2/2 
     2   55.35  2/2    82.86  2/2   100.05  2/2   105.71  2/2   105.79  2/2   105.79  2/2   105.79  2/2 
   all   65.34  6/6   100.99  6/6   113.92  6/6   116.12  6/6   116.31  6/6   116.31  6/6   116.31  6/6 
   change to the next S_max (paired over R_th and drops): 4->6 +35.65±2.99, 6->8 +12.93±2.43, 8->10 +2.20±2.79, 10->12 +0.19±0.33, 12->14 +0.00±0.00, 14->16 +0.00±0.00
-- K = 16: mean sum rate over the drops (QoS met / runs)
  R_th          S=4          S=6          S=8         S=10         S=12         S=14         S=16
     0   73.09  2/2   116.64  2/2   122.94  2/2   122.62  2/2   122.63  2/2   122.63  2/2   122.63  2/2 
     1   67.82  2/2    94.37  2/2   112.45  2/2   110.03  2/2   115.43  2/2   115.43  2/2   115.43  2/2 
     2   50.02  2/2    50.59  2/2    65.84  2/2    73.47  2/2    91.99  2/2    91.99  2/2    91.99  2/2 
   all   63.65  6/6    87.20  6/6   100.41  6/6   102.04  6/6   110.02  6/6   110.02  6/6   110.02  6/6 
   change to the next S_max (paired over R_th and drops): 4->6 +23.56±7.96, 6->8 +13.21±2.99, 8->10 +1.63±2.19, 10->12 +7.98±4.10, 12->14 +0.00±0.00, 14->16 +0.00±0.00
```


**R4.** The mean sum rate rises with $S^{\max}$ for every $K$ and flattens once all private and group-common streams
are served ($S^{\max}\ge10$–12). Negative mean steps: K = 8, 8→10: -0.49 ± 0.46. These are within one or two
standard errors of zero, on a flat curve, and come from single cells at $R_{th}\in\{1,2\}$ where the design changes
from BS-specific to joint group-common streams. The design changes with $S^{\max}$: in T2 with $N=3$,
$S^{\max}=4$ gives 3 private streams carried by one BS each, plus group-common streams; $S^{\max}=6$ gives 5 jointly
carried private streams and no group-common stream; $S^{\max}\ge8$ gives 6 jointly carried private streams plus
group-common streams. The rate still rises, including $K=16$ at $R_{th}=2$ from $S^{\max}=4$ to 6, where the
group-common streams of $S^{\max}=4$ meet the QoS.

### 3.5 Design requirement: where the BSs carry different streams

Per setting, over the 24 screen runs (the design does not depend on $R_{th}$; `p`$k$ = private stream of UE $k$, `g`$g$ = group-common stream of group $g$, `c` = super-common stream):

| Scen. | setting ($N, K, S^{\max}$) | $n$ / $m$ | runs with BS-specific streams | private streams on a subset of the BSs (per run) | group-common streams on a subset | group-common streams served | example: streams per BS, drop 901 |
|---|---|---|---|---|---|---|---|
| T2 | a (3, 8, 12) | 6 / 2 = B | 0/24 | 0.00 | 0.00 | 2.00 | all BSs: c g1 g2 p1 p3 p4 p6 p7 p8 |
| T2 | b (3, 12, 4) | 3 / 1 | 24/24 | 3.00 | 1.25 | 2.00 | BS1: c g1 p2 p3; BS2: c g1 g2 p7 |
| T2 | c (3, 16, 6) | 5 / 2 = B | 0/24 | 0.00 | 0.00 | 0.00 | all BSs: c p1 p6 p7 p12 p14 |
| T2 | d (5, 12, 6) | 5 / 1 | 24/24 | 5.00 | 1.00 | 2.75 | BS1: c g1 g3 p2 p3 p12; BS2: c g1 g3 g4 p6 p7 |
| T2 | e (5, 16, 10) | 9 / 2 = B | 0/24 | 0.00 | 0.00 | 0.00 | all BSs: c p1 p2 p4 p5 p6 p7 p8 p12 p15 |
| T2 | f (5, 16, 20) | 10 / 2 = B | 0/24 | 0.00 | 0.00 | 4.00 | all BSs: c g1 g2 g3 g4 p1 p2 p4 p5 p6 p7 p8 p12 p14 p15 |
| E3 | a (3, 8, 12) | 8 / 3 = B | 0/24 | 0.00 | 0.00 | 0.00 | all BSs: c p1 p2 p3 p4 p5 p6 p7 p8 |
| E3 | b (3, 12, 4) | 3 / 1 | 24/24 | 3.00 | 1.75 | 2.75 | BS1: c g1 g2 p9; BS2: c g1 g2 p6; BS3: c g1 g2 p3 |
| E3 | c (3, 16, 6) | 5 / 2 | 24/24 | 5.00 | 4.50 | 4.50 | BS1: c g2 g3 g4 p13 p15; BS2: c g3 p1 p6 p9 p13; BS3: c g1 p1 p6 p9 p15 |
| E3 | d (5, 12, 6) | 5 / 1 | 24/24 | 5.00 | 1.50 | 3.75 | BS1: c g1 g3 p1 p9 p12; BS2: c g1 g2 g3 g4 p6; BS3: c g1 g2 g3 g4 p3 |
| E3 | e (5, 16, 10) | 9 / 2 | 24/24 | 9.00 | 3.75 | 4.25 | BS1: c g1 g2 g4 p4 p7 p8 p11 p13 p15; BS2: c g1 g2 g3 g4 p1 p4 p6 p9 p13; BS3: c g1 g3 p1 p6 p7 p8 p9 p11 p15 |
| E3 | f (5, 16, 20) | 15 / 3 = B | 0/24 | 0.00 | 0.00 | 1.00 | all BSs: c g1 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p15 p16 |

In T2 settings b and d and E3 settings b and d, every core user is served by its strongest BS only: each BS has core
users of its own. In E3 settings c and e, every core user is served by 2 of the 3 BSs, so that every BS shares core
users with another BS. The group-common streams differ between BSs wherever the free slots are fewer than the groups.
Where the private streams need all BSs for zero-forcing ($m=B$), the transmission is joint, and so is every
group-common stream that fits (T2 a, f; E3 f; the benchmark setting).

### 3.6 Sensitivity check: a counterpart that fills the free slots with non-core private streams

`bsd.design_1layer_fill` keeps the design's core users and BSs and fills every BS's free slots with private
streams of non-core UEs, in decreasing order of its energy to them (as J1 fills the slots of the group-common streams with
the next private streams). Its screen runs are those of the study variant `XZ1c`, whose design function produces the
identical design (verified on all 239 grid, benchmark and $S^{\max}$ configurations). The design against it
(Q = 5, 24 paired runs per $R_{th}$):

| | $R_{th}$ = 0 | $R_{th}$ = 0.5 | $R_{th}$ = 1 | $R_{th}$ = 1.5 | $R_{th}$ = 2 | $R_{th}$ = 2.5 | all |
|---|---|---|---|---|---|---|---|
| T2 | +0.66 ± 0.31 | +0.69 ± 0.33 | +0.83 ± 0.38 | +1.18 ± 0.61 | +1.80 ± 0.99 | +1.31 ± 1.88 | +1.08 ± 0.38 |
| E3 | +0.33 ± 0.77 | -0.69 ± 0.89 | +0.82 ± 1.11 | +3.25 ± 1.37 | +4.56 ± 1.70 | +9.92 ± 3.81 | +3.03 ± 0.82 |

The design's mean is ahead of this stronger counterpart at every $R_{th}$ in T2 (by about two standard errors up
to $R_{th}=2$, within one at 2.5), and clearly ahead in E3 from $R_{th}=1.5$ on; in E3 at $R_{th}\le1$ the two tie.
There the counterpart's non-core private streams, carried by one BS
each, use spatial dimensions that the few private streams leave free: they carry 6.9 bit/s/Hz per run on average
in E3, against none in the current design's runs.

### 3.7 The AO start on the carried antennas

The alternative core size (`BSD`, carried-antenna start) against the identical design with the default start
(`XZDn`, design function `bsx.ZDn`, verified identical), screen tier, 144 paired runs per scenario. The runs with
joint transmission are bit-identical, so the difference comes from the BS-specific settings. (In T2 this is also the
final design, which is identical to `BSD` there.)

| | $R_{th}$ = 0 | $R_{th}$ = 0.5 | $R_{th}$ = 1 | $R_{th}$ = 1.5 | $R_{th}$ = 2 | $R_{th}$ = 2.5 | all | CPU per run |
|---|---|---|---|---|---|---|---|---|
| T2 | +1.49 ± 0.83 | +0.45 ± 0.65 | +0.95 ± 0.55 | +1.37 ± 0.59 | +1.23 ± 0.56 | +2.08 ± 1.60 | +1.26 ± 0.35 | 5.7 s vs 6.9 s |
| E3 | +0.84 ± 1.26 | +1.18 ± 1.24 | +0.54 ± 0.70 | +0.78 ± 0.80 | +0.47 ± 0.95 | +1.58 ± 0.81 | +0.90 ± 0.40 | 22.0 s vs 30.6 s |

From the carried-antenna start the AO converges to better solutions in fewer iterations.

## 4. Other methods tried

### 4.1 The alternative core size: as many private streams as the fronthaul can zero-force (`BSD`)

The first complete candidate, and the design's closest alternative, takes the largest number of private streams that the
whole fronthaul can carry with zero-forcing,
$$n=\max\{\nu\le\min\{K,BN\}:\ \nu\lceil\nu/N\rceil\le B(S^{\max}-1)\}$$
(`bsd.design_dof`; variants `BSD`, `BSD1`, `BSD@SC`, `BSD1f`), with everything else as in the design. It spends the
fronthaul freed by the zero-forcing-minimal carriage on more private streams instead of group-common streams. It is the
same design in T2 for every setting of the grid, the benchmark setting and the $S^{\max}$ sweep, and all its T2 runs
are bit-identical to the design's. In E3 it serves 4, 6, 7, 10 private streams in settings b–e, against the design's
3, 5, 5, 9, and leaves the group-common streams 1, 3, 1, 7 free slots. Its rate in E3 at Q = 50 is higher than the
design's at low $R_{th}$. The design minus BSD by $R_{th}$ (0…2.5): -10.9, -9.5, -8.5, -5.4, -2.6, +5.3
(by setting: b -8.9, c -3.5, d -25.0, e +5.8). Its private layer grows with the network
(Section 1.6). But its group-common layer is too small to beat its own 1-layer counterpart at $R_{th}=1$ in E3:

| BSD − BSD1 | $R_{th}$ = 0 | 0.5 | 1 | 1.5 | 2 | 2.5 |
|---|---|---|---|---|---|---|
| E3, Q = 50 (confirm) | +0.74 ± 1.04 | -2.35 ± 1.34 | -0.13 ± 1.31 | +1.82 ± 2.00 | +4.55 ± 2.79 | +8.10 ± 5.20 |
| E3, Q = 5 (screen) | +0.05 ± 0.17 | +0.70 ± 0.56 | +0.56 ± 0.23 | +2.29 ± 0.81 | +6.70 ± 2.58 | +8.68 ± 3.28 |
| T2 (= the design) | see Section 3 | | | | | |

It meets R2, R3 and R4 exactly as the design does, since those tiers are T2 runs. It was not chosen because R1 at
$R_{th}=1$ in E3 is a tie at Q = 50. It is the better choice for networks with many BSs (Section 1.6). Its full
reports: `results/BSD__report*.json`.

<details><summary>BSD: R1 reports (Q = 50 and Q = 5)</summary>

```
===== scenario T2, Q = 50: 108 common runs (BSD: 108, BSD1: 108, JH: 108, J1: 108)
-- T2, all settings
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0   18    117.20    116.59    124.68    123.91    123.76    +0.61± 1.26       -7.48± 2.99      18/18/18/18/18
    0.5   18    115.47    114.61    121.41    120.44    120.85    +0.87± 0.65       -5.93± 2.70      18/18/18/18/18
      1   18    110.32    109.01    113.54    113.02    112.95    +1.31± 0.72       -3.21± 1.79      18/18/18/18/18
    1.5   18    101.18    100.88    103.61    103.47    103.59    +0.30± 0.96       -2.43± 1.90      18/18/18/18/18
      2   18     94.28     90.65     93.54     94.06     93.22    +3.62± 1.28       +0.73± 0.81      18/18/18/18/18
    2.5   18     79.97     75.35     76.86     77.34     75.61    +4.61± 1.92       +3.11± 1.47      15/15/15/15/15
    all  108    103.07    101.18    105.61    105.37    105.00    +1.89± 0.51       -2.54± 0.90      105/105/105/105/105
-- BSD: mean served streams 9.11, group-common served 1.78, private served 6.33; BS-specific in 36/108 runs; per run, carried by a subset of the BSs: group-common 0.39, private 1.33
-- J: stream-set test made in 36/108 runs, 1-layer kept in 12
-- CPU per run (s): BSD 64.2, BSD1 44.1, J 67.8, JH 73.6, J1 49.9; BSD discrete design 0.001
===== scenario E3, Q = 50: 108 common runs (BSD: 108, BSD1: 108, JH: 108, J1: 108)
-- E3, all settings
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0   18    132.70    131.96    139.50    139.60    139.09    +0.74± 1.04       -6.80± 5.82      18/18/18/18/18
    0.5   18    129.10    131.46    133.02    133.37    132.32    -2.35± 1.34       -3.92± 6.06      18/18/18/18/18
      1   18    126.36    126.49    124.99    126.26    124.38    -0.13± 1.31       +1.37± 5.32      18/18/18/18/18
    1.5   18    120.12    118.30    112.79    112.47    112.00    +1.82± 2.00       +7.33± 4.26      18/18/18/18/18
      2   18    114.84    110.29     95.46     95.46     95.61    +4.55± 2.79      +19.38± 5.87      18/18/16/16/16
    2.5   18    102.54     94.44     78.43     79.12     78.16    +8.10± 5.20      +24.11± 5.98      17/15/12/12/12
    all  108    120.94    118.82    114.03    114.38    113.59    +2.12± 1.12       +6.91± 2.49      107/105/100/100/100
-- BSD: mean served streams 10.89, group-common served 1.56, private served 8.33; BS-specific in 72/108 runs; per run, carried by a subset of the BSs: group-common 1.39, private 4.50
-- J: stream-set test made in 18/108 runs, 1-layer kept in 9
-- CPU per run (s): BSD 195.4, BSD1 117.6, J 100.6, JH 89.8, J1 100.2; BSD discrete design 0.002
```

```
===== scenario T2, Q = 5: 144 common runs (BSD: 144, BSD1: 144, JH: 144, J1: 144)
-- T2, all settings
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0   24     96.48     95.91    104.77    104.76    104.41    +0.57± 0.47       -8.28± 2.69      24/24/24/24/24
    0.5   24     95.87     95.22    102.52    102.51    102.16    +0.65± 0.51       -6.65± 2.34      24/24/24/24/24
      1   24     91.16     90.11     95.64     95.63     95.34    +1.05± 0.50       -4.48± 1.83      24/24/24/24/24
    1.5   24     84.99     83.08     87.23     87.23     86.94    +1.91± 0.76       -2.24± 1.17      24/24/24/24/24
      2   24     70.37     66.73     70.13     70.40     69.41    +3.64± 1.06       +0.24± 0.84      20/20/20/20/20
    2.5   24     62.05     54.33     55.88     55.88     54.42    +7.72± 2.49       +6.17± 2.90      19/17/16/16/16
    all  144     83.49     80.90     86.03     86.07     85.45    +2.59± 0.52       -2.54± 0.93      135/133/132/132/132
-- BSD: mean served streams 9.12, group-common served 1.79, private served 6.33; BS-specific in 48/144 runs; per run, carried by a subset of the BSs: group-common 0.38, private 1.33
-- J: stream-set test made in 48/144 runs, 1-layer kept in 16
-- CPU per run (s): BSD 5.7, BSD1 4.3, J 7.7, JH 6.1, J1 4.4; BSD discrete design 0.001
===== scenario E3, Q = 5: 144 common runs (BSD: 144, BSD1: 144, JH: 144, J1: 144)
-- E3, all settings
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0   24    122.71    122.66    128.91    128.82    128.91    +0.05± 0.17       -6.19± 4.73      24/24/24/24/24
    0.5   24    121.22    120.52    122.90    122.83    122.90    +0.70± 0.56       -1.68± 4.24      24/24/24/24/24
      1   24    116.05    115.49    113.25    113.24    113.25    +0.56± 0.23       +2.80± 3.61      24/24/24/24/24
    1.5   24    109.56    107.28    101.61    101.60    101.71    +2.29± 0.81       +7.96± 2.91      24/24/24/24/24
      2   24    102.15     95.44     83.68     83.68     83.66    +6.70± 2.58      +18.47± 4.49      24/23/20/20/20
    2.5   24     92.70     84.02     70.64     70.64     70.63    +8.68± 3.28      +22.06± 4.44      22/20/16/16/16
    all  144    110.73    107.57    103.50    103.47    103.51    +3.16± 0.76       +7.24± 1.86      142/139/132/132/132
-- BSD: mean served streams 10.96, group-common served 1.62, private served 8.33; BS-specific in 96/144 runs; per run, carried by a subset of the BSs: group-common 1.46, private 4.50
-- J: stream-set test made in 24/144 runs, 1-layer kept in 13
-- CPU per run (s): BSD 22.0, BSD1 12.8, J 13.6, JH 8.8, J1 10.2; BSD discrete design 0.001
```

</details>

### 4.2 The design study

All at Q = 5 (screen tier, or its first 2 drops at $R_{th}\in\{0,1,2\}$ for the early pilots). R1 column: the variant
minus its counterpart for each $R_{th}$ present; then the variant's mean rate and its mean difference to J on the same runs.
Code: `code/bsx.py` (study variants) and `code/bsd.py`; variants in `harness/variants.py`; runs `results/X*__*__Q5.jsonl`.

| Variant | Idea | Scen. | Runs | R1 per $R_{th}$ | Rate | − J |
|---|---|---|---|---|---|---|
| XA | class order of the current design, per-BS ranking by own channel energy | T2 | 36 (2 drops, 3 $R_{th}$) | +0.57 / +0.57 / +0.56 | 81.6 | -7.6 |
| XA | class order of the current design, per-BS ranking by own channel energy | E3 | 36 (2 drops, 3 $R_{th}$) | -0.16 / -0.01 / +0.02 | 107.8 | -0.1 |
| XC | each BS: own core users, own groups, then cooperation | T2 | 36 (2 drops, 3 $R_{th}$) | -3.83 / -3.33 / -0.12 | 78.6 | -10.6 |
| XC | each BS: own core users, own groups, then cooperation | E3 | 36 (2 drops, 3 $R_{th}$) | -1.24 / -1.94 / -1.90 | 106.2 | -1.8 |
| XM | core private and group-common streams ranked together by energy to the weakest decoder | T2 | 36 (2 drops, 3 $R_{th}$) | -2.43 / -1.10 / +1.57 | 80.3 | -8.8 |
| XM | core private and group-common streams ranked together by energy to the weakest decoder | E3 | 36 (2 drops, 3 $R_{th}$) | -2.01 / -2.33 / -1.42 | 105.9 | -2.0 |
| XZ | zero-forcing-minimal carriage, $n=\min\{K,BN,S^{\max}-1\}$, counterpart fills free slots with non-core private streams | T2 | 36 (2 drops, 3 $R_{th}$) | +0.95 / +2.52 / +1.40 | 85.4 | -3.7 |
| XZ | zero-forcing-minimal carriage, $n=\min\{K,BN,S^{\max}-1\}$, counterpart fills free slots with non-core private streams | E3 | 36 (2 drops, 3 $R_{th}$) | -0.76 / -2.06 / +2.62 | 105.3 | -2.6 |
| XZc | XZ with the carried-antenna start | T2 | 144 (4 drops, 6 $R_{th}$) | +1.03 / +1.06 / +1.14 / +1.38 / +1.83 / +1.39 | 83.7 | -2.3 |
| XZc | XZ with the carried-antenna start | E3 | 144 (4 drops, 6 $R_{th}$) | +0.30 / -0.72 / +0.79 / +3.14 / +4.86 / +10.26 | 107.0 | +3.5 |
| XZDc | zero-forcing-minimal, DoF-max $n$, free slots: groups, then non-core private; counterpart: non-core private | T2 | 36 (2 drops, 3 $R_{th}$) | +1.27 / +1.53 / +1.80 | 86.4 | -2.7 |
| XZDc | zero-forcing-minimal, DoF-max $n$, free slots: groups, then non-core private; counterpart: non-core private | E3 | 144 (4 drops, 6 $R_{th}$) | -0.12 / +0.41 / +0.56 / +1.15 / +1.46 / +4.15 | 110.7 | +7.2 |
| XZDn | DoF-max $n$, core-only private streams (= BSD with the default start) | T2 | 144 (4 drops, 6 $R_{th}$) | +0.11 / +0.65 / +0.85 / +1.88 / +2.69 / +7.29 | 82.2 | -3.8 |
| XZDn | DoF-max $n$, core-only private streams (= BSD with the default start) | E3 | 144 (4 drops, 6 $R_{th}$) | -0.09 / -0.62 / +0.79 / +2.75 / +6.91 / +7.45 | 109.8 | +6.3 |
| BSD | DoF-max $n$, core-only private streams, carried-antenna start (Section 4.1) | T2 | 144 (4 drops, 6 $R_{th}$) | +0.57 / +0.65 / +1.05 / +1.91 / +3.64 / +7.72 | 83.5 | -2.5 |
| BSD | DoF-max $n$, core-only private streams, carried-antenna start (Section 4.1) | E3 | 144 (4 drops, 6 $R_{th}$) | +0.05 / +0.70 / +0.56 / +2.29 / +6.70 / +8.68 | 110.7 | +7.2 |
| BSD vs fill | BSD against the counterpart that fills free slots with non-core private streams | T2 | 144 (4 drops, 6 $R_{th}$) | +0.66 / +0.69 / +0.83 / +1.18 / +1.80 / +1.31 | 83.5 | -2.5 |
| BSD vs fill | BSD against the counterpart that fills free slots with non-core private streams | E3 | 144 (4 drops, 6 $R_{th}$) | -0.11 / +0.43 / +0.59 / +1.26 / +1.45 / +4.15 | 110.7 | +7.2 |
| XZnc (the design) | $n=\min\{K,BN,S^{\max}-1\}$, core-only private streams, carried-antenna start | T2 | 144 (4 drops, 6 $R_{th}$) | +0.57 / +0.65 / +1.05 / +1.91 / +3.64 / +7.72 | 83.5 | -2.5 |
| XZnc (the design) | $n=\min\{K,BN,S^{\max}-1\}$, core-only private streams, carried-antenna start | E3 | 144 (4 drops, 6 $R_{th}$) | +2.20 / +2.97 / +4.97 / +9.91 / +21.07 / +28.65 | 106.9 | +3.4 |
| XZnc vs fill | the design against the fill counterpart (Section 3.6) | T2 | 144 (4 drops, 6 $R_{th}$) | +0.66 / +0.69 / +0.83 / +1.18 / +1.80 / +1.31 | 83.5 | -2.5 |
| XZnc vs fill | the design against the fill counterpart (Section 3.6) | E3 | 144 (4 drops, 6 $R_{th}$) | +0.33 / -0.69 / +0.82 / +3.25 / +4.56 / +9.92 | 106.9 | +3.4 |

* **XA, XC, XM: per-BS rankings of the core streams.** Each BS keeps different core users, so the network serves many
  partially carried private streams: −16 to −44 bit/s/Hz against J at $R_{th}=0$ in settings b–e. Group-common
  streams either appear only where joint transmission already had room for them (XA: R1 a tie) or displace
  cooperation, which is worth more at low $R_{th}$ (XC, XM: R1 negative).
* **Zero-forcing-minimal carriage (XZ…, BSD, the design).** Carrying every private stream by just the BSs needed to
  zero-force it frees fronthaul at small $S^{\max}$. The study variants differ in how many private streams the freed
  fronthaul buys (the current design's number or the DoF-max) and in what the counterpart does with the free slots.
  The design's choice gives the group-common layer the most room.
* Q = 50 subset runs of `XZc`/`XZ1c` were queued but not run; the confirm tier of BSD and of the design took the
  time.

### 4.3 A probe: what the BSs not needed for zero-forcing contribute

In E3, setting e, drop 901, $R_{th}=0$, at the initial deployment, the current design (9 jointly carried private
streams over 15 antennas) reaches 170.9 bit/s/Hz. Carrying each of the same 9 streams by only its 2 strongest BSs
(still enough to zero-force them) reaches 120.9–122.2, and 124.4 when the AO starts from the joint solution with the
third block zeroed. In the dense 100 m area the BS that zero-forcing does not need still adds a large beamforming gain.
This is the price of the fronthaul the design gives to the group-common layer, and why the design is behind J at
low $R_{th}$ in the settings where it does not use all BSs.

## 5. Weak points

* **Lower rate than the current joint design at low $R_{th}$** where the design carries private streams by fewer than
  all BSs (T2 b, d; E3 b–e): the BSs that zero-forcing does not need still add beamforming gain in the dense 100 m
  area (Section 4.3). The user accepted this (BRIEF.md). At high $R_{th}$ the design is better than J.
* **The private layer does not grow with the network.** It is the current design's, $\min\{K,BN,S^{\max}-1\}$ streams.
  With many BSs over a large area most of the fronthaul then goes to group-common streams, which fill the free slots
  of almost every BS (Section 1.6, 36 BSs: each group-common stream at 32 BSs). Section 4.1's core size fixes this
  but loses the clear R1 margin in E3.
* **The 1-layer counterpart leaves free slots unused.** Section 3.6 compares the design with a counterpart that fills
  them with private streams of non-core UEs (Q = 5): ahead in T2, clearly ahead in E3 from $R_{th}=1.5$ on, a tie
  below.
* **Search variability at Q = 50.** Adding a stream changes the search path (its rate score counts every served
  stream), so paired runs differ by up to ±15 bit/s/Hz even where the extra streams carry no rate (Section 3.1).
  At 3 drops the R1 margins in T2 up to $R_{th}=1.5$ (weakest: +0.30 ± 0.96 at 1.5) and in E3 up to $R_{th}=0.5$ are
  within two standard errors.
* **Group-common share 9.2% in the benchmark setting**, just under 10% (super-common 15.6%).
* **Non-nested in $S^{\max}$:** $m$ and the free slots change with $S^{\max}$ (Section 3.4); R4 holds on the
  measured sweep but is not guaranteed by construction.
* **Cost in E3** per run is 2.1 times that of J at Q = 50 and 1.9 times at Q = 5 (Section 2): the design serves
  more distinct streams (11.1 against 8.8 per run for JH at Q = 50). In T2 it costs about as much as J.

## 6. Choices made without asking

* **Label** `cloud_xhigh`, from the requested zip name.
* **Final variant names** `XZnc` / `XZn1c` / `XZnc@SC`. They come from the design study; their design functions
  are `bsd.design` / `bsd.design_1layer`, verified identical to the study functions `bsx.Zn` / `bsx.Zn1` that
  produced the first E3 runs. I kept the names rather than re-running everything under new ones.
* **The 1-layer counterpart** keeps the design's core users and BSs and drops the group-common streams. Their slots
  stay free because the non-core UEs have no private stream in either design. The J1-style alternative is reported
  as a sensitivity check (Section 3.6).
* **AO start on the carried antennas** (Section 1.8): adopted because the design relies on per-BS zero-forcing; it
  is a switch, off by default.
* **Grouping:** the current design's (nearest core by channel correlation), applied to the design's core users.
* **Ties** in the energy rankings are broken by BS index and group index.
* **JH and J1 at Q = 50** were run for reference after the design's tiers, so that the Q = 50 reports show the
  current framework (J) on the same runs. They are not part of any requirement.
* **Machine:** 4 cores. The self-test passed with identical traces before and after the code changes.

## 7. Files

* `DESIGN.md` (this file).
* `code/bsd.py`: the design (`design`, `design_1layer`, `design_1layer_fill`) and the alternative core size
  (`design_dof`, `design_dof_1layer`, `design_dof_1layer_fill`).
* `code/bsx.py`: the study variants of Section 4.
* `code/algo.py`: the switch `INITS['core_carried']` / `SEEDS['rule_carried']` (off by default).
* `harness/variants.py`: the variants of this round; `tools/large_area_demo.py`: Section 1.6.
* `package.diff`: unified diff against the package (code, harness, tools).
* `results/*.jsonl`: every run (harness format, unedited); `results/*report*.json`: `tools/report.py --json`
  outputs.
