# DESIGN.md — a BS-specific discrete design: zero-forcing-minimal cooperation (round v10, label `cloud_xhigh`)

<!-- RESULTS-SUMMARY -->

## 1. The design (thesis-ready description)

### 1.1 Idea

Under joint transmission every served stream occupies one fronthaul slot at every BS, i.e. $B$ slots, whatever the
stream needs. The proposed design gives every stream the BSs it needs and no more, and spends the fronthaul this saves
on the layer of hierarchical rate splitting that exists to use it, the group-common streams:

* the **super-common stream** is decoded by every UE; it is carried by every BS (as required);
* a **private stream** is spatially multiplexed: it must be zero-forced at the other private streams' users. Its precoder
  is carried by a set $\mathcal A$ of BSs and has $|\mathcal A|N$ coefficients, so nulling it at $n-1$ other users needs
  $|\mathcal A|N\ge n$: it needs $m=\lceil n/N\rceil$ BSs, and the design gives it exactly the $m$ BSs with the strongest
  channels to its user;
* a **group-common stream** is decoded by the members of its group before their private streams (SIC). Each BS carries
  group-common streams in the slots its private streams leave free, choosing the groups whose weakest member (the
  decoder that limits the stream's rate) it reaches best.

Private streams are served to the core users only: the number $n$ of core users is the largest whose private streams
the fronthaul can carry with zero-forcing, so a private stream for a further user could not be zero-forced. Every other
UE is served by the common streams: the super-common stream and the group-common stream of its group.

### 1.2 The design

Let $\mathbf h_{d,k}=[\mathbf f_{1,k},\ldots,\mathbf f_{B,k}]$ be the stacked direct channel of UE $k$ and
$E_{k,b}=\|\mathbf f_{b,k}\|_2^2$ its energy from BS $b$; let $M_{b,j}\in\{0,1\}$ indicate that BS $b$ carries stream
$j$, so that \eqref{eq:c-front} reads $\sum_j M_{b,j}\le S^{\max}$.

**Core users.** The UEs are ordered by the greedy projection step of SUS \eqref{eq:core_selection} on the stacked direct
channels (as in the current design), $u_1,u_2,\ldots$. The core users are the first
$$
n=\max\Bigl\{\nu\le\min\{K,BN\}:\ \nu\Bigl\lceil\frac{\nu}{N}\Bigr\rceil\le B\,(S^{\max}-1)\Bigr\}\qquad\text{(eq:n)}
$$
UEs of this order, $\mathcal C=\{u_1,\ldots,u_n\}$: the largest number of private streams that can be zero-forced with
the fronthaul the super-common stream leaves ($n$ streams with $m=\lceil n/N\rceil$ BSs each take $nm$ of the
$B(S^{\max}-1)$ slots).

**Private streams.** In the SUS order, the private stream of core user $k$ is carried by the
$$
m=\lceil n/N\rceil\qquad\text{(eq:m)}
$$
BSs with the largest $E_{k,b}$ among those with a free slot (a BS carries at most $S^{\max}-1$ private streams).
When $n>(B-1)N$, $m=B$: every core user is served by every BS (joint transmission). Otherwise every core user is
served by its $m<B$ strongest BSs, so that different BSs carry different private streams and some core users are
carried by several BSs.

**Groups.** Every non-core UE joins the core user with the most correlated direct channel \eqref{eq:nearest_core},
and every core user with followers heads a group $\mathcal U_g$ (as in the current design; groups are disjoint, of size
$2\ldots K-1$, at most $\lfloor K/2\rfloor$ of them).

**Group-common streams.** Every BS $b$ fills the $S^{\max}-1-\sum_{k\in\mathcal C}M_{b,k}$ slots its private streams
leave free with group-common streams, in decreasing order of
$$
q_{g,b}=\min_{k\in\mathcal U_g}E_{k,b},\qquad\text{(eq:q)}
$$
its direct-channel energy to the weakest member of the group. A group-common stream carried by several BSs is precoded
jointly over their antennas.

**Super-common stream:** carried by every BS.

The non-core UEs have no private stream; the 1-layer counterpart of the design (Section 1.4) serves them by the
super-common stream only, the design by the super-common stream and their group-common stream.

### 1.3 Algorithm listing

```
Input: direct channels h_d,k (stacked, BN), E_kb = ||f_b,k||^2, B, N, K, S^max
1  u_1, u_2, ... <- SUS order on the stacked direct channels (min{K, BN} steps)
2  n <- largest nu <= min{K, BN} with nu * ceil(nu / N) <= B (S^max - 1);  m <- ceil(n / N);  C <- {u_1, ..., u_n}
3  for every BS b: J_b <- {c}                                              (super-common stream)
4  for k = u_1, ..., u_n:                                                   (private streams, SUS order)
5      add private stream k to J_b for the m BSs b with the largest E_kb among those with |J_b| < S^max
6  groups: every k not in C joins argmax_{a in C} |h_k h_a^H|^2 / (||h_k||^2 ||h_a||^2); U_a = {a} + its followers
7  for every BS b: add group-common streams to J_b in decreasing order of min_{k in U_g} E_kb while |J_b| < S^max
Output: J_1, ..., J_B (M_bj = 1 iff j in J_b), groups, core users C (start of the AO)
```

Code: `code/bsd.py` (`design`, `design_1layer`), 60 lines.

### 1.4 The 1-layer counterpart (requirement R1)

`bsd.design_1layer`: steps 1–5 unchanged (the same core users, carried by the same BSs), no groups and no
group-common streams; the non-core UEs are served by the super-common stream only. The comparison of the design with
it therefore isolates the group-common layer: the design is its 1-layer counterpart plus group-common streams in the
fronthaul slots the private streams leave free. (A variant of the counterpart that fills these slots with private
streams of non-core UEs, in the style of J1, is reported in Section 3.6 as a sensitivity check.)

### 1.5 Why it works

**The group-common layer pays for itself (R1).** With $R_{th}>0$ every UE must receive $R_{th}$. A core user receives
it on its private stream. A non-core UE has no private stream: in the 1-layer counterpart its QoS can only come from
the super-common stream, whose rate is limited by the weakest of all $K$ UEs and which must deliver about
$(K-n)R_{th}$ on top of its own share; the power this takes is taken from the private streams. The design adds, in
fronthaul slots that no further zero-forced private stream could use, group-common streams that carry the QoS of their
members at the rate of their own weakest member, from the BSs that reach that member best. The design contains its
counterpart (same core users, same BSs, the group-common streams start at the noise level and the AO can leave them
off), so it can do no worse in principle, and it gains wherever the super-common stream is loaded: increasingly with
$R_{th}$ and with the number of non-core UEs. At $R_{th}=0$ there is no QoS to carry and the two coincide (a tie).

**Zero-forcing first (DoF), then the QoS layer.** In the interference-limited regime of the 100 m area (direct-link
SNR around 60 dB) the sum rate comes from the spatially multiplexed private streams: in every run of the current design
the private streams of non-core users carry no rate (FINDINGS §3; `r_pn` = 0 in all 288 JH screen runs). The design
therefore spends the fronthaul first on as many zero-forced private streams as it can carry, (eq:n), with the fewest
BSs that can zero-force each, (eq:m), and gives the remaining slots to the group-common streams. Cooperation beyond the
zero-forcing need would add beamforming gain (Section 3.8 measures it), but the slots it takes are the ones the
group-common streams need at high $R_{th}$; the design leaves that trade to $R_{th}$-independent counting rather than
to a test.

**Against the current joint design.** Joint transmission spends $B$ slots per stream: with $S^{\max}-1<BN$ it serves
$S^{\max}-1$ private streams and no group-common stream, and its non-core UEs depend on the super-common stream alone
(this is why J misses the QoS at high $R_{th}$ in settings b and c). The design serves at least as many zero-forced
private streams, $n\ge\min\{K,BN,S^{\max}-1\}$, and group-common streams in the slots it saves, so it is better at high
$R_{th}$; at low $R_{th}$ the beamforming gain that joint transmission draws from the additional BSs makes J better.
Wherever the zero-forcing condition requires all BSs ($n>(B-1)N$), the design is the current design without the
private streams of non-core UEs (which carry no rate there).

### 1.6 Many BSs scattered over a large area

The design is defined for any number of BSs at any positions; nothing in it refers to $B=2$ or 3 or to the positions
of T2 and E3. With many BSs the fronthaul admits about $n\approx\sqrt{BN(S^{\max}-1)}$ zero-forced private streams, each
carried by $m=\lceil n/N\rceil$ BSs, a vanishing fraction $m/B\approx\sqrt{(S^{\max}-1)/(BN)}$ of the network as $B$
grows: every core user is served by its own cluster of strongest (nearby) BSs, some core users share BSs (their
clusters overlap), and a BS serves the core users near it ("its own"). Joint transmission by all BSs arises by itself
when the number of private streams the fronthaul can carry exceeds what $B-1$ BSs can zero-force, $n>(B-1)N$ (for
example T2 with $S^{\max}\ge2N$, E3 with $K=8$ or $N=5,S^{\max}=20$); then every core user needs every BS.
`tools/large_area_demo.py` (design only; UEs uniform, seed 1):

| Layout | $B$ | $K$ | $S^{\max}$ | $n$ | $m$ | mean / max distance to the BSs carrying a private stream vs. farthest BS | group-common streams served (BSs each) |
|---|---|---|---|---|---|---|---|
| T2 positions, 100 m | 2 | 16 | 12 | 6 | 2 (all) | 53 m vs 56 m | 3 (2) |
| T2 positions, 100 m | 2 | 16 | 4 | 3 | 1 | 13 m vs 56 m | 2 (1.5) |
| E3 positions, 100 m | 3 | 16 | 6 | 6 | 2 | 68 m vs 84 m | 3 (1) |
| 4 × 4 grid, 400 m | 16 | 64 | 12 | 22 | 8 | 220 m vs 355 m | 5 (1) |
| 6 × 6 grid, 1 km | 36 | 120 | 12 | 33 | 11 | 439 m vs 993 m | 17 (1.9) |
| 2 × 2 grid, 100 m | 4 | 8 | 12 | 8 | 3 | 56 m vs 67 m | — (all UEs core) |

(The column "distance" is the largest distance from a core user to the BSs that carry its private stream, averaged over
the core users, against its farthest BS.) The zero-forcing condition is counted against all other core users, which is
conservative over a large area (Section 5).

### 1.7 Nothing to tune

* $n$ and $m$ follow from $K$, $B$, $N$ and $S^{\max}$ by the zero-forcing condition $mN\ge n$ and the fronthaul
  constraint \eqref{eq:c-front}; no threshold, weight or constant enters.
* Which BSs carry a private stream (the $m$ strongest), which UEs are core users (SUS), the groups (nearest core) and
  which group-common streams each BS carries (largest $q_{g,b}$) are argmax/ordering decisions on the direct channels.
* The design is computed once per drop and setting, before the deployment search, in about 1–2 ms; it is deterministic
  and runs no AO and no evaluation of $F$.

### 1.8 One change outside the design: the AO start on the carried antennas (switch, off by default)

The AO start of the framework (`algo.init_core`) computes the regularized ZF directions of the core users' private
streams on the stacked $BN$-antenna channel and then zeroes the blocks of the BSs that do not carry a stream. With
BS-specific carriage this destroys the zero-forcing that the design is built on (a stream carried by $m$ BSs is nulled
only by its precoder on those $mN$ antennas). `algo.init_core_carried` (with `SEEDS['rule_carried']`, the coefficient
seed computed from it) computes every direction on the antennas of the BSs that carry the stream: the private stream of
core user $k$ carried by $\mathcal A_k$ uses column $k$ of the regularized ZF matrix of the core channels restricted to
the antennas of $\mathcal A_k$ (the same regularization), common streams the dominant eigenvector of their decoders'
normalized channels restricted to the carrying antennas; the power split is that of `init_core`. Under joint
transmission (`d.bsmask` None) it is `init_core` exactly, so it changes nothing for JH, J1 or any joint design; it is a
switch that is off by default and is set by the variant (`opt=dict(init='core_carried', seed='rule_carried')` in
`harness/variants.py`); `BSD@SC`, stream clustering on the design's streams, inherits it (harness/grid.py passes the
base variant's `opt`). Measured on the screen (Q = 5, the identical design with either start, 288 paired runs): see
Section 3.7.

## 2. Cost

* **The design itself:** 1.0–1.5 ms per drop and setting (`cpu_design`; SUS dominates), no AO, no evaluation of $F$.
* **Per run** (design + search + refinement; `cpu` of the records, mean over the runs):

| Tier | T2: BSD / BSD1 / JH / J1 | E3: BSD / BSD1 / JH / J1 |
|---|---|---|
| screen, Q = 5 | 5.7 / 4.3 / 6.1 / 4.4 s | 22.0 / 12.8 / 8.8 / 10.2 s |
| confirm, Q = 50 | <!-- CPU50T2 --> | <!-- CPU50E3 --> |
| bench (T2, Q = 50) | BSD 100.8 s (FINDINGS §7: the current design 90–120 s) | |
| smax (T2, Q = 50) | BSD 42.1 s | |

  In T2 the design costs about as much as the current design (it serves fewer distinct streams in settings b and d and
  the same in the others). In E3 it serves more distinct streams (private streams at 2 of 3 BSs plus group-common
  streams; up to 15 in setting e against 10) and costs about 1.6 times the current framework J at Q = 5; the most
  expensive setting is E3 e (Q = 50: about 500 CPU-s per run). With the default AO start the design costs 20–40%
  more (Section 3.7).

## 3. Results

Rates in bit/s/Hz (0 when the QoS is missed), differences paired (mean ± standard error over the runs). `BSD` = the
design, `BSD1` = its 1-layer counterpart, J/JH/J1 = the current framework, its design without the stream-set test and
its 1-layer RSMA (reference). All outputs below are those of `tools/report.py` (unedited).

### 3.1 R1 at Q = 50 (`--tier confirm`, 3 drops)

<!-- CONFIRM -->

### 3.2 R1 at Q = 5 (`--tier screen`, 4 drops), by setting

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
-- T2, setting a
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    0.5    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
      1    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    1.5    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
      2    4     93.80     93.80     93.81     93.81     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    2.5    4     93.55     93.55     93.56     93.56     93.56    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    all   24     93.76     93.76     93.77     93.77     93.77    +0.01± 0.00       -0.01± 0.00      24/24/24/24/24
-- T2, setting b
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4     60.56     59.71     73.49     73.49     73.49    +0.85± 0.68      -12.93± 1.34      4/4/4/4/4
    0.5    4     60.13     59.28     67.64     67.64     67.64    +0.85± 0.69       -7.51± 1.33      4/4/4/4/4
      1    4     57.26     55.33     59.67     59.67     59.67    +1.93± 0.82       -2.42± 1.00      4/4/4/4/4
    1.5    4     50.48     47.88     50.97     50.97     50.97    +2.60± 1.66       -0.49± 1.41      4/4/4/4/4
      2    4     43.81     39.69     42.45     42.45     42.45    +4.12± 2.00       +1.35± 1.88      4/4/4/4/4
    2.5    4     30.11      8.08      0.00      0.00      0.00   +22.02± 9.59      +30.11±10.15      3/1/0/0/0
    all   24     50.39     44.99     49.04     49.04     49.04    +5.40± 2.15       +1.35± 3.26      23/21/20/20/20
-- T2, setting c
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4     96.70     96.70     96.70     96.70     96.70    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    0.5    4     93.91     93.91     93.91     93.91     93.91    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      1    4     79.27     79.27     79.27     79.27     79.27    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    1.5    4     58.57     58.57     58.57     58.57     58.57    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      2    4      0.00      0.00      0.00      0.00      0.00    +0.00± 0.00       +0.00± 0.00      0/0/0/0/0
    2.5    4      0.00      0.00      0.00      0.00      0.00    +0.00± 0.00       +0.00± 0.00      0/0/0/0/0
    all   24     54.74     54.74     54.74     54.74     54.74    +0.00± 0.00       +0.00± 0.00      16/16/16/16/16
-- T2, setting d
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4     78.77     74.75    113.38    113.38    113.38    +4.02± 1.40      -34.61± 3.12      4/4/4/4/4
    0.5    4     78.77     74.31    109.01    109.01    109.01    +4.46± 1.57      -30.24± 3.13      4/4/4/4/4
      1    4     77.17     72.53     99.85     99.85     99.85    +4.65± 2.14      -22.67± 3.92      4/4/4/4/4
    1.5    4     76.40     68.05     88.14     88.14     88.14    +8.35± 2.37      -11.75± 4.75      4/4/4/4/4
      2    4     72.82     61.37     74.76     74.76     74.76   +11.45± 2.39       -1.95± 4.68      4/4/4/4/4
    2.5    4     67.52     52.61     60.94     60.94     60.94   +14.91± 4.75       +6.58± 6.56      4/4/4/4/4
    all   24     75.24     67.27     91.01     91.01     91.01    +7.97± 1.29      -15.77± 3.50      24/24/24/24/24
-- T2, setting e
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4    133.91    133.91    133.91    133.91    133.91    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    0.5    4    133.43    133.43    133.43    133.43    133.43    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      1    4    124.60    124.60    124.60    124.60    124.60    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    1.5    4    116.82    116.82    116.82    116.82    116.82    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      2    4    101.53    101.53    101.53    101.53    101.53    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    2.5    4     80.51     80.51     80.51     80.51     80.51    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    all   24    115.13    115.13    115.13    115.13    115.13    +0.00± 0.00       +0.00± 0.00      24/24/24/24/24
-- T2, setting f
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4    115.15    116.60    117.30    117.27    115.16    -1.44± 1.44       -2.14± 1.66      4/4/4/4/4
    0.5    4    115.15    116.60    117.30    117.27    115.16    -1.44± 1.44       -2.14± 1.66      4/4/4/4/4
      1    4    114.84    115.14    116.61    116.58    114.85    -0.30± 0.29       -1.77± 1.74      4/4/4/4/4
    1.5    4    113.89    113.40    115.09    115.08    113.33    +0.49± 0.44       -1.20± 1.25      4/4/4/4/4
      2    4    110.26    104.01    108.19    109.85    103.91    +6.25± 2.51       +2.07± 1.91      4/4/4/4/4
    2.5    4    100.61     91.23    100.26    100.26     91.50    +9.39± 4.87       +0.35± 0.12      4/4/4/4/4
    all   24    111.65    109.50    112.46    112.72    108.99    +2.16± 1.23       -0.80± 0.63      24/24/24/24/24
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
-- E3, setting a
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4    125.65    125.65    125.65    125.65    125.65    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    0.5    4    125.56    125.56    125.56    125.56    125.56    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      1    4    125.42    125.42    125.42    125.42    125.42    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    1.5    4    125.25    125.25    125.25    125.25    125.25    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      2    4    125.04    125.04    125.04    125.04    125.04    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    2.5    4    125.14    125.14    125.14    125.14    125.14    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    all   24    125.34    125.34    125.34    125.34    125.34    +0.00± 0.00       +0.00± 0.00      24/24/24/24/24
-- E3, setting b
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4     86.91     86.91     72.05     72.05     72.05    -0.00± 0.00      +14.86± 1.07      4/4/4/4/4
    0.5    4     80.13     80.13     64.42     64.42     64.42    -0.00± 0.00      +15.71± 0.92      4/4/4/4/4
      1    4     70.00     70.00     55.60     55.60     55.60    +0.00± 0.00      +14.40± 1.26      4/4/4/4/4
    1.5    4     61.32     59.29     46.85     46.85     46.85    +2.03± 2.03      +14.47± 0.66      4/4/4/4/4
      2    4     51.61     47.63     38.17     38.17     38.17    +3.98± 2.73      +13.43± 1.14      4/4/4/4/4
    2.5    4     41.25     34.89      0.00      0.00      0.00    +6.36± 3.75      +41.25± 3.98      4/4/0/0/0
    all   24     65.20     63.14     46.18     46.18     46.18    +2.06± 0.90      +19.02± 2.19      24/24/20/20/20
-- E3, setting c
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4    103.90    103.91    113.87    113.87    113.87    -0.00± 0.00       -9.96± 4.05      4/4/4/4/4
    0.5    4     99.38     98.94     97.36     97.36     97.36    +0.44± 0.44       +2.02± 4.01      4/4/4/4/4
      1    4     85.94     82.94     76.29     76.29     76.29    +3.01± 0.12       +9.65± 3.11      4/4/4/4/4
    1.5    4     70.35     61.00     54.52     54.52     54.52    +9.35± 1.71      +15.83± 2.45      4/4/4/4/4
      2    4     57.21     28.72      0.00      0.00      0.00   +28.49± 8.97      +57.21± 2.20      4/3/0/0/0
    2.5    4     27.71      0.00      0.00      0.00      0.00   +27.71±16.01      +27.71±16.01      2/0/0/0/0
    all   24     74.08     62.58     57.01     57.01     57.01   +11.50± 3.71      +17.08± 5.15      22/19/16/16/16
-- E3, setting d
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4    125.68    125.67    115.43    115.43    115.43    +0.00± 0.00      +10.25± 4.74      4/4/4/4/4
    0.5    4    124.91    124.91    109.73    109.73    109.73    +0.00± 0.00      +15.19± 4.14      4/4/4/4/4
      1    4    121.48    121.48     99.18     99.18     99.18    +0.00± 0.00      +22.31± 3.53      4/4/4/4/4
    1.5    4    114.24    114.01     86.12     86.12     86.12    +0.23± 0.23      +28.12± 3.39      4/4/4/4/4
      2    4    105.47    103.29     72.35     72.35     72.35    +2.18± 1.53      +33.12± 4.33      4/4/4/4/4
    2.5    4     95.66     91.14     58.64     58.64     58.64    +4.52± 3.10      +37.02± 5.54      4/4/4/4/4
    all   24    114.57    113.42     90.24     90.24     90.24    +1.16± 0.62      +24.33± 2.52      24/24/24/24/24
-- E3, setting e
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4    127.48    126.62    179.41    179.41    179.41    +0.86± 0.83      -51.93± 5.15      4/4/4/4/4
    0.5    4    130.75    126.60    173.48    173.48    173.48    +4.15± 3.07      -42.73± 5.41      4/4/4/4/4
      1    4    127.76    127.38    157.45    157.45    157.45    +0.38± 0.29      -29.69± 5.49      4/4/4/4/4
    1.5    4    123.76    121.64    135.06    135.06    135.06    +2.11± 1.34      -11.30± 6.23      4/4/4/4/4
      2    4    117.28    111.70    110.21    110.21    110.21    +5.57± 4.31       +7.07± 7.17      4/4/4/4/4
    2.5    4    111.13     97.74     84.76     84.76     84.76   +13.39± 5.37      +26.38± 8.00      4/4/4/4/4
    all   24    123.03    118.61    140.06    140.06    140.06    +4.41± 1.46      -17.03± 6.17      24/24/24/24/24
-- E3, setting f
   R_th    n       BSD      BSD1         J        JH        J1          BSD-BSD1             BSD-J   QoS met BSD/BSD1/J/JH/J1
      0    4    166.66    167.20    167.03    166.53    167.04    -0.54± 0.50       -0.37± 0.58      4/4/4/4/4
    0.5    4    166.61    166.99    166.86    166.43    166.87    -0.39± 0.34       -0.25± 0.47      4/4/4/4/4
      1    4    165.67    165.70    165.53    165.51    165.54    -0.03± 0.02       +0.14± 0.15      4/4/4/4/4
    1.5    4    162.45    162.47    161.85    161.82    162.46    -0.02± 0.03       +0.61± 0.58      4/4/4/4/4
      2    4    156.27    156.27    156.28    156.28    156.22    -0.00± 0.02       -0.01± 0.01      4/4/4/4/4
    2.5    4    155.31    155.22    155.32    155.32    155.25    +0.09± 0.07       -0.01± 0.00      4/4/4/4/4
    all   24    162.16    162.31    162.15    161.98    162.23    -0.15± 0.10       +0.02± 0.16      24/24/24/24/24
-- BSD: mean served streams 10.96, group-common served 1.62, private served 8.33; BS-specific in 96/144 runs; per run, carried by a subset of the BSs: group-common 1.46, private 4.50
-- J: stream-set test made in 24/144 runs, 1-layer kept in 13
-- CPU per run (s): BSD 22.0, BSD1 12.8, J 13.6, JH 8.8, J1 10.2; BSD discrete design 0.001
```

### 3.3 R2 and R3: the benchmark setting (`--tier bench`, Q = 50)

```
===== benchmark setting (T2, K = 16, N = 3, S_max = 12, R_th = 2, Q = 50): 15 common runs (BSD: 15, sparse_bf: 15, BSD@SC: 15, J: 15, J@SC: 15)
sparse_bf, J (the current framework) and J@SC (stream clustering on its streams): stored runs of Fig. 5.10
-- by M
      M    n       BSD sparse_bf    BSD@SC         J      J@SC     BSD-sparse_bf        BSD-BSD@SC   QoS met BSD/sparse_bf/BSD@SC/J/J@SC
    100    5     74.68     63.08     46.34     73.51     44.18   +11.61± 3.43      +28.35± 2.43      5/5/5/5/5
    300    5     81.44     75.09     55.63     81.95     54.38    +6.35± 6.04      +25.81± 2.44      5/5/5/5/5
    500    5     85.68     81.03     62.09     88.72     64.51    +4.65± 4.33      +23.59± 4.15      5/5/5/5/5
    all   15     80.60     73.07     54.69     81.39     54.36    +7.54± 2.65      +25.92± 1.74      15/15/15/15/15
-- BSD: mean served streams 11.40, group-common served 4.40, private served 6.00; BS-specific in 0/15 runs; per run, carried by a subset of the BSs: group-common 0.00, private 0.00
-- BSD: shares of the sum rate over its 15 runs meeting the QoS: super-common 15.6%, group-common 9.2%, private 75.1%
```

* **R2:** the design beats sparse beamforming by +7.5 ± 2.7 bit/s/Hz (80.6 against 73.1; better at every $M$) and
  stream clustering on its own streams and groups (`BSD@SC`, built from `d.groups`, `d.priority`, `d.core`; with the
  design's AO start, Section 1.8) by +25.9 ± 1.7 (54.7). In this setting ($n=6=BN$, $m=2=B$) every BS carries every
  private stream, and the 5 free slots per BS carry all group-common streams (4.4 on average), so the design is the
  current design without its non-core private stream (where J serves 4 groups); its rate is that of J (81.4).
* **R3:** shares of the sum rate (Fig. 5.7 definition, over the 15 runs, all meeting the QoS): super-common 15.6%,
  group-common 9.2%. In every run the two common layers together carry the QoS of the 10 non-core UEs
  ($r_c+r_g=20.0$–$20.3$ bit/s/Hz, $10R_{th}=20$); the group-common streams carry 4.0–9.6 of it.

### 3.4 R4: sum rate versus $S^{\max}$ (`--tier smax`, Q = 50)

```
===== sum rate versus S_max (T2, N = 3, M = 300, Q = 50): 126 runs of BSD
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

<!-- SMAXNOTE -->

### 3.5 Design requirement: where the BSs carry different streams

Per setting, over the 24 screen runs (4 drops × 6 $R_{th}$; the design does not depend on $R_{th}$): runs in which the
BSs carry different streams, and per run the private and group-common streams carried by a subset of the BSs only
(the report's "carried by a subset" count), with an example (drop 901).

| Scen. | Setting ($N,K,S^{\max}$) | $n$ / $m$ | BS-specific runs | private on a subset | group-common on a subset | example (streams per BS, drop 901) |
|---|---|---|---|---|---|---|
| T2 | a (3, 8, 12) | 6 / 2 (all) | 0/24 | 0 | 0 | joint |
| T2 | b (3, 12, 4) | 3 / 1 | 24/24 | 3.00 | 1.25 | BS1: c g1 p2 p3; BS2: c g1 g2 p7 |
| T2 | c (3, 16, 6) | 5 / 2 (all) | 0/24 | 0 | 0 | joint |
| T2 | d (5, 12, 6) | 5 / 1 | 24/24 | 5.00 | 1.00 | BS1: c g1 g3 p2 p3 p12; BS2: c g1 g3 g4 p6 p7 |
| T2 | e (5, 16, 10) | 9 / 2 (all) | 0/24 | 0 | 0 | joint |
| T2 | f (5, 16, 20) | 10 / 2 (all) | 0/24 | 0 | 0 | joint |
| E3 | a (3, 8, 12) | 8 / 3 (all) | 0/24 | 0 | 0 | joint (all UEs are core users) |
| E3 | b (3, 12, 4) | 4 / 2 | 24/24 | 4.00 | 1.00 | BS1: c p3 p9 p12; BS2: c g1 p6 p12; BS3: c p3 p6 p9 |
| E3 | c (3, 16, 6) | 6 / 2 | 24/24 | 6.00 | 2.75 | BS1: c g1 g2 p7 p13 p15; BS2: c g1 p1 p6 p9 p13; BS3: c p1 p6 p7 p9 p15 |
| E3 | d (5, 12, 6) | 7 / 2 | 24/24 | 7.00 | 1.00 | BS1: c p1 p3 p8 p9 p12; BS2: c g1 p1 p6 p11 p12; BS3: c p3 p6 p8 p9 p11 |
| E3 | e (5, 16, 10) | 10 / 2 | 24/24 | 10.00 | 4.00 | BS1: c g2 g4 + 7 private; BS2: c g1 g2 g3 g4 + 5 private; BS3: c g3 + 8 private |
| E3 | f (5, 16, 20) | 15 / 3 (all) | 0/24 | 0 | 0 | joint |

In T2 settings b and d every core user is served by its strongest BS only ("BSs with core users of their own"); in E3
settings b–e every core user is served by 2 of the 3 BSs, so that every BS carries core users that one other BS also
carries; where (eq:n) admits more private streams than $B-1$ BSs can zero-force, all BSs carry all of them. The
group-common streams differ between BSs wherever the free slots are fewer than the groups. In the benchmark setting
($S^{\max}=12$, $n=6$, $m=2$) the transmission is joint and every BS carries every group-common stream (all fit).

### 3.6 Sensitivity check: a counterpart that fills the free slots with non-core private streams

<!-- FILL -->

### 3.7 The AO start on the carried antennas

The design with the default start (`XZDn`, same design function `bsx.ZDn` ≡ `bsd.design`, verified identical on all
grid drops) against the carried-antenna start (`BSD`), screen tier, paired over the 144 runs per scenario
(`BSD − XZDn`; the runs with joint transmission are identical, so the difference comes from settings b, d (T2) and
b–e (E3)):

| $R_{th}$ | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 | all | CPU per run |
|---|---|---|---|---|---|---|---|---|
| T2 | +1.49 ± 0.83 | +0.45 ± 0.65 | +0.95 ± 0.55 | +1.37 ± 0.59 | +1.23 ± 0.56 | +2.08 ± 1.60 | +1.26 ± 0.35 | 5.7 s vs 6.9 s |
| E3 | +0.84 ± 1.26 | +1.18 ± 1.24 | +0.54 ± 0.70 | +0.78 ± 0.80 | +0.47 ± 0.95 | +1.58 ± 0.81 | +0.90 ± 0.40 | 22.0 s vs 30.6 s |

The AO converges from the carried-antenna start to better solutions and in fewer iterations.

## 4. Other methods tried

All at Q = 5 (`--tier screen`); "pilot" = the first 2 drops and $R_{th}\in\{0,1,2\}$ (36 runs per scenario), "full" =
the screen tier (144 runs per scenario). R1 columns: the variant minus its own 1-layer counterpart, mean ± SE over the
runs; last columns: the variant's mean rate and its mean difference to J on the same runs. Code: `code/bsx.py`
(variants `X*` in `harness/variants.py`), results `results/X*__*__Q5.jsonl`.

| Variant − counterpart | Scen. | Runs | R1 at $R_{th}$ = 0 / 0.5 / 1 / 1.5 / 2 / 2.5 (pilot: 0 / 1 / 2) | Rate | − J |
|---|---|---|---|---|---|
| XA − XA1 | T2 | 36 (2 drops) | +0.57 ± 0.58 / +0.57 ± 0.58 / +0.56 ± 0.57 | 81.56 | -7.58 |
| XA − XA1 | E3 | 36 (2 drops) | -0.16 ± 0.15 / -0.01 ± 0.01 / +0.02 ± 0.02 | 107.81 | -0.12 |
| XC − XC1 | T2 | 36 (2 drops) | -3.83 ± 2.80 / -3.33 ± 2.29 / -0.12 ± 3.89 | 78.57 | -10.57 |
| XC − XC1 | E3 | 36 (2 drops) | -1.24 ± 0.73 / -1.94 ± 0.73 / -1.90 ± 0.91 | 106.17 | -1.76 |
| XM − XA1 | T2 | 36 (2 drops) | -2.43 ± 3.05 / -1.10 ± 3.51 / +1.57 ± 3.95 | 80.35 | -8.80 |
| XM − XA1 | E3 | 36 (2 drops) | -2.01 ± 0.77 / -2.33 ± 0.97 / -1.42 ± 1.01 | 105.94 | -1.99 |
| XZ − XZ1 | T2 | 36 (2 drops) | +0.95 ± 0.68 / +2.52 ± 1.66 / +1.40 ± 1.10 | 85.44 | -3.70 |
| XZ − XZ1 | E3 | 36 (2 drops) | -0.76 ± 0.54 / -2.06 ± 1.46 / +2.62 ± 2.25 | 105.33 | -2.60 |
| XZc − XZ1c | T2 | 144 (4 drops) | +1.03 ± 0.41 / +1.06 ± 0.42 / +1.14 ± 0.46 / +1.38 ± 0.63 / +1.83 ± 0.95 / +1.39 ± 1.89 | 83.71 | -2.31 |
| XZc − XZ1c | E3 | 144 (4 drops) | +0.30 ± 0.77 / -0.72 ± 0.89 / +0.79 ± 1.11 / +3.14 ± 1.38 / +4.86 ± 1.65 / +10.26 ± 3.76 | 106.97 | +3.47 |
| XZDc − XZD1c | T2 | 36 (2 drops) | +1.27 ± 0.75 / +1.53 ± 0.85 / +1.80 ± 1.39 | 86.41 | -2.73 |
| XZDc − XZD1c | E3 | 144 (4 drops) | -0.12 ± 0.10 / +0.41 ± 0.40 / +0.56 ± 0.23 / +1.15 ± 0.88 / +1.46 ± 1.20 / +4.15 ± 3.37 | 110.70 | +7.21 |
| XZDn − XZDn1 | T2 | 144 (4 drops) | +0.11 ± 0.36 / +0.65 ± 0.59 / +0.85 ± 0.83 / +1.88 ± 0.76 / +2.69 ± 0.97 / +7.29 ± 2.53 | 82.23 | -3.80 |
| XZDn − XZDn1 | E3 | 144 (4 drops) | -0.09 ± 0.08 / -0.62 ± 0.77 / +0.79 ± 0.60 / +2.75 ± 1.22 / +6.91 ± 2.78 / +7.45 ± 3.13 | 109.83 | +6.34 |
| BSD − BSD1 | T2 | 144 (4 drops) | +0.57 ± 0.47 / +0.65 ± 0.51 / +1.05 ± 0.50 / +1.91 ± 0.76 / +3.64 ± 1.06 / +7.72 ± 2.49 | 83.49 | -2.54 |
| BSD − BSD1 | E3 | 144 (4 drops) | +0.05 ± 0.17 / +0.70 ± 0.56 / +0.56 ± 0.23 / +2.29 ± 0.81 / +6.70 ± 2.58 / +8.68 ± 3.28 | 110.73 | +7.24 |

* **XA** (per-BS ranking within the current class order: each BS takes the super-common stream, then the core users'
  private streams in decreasing order of its own channel energy, then group-common streams, then non-core private
  streams). BS-specific core streams appear whenever $S^{\max}-1<|\mathcal C|$, but each BS keeps different core users
  and every core stream is carried by few BSs: 9–11 partially carried private streams, a large loss to J at low
  $R_{th}$ (−16 to −44 bit/s/Hz at $R_{th}=0$ in settings b–e), and group-common streams only where joint
  transmission already had room for them, so R1 is a tie.
* **XC** (each BS first serves its own core users and its own groups, then cooperates): the group-common streams
  displace cooperation, which is worth more at low $R_{th}$; R1 negative.
* **XM** (core private and group-common streams ranked together by each BS's energy to their weakest decoder; its counterpart is the same design as XA1, whose runs are used): R1
  negative in E3.
* **XZ / XZc** (zero-forcing-minimal carriage with the current design's number of private streams,
  $n=\min\{K,BN,S^{\max}-1\}$; free slots to group-common streams, then to non-core private streams; counterpart: free
  slots to non-core private streams; XZc with the AO start of Section 1.8). R1 positive in T2 at every $R_{th}$ and in
  E3 from $R_{th}=1.5$, a tie at 0.5–1 in E3; much below the final design in E3 at low $R_{th}$ (fewer private
  streams).
* **XZDc** (the final core size (eq:n), counterpart with non-core private streams in the free slots) = the final
  design's streams plus non-core private streams after the group-common streams; see Section 3.6.
* **XZDn** = the final design with the default AO start (identical design function; Section 3.7).

A probe that motivated the zero-forcing count: in E3, setting e, drop 901, $R_{th}=0$, at the initial deployment, the
current design (9 jointly carried private streams over 15 antennas) reaches 170.9 bit/s/Hz; carrying each of the same
9 streams by only its 2 strongest BSs (still enough to zero-force them) reaches 120.9–124.4 (also when the AO starts
from the joint solution with the third block zeroed). In this dense 100 m area the third BS adds a large beamforming
gain even where it is not needed for zero-forcing; this is the price of the fronthaul the design gives to the
group-common layer, and why the design is behind J at low $R_{th}$.

## 5. Weak points

* **Lower rate than the current joint design at low $R_{th}$** where the design carries private streams by fewer than
  all BSs (T2 settings b, d; E3 settings b–e): the BSs it leaves out of a private stream are not needed for
  zero-forcing but add beamforming gain in the dense 100 m area (Section 4, probe). The user accepted this
  (BRIEF.md); at high $R_{th}$ the design is better than J (more zero-forced private streams and the group-common layer).
* **The 1-layer counterpart leaves fronthaul slots free.** By the design's logic no further private stream can be
  zero-forced, so its counterpart has nothing to put there; a counterpart that fills them with non-zero-forced private
  streams of non-core UEs (J1 style, `BSD1f`) is a stronger baseline at low $R_{th}$ in E3 (Section 3.6).
* **The margin at $R_{th}=1$ is small** (the group-common streams pay off as the QoS load grows); see Section 3.1.
* **Global zero-forcing count.** (eq:n)–(eq:m) count every private stream against all other core users. Over a very
  large area distant users barely interfere and fewer BSs per stream would do, so the design is conservative there:
  with 36 BSs over 1 km² it serves 33 private streams with 11 BSs each and few group-common streams. A local count
  would need a notion of "distant" (a threshold), which the ground rules exclude.
* **Group-common streams are few where the fronthaul is tight** (E3 settings b and d: one free slot), and the group
  share in the benchmark setting is 9.2%, just under 10% (the current design's runs of the thesis had 10.8%).
* **Non-nested in $S^{\max}$.** $n$ and $m$ change with $S^{\max}$ (e.g. T2: $S^{\max}=4$: 3 private streams carried by
  one BS each plus group-common streams; $S^{\max}=6$: 5 jointly carried private streams and no group-common stream),
  so monotonicity in $S^{\max}$ is not structural; Section 3.3 measures it.
* **Cost in E3** is about 1.6 times the current design's (more distinct streams per run).

## 6. Choices made without asking

* **Label:** `cloud_xhigh` (from the requested file name `design_return_cloud_xhigh.zip`).
* **Variant names:** `BSD` (the design), `BSD1` (its 1-layer counterpart), `BSD@SC` (stream clustering on its streams),
  `BSD1f` (sensitivity check of the counterpart). The design-study variants are `X*` (code/bsx.py).
* **The 1-layer counterpart** keeps the design's core users and BSs and drops the group-common streams; their slots
  stay free because, by the definition of the core, no further private stream could be zero-forced. The J1-style
  alternative (free slots to private streams of non-core UEs) is reported as `BSD1f` (Section 3.6).
* **AO start on the carried antennas** (switch, Section 1.8): adopted because the design is built on per-BS
  zero-forcing and the default start zeroes the blocks after computing joint ZF directions; measured better and
  cheaper (Section 3.7).
* **Grouping** is the current design's (nearest core by channel correlation), applied to the design's core users.
* **Ties in (eq:m)** (equal energies) are broken by BS index; ties in (eq:q) by group index.
* **Reference runs:** JH and J1 at Q = 5 (screen) and <!-- JQ50 -->.
* **Machine:** 4 cores; the self-test passed with identical traces (all five cases and the derived J).
