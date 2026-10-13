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

<!-- COST -->

## 3. Results

<!-- RESULTS -->

## 4. Other methods tried

All at Q = 5 (`--tier screen`); "pilot" = the first 2 drops and $R_{th}\in\{0,1,2\}$ (36 runs per scenario), "full" =
the screen tier (144 runs per scenario). R1 columns: the variant minus its own 1-layer counterpart, mean ± SE over the
runs; last columns: the variant's mean rate and its mean difference to J on the same runs. Code: `code/bsx.py`
(variants `X*` in `harness/variants.py`), results `results/X*__*__Q5.jsonl`.

| Variant (counterpart) | Idea | Scen. | Runs | R1 at $R_{th}$ = 0 / 0.5 / 1 / 1.5 / 2 / 2.5 (pilot: 0 / 1 / 2) | Rate | − J |
|---|---|---|---|---|---|---|
<!-- OTHERS-TABLE -->

* **XA** (per-BS ranking within the current class order: each BS takes the super-common stream, then the core users'
  private streams in decreasing order of its own channel energy, then group-common streams, then non-core private
  streams). BS-specific core streams appear whenever $S^{\max}-1<|\mathcal C|$, but each BS keeps different core users
  and every core stream is carried by few BSs: 9–11 partially carried private streams, a large loss to J at low
  $R_{th}$ (−16 to −44 bit/s/Hz at $R_{th}=0$ in settings b–e), and group-common streams only where joint
  transmission already had room for them, so R1 is a tie.
* **XC** (each BS first serves its own core users and its own groups, then cooperates): the group-common streams
  displace cooperation, which is worth more at low $R_{th}$; R1 negative.
* **XM** (core private and group-common streams ranked together by each BS's energy to their weakest decoder): R1
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
