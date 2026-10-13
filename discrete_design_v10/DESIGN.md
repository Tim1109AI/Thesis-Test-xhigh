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

<!-- WHY -->

### 1.6 Many BSs scattered over a large area

<!-- LARGE -->

### 1.7 Nothing to tune

* $n$ and $m$ follow from $K$, $B$, $N$ and $S^{\max}$ by the zero-forcing condition $mN\ge n$ and the fronthaul
  constraint \eqref{eq:c-front}; no threshold, weight or constant enters.
* Which BSs carry a private stream (the $m$ strongest), which UEs are core users (SUS), the groups (nearest core) and
  which group-common streams each BS carries (largest $q_{g,b}$) are argmax/ordering decisions on the direct channels.
* The design is computed once per drop and setting, before the deployment search, in about 1–2 ms; it is deterministic
  and runs no AO and no evaluation of $F$.

### 1.8 One change outside the design: the AO start on the carried antennas (switch, off by default)

<!-- START -->

## 2. Cost

<!-- COST -->

## 3. Results

<!-- RESULTS -->

## 4. Other methods tried

<!-- OTHERS -->

## 5. Weak points

<!-- WEAK -->

## 6. Choices made without asking

<!-- CHOICES -->
