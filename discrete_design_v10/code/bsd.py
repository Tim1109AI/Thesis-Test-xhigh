"""BS-specific discrete design: zero-forcing-minimal cooperation (design round v10).

Every stream is carried by the BSs it needs, and the fronthaul this saves carries group-common streams.

  1. Core users. The UEs are ordered by the greedy projection step of SUS on the stacked direct channels (as in the
     current design). The core users are the first n of this order, n the largest number (at most min(K, BN)) whose
     private streams can be zero-forced within the fronthaul: a private stream must be nulled at the other n - 1 core
     users, which takes at least m = ceil(n / N) BSs (mN >= n antennas), so the n core streams need n m fronthaul
     slots, and n m <= B (S_max - 1) (one slot per BS is the super-common stream's).
  2. Private streams. The private stream of every core user is carried by the m BSs with the strongest direct
     channels to it (in SUS order; a BS whose S_max - 1 slots are taken is skipped). When n > (B - 1) N, m = B and
     every BS carries every core stream (joint transmission); otherwise each core user is served by its m strongest
     BSs only.
  3. Groups. Every non-core UE joins the core user with the most correlated direct channel (as in the current
     design); every core user with followers heads a group.
  4. Group-common streams. Each BS fills the slots that its private streams leave free with group-common streams, in
     decreasing order of its direct-channel energy to the weakest member of the group (the decoder that limits the
     stream's rate). The super-common stream is carried by every BS.
Only core users have private streams; every other UE is served by the super-common stream and its group-common
stream (a further private stream could not be zero-forced within the fronthaul).

design_1layer: its 1-layer RSMA: steps 1-2 unchanged, no groups (the non-core UEs are served by the super-common
stream only).
Nothing in the design is tuned: n and m follow from K, B, N and S_max; the BSs and the order of the group-common
streams from the direct channels.
"""
import math
import numpy as np
import hrs as H_


def _energy(Hd, B, N):
    """E[k, b] = ||h_d,kb||^2."""
    return np.stack([np.sum(np.abs(Hd[:, b * N:(b + 1) * N]) ** 2, axis=1) for b in range(B)], axis=1)


def core_size(K, B, N, S):
    """Largest n <= min(K, BN) with n ceil(n / N) <= B (S - 1)."""
    n = 0
    while n < min(K, B * N) and (n + 1) * math.ceil((n + 1) / N) <= B * (S - 1):
        n += 1
    return n


def design(net, p, groups_on=True):
    K, B, N, S = p['K'], p['B'], p['N'], p['Smax']
    Hd = net['Hd']; E = _energy(Hd, B, N)
    order = H_.sus(Hd, min(K, B * N))
    n = core_size(K, B, N, S); m = math.ceil(n / N) if n else 0
    core = order[:n]; others = [k for k in range(K) if k not in core]
    # groups: every non-core UE joins the most correlated core user
    groups = []
    if groups_on and core:
        Hn = Hd / np.linalg.norm(Hd, axis=1)[:, None]
        corr = np.abs(Hn @ Hn.conj().T) ** 2
        follow = {a: [] for a in core}
        for k in others:
            follow[core[int(np.argmax(corr[k, core]))]].append(k)
        groups = [[a] + follow[a] for a in core if follow[a] and len(follow[a]) < K - 1]     # (n = 1: no group of all K UEs)
    G = len(groups)
    priv = lambda k: 1 + G + k
    # private streams: the m strongest BSs of each core user, in SUS order
    sets = [[0] for _ in range(B)]
    for k in core:
        for b in [b for b in np.argsort(-E[k], kind='stable') if len(sets[b]) < S][:m]:
            sets[b].append(priv(k))
    # group-common streams: each BS's free slots, by its energy to the weakest member
    weakest = np.array([E[g].min(0) for g in groups]) if G else np.zeros((0, B))
    for b in range(B):
        for g in sorted(range(G), key=lambda g: (-weakest[g, b], g)):
            if len(sets[b]) >= S: break
            sets[b].append(1 + g)
    carried = set().union(*map(set, sets))
    pr = [0] + [priv(k) for k in core] + [1 + g for g in range(G)] + [priv(k) for k in others]
    d = H_.Design(groups, [j for j in pr if j in carried], K)
    d.core = list(core); d.priority = pr; d.rule = 'zf_minimal'
    if any(set(s) != carried for s in sets):
        d.bsmask = np.array([[j in s for j in d.served] for s in sets])
        d.bs_sets = [sorted(s) for s in sets]
    return d


def design_1layer(net, p):
    """1-layer RSMA of the design: the same core users and BSs, no group-common streams."""
    return design(net, p, groups_on=False)


def design_1layer_fill(net, p):
    """Sensitivity check of the 1-layer counterpart (not the counterpart of the design): the same core users and BSs,
    and every BS fills its free slots with private streams of non-core UEs, in decreasing order of its energy to them
    (as J1 fills the slots of the group-common streams with the next private streams)."""
    d = design(net, p, groups_on=False)
    B, N, S, K = p['B'], p['N'], p['Smax'], p['K']
    E = _energy(net['Hd'], B, N)
    bm = np.ones((B, d.nS), bool) if d.bsmask is None else np.asarray(d.bsmask, bool)
    sets = [[d.served[i] for i in np.flatnonzero(bm[b])] for b in range(B)]
    others = [k for k in range(K) if k not in d.core]
    for b in range(B):
        for k in sorted(others, key=lambda k: (-E[k, b], k)):
            if len(sets[b]) >= S: break
            sets[b].append(1 + k)
    carried = set().union(*map(set, sets))
    dn = H_.Design([], [j for j in d.priority if j in carried], K)
    dn.core = d.core; dn.priority = d.priority; dn.rule = 'zf_minimal_1layer_fill'
    if any(set(s) != carried for s in sets):
        dn.bsmask = np.array([[j in s for j in dn.served] for s in sets])
        dn.bs_sets = [sorted(s) for s in sets]
    return dn
