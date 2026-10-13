"""Experimental BS-specific discrete designs (design study; the final design is in bsd.py)."""
import math
import numpy as np
import hrs as H_


def energy(net, p):
    """E[k, b] = ||h_d,kb||^2 (noise-normalized direct-channel energy of UE k from BS b)."""
    Hd, N, B = net['Hd'], p['N'], p['B']
    return np.stack([np.sum(np.abs(Hd[:, b * N:(b + 1) * N]) ** 2, axis=1) for b in range(B)], axis=1)


def finish(groups, lists, K, p, core, priority, rule):
    """Design from per-BS ordered stream lists (global ids; 0 first): BS b carries the first S_max of its list."""
    S, B = p['Smax'], p['B']
    sets = [list(dict.fromkeys(l))[:S] for l in lists]
    carried = set().union(*map(set, sets))
    served = [j for j in priority if j in carried]
    d = H_.Design(groups, served, K)
    d.core = list(core); d.priority = list(priority); d.rule = rule
    if any(set(s) != carried for s in sets):
        d.bsmask = np.array([[j in sets[b] for j in d.served] for b in range(B)])
        d.bs_sets = [sorted(s) for s in sets]
    return d


def generic(net, p, order, use_groups=True, core_rule='sus', home_rule='energy', grp_key='min'):
    """order: sequence of class names for every BS list after the super-common stream:
       'core'  all core privates by E[k,b];  'own' / 'coop' core privates with / without home b
       'grp'   all group-common streams by the BS's energy to the weakest member; 'ogrp' / 'cgrp' own / other groups
       'non'   non-core privates by E[k,b]."""
    K, B, BN = p['K'], p['B'], p['BN']
    Hd = net['Hd']; E = energy(net, p)
    core, groups, _, _ = H_.joint_structure(Hd, p, core_rule)
    if not use_groups: groups = []
    G = len(groups)
    home = np.argmax(E, axis=1)
    priv = lambda k: 1 + G + k
    gq = np.array([[E[m, b].min() for b in range(B)] for m in groups]) if G else np.zeros((0, B))
    ghome = np.argmax(gq, axis=1) if G else np.zeros(0, int)
    noncore = [k for k in range(K) if k not in core]
    lists = []
    for b in range(B):
        lst = [0]
        for cl in order:
            if cl == 'core': lst += [priv(k) for k in sorted(core, key=lambda k: -E[k, b])]
            elif cl == 'own': lst += [priv(k) for k in sorted([k for k in core if home[k] == b], key=lambda k: -E[k, b])]
            elif cl == 'coop': lst += [priv(k) for k in sorted([k for k in core if home[k] != b], key=lambda k: -E[k, b])]
            elif cl == 'grp': lst += [1 + g for g in sorted(range(G), key=lambda g: -gq[g, b])]
            elif cl == 'ogrp': lst += [1 + g for g in sorted([g for g in range(G) if ghome[g] == b], key=lambda g: -gq[g, b])]
            elif cl == 'cgrp': lst += [1 + g for g in sorted([g for g in range(G) if ghome[g] != b], key=lambda g: -gq[g, b])]
            elif cl == 'non': lst += [priv(k) for k in sorted(noncore, key=lambda k: -E[k, b])]
            elif cl == 'mix':      # core privates and group-common streams ranked together by the energy to the weakest decoder
                cand = [(E[k, b], priv(k)) for k in core] + [(gq[g, b], 1 + g) for g in range(G)]
                lst += [j for _, j in sorted(cand, key=lambda t: (-t[0], t[1]))]
            else: raise ValueError(cl)
        lists.append(lst)
    # global order (for the records / stream clustering): super-common, core (SUS order), groups by size, non-core by energy
    tot = E.sum(1)
    pr = [0] + [priv(k) for k in core] + [1 + g for g in sorted(range(G), key=lambda g: (-len(groups[g]), g))] \
         + [priv(k) for k in sorted(noncore, key=lambda k: (-tot[k], k))]
    return finish(groups, lists, K, p, core, pr, 'bsx_' + '_'.join(order))


# P6: class order c, core, groups, non-core; per-BS ranking within each class
def A(net, p): return generic(net, p, ('core', 'grp', 'non'))
def A1(net, p): return generic(net, p, ('core', 'grp', 'non'), use_groups=False)
# own core first, then own groups, then cooperation
def C(net, p): return generic(net, p, ('own', 'ogrp', 'coop', 'cgrp', 'non'))
def C1(net, p): return generic(net, p, ('own', 'ogrp', 'coop', 'cgrp', 'non'), use_groups=False)
# mixed ranking of core privates and group-commons
def M(net, p): return generic(net, p, ('mix', 'non'))
def M1(net, p): return generic(net, p, ('mix', 'non'), use_groups=False)


# ---------------------------------------------------------------- ZF-minimal carriage
def zf_minimal(net, p, use_groups=True, ncore='j', groups_on='served', leftover=('grp', 'non')):
    """Core users: the first n of the SUS order (n = min(K, BN, S_max - 1) for ncore='j', or the largest n with
    n * ceil(n / N) <= B (S_max - 1) for ncore='dof'); every core stream is carried by the m = ceil(n / N) BSs with
    the strongest channels to its UE (the fewest BSs whose antennas can zero-force it against the other core streams);
    groups: every non-core UE joins the most correlated core user; the remaining slots of each BS: group-common
    streams by the BS's energy to the weakest member, then non-core privates by energy."""
    K, B, N, BN, S = p['K'], p['B'], p['N'], p['BN'], p['Smax']
    Hd = net['Hd']; E = energy(net, p)
    order = H_.sus(Hd, min(K, BN))
    if ncore == 'j':
        n = min(K, BN, S - 1)
    else:
        n = 1
        while n + 1 <= min(K, BN) and (n + 1) * math.ceil((n + 1) / N) <= B * (S - 1): n += 1
    core = order[:n]; m = math.ceil(n / N)
    Hn = Hd / np.linalg.norm(Hd, axis=1)[:, None]; corr = np.abs(Hn @ Hn.conj().T) ** 2
    pool = core if groups_on == 'served' else order
    follow = {c: [] for c in pool}
    for k in range(K):
        if k in pool: continue
        follow[pool[int(np.argmax(corr[k, pool]))]].append(k)
    groups = [[c] + follow[c] for c in pool if follow[c]] if use_groups else []
    groups = [g for g in groups if len(g) <= K - 1]
    G = len(groups); priv = lambda k: 1 + G + k
    load = np.zeros(B, int); carriers = {}
    for k in core:          # SUS order; the m strongest BSs with a free core slot
        bs = [b for b in np.argsort(-E[k]) if load[b] < S - 1][:m]
        carriers[k] = bs; load[bs] += 1
    gq = np.array([[E[mm, b].min() for b in range(B)] for mm in groups]) if G else np.zeros((0, B))
    noncore = [k for k in range(K) if k not in core]
    lists = []
    for b in range(B):
        lst = [0] + [priv(k) for k in core if b in carriers[k]]
        for cl in leftover:
            if cl == 'grp': lst += [1 + g for g in sorted(range(G), key=lambda g: -gq[g, b])]
            elif cl == 'non': lst += [priv(k) for k in sorted(noncore, key=lambda k: -E[k, b])]
            elif cl == 'coop': lst += [priv(k) for k in sorted([k for k in core if b not in carriers[k]], key=lambda k: -E[k, b])]
        lists.append(lst)
    tot = E.sum(1)
    pr = [0] + [priv(k) for k in core] + [1 + g for g in sorted(range(G), key=lambda g: (-len(groups[g]), g))] \
         + [priv(k) for k in sorted(noncore, key=lambda k: (-tot[k], k))]
    return finish(groups, lists, K, p, core, pr, f'zfmin_{ncore}')


def Z(net, p): return zf_minimal(net, p)
def Z1(net, p): return zf_minimal(net, p, use_groups=False)
def ZD(net, p): return zf_minimal(net, p, ncore='dof')
def ZD1(net, p): return zf_minimal(net, p, ncore='dof', use_groups=False)


# private streams for the core users only; the remaining fronthaul carries group-common streams (HRS) or nothing (1L)
def ZDn(net, p): return zf_minimal(net, p, ncore='dof', leftover=('grp',))
def ZDn1(net, p): return zf_minimal(net, p, ncore='dof', use_groups=False, leftover=('grp',))
def Zn(net, p): return zf_minimal(net, p, leftover=('grp',))
def Zn1(net, p): return zf_minimal(net, p, use_groups=False, leftover=('grp',))
