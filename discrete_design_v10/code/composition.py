"""Rate composition of an HRS solution and the no-group ablation."""
import numpy as np
import hrs as H_


def composition(W, H, d, p):
    Y = H @ W
    cap, _ = H_.capacities(Y, d)
    G, K, Rth = d.G, d.K, p['Rth']
    core = set(d.core)
    rc = float(cap[d.col[0]]) if 0 in d.col else 0.0
    rg = {g: float(cap[d.col[1 + g]]) for g in d.served_groups}
    rk = np.zeros(K)
    for k in d.private_served:
        rk[k] = cap[d.col[1 + G + k]]
    need = np.maximum(Rth - rk, 0.0)
    covered = np.zeros(K, bool); carried_g = 0.0; resid = 0.0
    for g in d.served_groups:
        m = d.groups[g]; covered[m] = True
        ng = need[m].sum(); carried_g += min(rg[g], ng); resid += max(ng - rg[g], 0.0)
    resid += need[~covered].sum()
    carried_c = min(rc, resid)
    pw = np.sum(np.abs(W) ** 2, axis=0); ptot = pw.sum()
    typ = {0: 'c'} if 0 in d.col else {}      # (a design may leave the super-common stream unserved)
    for g in d.served_groups: typ[1 + g] = 'g'
    for k in d.private_served: typ[1 + G + k] = 'pc' if k in core else 'pn'
    pshare = {t: float(sum(pw[d.col[j]] for j in typ if typ[j] == t) / ptot) for t in ['c', 'g', 'pc', 'pn']}
    total = rc + sum(rg.values()) + rk.sum()
    nongroup_noncore = [k for k in range(K) if k not in core and not covered[k]]
    return dict(rate=float(total), r_c=rc, r_g=float(sum(rg.values())), r_pc=float(sum(rk[k] for k in core)),
                r_pn=float(sum(rk[k] for k in range(K) if k not in core)),
                n_groups_served=len(d.served_groups), n_groups=G, n_groups_active=int(sum(v > 0.1 for v in rg.values())),
                group_rates=[rg[g] for g in d.served_groups], group_sizes=[len(d.groups[g]) for g in d.served_groups],
                load=float(need.sum()), load_nominal=float(K * Rth), carried_g=float(carried_g), carried_c=float(carried_c),
                n_qos_by_private=int(np.sum(rk >= Rth - 1e-9)) if Rth > 0 else K,
                n_noncore_in_groups=int(sum(1 for k in range(K) if k not in core and covered[k])),
                n_noncore_outside=len(nongroup_noncore), power_share=pshare)


def nogroup_design(d, p):
    """Same core users and order, group-common streams removed; the slots go to the next private streams."""
    K = d.K
    others = [k for k in range(K) if k not in d.core]
    others = sorted(others, key=lambda k: d.priority.index(1 + d.G + k))
    pr = [0] + [1 + k for k in d.core] + [1 + k for k in others]
    served = pr[:min(p['Smax'], len(pr))]
    dn = H_.Design([], served, K)
    dn.core = d.core; dn.rule = 'no_groups'; dn.priority = pr
    return dn


def ue_allocation(W, H, d, p):
    """Per-UE split of the sum rate into private, group-common and super-common parts. Each private stream carries
    its UE only. The rates of the common streams are first allocated to the QoS deficits R_th - r_k (group-common
    streams to their members, then the super-common stream to the remaining deficits, in proportion to the deficits),
    and the remainder of each common stream is shared equally among the UEs that decode it. The split of the
    remainder does not affect the sum rate; this rule is only a way to display it."""
    Y = H @ W
    cap, _ = H_.capacities(Y, d)
    K, G, Rth = d.K, d.G, p['Rth']
    priv = np.zeros(K); grp = np.zeros(K); sup = np.zeros(K)
    for k in d.private_served:
        priv[k] = cap[d.col[1 + G + k]]
    need = np.maximum(Rth - priv, 0.0)
    for g in d.served_groups:
        m = list(d.carriers[g]); rg = float(cap[d.col[1 + g]])
        nd = need[m]; tot = nd.sum()
        give = nd * min(1.0, rg / tot) if tot > 0 else np.zeros(len(m))
        grp[m] += give; need[m] -= give
        grp[m] += (rg - give.sum()) / len(m)
    if 0 in d.col:
        rc = float(cap[d.col[0]]); tot = need.sum()
        give = need * min(1.0, rc / tot) if tot > 0 else np.zeros(K)
        sup += give + (rc - give.sum()) / K
    return dict(private=priv.tolist(), group=grp.tolist(), super=sup.tolist(),
                total=(priv + grp + sup).tolist(), cap=[float(x) for x in cap], served=[d.name(j) for j in d.served])
