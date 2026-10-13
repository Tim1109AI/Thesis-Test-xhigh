"""The two discrete-design benchmarks of the thesis benchmark figure (Fig. 5.10) that requirement 2 refers to.

  stream_cluster_design(net, p, d=None)   stream clustering [Ahmad2021cran, Algorithm 1] applied to the streams of the
                                          design d (its groups); d=None: the groups of the current design (SUS core
                                          users, nearest-core groups), as in the stored results
  sparse_bf_design(net, p, seed, lamQ)    sparse beamforming [Dai2014sparse, Algorithm 1]; candidate streams: the HRS
                                          order of the current design (independent of the design under test)

Both return an hrs.Design with a BS-specific association (d.bsmask) and are run by discrete.run like any design
(framing=False: they may leave the super-common stream to some BSs). The stored results of Fig. 5.10 (drops 801-805)
are in reference/bench_fig510.jsonl; harness/bench.py runs and compares.
"""
import math
import numpy as np
import scipy.sparse as sp
import clarabel
import hrs as H_, algo as A


def stream_cluster_design(net, p, d=None):
    """Ahmad2021cran, Algorithm 1 (stream-based clustering with load balancing), applied to all streams of an HRS
    design (super-common, the group-common streams of its groups, the private streams of all UEs): the channel
    quality of BS n to UE j is q_nj = -path loss in dB (statistical CSI: the direct-link distance), the collective
    quality of BS n to a stream is the mean of q_nj over its decoders, and the candidate cluster of a stream is its
    mu = B best BSs. Pooling rounds: every stream with a non-empty candidate set is joined by its strongest remaining
    candidate; then every BS above A_max = S_max streams keeps its S_max strongest streams (collective quality) and
    leaves the network and all candidate sets. The rounds repeat until every candidate set is used up or every BS has
    left; a stream carried by several BSs is precoded jointly over their antennas.
    d: the design whose groups (and stream order, d.priority, when it lists every stream) are used; None: the current
    design's groups and order (the stored results)."""
    K, B, S = p['K'], p['B'], p['Smax']
    if d is None:
        order, groups, energy, _ = H_.joint_structure(net['Hd'], p, 'sus')
        pr = H_.priority_list(order, groups, energy, K, 'private_first')
        core = order
    else:
        groups = [list(m) for m in d.groups]; J = 1 + len(groups) + K
        pr = list(d.priority) if getattr(d, 'priority', None) is not None and sorted(d.priority) == list(range(J)) else list(range(J))
        core = list(d.core)
    dall = H_.Design(groups, pr, K)
    qdb = np.zeros((K, B))
    for k in range(K):
        for b in range(B):
            qdb[k, b] = -10 * p['alpha'] * math.log10(np.linalg.norm(net['UE'][k] - net['BS'][b]))
    q = {j: qdb[dall.decoders[j]].mean(0) for j in pr}
    cand = {j: list(np.argsort(-q[j])) for j in pr}
    sets = [[] for _ in range(B)]; open_ = [True] * B
    for _round in range(B):
        for j in pr:
            while cand[j] and not open_[cand[j][0]]: cand[j].pop(0)
            if cand[j]:
                sets[cand[j].pop(0)].append(j)
        for b in range(B):
            if open_[b] and len(sets[b]) > S:
                sets[b] = sorted(sets[b], key=lambda j: -q[j][b])[:S]; open_[b] = False
    served = [j for j in pr if any(j in s for s in sets)]
    dn = H_.Design(groups, served, K); dn.core = core; dn.priority = pr; dn.rule = 'stream_cluster'
    dn.bsmask = np.array([[j in sets[b] for j in dn.served] for b in range(B)]); dn.bs_sets = [sorted(s) for s in sets]
    return dn


class QTPrecoderFronthaul(H_.QTPrecoderDirect):
    """QT precoder step with the reweighted-l1 stream-count constraint of Dai2014sparse per BS:
    sum_i beta_{b,i} ||w_{b,i}||^2 <= S_max, beta from the previous iterate."""
    def __init__(self, d, p, lamQ, beta):
        super().__init__(d, p, lamQ); self.beta = beta

    def solve(self, H, W):
        p, d = self.p, self.d; BN, N = p['BN'], p['N']; n = self.n
        orig = clarabel.DefaultSolver
        extra = []
        for b in range(p['B']):
            idx = [(i, i * BN + b * N + m) for i in range(d.nS) for m in range(N)]
            m_ = 1 + 2 * len(idx); Ae = np.zeros((m_, n)); be = np.zeros(m_); be[0] = math.sqrt(p['Smax'])
            for qq, (i, ii) in enumerate(idx):
                w = math.sqrt(self.beta[b, i])
                Ae[1 + qq, self.iwr + ii] = -w; Ae[1 + len(idx) + qq, self.iwi + ii] = -w
            extra.append((Ae, be, clarabel.SecondOrderConeT(m_)))

        def patched(P, q, A_, b_, cones, st):
            A2 = sp.vstack([A_] + [sp.csc_matrix(e[0]) for e in extra]).tocsc()
            b2 = np.concatenate([b_] + [e[1] for e in extra])
            return orig(P, q, A2, b2, list(cones) + [e[2] for e in extra], st)
        clarabel.DefaultSolver = patched
        try:
            return super().solve(H, W)
        finally:
            clarabel.DefaultSolver = orig


def sparse_bf_design(net, p, seed, lamQ, max_iter=50, tau=1e-10, thr=1e-6, stable=5):
    """Dai2014sparse, Algorithm 1 (reweighted-l1 sparse beamforming under a per-BS fronthaul limit), with the
    fronthaul counted in streams: every stream of the current HRS order is a candidate at every BS, and the precoder
    update (the QT step of the framework in place of WMMSE) carries the constraint sum_i beta_{b,i} ||w_{b,i}||^2 <=
    S_max per BS, beta_{b,i} = 1 / (||w_{b,i}||^2 + tau) from the previous iterate (tau = 1e-10, as in the paper).
    Reweighted l1 starts from unit weights, under which the constraint is inactive: the first iterate is the precoder
    optimum without the limit (QT loop to convergence). Links whose power falls below thr * P_max are removed, and the
    iterations run until every BS carries at most S_max streams or the supports no longer change (`stable`
    iterations) or max_iter (a failed solver step also ends them). Computed once at the prescribed deployment
    (closed-form coefficients); every BS then keeps at most S_max of its strongest streams, and the supports stay
    fixed during the search."""
    K, B, N, S = p['K'], p['B'], p['N'], p['Smax']
    order, groups, energy, _ = H_.joint_structure(net['Hd'], p, 'sus')
    pr = H_.priority_list(order, groups, energy, K, 'private_first')
    dall = H_.Design(groups, pr, K); dall.core = order; dall.priority = pr
    pose = H_.pose_from(net['p0'], net['angles0']); ch = H_.channel_terms(pose, net, p)
    theta = A.SEEDS['rule'](ch, net, dall, p); Hh = H_.effective_channel(theta, ch, net, p)

    def powers(W):
        return np.array([[np.sum(np.abs(W[b * N:(b + 1) * N, i]) ** 2) for i in range(dall.nS)] for b in range(B)])
    W, _, _, _ = A.qt_precoder_loop(Hh, A.INITS['eqc'](Hh, dall, p), dall, p, H_.QTPrecoderDirect(dall, p, lamQ), lamQ, 30, 0.1)
    pw = powers(W); act = pw > thr * p['Pmax']; same = 0; it = 0
    for it in range(1, max_iter + 1):
        if act.sum(1).max() <= S:
            it -= 1; break
        dw = H_.Design(groups, pr, K); dw.core = order; dw.priority = pr; dw.bsmask = act.copy()
        beta = np.where(act, 1.0 / (pw + tau), 0.0)
        try:
            W = QTPrecoderFronthaul(dw, p, lamQ, beta).solve(Hh, W)
        except Exception:
            break
        pw = powers(W); new = act & (pw > thr * p['Pmax'])
        same = same + 1 if np.array_equal(new, act) else 0
        act = new
        if same >= stable: break
    sets = []
    for b in range(B):
        on = [i for i in np.argsort(-pw[b]) if act[b, i]][:S]
        sets.append([dall.served[i] for i in on])
    served = [j for j in pr if any(j in s for s in sets)]
    if not served: served = pr[:S]; sets = [served] * B
    d = H_.Design(groups, served, K); d.core = order; d.priority = pr; d.rule = 'sparse_bf'; d.sbf_iters = it
    d.bsmask = np.array([[j in sets[b] for j in d.served] for b in range(B)]); d.bs_sets = [sorted(s) for s in sets]
    if d.bsmask.all(): d.bsmask = None
    return d
