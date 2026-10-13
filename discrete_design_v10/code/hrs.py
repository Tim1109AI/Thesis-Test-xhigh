"""Python port of rsma_evaluator_test.m (system model, joint discrete design,
QT precoder / coefficient blocks, AO, BO) for algorithm design experiments.

Conventions follow the MATLAB file: noise normalised to 1 (channels scaled by
1/sqrt(noise)), theta is (M,4,L) with modes AT,AR,BT,BR, precoders W are
(B*N, J) with column index over the SERVED streams only.
"""
import json, math, time
import numpy as np
import cvxpy as cp

LN2 = math.log(2.0)
BANK = None


def load_bank(path):
    global BANK
    if BANK is None:
        with open(path) as f:
            BANK = json.load(f)
    return BANK


# --------------------------------------------------------------------------- geometry
def element_grid(nc, nr, spacing):
    j = np.arange(nc * nr)
    return np.stack([(j % nc - (nc - 1) / 2) * spacing, np.zeros(nc * nr),
                     (j // nc - (nr - 1) / 2) * spacing], axis=1)


def rot(phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def params(K=8, L=3, N=3, M=100, Smax=9, Pmax=1.0, Dmax=10.0, Rth=0.0, bank=None, B=2):
    """B: number of BSs (their positions: make_network(..., BS=...); code/scenarios.py)."""
    p = dict(B=B, Pmax=Pmax, noise=1e-12, Smax=Smax, Rth=Rth, lam=0.1, PL0=1e-3, alpha=2.2, q=3)
    p['gainScale'] = 10 ** (0.4 * (p['q'] + 1))
    reg = bank['region']
    for key in ['X', 'Y', 'zBS', 'zUE', 'zmin', 'zmax', 'dmin', 'd0']:
        p[key] = reg[key]
    p.update(K=K, L=L, N=N, M=M, Dmax=np.full(L, float(Dmax)))
    p['Nc'], p['Nr'] = 1, N
    divisors = [d for d in range(1, min(10, M) + 1) if M % d == 0]
    p['Mr'] = divisors[-1]; p['Mc'] = M // p['Mr']
    p['uBS'] = element_grid(p['Nc'], p['Nr'], p['lam'] / 2)
    p['uR'] = element_grid(p['Mc'], p['Mr'], p['lam'] / 2)
    p['BN'] = p['B'] * N
    return p


def make_network(p, bank, kind='random', drop=801, setup=1, bs_setup='even', BS=None, UE=None):
    """BS, UE: positions (B x 3, K x 3) replacing the bank's BS layout and UE drop (code/scenarios.py)."""
    layout = bank['bsSetups'][bs_setup]
    users = np.array(layout['UE']['K%d' % p['K']][kind])
    star = bank['starSetups']['L%d' % p['L']]
    net = {}
    net['BS'] = np.array(layout['BS'] if BS is None else BS, float)
    net['UE'] = users[(drop - 1) * p['K']:drop * p['K'], :] if UE is None else np.array(UE, float)
    assert len(net['BS']) == p['B'] and len(net['UE']) == p['K'], 'BS / UE positions do not match p'
    rows = slice((setup - 1) * p['L'], setup * p['L'])
    net['p0'] = np.array(star['positions'], float)[rows]
    net['angles0'] = np.array(star['angles'], float)[rows, 0]
    K, B, N = p['K'], p['B'], p['N']
    k0 = 2 * math.pi / p['lam']
    Hd = np.zeros((K, B * N), complex)
    for k in range(K):
        for b in range(B):
            v = net['UE'][k] - net['BS'][b]; d = np.linalg.norm(v)
            a = np.exp(1j * k0 * p['uBS'] @ (v / d))
            Hd[k, b * N:(b + 1) * N] = math.sqrt(p['PL0'] / p['noise']) * d ** (-p['alpha'] / 2) * np.exp(-1j * k0 * d) * a
    net['Hd'] = Hd
    return net


def pose_from(pos, angles):
    return {'pos': np.array(pos, float), 'angles': np.array(angles, float)}


def geometry_violation(pose, net, p):
    pos, ang = pose['pos'], pose['angles']
    move = np.linalg.norm(pos - net['p0'], axis=1)
    viol = max(0.0, np.max(move - p['Dmax']))
    viol = max(viol, np.max(np.abs(ang)) - math.pi / 2)
    box = max(0.0, np.max(-pos[:, 0]), np.max(pos[:, 0] - p['X']), np.max(-pos[:, 1]), np.max(pos[:, 1] - p['Y']),
              np.max(p['zmin'] - pos[:, 2]), np.max(pos[:, 2] - p['zmax']))
    viol = max(viol, box)
    L = p['L']; sep = math.inf; link = math.inf
    fixed = np.vstack([net['BS'], net['UE']])
    for l in range(L):
        link = min(link, np.min(np.linalg.norm(fixed - pos[l], axis=1)))
        for h in range(l + 1, L):
            sep = min(sep, np.linalg.norm(pos[l] - pos[h]))
    viol = max(viol, p['dmin'] - sep)
    return viol, max(0.0, p['d0'] - link)


# --------------------------------------------------------------------------- channels
def channel_terms(pose, net, p):
    """LoS single hop: A_{k,b,l} = s_kbl * e_kbl aBS_bl^T (rank one).
    Returns dict with s (K,B,L), e (K,B,L,M), aBS (B,L,N), mode (L,B,K) in 0..3."""
    K, B, L, M, N = p['K'], p['B'], p['L'], p['M'], p['N']
    k0 = 2 * math.pi / p['lam']
    s = np.zeros((K, B, L), complex); e = np.zeros((K, B, L, M), complex)
    aBS = np.zeros((B, L, N), complex); mode = np.zeros((L, B, K), int)
    for l in range(L):
        R = rot(pose['angles'][l]); pos = pose['pos'][l]
        for b in range(B):
            rb = net['BS'][b] - pos; db = np.linalg.norm(rb); ub = rb / db; vb = R.T @ ub
            aBS[b, l] = np.exp(1j * k0 * p['uBS'] @ (-ub))
            faceA = vb[1] >= 0
            for k in range(K):
                ru = net['UE'][k] - pos; du = np.linalg.norm(ru); vu = R.T @ (ru / du)
                same = faceA == (vu[1] >= 0)
                mode[l, b, k] = 2 * (0 if faceA else 1) + (1 if same else 0)
                e[k, b, l] = np.exp(1j * k0 * p['uR'] @ (vb + vu))
                s[k, b, l] = (p['gainScale'] * p['PL0'] / math.sqrt(p['noise']) * db ** (-p['alpha'] / 2)
                              * du ** (-p['alpha'] / 2) * np.exp(-1j * k0 * (db + du))
                              * abs(vb[1]) ** p['q'] * abs(vu[1]) ** p['q'])
    return {'s': s, 'e': e, 'aBS': aBS, 'mode': mode}


def cascade_scalars(theta, ch):
    """c_kbl = theta_{l,mode}^T e_kbl."""
    K, B, L = ch['s'].shape
    c = np.zeros((K, B, L), complex)
    for l in range(L):
        for b in range(B):
            th = theta[:, ch['mode'][l, b, :], l]  # (M,K)
            c[:, b, l] = np.einsum('km,mk->k', ch['e'][:, b, l, :], th)
    return c


def effective_channel_from_c(c, ch, net, p):
    H = net['Hd'].copy(); N = p['N']
    for b in range(p['B']):
        H[:, b * N:(b + 1) * N] += np.einsum('kl,ln->kn', ch['s'][:, b, :] * c[:, b, :], ch['aBS'][b])
    return H


def effective_channel(theta, ch, net, p):
    return effective_channel_from_c(cascade_scalars(theta, ch), ch, net, p)


def project_coupled(relaxed):
    theta = np.zeros_like(relaxed)
    for col in (0, 2):
        t, r = relaxed[:, col, :], relaxed[:, col + 1, :]
        u = unit_phase(t + r); v = unit_phase(t - r)
        theta[:, col, :] = (u + v) / 2; theta[:, col + 1, :] = (u - v) / 2
    return theta


def unit_phase(x):
    x = np.asarray(x); z = np.ones_like(x, dtype=complex)
    nz = np.abs(x) > 0
    z[nz] = x[nz] / np.abs(x[nz])
    return z


def theta_from_field(field):
    """max Re{t zT + r zR} s.t. coupled hardware: u=ph(conj(zT+zR)), v=ph(conj(zT-zR))."""
    theta = np.zeros_like(field)
    for q in (0, 2):
        plus = unit_phase(np.conj(field[:, q, :] + field[:, q + 1, :]))
        minus = unit_phase(np.conj(field[:, q, :] - field[:, q + 1, :]))
        theta[:, q, :] = (plus + minus) / 2; theta[:, q + 1, :] = (plus - minus) / 2
    return theta


def hardware_residual(theta):
    res = 0.0
    for q in (0, 2):
        T, R = theta[:, q, :], theta[:, q + 1, :]
        res = max(res, np.max(np.abs(np.abs(T) ** 2 + np.abs(R) ** 2 - 1)), np.max(np.abs(np.real(T * np.conj(R)))))
    return res


# --------------------------------------------------------------------------- discrete design
def sus(H, count):
    K = H.shape[0]; Q = np.zeros((H.shape[1], 0), complex); order = []
    gain = np.sum(np.abs(H) ** 2, axis=1)
    for _ in range(min(K, count)):
        best, chosen, res_c = -1, -1, None
        for k in range(K):
            if k in order: continue
            res = H[k] - (H[k] @ Q) @ Q.conj().T
            val = np.sum(np.abs(res) ** 2)
            if val > best: best, chosen, res_c = val, k, res
        order.append(chosen)
        if best > 1e-12 * max(gain[chosen], 1e-300):
            Q = np.hstack([Q, (res_c.conj() / math.sqrt(best))[:, None]])
    return order


class Design:
    """Stream bookkeeping. Global stream ids: 0=super-common, 1..G groups, G+1+k private of UE k."""

    def __init__(self, groups, served, K):
        self.groups = groups; self.G = len(groups); self.K = K; self.J = 1 + self.G + K
        # carriers[g]: members of group g whose rate the group-common stream may carry (all members in HRS;
        # only the weak UE of a NOMA pair). bsmask: optional (B, nS) bool, BS b transmits served column i.
        self.carriers = [list(m) for m in groups]; self.bsmask = None
        self.gid = -np.ones(K, int)
        for g, members in enumerate(groups):
            self.gid[members] = g
        self.served = sorted(served)
        self.col = {j: i for i, j in enumerate(self.served)}  # global id -> column
        self.nS = len(self.served)
        self.decoders = {}
        for j in self.served:
            if j == 0: self.decoders[j] = list(range(K))
            elif j <= self.G: self.decoders[j] = list(groups[j - 1])
            else: self.decoders[j] = [j - 1 - self.G]
        # interference sets (global ids) per (k, j)
        self.pairs = []  # (k, j, [interfering global ids])
        S = set(self.served)
        for j in self.served:
            for k in self.decoders[j]:
                if j == 0: canc = {0}
                elif j <= self.G: canc = {0, j}
                else:
                    canc = {0, j}
                    if self.gid[k] >= 0: canc.add(1 + self.gid[k])
                self.pairs.append((k, j, sorted(S - canc)))
        self.served_groups = [g for g in range(self.G) if (1 + g) in S]
        self.private_served = [k for k in range(K) if (1 + self.G + k) in S]

    def name(self, j):
        if j == 0: return 'c'
        if j <= self.G: return 'g%d' % (j)
        return 'p%d' % (j - self.G)


def zf_greedy(H, P):
    """Greedy ZF sum-rate user selection (Dimic-Sidiropoulos): add the user that most increases
    sum_k log2(1 + (P/|C|)/[(H_C H_C^H)^-1]_kk); stop when no user increases it."""
    K, BN = H.shape; sel = []; best = 0.0
    while len(sel) < min(K, BN):
        cand, cr = -1, -np.inf
        for k in range(K):
            if k in sel: continue
            S = sel + [k]; Hs = H[S]
            try:
                g = 1.0 / np.maximum(np.real(np.diag(np.linalg.inv(Hs @ Hs.conj().T))), 1e-300)
            except np.linalg.LinAlgError:
                continue
            r = np.sum(np.log2(1 + P / len(S) * g))
            if r > best + 1e-9 and r > cr: cand, cr = k, r
        if cand < 0: break
        sel.append(cand); best = cr
    return sel


def joint_structure(Hd, p, core_rule='sus'):
    """Core selection + nearest-core grouping."""
    BN, K = p['BN'], p['K']
    order = sus(Hd, BN) if core_rule == 'sus' else zf_greedy(Hd, p['B'] * p['Pmax'])
    energy = np.sum(np.abs(Hd) ** 2, axis=1)
    Hn = Hd / np.sqrt(energy)[:, None]
    corr = np.abs(Hn @ Hn.conj().T) ** 2
    follow = {c: [] for c in order}
    for k in range(K):
        if k in order: continue
        c = order[int(np.argmax(corr[k, order]))]
        follow[c].append(k)
    groups = [[c] + follow[c] for c in order if follow[c]]
    return order, groups, energy, corr


def priority_list(order, groups, energy, K, rule):
    G = len(groups)
    sizes = [len(g) for g in groups]
    grank = sorted(range(G), key=lambda g: (-sizes[g], g))
    core_priv = [1 + G + k for k in order]
    others = sorted([k for k in range(K) if k not in order], key=lambda k: (-energy[k], k))
    other_priv = [1 + G + k for k in others]
    grp = [1 + g for g in grank]
    if rule == 'private_first':
        return [0] + core_priv + grp + other_priv
    if rule == 'group_first':
        return [0] + grp + core_priv + other_priv
    if rule == 'cluster':
        # each core: private then its group (if any), in SUS order
        head = {groups[g][0]: 1 + g for g in range(G)}
        pr = [0]
        for c in order:
            pr.append(1 + G + c)
            if c in head: pr.append(head[c])
        return pr + other_priv
    raise ValueError(rule)


def joint_design(net, p, rule='private_first', core_rule='sus'):
    order, groups, energy, corr = joint_structure(net['Hd'], p, core_rule)
    pr = priority_list(order, groups, energy, p['K'], rule)
    served = pr[:min(p['Smax'], len(pr))]
    d = Design(groups, served, p['K'])
    d.core = order; d.rule = rule; d.priority = pr
    return d


# --------------------------------------------------------------------------- rates / merit
def capacities(Y, d):
    """Y: (K, nS) received amplitudes over served columns. Returns cap per served column."""
    cap = np.full(d.nS, np.inf); dec = {}
    for (k, j, I) in d.pairs:
        cj = d.col[j]; icols = [d.col[i] for i in I]
        sinr = abs(Y[k, cj]) ** 2 / (1.0 + np.sum(np.abs(Y[k, icols]) ** 2))
        r = math.log2(1.0 + sinr); dec[(k, j)] = r
        cap[cj] = min(cap[cj], r)
    return cap, dec


def deficit(cap, d, Rth):
    """Minimum total QoS deficit given stream capacities (exact routability)."""
    if Rth <= 0: return 0.0
    K = d.K; G = d.G
    rk = np.zeros(K)
    for k in d.private_served:
        rk[k] = cap[d.col[1 + G + k]]
    need = np.maximum(Rth - rk, 0.0)
    total = 0.0; covered = np.zeros(K, bool)
    for g in d.served_groups:
        m = d.carriers[g]; covered[m] = True
        total += max(0.0, need[m].sum() - cap[d.col[1 + g]])
    total += need[~covered].sum()
    return max(0.0, total - cap[d.col[0]]) if 0 in d.col else total


def merit(W, H, d, p, lamQ):
    Y = H @ W
    cap, _ = capacities(Y, d)
    rate = float(np.sum(cap)); df = deficit(cap, d, p['Rth'])
    return rate - lamQ * df, rate, df


def power_ok(W, p):
    N = p['N']
    return max(np.sum(np.abs(W[b * N:(b + 1) * N]) ** 2) for b in range(p['B'])) <= p['Pmax'] * (1 + 1e-6)


def scale_to_power(W, p):
    W = W.copy(); N = p['N']
    for b in range(p['B']):
        blk = W[b * N:(b + 1) * N]; m = np.linalg.norm(blk)
        if m > 0: W[b * N:(b + 1) * N] = blk * math.sqrt(p['Pmax']) / m
    return W


# --------------------------------------------------------------------------- QT blocks (cvxpy DPP)
class QTPrecoder:
    """max sum_j r_j - lamQ sum xi  s.t. QT rate constraints, per-BS power, QoS with slack."""

    def __init__(self, d, p, lamQ):
        self.d, self.p, self.lamQ = d, p, lamQ
        BN, nS, K = p['BN'], d.nS, d.K
        self.W = cp.Variable((BN, nS), complex=True)
        self.r = cp.Variable(nS, nonneg=True)
        self.xi = cp.Variable(K, nonneg=True)
        self.ac = cp.Variable(K, nonneg=True)
        grouped = [k for g in d.served_groups for k in d.carriers[g]]
        self.ag = cp.Variable(len(grouped), nonneg=True) if grouped else None
        gpos = {k: i for i, k in enumerate(grouped)}
        cons = []
        N = p['N']
        for b in range(p['B']):
            cons.append(cp.sum_squares(self.W[b * N:(b + 1) * N, :]) <= p['Pmax'])
        self.par = []
        for (k, j, I) in d.pairs:
            g = cp.Parameter(BN, complex=True)
            qv = cp.Parameter(BN, complex=True)
            c1 = cp.Parameter(); c2 = cp.Parameter()
            cj = d.col[j]; icols = [d.col[i] for i in I]
            lhs = cp.exp(LN2 * self.r[cj] + c1)
            if icols:
                lhs = lhs + cp.sum_squares(qv @ self.W[:, icols])
            cons.append(lhs <= c2 + 2 * cp.real(g @ self.W[:, cj]))
            self.par.append((g, qv, c1, c2))
        # rate budgets / QoS
        if 0 in d.col:
            cons.append(cp.sum(self.ac) <= self.r[d.col[0]])
        else:
            cons.append(self.ac == 0)
        for g in d.served_groups:
            idx = [gpos[k] for k in d.carriers[g]]
            cons.append(cp.sum(self.ag[idx]) <= self.r[d.col[1 + g]])
        for k in range(K):
            tot = self.ac[k] + self.xi[k]
            if k in gpos: tot = tot + self.ag[gpos[k]]
            if (1 + d.G + k) in d.col: tot = tot + self.r[d.col[1 + d.G + k]]
            cons.append(tot >= p['Rth'])
        obj = cp.Maximize(cp.sum(self.r) - lamQ * cp.sum(self.xi))
        self.prob = cp.Problem(obj, cons)

    def set_aux(self, H, W):
        Y = H @ W; d = self.d
        for (k, j, I), (g, qv, c1, c2) in zip(d.pairs, self.par):
            cj = d.col[j]; icols = [d.col[i] for i in I]
            Iint = 1.0 + np.sum(np.abs(Y[k, icols]) ** 2)
            mu = Y[k, cj] / Iint
            rho = 1.0 + abs(Y[k, cj]) ** 2 / Iint
            # conj(mu) h_k w_j  ->  row vector acting on W
            g.value = (np.conj(mu) / rho) * H[k]
            qv.value = (abs(mu) / math.sqrt(rho)) * H[k]
            c1.value = -math.log(rho)
            c2.value = (1.0 - abs(mu) ** 2) / rho

    def solve(self, H, W):
        self.set_aux(H, W)
        self.prob.solve(solver=cp.CLARABEL)
        if self.prob.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError('QT precoder: ' + str(self.prob.status))
        return np.asarray(self.W.value)


class QTCoefficient:
    """Reduced form of the MATLAB coefficient block. The relaxed coefficients enter only through the
    cascade scalars c_kbl = theta^T e_kbl, so the proximal x-space problem is solved exactly over
    u = (c - c0)/sqrt(M) with the Gram-weighted proximal term eta*(c-c0)^H S^{-1} (c-c0).
    Received amplitudes are auxiliary variables Y[k,:] = Y0[k,:] + u_k^T PHs[k] (one parameter
    matrix per UE), and every rate constraint uses scalar parameters only (small DPP tensors)."""

    def __init__(self, d, p, lamQ):
        self.d, self.p, self.lamQ = d, p, lamQ
        K, B, L = p['K'], p['B'], p['L']
        nS = d.nS; self.nc = K * B * L; BL = B * L
        self.u = cp.Variable(self.nc, complex=True)
        self.Y = cp.Variable((K, nS), complex=True)
        self.r = cp.Variable(nS, nonneg=True)
        self.xi = cp.Variable(K, nonneg=True)
        self.ac = cp.Variable(K, nonneg=True)
        grouped = [k for g in d.served_groups for k in d.carriers[g]]
        self.ag = cp.Variable(len(grouped), nonneg=True) if grouped else None
        gpos = {k: i for i, k in enumerate(grouped)}
        self.Lprox = cp.Parameter((self.nc, self.nc), complex=True)
        self.PH = [cp.Parameter((BL, nS), complex=True) for _ in range(K)]
        self.Y0 = cp.Parameter((K, nS), complex=True)
        cons = [self.Y[k, :] == self.Y0[k, :] + self.u[k * BL:(k + 1) * BL] @ self.PH[k] for k in range(K)]
        self.par = []
        for (k, j, I) in d.pairs:
            cj = d.col[j]; icols = [d.col[i] for i in I]
            gm = cp.Parameter(complex=True)
            sc = cp.Parameter(nonneg=True)
            c2 = cp.Parameter(); c1 = cp.Parameter()
            lhs = cp.exp(LN2 * self.r[cj] + c1)
            if icols:
                lhs = lhs + cp.sum_squares(sc * self.Y[k, icols])
            cons.append(lhs <= c2 + 2 * cp.real(gm * self.Y[k, cj]))
            self.par.append((gm, sc, c2, c1))
        if 0 in d.col:
            cons.append(cp.sum(self.ac) <= self.r[d.col[0]])
        else:
            cons.append(self.ac == 0)
        for g in d.served_groups:
            idx = [gpos[k] for k in d.carriers[g]]
            cons.append(cp.sum(self.ag[idx]) <= self.r[d.col[1 + g]])
        for k in range(K):
            tot = self.ac[k] + self.xi[k]
            if k in gpos: tot = tot + self.ag[gpos[k]]
            if (1 + d.G + k) in d.col: tot = tot + self.r[d.col[1 + d.G + k]]
            cons.append(tot >= p['Rth'])
        obj = cp.Maximize(cp.sum(self.r) - lamQ * cp.sum(self.xi) - cp.sum_squares(self.Lprox @ self.u))
        self.prob = cp.Problem(obj, cons)

    @staticmethod
    def phi(ch, W, p):
        """phi[k, i, b*L+l] = s_kbl aBS_bl^T w_{i,b}."""
        K, B, L = ch['s'].shape; N = p['N']
        out = np.zeros((K, W.shape[1], B * L), complex)
        for b in range(B):
            aw = np.einsum('ln,ni->li', ch['aBS'][b], W[b * N:(b + 1) * N])
            out[:, :, b * L:(b + 1) * L] = np.einsum('kl,li->kil', ch['s'][:, b, :], aw)
        return out

    def solve(self, Hd, W, ch, crel, c0, Lprox):
        p, d = self.p, self.d
        K, B, L = ch['s'].shape; BL = B * L; sM = math.sqrt(p['M'])
        PH = self.phi(ch, W, p)
        H0 = Hd @ W
        Y0 = H0 + np.einsum('kl,kil->ki', c0.reshape(K, BL), PH)
        Y = H0 + np.einsum('kl,kil->ki', crel.reshape(K, BL), PH)
        for k in range(K):
            self.PH[k].value = sM * PH[k].T
        self.Y0.value = Y0
        for (k, j, I), (gm, sc, c2, c1) in zip(d.pairs, self.par):
            cj = d.col[j]; icols = [d.col[i] for i in I]
            Iint = 1.0 + np.sum(np.abs(Y[k, icols]) ** 2)
            mu = Y[k, cj] / Iint; rho = 1.0 + abs(Y[k, cj]) ** 2 / Iint
            gm.value = np.conj(mu) / rho
            sc.value = abs(mu) / math.sqrt(rho)
            c2.value = (1.0 - abs(mu) ** 2) / rho
            c1.value = -math.log(rho)
        self.Lprox.value = Lprox
        self.prob.solve(solver=cp.CLARABEL)
        if self.prob.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError('QT coefficient: ' + str(self.prob.status))
        return c0 + sM * np.asarray(self.u.value)


def cascade_map(ch, p):
    """Linear map theta (M,4,L) -> c (K*B*L), as dense per-surface blocks.
    Returns list over l of (rows (K*B), cols (4M)) matrices T_l with c_{k,b,l} = T_l[k*B+b] . vec(theta[:,:,l])."""
    K, B, L, M = p['K'], p['B'], p['L'], p['M']
    Ts = []
    for l in range(L):
        T = np.zeros((K * B, 4 * M), complex)
        for k in range(K):
            for b in range(B):
                o = ch['mode'][l, b, k]
                T[k * B + b, o * M:(o + 1) * M] = ch['e'][k, b, l]
        Ts.append(T)
    return Ts


def c_order_index(p):
    """Index mapping between (k,b,l) flat order used in QTCoefficient (k*B*L + b*L + l)
    and per-surface blocks (l, k*B+b)."""
    K, B, L = p['K'], p['B'], p['L']
    idx = np.zeros((L, K * B), int)
    for l in range(L):
        for k in range(K):
            for b in range(B):
                idx[l, k * B + b] = k * B * L + b * L + l
    return idx


# --------------------------------------------------------------------------- AO (benchmark and refinement)
def _psd_factor(A):
    """Cholesky factor of a Hermitian positive definite matrix; if rounding makes it fail (nearly singular
    cascade Gram matrix S), a factor from the eigendecomposition with eigenvalues floored at 1e-12 of the largest."""
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh((A + A.conj().T) / 2)
        w = np.maximum(w, 1e-12 * max(float(w.max()), 1e-300))
        return U * np.sqrt(w)


def ao(theta, W, ch, net, d, p, lamQ, max_iter=20, prec_steps=3, coef_steps=3, tol=0.1, stable_req=2,
       pen0=0.2, pen_growth=1.8, qtp=None, qtc=None, log=None, time_budget=None, take_all=False):
    """QT alternating optimisation. Every block update is kept only if the merit does not
    decrease (the precoder step is monotone by construction)."""
    qtp = qtp or QTPrecoderDirect(d, p, lamQ)
    qtc = qtc or QTCoefficientDirect(d, p, lamQ)
    Ts = cascade_map(ch, p); idx = c_order_index(p)
    H = effective_channel(theta, ch, net, p)
    cur, rate, df = merit(W, H, d, p, lamQ)
    penalty = pen0; stable = 0; calls = 0; it = 0
    hist = [cur]; best = (cur, rate, df, theta, W, H)
    failed = False
    for it in range(1, max_iter + 1):
        old = cur
        for _ in range(prec_steps):
            try:
                Wn = qtp.solve(H, W); calls += 1
            except Exception:
                calls += 1; failed = True; break
            mn = merit(Wn, H, d, p, lamQ)
            if take_all or mn[0] >= cur - 1e-9:
                W, (cur, rate, df) = Wn, mn
                if cur > best[0]: best = (cur, rate, df, theta, W, H)
        if failed: break
        # coefficient block
        cvec = np.zeros(p['K'] * p['B'] * p['L'], complex)
        for l in range(p['L']):
            cvec[idx[l]] = Ts[l] @ theta[:, :, l].reshape(-1, order='F')
        S_inv_blocks = []
        Lp = np.zeros((cvec.size, cvec.size), complex)
        for l in range(p['L']):
            S = Ts[l] @ Ts[l].conj().T
            Si = np.linalg.inv(S + 1e-12 * np.trace(S).real / S.shape[0] * np.eye(S.shape[0]))
            S_inv_blocks.append(Si)
            Lc = _psd_factor(penalty * p['M'] * Si + 0j)   # Lc Lc^H = eta M S^-1
            ii = idx[l]
            Lp[np.ix_(ii, ii)] = Lc.conj().T               # ||Lc^H v||^2 = v^H eta S^-1 v
        crel = cvec.copy()
        for _ in range(coef_steps):
            try:
                crel = qtc.solve(net['Hd'], W, ch, crel, cvec, Lp); calls += 1
            except Exception:
                calls += 1; failed = True; break
        if failed: break
        # recover relaxed theta: theta + T^H S^-1 (c - c0)
        relaxed = theta.copy()
        for l in range(p['L']):
            dc = crel[idx[l]] - cvec[idx[l]]
            dx = Ts[l].conj().T @ (S_inv_blocks[l] @ dc)
            relaxed[:, :, l] = relaxed[:, :, l] + dx.reshape(p['M'], 4, order='F')
        proj = project_coupled(relaxed)
        residual = float(np.sum(np.abs(relaxed - proj) ** 2))
        Hn = effective_channel(proj, ch, net, p)
        mn = merit(W, Hn, d, p, lamQ)
        kept = take_all or mn[0] >= cur - 1e-9
        if kept:
            theta, H, (cur, rate, df) = proj, Hn, mn
            if cur > best[0]: best = (cur, rate, df, theta, W, H)
        if residual > 1e-3:
            penalty *= pen_growth
        hist.append(cur)
        if log: log(f'    AO {it:2d}: merit {cur:.4f} rate {rate:.4f} deficit {df:.3g} proj {residual:.3g} coef kept {kept} (coef merit {mn[0]:.4f})')
        if abs(cur - old) < tol: stable += 1
        else: stable = 0
        if stable >= stable_req: break
        if time_budget is not None and time_budget(): break
    cur, rate, df, theta, W, H = best
    return dict(theta=theta, W=W, H=H, merit=cur, rate=rate, deficit=df, iterations=it, calls=calls, hist=hist, failed=failed)


class QTCoefficientHull:
    """Coefficient block over the convex hull of the coupled hardware set. With u = t + r and
    v = t - r per face, the coupled constraints are |u_m| = |v_m| = 1; the QT subproblem is solved
    over |u_m| <= 1, |v_m| <= 1 and then projected (u/|u|, v/|v|)."""

    def __init__(self, d, p, lamQ):
        self.d, self.p, self.lamQ = d, p, lamQ
        self.prob = None

    def set_geometry(self, ch):
        """(Re)build the problem for a pose: E_l maps [uA vA uB vB] to c_{k,b,l} with
        theta_AT=(uA+vA)/2, AR=(uA-vA)/2, BT=(uB+vB)/2, BR=(uB-vB)/2."""
        d, p, lamQ = self.d, self.p, self.lamQ
        K, B, L, M = p['K'], p['B'], p['L'], p['M']
        nS = d.nS; BL = B * L
        sgn = {0: (0, 1, 1), 1: (0, 1, -1), 2: (2, 3, 1), 3: (2, 3, -1)}
        self.x = [cp.Variable(4 * M, complex=True) for _ in range(L)]
        self.cvar = cp.Variable(K * B * L, complex=True)
        self.r = cp.Variable(nS, nonneg=True)
        self.xi = cp.Variable(K, nonneg=True)
        self.ac = cp.Variable(K, nonneg=True)
        grouped = [k for g in d.served_groups for k in d.carriers[g]]
        self.ag = cp.Variable(len(grouped), nonneg=True) if grouped else None
        gpos = {k: i for i, k in enumerate(grouped)}
        cons = []
        idx = c_order_index(p)
        for l in range(L):
            E = np.zeros((K * B, 4 * M), complex)
            for k in range(K):
                for b in range(B):
                    o = ch['mode'][l, b, k]; ub, vb, sv = sgn[o]
                    e = ch['e'][k, b, l] / 2
                    E[k * B + b, ub * M:(ub + 1) * M] = e
                    E[k * B + b, vb * M:(vb + 1) * M] = sv * e
            cons.append(cp.abs(self.x[l]) <= 1)
            cons.append(self.cvar[idx[l]] == E @ self.x[l])
        self.par = []
        for (k, j, I) in d.pairs:
            cj = d.col[j]; icols = [d.col[i] for i in I]
            ck = self.cvar[k * BL:(k + 1) * BL]
            gph = cp.Parameter(BL, complex=True); c2 = cp.Parameter(); c1 = cp.Parameter()
            if icols:
                Qp = cp.Parameter((BL, len(icols)), complex=True); q0 = cp.Parameter(len(icols), complex=True)
                lhs = cp.exp(LN2 * self.r[cj] + c1) + cp.sum_squares(ck @ Qp + q0)
            else:
                Qp = q0 = None; lhs = cp.exp(LN2 * self.r[cj] + c1)
            cons.append(lhs <= c2 + 2 * cp.real(ck @ gph))
            self.par.append((gph, c2, c1, Qp, q0))
        if 0 in d.col:
            cons.append(cp.sum(self.ac) <= self.r[d.col[0]])
        else:
            cons.append(self.ac == 0)
        for g in d.served_groups:
            ii = [gpos[k] for k in d.carriers[g]]
            cons.append(cp.sum(self.ag[ii]) <= self.r[d.col[1 + g]])
        for k in range(K):
            tot = self.ac[k] + self.xi[k]
            if k in gpos: tot = tot + self.ag[gpos[k]]
            if (1 + d.G + k) in d.col: tot = tot + self.r[d.col[1 + d.G + k]]
            cons.append(tot >= p['Rth'])
        self.prob = cp.Problem(cp.Maximize(cp.sum(self.r) - lamQ * cp.sum(self.xi)), cons)

    @staticmethod
    def theta_to_x(theta):
        L = theta.shape[2]
        return [np.concatenate([theta[:, 0, l] + theta[:, 1, l], theta[:, 0, l] - theta[:, 1, l],
                                theta[:, 2, l] + theta[:, 3, l], theta[:, 2, l] - theta[:, 3, l]]) for l in range(L)]

    @staticmethod
    def x_to_theta(xs, M):
        L = len(xs); th = np.zeros((M, 4, L), complex)
        for l, x in enumerate(xs):
            uA, vA, uB, vB = x[:M], x[M:2 * M], x[2 * M:3 * M], x[3 * M:]
            th[:, 0, l] = (uA + vA) / 2; th[:, 1, l] = (uA - vA) / 2
            th[:, 2, l] = (uB + vB) / 2; th[:, 3, l] = (uB - vB) / 2
        return th

    def solve(self, Hd, W, ch, theta_rel):
        p, d = self.p, self.d
        K, B, L = ch['s'].shape; BL = B * L
        PH = QTCoefficient.phi(ch, W, p)
        H0 = Hd @ W
        crel = np.zeros(K * B * L, complex); idx = c_order_index(p)
        cs = cascade_scalars(theta_rel, ch)  # (K,B,L)
        for k in range(K):
            for b in range(B):
                for l in range(L):
                    crel[k * BL + b * L + l] = cs[k, b, l]
        Y = H0 + np.einsum('kl,kil->ki', crel.reshape(K, BL), PH)
        for (k, j, I), (gph, c2, c1, Qp, q0) in zip(d.pairs, self.par):
            cj = d.col[j]; icols = [d.col[i] for i in I]
            Iint = 1.0 + np.sum(np.abs(Y[k, icols]) ** 2)
            mu = Y[k, cj] / Iint; rho = 1.0 + abs(Y[k, cj]) ** 2 / Iint
            gph.value = (np.conj(mu) / rho) * PH[k, cj]
            c2.value = (1.0 - abs(mu) ** 2) / rho + 2 * np.real(np.conj(mu) / rho * H0[k, cj])
            c1.value = -math.log(rho)
            if Qp is not None:
                sc = abs(mu) / math.sqrt(rho)
                Qp.value = sc * PH[k][icols].T; q0.value = sc * H0[k, icols]
        self.prob.solve(solver=cp.CLARABEL)
        if self.prob.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError('QT hull coefficient: ' + str(self.prob.status))
        return self.x_to_theta([np.asarray(x.value) for x in self.x], p['M'])


def ao_hull(theta, W, ch, net, d, p, lamQ, max_iter=20, coef_steps=1, tol=0.1, stable_req=2, qtp=None, qtc=None,
            log=None):
    """AO: one QT precoder step, then QT coefficient step(s) over the convex hull followed by the
    closed-form projection; each block update is kept if the merit does not decrease."""
    qtp = qtp or QTPrecoder(d, p, lamQ)
    qtc = qtc or QTCoefficientHull(d, p, lamQ)
    qtc.set_geometry(ch)
    H = effective_channel(theta, ch, net, p)
    cur, rate, df = merit(W, H, d, p, lamQ)
    stable = 0; calls = 0; hist = [cur]; failed = False; it = 0
    for it in range(1, max_iter + 1):
        old = cur
        try:
            Wn = qtp.solve(H, W); calls += 1
            mn = merit(Wn, H, d, p, lamQ)
            if mn[0] >= cur - 1e-9: W, (cur, rate, df) = Wn, mn
            rel = theta
            for _ in range(coef_steps):
                rel = qtc.solve(net['Hd'], W, ch, rel); calls += 1
        except Exception:
            calls += 1; failed = True; break
        proj = project_coupled(rel)
        Hn = effective_channel(proj, ch, net, p)
        mn = merit(W, Hn, d, p, lamQ)
        mrel = merit(W, effective_channel(rel, ch, net, p), d, p, lamQ)[0]
        acc = mn[0] >= cur - 1e-9
        if acc: theta, H, (cur, rate, df) = proj, Hn, mn
        hist.append(cur)
        if log: log(f'    AOh {it:2d}: merit {cur:.4f} (relaxed {mrel:.3f}, projected {mn[0]:.3f}, kept={acc})')
        if abs(cur - old) < tol: stable += 1
        else: stable = 0
        if stable >= stable_req: break
    cur, rate, df, theta, W, H = best
    return dict(theta=theta, W=W, H=H, merit=cur, rate=rate, deficit=df, iterations=it, calls=calls, hist=hist, failed=failed)


# --------------------------------------------------------------------------- direct Clarabel coefficient block
import clarabel
import scipy.sparse as sp


class QTCoefficientDirect(QTCoefficient):
    """Same problem as QTCoefficient, assembled directly for Clarabel (no cvxpy compile)."""

    def __init__(self, d, p, lamQ):
        self.d, self.p, self.lamQ = d, p, lamQ
        K, B, L = p['K'], p['B'], p['L']
        self.nc = K * B * L
        grouped = [k for g in d.served_groups for k in d.carriers[g]]
        self.gpos = {k: i for i, k in enumerate(grouped)}
        nS = d.nS; nG = len(grouped); nP = len(d.pairs)
        # variable layout
        o = 0
        self.iur = o; o += self.nc
        self.iui = o; o += self.nc
        self.ir = o; o += nS
        self.ixi = o; o += K
        self.iac = o; o += K
        self.iag = o; o += nG
        self.it = o; o += nP
        self.itau = o; o += 1
        self.n = o

    def solve(self, Hd, W, ch, crel, c0, Lprox):
        p, d = self.p, self.d
        K, B, L = ch['s'].shape; BL = B * L; sM = math.sqrt(p['M']); nc = self.nc
        PH = self.phi(ch, W, p) * sM                 # (K,nS,BL), u-scaled
        H0 = Hd @ W
        Y0 = H0 + np.einsum('kl,kil->ki', c0.reshape(K, BL), PH / sM)
        Y = H0 + np.einsum('kl,kil->ki', crel.reshape(K, BL), PH / sM)
        n = self.n
        rows = []; vals_b = []; cones = []
        Arows = []  # list of (dict col->val) rows ; build with lil
        A = sp.lil_matrix((0, n))
        blocks_A = []; blocks_b = []

        def yrow(k, i):
            """Return (row_re, row_im, const_re, const_im) as dense arrays over x for Y[k,i]."""
            re = np.zeros(n); im = np.zeros(n)
            ph = PH[k, i]  # (BL,)
            cols = np.arange(k * BL, (k + 1) * BL)
            re[self.iur + cols] = ph.real; re[self.iui + cols] = -ph.imag
            im[self.iur + cols] = ph.imag; im[self.iui + cols] = ph.real
            return re, im, Y0[k, i].real, Y0[k, i].imag

        Ablocks = []; bblocks = []
        for pi, (k, j, I) in enumerate(d.pairs):
            cj = d.col[j]; icols = [d.col[i] for i in I]
            Iint = 1.0 + np.sum(np.abs(Y[k, icols]) ** 2)
            mu = Y[k, cj] / Iint; rho = 1.0 + abs(Y[k, cj]) ** 2 / Iint
            gm = np.conj(mu) / rho; sc = abs(mu) / math.sqrt(rho)
            c2 = (1.0 - abs(mu) ** 2) / rho; c1 = -math.log(rho)
            # exp cone (LN2 r + c1, 1, t) : s = b - A x
            Ae = np.zeros((3, n)); be = np.zeros(3)
            Ae[0, self.ir + cj] = -LN2; be[0] = c1
            be[1] = 1.0
            Ae[2, self.it + pi] = -1.0
            Ablocks.append(Ae); bblocks.append(be); cones.append(clarabel.ExponentialConeT())
            # s_p = c2 - t + 2(Re(gm)Re(Y) - Im(gm)Im(Y))   (affine: a.x + a0)
            yre, yim, y0r, y0i = yrow(k, cj)
            a = 2 * (gm.real * yre - gm.imag * yim); a[self.it + pi] -= 1.0
            a0 = c2 + 2 * (gm.real * y0r - gm.imag * y0i)
            if icols:
                m = 2 + 2 * len(icols)
                As = np.zeros((m, n)); bs = np.zeros(m)
                # (s+1)/2 >= ||[(s-1)/2, sc*Re Y_I, sc*Im Y_I]||
                As[0] = -a / 2; bs[0] = (a0 + 1) / 2
                As[1] = -a / 2; bs[1] = (a0 - 1) / 2
                for q, i in enumerate(icols):
                    re, im, cr, ci = yrow(k, i)
                    As[2 + q] = -sc * re; bs[2 + q] = sc * cr
                    As[2 + len(icols) + q] = -sc * im; bs[2 + len(icols) + q] = sc * ci
                Ablocks.append(As); bblocks.append(bs); cones.append(clarabel.SecondOrderConeT(m))
            else:
                Ablocks.append(-a[None, :]); bblocks.append(np.array([a0])); cones.append(clarabel.NonnegativeConeT(1))
        # linear inequalities (nonneg cone: b - Ax >= 0)
        lin = []; linb = []
        c_on = 0 in d.col
        row = np.zeros(n)
        if c_on:
            row[self.iac:self.iac + K] = 1.0; row[self.ir + d.col[0]] = -1.0; lin.append(row); linb.append(0.0)
        for g in d.served_groups:
            row = np.zeros(n)
            for k in d.carriers[g]: row[self.iag + self.gpos[k]] = 1.0
            row[self.ir + d.col[1 + g]] = -1.0; lin.append(row); linb.append(0.0)
        for k in range(K):
            row = np.zeros(n)
            row[self.iac + k] = -1.0; row[self.ixi + k] = -1.0
            if k in self.gpos: row[self.iag + self.gpos[k]] = -1.0
            if (1 + d.G + k) in d.col: row[self.ir + d.col[1 + d.G + k]] = -1.0
            lin.append(row); linb.append(-p['Rth'])
        nn = self.n - self.ir - len(d.pairs) - 1   # r, xi, ac, ag nonneg
        for q in range(nn):
            row = np.zeros(n); row[self.ir + q] = -1.0; lin.append(row); linb.append(0.0)
        if not c_on:
            for k in range(K):
                row = np.zeros(n); row[self.iac + k] = 1.0; lin.append(row); linb.append(0.0)
        Ablocks.insert(0, np.array(lin)); bblocks.insert(0, np.array(linb)); cones.insert(0, clarabel.NonnegativeConeT(len(lin)))
        # proximal: (tau+1)/2 >= ||[(tau-1)/2, Re(Lp u), Im(Lp u)]||
        Lr, Li = Lprox.real, Lprox.imag
        m = 2 + 2 * nc
        Ap = np.zeros((m, n)); bp = np.zeros(m)
        Ap[0, self.itau] = -0.5; bp[0] = 0.5
        Ap[1, self.itau] = -0.5; bp[1] = -0.5
        Ap[2:2 + nc, self.iur:self.iur + nc] = -Lr; Ap[2:2 + nc, self.iui:self.iui + nc] = Li
        Ap[2 + nc:, self.iur:self.iur + nc] = -Li; Ap[2 + nc:, self.iui:self.iui + nc] = -Lr
        Ablocks.append(Ap); bblocks.append(bp); cones.append(clarabel.SecondOrderConeT(m))
        Afull = sp.csc_matrix(np.vstack(Ablocks)); bfull = np.concatenate(bblocks)
        q = np.zeros(n); q[self.ir:self.ir + d.nS] = -1.0; q[self.ixi:self.ixi + K] = self.lamQ; q[self.itau] = 1.0
        P = sp.csc_matrix((n, n))
        st = clarabel.DefaultSettings(); st.verbose = False; st.max_threads = 1
        solver = clarabel.DefaultSolver(P, q, Afull, bfull, cones, st)
        sol = solver.solve()
        status = str(sol.status)
        if 'Solved' not in status:
            raise RuntimeError('QT coefficient (direct): ' + status)
        x = np.array(sol.x)
        u = x[self.iur:self.iur + nc] + 1j * x[self.iui:self.iui + nc]
        return c0 + sM * u


class QTPrecoderDirect:
    """QTPrecoder assembled directly for Clarabel (identical problem)."""

    def __init__(self, d, p, lamQ):
        self.d, self.p, self.lamQ = d, p, lamQ
        K = p['K']; BN = p['BN']; nS = d.nS
        grouped = [k for g in d.served_groups for k in d.carriers[g]]
        self.gpos = {k: i for i, k in enumerate(grouped)}
        nG = len(grouped); nP = len(d.pairs)
        o = 0
        self.iwr = o; o += BN * nS          # column-major: W[:, i] at iwr + i*BN
        self.iwi = o; o += BN * nS
        self.ir = o; o += nS
        self.ixi = o; o += K
        self.iac = o; o += K
        self.iag = o; o += nG
        self.it = o; o += nP
        self.n = o
        self.settings = clarabel.DefaultSettings(); self.settings.verbose = False; self.settings.max_threads = 1

    def _yrow(self, H, k, i):
        n, BN = self.n, self.p['BN']
        re = np.zeros(n); im = np.zeros(n)
        h = H[k]; c = i * BN
        re[self.iwr + c:self.iwr + c + BN] = h.real; re[self.iwi + c:self.iwi + c + BN] = -h.imag
        im[self.iwr + c:self.iwr + c + BN] = h.imag; im[self.iwi + c:self.iwi + c + BN] = h.real
        return re, im

    def solve(self, H, W):
        p, d = self.p, self.d
        K, BN, N = p['K'], p['BN'], p['N']; n = self.n
        Y = H @ W
        Ab = []; bb = []; cones = []
        lin = []; linb = []
        c_on = 0 in d.col
        if c_on:
            row = np.zeros(n); row[self.iac:self.iac + K] = 1.0; row[self.ir + d.col[0]] = -1.0; lin.append(row); linb.append(0.0)
        for g in d.served_groups:
            row = np.zeros(n)
            for k in d.carriers[g]: row[self.iag + self.gpos[k]] = 1.0
            row[self.ir + d.col[1 + g]] = -1.0; lin.append(row); linb.append(0.0)
        for k in range(K):
            row = np.zeros(n); row[self.iac + k] = -1.0; row[self.ixi + k] = -1.0
            if k in self.gpos: row[self.iag + self.gpos[k]] = -1.0
            if (1 + d.G + k) in d.col: row[self.ir + d.col[1 + d.G + k]] = -1.0
            lin.append(row); linb.append(-p['Rth'])
        for q in range(self.it - self.ir):
            row = np.zeros(n); row[self.ir + q] = -1.0; lin.append(row); linb.append(0.0)
        if not c_on:
            for k in range(K):
                row = np.zeros(n); row[self.iac + k] = 1.0; lin.append(row); linb.append(0.0)
        Ab.append(np.array(lin)); bb.append(np.array(linb)); cones.append(clarabel.NonnegativeConeT(len(lin)))
        # per-BS power: sqrt(Pmax) >= ||[Re, Im of W_b]||
        for b in range(p['B']):
            idx = [i * BN + b * N + m for i in range(d.nS) for m in range(N)]
            m_ = 1 + 2 * len(idx)
            Apw = np.zeros((m_, n)); bpw = np.zeros(m_); bpw[0] = math.sqrt(p['Pmax'])
            for q, ii in enumerate(idx):
                Apw[1 + q, self.iwr + ii] = -1.0; Apw[1 + len(idx) + q, self.iwi + ii] = -1.0
            Ab.append(Apw); bb.append(bpw); cones.append(clarabel.SecondOrderConeT(m_))
        for pi, (k, j, I) in enumerate(d.pairs):
            cj = d.col[j]; icols = [d.col[i] for i in I]
            Iint = 1.0 + np.sum(np.abs(Y[k, icols]) ** 2)
            mu = Y[k, cj] / Iint; rho = 1.0 + abs(Y[k, cj]) ** 2 / Iint
            gm = np.conj(mu) / rho; sc = abs(mu) / math.sqrt(rho)
            c2 = (1.0 - abs(mu) ** 2) / rho; c1 = -math.log(rho)
            Ae = np.zeros((3, n)); be = np.zeros(3)
            Ae[0, self.ir + cj] = -LN2; be[0] = c1; be[1] = 1.0; Ae[2, self.it + pi] = -1.0
            Ab.append(Ae); bb.append(be); cones.append(clarabel.ExponentialConeT())
            yre, yim = self._yrow(H, k, cj)
            a = 2 * (gm.real * yre - gm.imag * yim); a[self.it + pi] -= 1.0
            a0 = c2
            if icols:
                m_ = 2 + 2 * len(icols)
                As = np.zeros((m_, n)); bs = np.zeros(m_)
                As[0] = -a / 2; bs[0] = (a0 + 1) / 2
                As[1] = -a / 2; bs[1] = (a0 - 1) / 2
                for q, i in enumerate(icols):
                    re, im = self._yrow(H, k, i)
                    As[2 + q] = -sc * re; As[2 + len(icols) + q] = -sc * im
                Ab.append(As); bb.append(bs); cones.append(clarabel.SecondOrderConeT(m_))
            else:
                Ab.append(-a[None, :]); bb.append(np.array([a0])); cones.append(clarabel.NonnegativeConeT(1))
        if d.bsmask is not None:
            # w_{j,b} = 0 for the (stream, BS) pairs the BS does not carry (zero cone: A x = b)
            off = [i * BN + b * N + m for i in range(d.nS) for b in range(p['B']) if not d.bsmask[b, i] for m in range(N)]
            if off:
                Az = np.zeros((2 * len(off), n))
                for q_, ii in enumerate(off):
                    Az[q_, self.iwr + ii] = 1.0; Az[len(off) + q_, self.iwi + ii] = 1.0
                Ab.append(Az); bb.append(np.zeros(2 * len(off))); cones.append(clarabel.ZeroConeT(2 * len(off)))
        Afull = sp.csc_matrix(np.vstack(Ab)); bfull = np.concatenate(bb)
        q = np.zeros(n); q[self.ir:self.ir + d.nS] = -1.0; q[self.ixi:self.ixi + K] = self.lamQ
        solver = clarabel.DefaultSolver(sp.csc_matrix((n, n)), q, Afull, bfull, cones, self.settings)
        sol = solver.solve()
        status = str(sol.status)
        if 'Solved' not in status:
            raise RuntimeError('QT precoder (direct): ' + status)
        x = np.array(sol.x)
        Wn = (x[self.iwr:self.iwr + BN * d.nS] + 1j * x[self.iwi:self.iwi + BN * d.nS]).reshape(d.nS, BN).T
        return Wn


class QTCoefficientX:
    """The MATLAB coefficient block in its original form: x in C^{4LM} (all modes of all surfaces),
    Y(k,:) = Hd(k,:) W + x^T D_k, proximal penalty eta*||x - anchor||^2, assembled for Clarabel.
    solve() has the same signature/return (cascade scalars) as QTCoefficient via theta mapping."""

    def __init__(self, d, p, lamQ):
        self.d, self.p, self.lamQ = d, p, lamQ
        K, L, M = p['K'], p['L'], p['M']
        grouped = [k for g in d.served_groups for k in d.carriers[g]]
        self.gpos = {k: i for i, k in enumerate(grouped)}
        nS = d.nS; nG = len(grouped); nP = len(d.pairs); nx = 4 * L * M
        o = 0
        self.ixr = o; o += nx
        self.ixi = o; o += nx
        self.iyr = o; o += K * nS
        self.iyi = o; o += K * nS
        self.ir = o; o += nS
        self.ixi_ = o; o += K
        self.iac = o; o += K
        self.iag = o; o += nG
        self.it = o; o += nP
        self.n = o; self.nx = nx
        self.settings = clarabel.DefaultSettings(); self.settings.verbose = False; self.settings.max_threads = 1

    def solve_theta(self, Hd, W, ch, theta_rel, theta_anchor, eta):
        p, d = self.p, self.d
        K, B, L, M, N = p['K'], p['B'], p['L'], p['M'], p['N']
        nS = d.nS; n = self.n; nx = self.nx
        # D_k(:, i) = sum_{l,b} [mode block] s e (aBS^T w_{b,i})
        rowsD = []; colsD = []; valsD = []   # equality rows for Y
        H0 = Hd @ W
        Drows = {}
        for k in range(K):
            for i in range(nS):
                vec = {}
                for l in range(L):
                    for b in range(B):
                        o_ = ch['mode'][l, b, k]
                        a = ch['s'][k, b, l] * (ch['aBS'][b, l] @ W[b * N:(b + 1) * N, i])
                        base = l * 4 * M + o_ * M
                        cur = vec.get(base)
                        vec[base] = (a * ch['e'][k, b, l]) if cur is None else cur + a * ch['e'][k, b, l]
                Drows[(k, i)] = vec
        # relaxed responses for the auxiliaries
        xrel = theta_rel.transpose(2, 1, 0).reshape(-1)       # index l*4M + o*M + m
        xa = theta_anchor.transpose(2, 1, 0).reshape(-1)
        Y = np.zeros((K, nS), complex)
        for (k, i), vec in Drows.items():
            Y[k, i] = H0[k, i] + sum(xrel[bs:bs + M] @ v for bs, v in vec.items())
        # equality constraints: Yre - Re(x^T D) = Re(H0), Yim - Im(x^T D) = Im(H0)
        er = []; ec = []; ev = []; eb = []
        row = 0
        for (k, i), vec in Drows.items():
            yi = k * nS + i
            # real row
            er.append(row); ec.append(self.iyr + yi); ev.append(1.0)
            for bs, v in vec.items():
                idx = np.arange(bs, bs + M)
                er += [row] * M; ec += list(self.ixr + idx); ev += list(-v.real)
                er += [row] * M; ec += list(self.ixi + idx); ev += list(v.imag)
            eb.append(H0[k, i].real); row += 1
            er.append(row); ec.append(self.iyi + yi); ev.append(1.0)
            for bs, v in vec.items():
                idx = np.arange(bs, bs + M)
                er += [row] * M; ec += list(self.ixr + idx); ev += list(-v.imag)
                er += [row] * M; ec += list(self.ixi + idx); ev += list(-v.real)
            eb.append(H0[k, i].imag); row += 1
        Aeq = sp.csc_matrix((ev, (er, ec)), shape=(row, n)); beq = np.array(eb)
        blocks = [Aeq]; bvec = [beq]; cones = [clarabel.ZeroConeT(row)]
        # linear rows
        lin = []; linb = []
        c_on = 0 in d.col
        def zr():
            return np.zeros(n)
        if c_on:
            r_ = zr(); r_[self.iac:self.iac + K] = 1.0; r_[self.ir + d.col[0]] = -1.0; lin.append(r_); linb.append(0.0)
        for g in d.served_groups:
            r_ = zr()
            for k in d.carriers[g]: r_[self.iag + self.gpos[k]] = 1.0
            r_[self.ir + d.col[1 + g]] = -1.0; lin.append(r_); linb.append(0.0)
        for k in range(K):
            r_ = zr(); r_[self.iac + k] = -1.0; r_[self.ixi_ + k] = -1.0
            if k in self.gpos: r_[self.iag + self.gpos[k]] = -1.0
            if (1 + d.G + k) in d.col: r_[self.ir + d.col[1 + d.G + k]] = -1.0
            lin.append(r_); linb.append(-p['Rth'])
        for q in range(self.it - self.ir):
            r_ = zr(); r_[self.ir + q] = -1.0; lin.append(r_); linb.append(0.0)
        if not c_on:
            for k in range(K):
                r_ = zr(); r_[self.iac + k] = 1.0; lin.append(r_); linb.append(0.0)
        blocks.append(sp.csc_matrix(np.array(lin))); bvec.append(np.array(linb)); cones.append(clarabel.NonnegativeConeT(len(lin)))
        for pi, (k, j, I) in enumerate(d.pairs):
            cj = d.col[j]; icols = [d.col[i] for i in I]
            Iint = 1.0 + np.sum(np.abs(Y[k, icols]) ** 2)
            mu = Y[k, cj] / Iint; rho = 1.0 + abs(Y[k, cj]) ** 2 / Iint
            gm = np.conj(mu) / rho; sc = abs(mu) / math.sqrt(rho)
            c2 = (1.0 - abs(mu) ** 2) / rho; c1 = -math.log(rho)
            Ae = sp.lil_matrix((3, n)); be = np.zeros(3)
            Ae[0, self.ir + cj] = -LN2; be[0] = c1; be[1] = 1.0; Ae[2, self.it + pi] = -1.0
            blocks.append(Ae.tocsc()); bvec.append(be); cones.append(clarabel.ExponentialConeT())
            yj = k * nS + cj
            a = {self.iyr + yj: 2 * gm.real, self.iyi + yj: -2 * gm.imag, self.it + pi: -1.0}
            if icols:
                m_ = 2 + 2 * len(icols)
                As = sp.lil_matrix((m_, n)); bs_ = np.zeros(m_)
                for c_, v in a.items():
                    As[0, c_] = -v / 2; As[1, c_] = -v / 2
                bs_[0] = (c2 + 1) / 2; bs_[1] = (c2 - 1) / 2
                for q, i in enumerate(icols):
                    yi = k * nS + i
                    As[2 + q, self.iyr + yi] = -sc; As[2 + len(icols) + q, self.iyi + yi] = -sc
                blocks.append(As.tocsc()); bvec.append(bs_); cones.append(clarabel.SecondOrderConeT(m_))
            else:
                As = sp.lil_matrix((1, n))
                for c_, v in a.items(): As[0, c_] = -v
                blocks.append(As.tocsc()); bvec.append(np.array([c2])); cones.append(clarabel.NonnegativeConeT(1))
        Afull = sp.vstack(blocks).tocsc(); bfull = np.concatenate(bvec)
        Pd = np.zeros(n); Pd[self.ixr:self.ixr + nx] = 2 * eta; Pd[self.ixi:self.ixi + nx] = 2 * eta
        P = sp.diags(Pd).tocsc()
        q = np.zeros(n); q[self.ir:self.ir + nS] = -1.0; q[self.ixi_:self.ixi_ + K] = self.lamQ
        q[self.ixr:self.ixr + nx] = -2 * eta * xa.real; q[self.ixi:self.ixi + nx] = -2 * eta * xa.imag
        solver = clarabel.DefaultSolver(sp.triu(P).tocsc(), q, Afull, bfull, cones, self.settings)
        sol = solver.solve()
        if 'Solved' not in str(sol.status):
            raise RuntimeError('QT coefficient (x-space): ' + str(sol.status))
        x = np.array(sol.x); xv = x[self.ixr:self.ixr + nx] + 1j * x[self.ixi:self.ixi + nx]
        return xv.reshape(L, 4, M).transpose(2, 1, 0)


def ao_x(theta, W, ch, net, d, p, lamQ, max_iter=20, prec_steps=3, coef_steps=3, tol=0.1, stable_req=2,
         pen0=0.2, pen_growth=1.8, qtp=None, qtc=None, log=None):
    """ao() with the coefficient block in its original 4LM-variable form (MATLAB hrs_cvx_block)."""
    qtp = qtp or QTPrecoderDirect(d, p, lamQ)
    qtc = qtc or QTCoefficientX(d, p, lamQ)
    H = effective_channel(theta, ch, net, p)
    cur, rate, df = merit(W, H, d, p, lamQ)
    penalty = pen0; stable = 0; calls = 0; it = 0; hist = [cur]; failed = False
    for it in range(1, max_iter + 1):
        old = cur
        for _ in range(prec_steps):
            try:
                Wn = qtp.solve(H, W); calls += 1
            except Exception:
                calls += 1; failed = True; break
            mn = merit(Wn, H, d, p, lamQ)
            if mn[0] >= cur - 1e-9: W, (cur, rate, df) = Wn, mn
        if failed: break
        rel = theta
        for _ in range(coef_steps):
            try:
                rel = qtc.solve_theta(net['Hd'], W, ch, rel, theta, penalty); calls += 1
            except Exception:
                calls += 1; failed = True; break
        if failed: break
        proj = project_coupled(rel); residual = float(np.sum(np.abs(rel - proj) ** 2))
        Hn = effective_channel(proj, ch, net, p); mn = merit(W, Hn, d, p, lamQ)
        if mn[0] >= cur - 1e-9: theta, H, (cur, rate, df) = proj, Hn, mn
        if residual > 1e-3: penalty *= pen_growth
        hist.append(cur)
        if log: log(f'    AOx {it:2d}: merit {cur:.4f} proj {residual:.3g}')
        if abs(cur - old) < tol: stable += 1
        else: stable = 0
        if stable >= stable_req: break
    return dict(theta=theta, W=W, H=H, merit=cur, rate=rate, deficit=df, iterations=it, calls=calls, hist=hist, failed=failed)
