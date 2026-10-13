"""TuRBO-1 deployment search (Eriksson et al., "Scalable global optimization via local Bayesian optimization",
NeurIPS 2019), adapted to the constrained deployment domain.

* Search space: per surface (x, y, z, azimuth), each mapped to [0, 1] over the bounding box of the admissible
  set: [x0 - Dmax, x0 + Dmax] clipped to the region, likewise y, z within [zmin, zmax], azimuth in [-pi/2, pi/2].
  Coordinates with an empty range (Dmax = 0) are dropped, so d = 4L, or L with Dmax = 0. The ball, separation and
  distance constraints are enforced by discarding inadmissible candidates, as in the BO candidate generation.
* Initial design: the same n_init deployments as the BO (the prescribed deployment and random admissible ones).
* Local GP: Matern-5/2 with ARD length scales and a constant mean, fitted to the standardized values by maximizing
  the marginal likelihood with 50 Adam steps (learning rate 0.1) as in the reference implementation; bounds as in
  the paper: length scales [0.005, 2], signal variance [0.05, 20], noise variance [0.0005, 0.1].
* Trust region: hyperrectangle centred at the best evaluated point, side lengths L * lambda_i / (prod lambda)^(1/d),
  clipped to [0, 1]^d. L_init = 0.8, L_min = 2^-7, L_max = 1.6, tau_succ = 3, tau_fail = ceil(d / q) with q = 1.
  A success is an evaluation that improves the best value by more than 1e-3 |best|. When L < L_min the region is
  restarted with L_init and a new initial design (n_init random admissible deployments) and a fresh GP.
* Acquisition: Thompson sampling on min(100 d, 5000) candidates from a scrambled Sobol sequence in the trust
  region; each coordinate is perturbed with probability min(1, 20 / d) (all coordinates here, d <= 20)."""
import math, time, warnings
import numpy as np
from scipy.stats import qmc
import hrs as H_
import algo as A

TURBO = dict(L_init=0.8, L_min=2.0 ** -7, L_max=1.6, succ_tol=3, n_cand_per_dim=100, n_cand_max=5000, acq='ts', fail_tol=None)
BOUNDS = dict(ls=(0.005, 2.0), sig=(0.05, 20.0), noise=(0.0005, 0.1))


# --------------------------------------------------------------------------- coordinates
class Box:
    def __init__(self, net, p):
        L = p['L']; D = p['Dmax']
        lo = np.zeros((L, 4)); hi = np.zeros((L, 4))
        for l in range(L):
            x0, y0, z0 = net['p0'][l]
            lo[l] = [max(0.0, x0 - D[l]), max(0.0, y0 - D[l]), max(p['zmin'], z0 - D[l]), -math.pi / 2]
            hi[l] = [min(p['X'], x0 + D[l]), min(p['Y'], y0 + D[l]), min(p['zmax'], z0 + D[l]), math.pi / 2]
        self.lo = lo.reshape(-1); self.hi = hi.reshape(-1)
        self.active = np.nonzero(self.hi - self.lo > 1e-9)[0]
        self.d = len(self.active); self.L = L; self.net = net; self.p = p

    def to_unit(self, pose):
        v = np.concatenate([pose['pos'], pose['angles'][:, None]], axis=1).reshape(-1)
        return ((v - self.lo) / np.where(self.hi > self.lo, self.hi - self.lo, 1.0))[self.active]

    def from_unit(self, U):
        """U: (n, d) -> positions (n, L, 3), angles (n, L)."""
        U = np.atleast_2d(U)
        V = np.tile(self.lo, (len(U), 1))
        V[:, self.active] = self.lo[self.active] + U * (self.hi - self.lo)[self.active]
        V = V.reshape(len(U), self.L, 4)
        return V[:, :, :3], V[:, :, 3]

    def admissible(self, pos, ang):
        p, net = self.p, self.net
        ok = np.all(np.abs(ang) <= math.pi / 2 + 1e-12, axis=1)
        ok &= np.all(np.linalg.norm(pos - net['p0'][None], axis=2) <= p['Dmax'][None] + 1e-7, axis=1)
        ok &= np.all((pos[:, :, 0] >= -1e-7) & (pos[:, :, 0] <= p['X'] + 1e-7) & (pos[:, :, 1] >= -1e-7) &
                     (pos[:, :, 1] <= p['Y'] + 1e-7) & (pos[:, :, 2] >= p['zmin'] - 1e-7) & (pos[:, :, 2] <= p['zmax'] + 1e-7), axis=1)
        fixed = np.vstack([net['BS'], net['UE']])
        for l in range(self.L):
            for h in range(l + 1, self.L):
                ok &= np.linalg.norm(pos[:, l] - pos[:, h], axis=1) >= p['dmin'] - 1e-7
            ok &= np.min(np.linalg.norm(pos[:, l][:, None, :] - fixed[None], axis=2), axis=1) >= p['d0'] - 1e-7
        return ok


# --------------------------------------------------------------------------- ARD Matern-5/2 GP
def _sqdist(X, Z, ls):
    Xs = X / ls; Zs = Z / ls
    return np.maximum(np.sum(Xs * Xs, 1)[:, None] + np.sum(Zs * Zs, 1)[None, :] - 2.0 * Xs @ Zs.T, 0.0)


def matern52(X, Z, ls):
    r = np.sqrt(5.0 * _sqdist(X, Z, ls))
    return (1.0 + r + r * r / 3.0) * np.exp(-r)


def _nll_grad(theta, X, y):
    """Negative log marginal likelihood (constant mean profiled out) and its gradient w.r.t. the
    log-parameters (log length scales, log signal variance, log noise variance)."""
    n, d = X.shape
    ls = np.exp(theta[:d]); s2 = np.exp(theta[d]); n2 = np.exp(theta[d + 1])
    D2 = [((X[:, None, i] - X[None, :, i]) / ls[i]) ** 2 for i in range(d)]
    r2 = np.sum(D2, axis=0); r = np.sqrt(5.0 * r2); e = np.exp(-r)
    Km = (1.0 + r + r * r / 3.0) * e
    K = s2 * Km + (n2 + 1e-8) * np.eye(n)
    try:
        Lc = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10, np.zeros_like(theta)
    Ki = np.linalg.solve(Lc.T, np.linalg.solve(Lc, np.eye(n)))
    one = np.ones(n)
    c = (one @ Ki @ y) / (one @ Ki @ one)
    rr = y - c; a = Ki @ rr
    nll = 0.5 * rr @ a + np.sum(np.log(np.diag(Lc)))
    W = 0.5 * (Ki - np.outer(a, a))             # dNLL/dK
    g = np.zeros_like(theta)
    base = s2 * (5.0 / 3.0) * (1.0 + r) * e     # dk/dlog l_i = base * 5 * (D_i/l_i)^2 / 5  (r already includes sqrt5)
    for i in range(d):
        g[i] = np.sum(W * base * D2[i])
    g[d] = np.sum(W * s2 * Km)
    g[d + 1] = np.trace(W) * n2
    return nll, g


def gp_fit_ard(X, y, steps=50, lr=0.1):
    """Marginal-likelihood fit as in the reference implementation: Adam (learning rate 0.1, 50 steps) on the
    log-parameters from length scale 0.5, signal variance 1 and noise variance 0.005, within the bounds above."""
    off = np.mean(y); sc = np.std(y) if len(y) > 1 else 1.0
    if not np.isfinite(sc) or sc < 1e-12: sc = 1.0
    yn = (y - off) / sc; d = X.shape[1]
    th = np.concatenate([np.full(d, math.log(0.5)), [0.0, math.log(0.005)]])
    lo = np.array([math.log(BOUNDS['ls'][0])] * d + [math.log(BOUNDS['sig'][0]), math.log(BOUNDS['noise'][0])])
    hi = np.array([math.log(BOUNDS['ls'][1])] * d + [math.log(BOUNDS['sig'][1]), math.log(BOUNDS['noise'][1])])
    m1 = np.zeros_like(th); m2 = np.zeros_like(th); b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        f, g = _nll_grad(th, X, yn)
        if not np.isfinite(f): break
        m1 = b1 * m1 + (1 - b1) * g; m2 = b2 * m2 + (1 - b2) * g * g
        th = np.clip(th - lr * (m1 / (1 - b1 ** t)) / (np.sqrt(m2 / (1 - b2 ** t)) + eps), lo, hi)
    ls = np.exp(th[:d]); s2 = np.exp(th[d]); n2 = np.exp(th[d + 1])
    K = s2 * matern52(X, X, ls) + (n2 + 1e-8) * np.eye(len(y))
    Lc = np.linalg.cholesky(K); one = np.ones(len(y))
    Ki = lambda v: np.linalg.solve(Lc.T, np.linalg.solve(Lc, v))
    c = (one @ Ki(yn)) / (one @ Ki(one))
    return dict(X=X, ls=ls, s2=s2, n2=n2, c=c, L=Lc, a=Ki(yn - c), off=off, sc=sc)


def thompson(m, Z, rng):
    """One joint posterior sample of the latent function on the candidates (standardized scale)."""
    Kz = m['s2'] * matern52(m['X'], Z, m['ls'])
    mu = m['c'] + Kz.T @ m['a']
    V = np.linalg.solve(m['L'], Kz)
    C = m['s2'] * matern52(Z, Z, m['ls']) - V.T @ V
    C = 0.5 * (C + C.T)
    jit = 1e-8 * m['s2']
    for _ in range(8):
        try:
            Lc = np.linalg.cholesky(C + jit * np.eye(len(Z))); break
        except np.linalg.LinAlgError:
            jit *= 10
    else:
        w, U = np.linalg.eigh(C); Lc = U * np.sqrt(np.maximum(w, 0))
    return mu + Lc @ rng.standard_normal(len(Z))


# --------------------------------------------------------------------------- TuRBO-1
def turbo(evaluate, net, p, budget, seed, n_init=5, log=None, **kw):
    o = dict(TURBO); o.update(kw)
    box = Box(net, p); d = box.d
    fail_tol = int(math.ceil(d / 1.0)) if o['fail_tol'] is None else int(o['fail_tol'])
    n_cand = min(o['n_cand_per_dim'] * d, o['n_cand_max'])
    base = H_.pose_from(net['p0'], net['angles0'])
    rng0 = np.random.default_rng(seed + 200000)
    init = [base] + A.proposals_fast(net, p, rng0, n_init - 1, base, A.features([base], p), False)   # same as BO
    recs = []; Xall = []; yall = []
    Xtr = []; ytr = []                       # data of the current trust region
    Lside = o['L_init']; succ = fail = 0; restarts = 0; pending = list(init); trace_L = []
    t0 = time.time()
    for q in range(budget):
        from_ts = not pending
        if pending:
            pose = pending.pop(0)
        else:
            rng = np.random.default_rng([seed + 400000, q])
            Xa = np.array(Xtr); ya = np.array(ytr)
            m = gp_fit_ard(Xa, ya)
            xc = Xa[int(np.argmax(ya))]
            w = m['ls'] / np.prod(m['ls']) ** (1.0 / d)
            lb = np.clip(xc - w * Lside / 2, 0.0, 1.0); ub = np.clip(xc + w * Lside / 2, 0.0, 1.0)
            prob = min(1.0, 20.0 / d)
            cand = []; tries = 0
            while sum(len(c) for c in cand) < n_cand and tries < 20:
                tries += 1
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sob = qmc.Sobol(d, scramble=True, seed=rng).random(n_cand)
                U = lb + (ub - lb) * sob
                if prob < 1.0:
                    mask = rng.random((n_cand, d)) <= prob
                    mask[np.arange(n_cand), rng.integers(0, d, n_cand)] = True
                    U = np.where(mask, U, xc)
                pos, ang = box.from_unit(U)
                ok = box.admissible(pos, ang)
                seen = np.array(Xall)
                if len(seen):
                    dmin = np.min(np.max(np.abs(U[:, None, :] - seen[None]), axis=2), axis=1)
                    ok &= dmin > 1e-9
                cand.append(U[ok])
            U = np.vstack(cand)[:n_cand]
            if len(U) == 0:                  # no admissible point in the region: shrink and use the BO candidates
                pose = A.proposals_fast(net, p, rng, 1, recs[int(np.argmax(yall))]['pose'], np.array([A.features([r['pose']], p)[0] for r in recs]), True)[0]
            else:
                if o['acq'] == 'ei':           # variant (not in the paper): expected improvement inside the trust region
                    Kz = m['s2'] * matern52(m['X'], U, m['ls']); mu = m['c'] + Kz.T @ m['a']
                    V = np.linalg.solve(m['L'], Kz); sd = np.sqrt(np.maximum(m['s2'] - np.sum(V * V, 0), 1e-12))
                    ybest = (np.max(ya) - m['off']) / m['sc']
                    f = A.log_ei((mu - ybest) / sd, sd)
                else:
                    f = thompson(m, U, rng)
                pos, ang = box.from_unit(U[int(np.argmax(f))][None])
                pose = H_.pose_from(pos[0], ang[0])
        r = evaluate(pose); r['pose'] = pose; r['L'] = Lside; recs.append(r)
        u = box.to_unit(pose); Xall.append(u); yall.append(r['merit'])
        if from_ts:
            best = max(ytr)
            if r['merit'] > best + 1e-3 * abs(best): succ += 1; fail = 0
            else: succ = 0; fail += 1
            if succ == o['succ_tol']: Lside = min(2.0 * Lside, o['L_max']); succ = 0
            elif fail == fail_tol: Lside /= 2.0; fail = 0
        Xtr.append(u); ytr.append(r['merit'])
        trace_L.append(Lside)
        if Lside < o['L_min'] and q + 1 < budget:   # restart the trust region
            restarts += 1; Lside = o['L_init']; succ = fail = 0; Xtr = []; ytr = []
            rr = np.random.default_rng([seed + 500000, restarts])
            pending = A.proposals_fast(net, p, rr, n_init, base, np.array(Xall_feats(recs, p)), False)
        if log: log(f'  q{q + 1:2d}: merit {r["merit"]:9.4f} L {Lside:.3f} succ {succ} fail {fail}')
    ib = int(np.argmax(yall))
    return recs, ib, time.time() - t0, dict(restarts=restarts, L=trace_L, d=d, fail_tol=fail_tol, n_cand=n_cand)


def Xall_feats(recs, p):
    return [A.features([r['pose']], p)[0] for r in recs]
