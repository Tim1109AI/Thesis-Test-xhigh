"""Seeds, precoder initialisations, evaluators, BO and full runs."""
import math, time
import numpy as np
from scipy.special import erfcx, erfc
import hrs as H_

LAMQ = 200.0


# --------------------------------------------------------------------------- precoder initialisations
def unit_vector(x):
    n = np.linalg.norm(x)
    if n < 1e-14:
        x = np.zeros_like(x); x[0] = 1; n = 1
    v = x / n; j = int(np.argmax(np.abs(v)))
    return v * np.exp(-1j * np.angle(v[j]))


def reference_beams(H, d, p):
    """Port of hrs_reference (old AO initialisation / seed reference beams)."""
    BN = p['BN']; W = np.zeros((BN, d.nS), complex)
    priv = d.private_served
    V = {}
    if priv:
        Hp = H[priv]; ridge = max(len(priv) / max(p['B'] * p['Pmax'], 1e-300), 1e-12)
        Vp = Hp.conj().T @ np.linalg.inv(Hp @ Hp.conj().T + ridge * np.eye(len(priv)))
        for i, k in enumerate(priv): V[k] = Vp[:, i]
    for j in d.served:
        users = d.decoders[j]
        if j > d.G:
            vec = V[j - 1 - d.G]
        else:
            F = H[users]; st = np.sum(np.abs(F) ** 2, axis=1)
            ref = F[int(np.argmax(st))].conj()
            vec = F.conj().T @ ((1.0 / st) * H_.unit_phase(F @ ref))
        if np.linalg.norm(vec) < 1e-14: vec = H[users[0]].conj()
        W[:, d.col[j]] = unit_vector(vec)
    return H_.scale_to_power(W, p)


def init_rzf_eig(H, d, p):
    """New closed-form start: MMSE-regularised ZF over the served private streams,
    dominant eigenvector of the normalised channel covariance for every common stream,
    equal power per stream before per-BS normalisation."""
    BN = p['BN']; W = np.zeros((BN, d.nS), complex)
    priv = d.private_served
    if priv:
        Hp = H[priv]
        alpha = len(priv) / (p['B'] * p['Pmax'])
        Vp = np.linalg.solve(Hp.conj().T @ Hp + alpha * np.eye(BN), Hp.conj().T)
        for i, k in enumerate(priv):
            W[:, d.col[1 + d.G + k]] = Vp[:, i] / np.linalg.norm(Vp[:, i])
    for j in d.served:
        if j > d.G: continue
        F = H[d.decoders[j]]
        Fn = F / np.linalg.norm(F, axis=1)[:, None]
        R = Fn.conj().T @ Fn
        w, U = np.linalg.eigh(R)
        W[:, d.col[j]] = U[:, -1]
    return H_.scale_to_power(W, p)


def init_rzf_eig_eqtotal(H, d, p):
    """Same directions; streams share the TOTAL power equally (joint normalisation)."""
    W = init_rzf_eig(H, d, p)
    for i in range(W.shape[1]):
        W[:, i] /= np.linalg.norm(W[:, i])
    return H_.scale_to_power(W, p)


INITS = {'reference': reference_beams, 'rzf_eig': init_rzf_eig}


# --------------------------------------------------------------------------- coefficient seeds
def seed_old(ch, net, d, p, which='core'):
    """Port of hrs_coherent_seed_field ('core' or 'coherent') + closed-form coupled maximiser."""
    if which == 'core':
        core = [k for k in d.core if (1 + d.G + k) in d.col]
        dd = H_.Design(d.groups, [0] + [1 + d.G + k for k in core], d.K)
        dd.core = d.core
    else:
        dd = d
    V = reference_beams(net['Hd'], dd, p)
    for i in range(V.shape[1]): V[:, i] = unit_vector(V[:, i])
    wts = 1.0 / np.sqrt(np.maximum(np.sum(np.abs(net['Hd']) ** 2, axis=1), 1e-300))
    sw = 1.0 / dd.nS
    M, L, B, N = p['M'], p['L'], p['B'], p['N']
    field = np.zeros((M, 4, L), complex)
    for j in dd.served:
        users = dd.decoders[j]; v = V[:, dd.col[j]]
        w = wts[users] / np.sum(wts[users]) * sw
        phase = np.conj(H_.unit_phase(net['Hd'][users] @ v))
        for h, k in enumerate(users):
            for l in range(L):
                for b in range(B):
                    o = ch['mode'][l, b, k]
                    av = ch['aBS'][b, l] @ v[b * N:(b + 1) * N]
                    field[:, o, l] += w[h] * phase[h] * ch['s'][k, b, l] * av * ch['e'][k, b, l]
    return H_.theta_from_field(field)


def seed_loggain(ch, net, d, p):
    """Maximiser of the first-order expansion of sum_k log ||h_k||^2 at the direct channels:
    field z_{l,o} = sum_k sum_{b: o(l,b,k)=o} A_{k,b,l} h_{d,k,b}^H / ||h_{d,k}||^2."""
    M, L, B, N, K = p['M'], p['L'], p['B'], p['N'], p['K']
    Hd = net['Hd']; g = np.sum(np.abs(Hd) ** 2, axis=1)
    field = np.zeros((M, 4, L), complex)
    for k in range(K):
        for l in range(L):
            for b in range(B):
                o = ch['mode'][l, b, k]
                av = ch['aBS'][b, l] @ Hd[k, b * N:(b + 1) * N].conj()
                field[:, o, l] += ch['s'][k, b, l] * av / g[k] * ch['e'][k, b, l]
    return H_.theta_from_field(field)


def seed_logdet(ch, net, d, p):
    """Maximiser of the first-order expansion of log det(I + rho Hd^H Hd) (rho=P/N):
    z = sum_k A_k (I + rho Hd^H Hd)^{-1} h_{d,k}^H."""
    M, L, B, N, K = p['M'], p['L'], p['B'], p['N'], p['K']
    Hd = net['Hd']; rho = p['Pmax'] / N
    G = np.linalg.solve(np.eye(p['BN']) + rho * Hd.conj().T @ Hd, Hd.conj().T)  # (BN,K)
    field = np.zeros((M, 4, L), complex)
    for k in range(K):
        v = G[:, k]
        for l in range(L):
            for b in range(B):
                o = ch['mode'][l, b, k]
                av = ch['aBS'][b, l] @ v[b * N:(b + 1) * N]
                field[:, o, l] += ch['s'][k, b, l] * av * ch['e'][k, b, l]
    return H_.theta_from_field(field)


SEEDS = {'old_core': lambda ch, net, d, p: seed_old(ch, net, d, p, 'core'),
         'old_coherent': lambda ch, net, d, p: seed_old(ch, net, d, p, 'coherent'),
         'loggain': seed_loggain, 'logdet': seed_logdet}


# --------------------------------------------------------------------------- evaluators
def qt_precoder_loop(H, W, d, p, qtp, lamQ=LAMQ, max_iter=20, tol=0.1, rel=None):
    cur = H_.merit(W, H, d, p, lamQ); calls = 0; hist = [cur[0]]
    for it in range(1, max_iter + 1):
        try:
            Wn = qtp.solve(H, W); calls += 1
        except Exception:
            calls += 1; break
        mn = H_.merit(Wn, H, d, p, lamQ)
        if mn[0] < cur[0]: break          # solver-accuracy guard (QT is monotone in exact arithmetic)
        W = Wn; gain = mn[0] - cur[0]; cur = mn; hist.append(cur[0])
        thr = tol if rel is None else rel * max(1.0, abs(cur[0]))
        if gain < thr: break
    return W, cur, calls, hist


class Evaluator:
    def __init__(self, net, d, p, seed='loggain', init='rzf_eig', lamQ=LAMQ, max_iter=20, tol=0.1):
        self.net, self.d, self.p = net, d, p
        self.seed, self.init, self.lamQ = seed, init, lamQ
        self.max_iter, self.tol = max_iter, tol
        self.qtp = H_.QTPrecoderDirect(d, p, lamQ)

    def start(self, pose):
        ch = H_.channel_terms(pose, self.net, self.p)
        theta = SEEDS[self.seed](ch, self.net, self.d, self.p)
        H = H_.effective_channel(theta, ch, self.net, self.p)
        W0 = INITS[self.init](H, self.d, self.p)
        return ch, theta, H, W0

    def __call__(self, pose):
        t0 = time.time(); c0 = time.process_time()
        ch, theta, H, W0 = self.start(pose)
        m0 = H_.merit(W0, H, self.d, self.p, self.lamQ)
        W, cur, calls, hist = qt_precoder_loop(H, W0, self.d, self.p, self.qtp, self.lamQ, self.max_iter, self.tol)
        return dict(merit=cur[0], rate=cur[1], deficit=cur[2], theta=theta, W=W, calls=calls,
                    iters=len(hist) - 1, init_merit=m0[0], seconds=time.time() - t0, cpu=time.process_time() - c0, hist=hist)


class AOEvaluator:
    def __init__(self, net, d, p, seed='loggain', init='rzf_eig', lamQ=LAMQ, **aokw):
        self.net, self.d, self.p = net, d, p
        self.seed, self.init, self.lamQ = seed, init, lamQ
        self.qtp = H_.QTPrecoderDirect(d, p, lamQ); self.qtc = H_.QTCoefficientDirect(d, p, lamQ)
        self.aokw = aokw

    def __call__(self, pose):
        t0 = time.time(); c0 = time.process_time()
        ch = H_.channel_terms(pose, self.net, self.p)
        theta = SEEDS[self.seed](ch, self.net, self.d, self.p)
        H = H_.effective_channel(theta, ch, self.net, self.p)
        W0 = INITS[self.init](H, self.d, self.p)
        m0 = H_.merit(W0, H, self.d, self.p, self.lamQ)
        out = H_.ao(theta, W0, ch, self.net, self.d, self.p, self.lamQ, qtp=self.qtp, qtc=self.qtc, **self.aokw)
        return dict(merit=out['merit'], rate=out['rate'], deficit=out['deficit'], theta=out['theta'], W=out['W'],
                    calls=out['calls'], iters=out['iterations'], init_merit=m0[0], seconds=time.time() - t0,
                    cpu=time.process_time() - c0, hist=out['hist'])


# --------------------------------------------------------------------------- BO
PERIODIC = True      # azimuths modulo pi (a two-sided surface: phi and phi + pi are the same deployment); design P / R


def features(poses, p):
    X = np.zeros((len(poses), (5 if PERIODIC else 4) * p['L']))
    for i, ps in enumerate(poses):
        pos = ps['pos']
        if PERIODIC:     # the azimuth as a point on a circle of circumference 1 (the same local scale as before)
            a = 2.0 * ps['angles']
            blk = np.stack([pos[:, 0] / p['X'], pos[:, 1] / p['Y'], (pos[:, 2] - p['zmin']) / (p['zmax'] - p['zmin']),
                            np.cos(a) / (2 * math.pi), np.sin(a) / (2 * math.pi)], axis=1)
        else:
            blk = np.stack([pos[:, 0] / p['X'], pos[:, 1] / p['Y'], (pos[:, 2] - p['zmin']) / (p['zmax'] - p['zmin']),
                            (ps['angles'] + math.pi / 2) / math.pi], axis=1)
        X[i] = blk.reshape(-1)
    return X


def wrap_angle(a):
    """Azimuth modulo pi, in [-pi/2, pi/2)."""
    return np.mod(a + math.pi / 2, math.pi) - math.pi / 2


def features_m(poses, p, length=100.0):
    """Features with all coordinates in the same unit (length, m) and the azimuth in units of pi."""
    X = np.zeros((len(poses), 4 * p['L']))
    for i, ps in enumerate(poses):
        pos = ps['pos']
        X[i] = np.stack([pos[:, 0] / length, pos[:, 1] / length, (pos[:, 2] - p['zmin']) / length, (ps['angles'] + math.pi / 2) / math.pi], axis=1).reshape(-1)
    return X


def features_w(poses, center, r1=1.0, rref=100.0, a1=0.05, aref=math.pi):
    """Features measured from the deployment center on a logarithmic scale: every surface displacement d from its
    position in center is mapped to log(1 + |d|/r1) / log(1 + rref/r1) in the direction of d, every azimuth difference
    a to sign(a) log(1 + |a|/a1) / log(1 + aref/a1)."""
    X = []
    for ps in poses:
        d = ps['pos'] - center['pos']; r = np.linalg.norm(d, axis=1)
        u = (np.log1p(r / r1) / math.log1p(rref / r1) / np.maximum(r, 1e-12))[:, None] * d
        a = ps['angles'] - center['angles']
        v = np.sign(a) * np.log1p(np.abs(a) / a1) / math.log1p(aref / a1)
        X.append(np.concatenate([u, v[:, None]], axis=1).reshape(-1))
    return np.array(X)


def proposals(net, p, rng, count, center, seen, use_local, local_scale=0.15):
    L = p['L']; out = []; cand = [] if seen is None else list(seen)
    attempts = 0
    while len(out) < count and attempts < max(5000, 200 * count):
        attempts += 1
        local = use_local and attempts % 2 == 0
        pos = net['p0'].copy(); ang = net['angles0'].copy()
        for l in range(L):
            if local:
                pos[l] = center['pos'][l] + local_scale * p['Dmax'][l] * rng.standard_normal(3)
                ang[l] = center['angles'][l] + local_scale * math.pi * rng.standard_normal()
            else:
                dirn = rng.standard_normal(3); dirn /= max(np.linalg.norm(dirn), 1e-300)
                pos[l] = net['p0'][l] + p['Dmax'][l] * rng.random() ** (1 / 3) * dirn
                ang[l] = math.pi * rng.random() - math.pi / 2
        if np.any(np.abs(ang) > math.pi / 2): continue
        pose = H_.pose_from(pos, ang)
        v, lk = H_.geometry_violation(pose, net, p)
        if v > 1e-7 or lk > 1e-7: continue
        f = features([pose], p)[0]
        if cand and np.min(np.max(np.abs(np.array(cand) - f), axis=1)) < 1e-10: continue
        out.append(pose); cand.append(f)
    return out


def kernel(X, Z, ell, kern='iso'):
    if kern == 'pa':
        # two length scales: ell = (positions, azimuths); features are [x y z phi] per surface
        lp, la = ell
        w = np.tile([1 / lp, 1 / lp, 1 / lp, 1 / la], X.shape[1] // 4)
        Xs, Zs = X * w, Z * w
        sq = np.maximum(0, np.sum(Xs ** 2, 1)[:, None] + np.sum(Zs ** 2, 1)[None, :] - 2 * Xs @ Zs.T) / X.shape[1]
        r = np.sqrt(5 * sq)
        return (1 + r + r ** 2 / 3) * np.exp(-r)
    if kern == 'add':
        # additive kernel: mean of Matern-5/2 kernels on the 4-coordinate blocks of the surfaces
        nb = X.shape[1] // 4; out = 0.0
        for b in range(nb):
            out = out + kernel(X[:, 4 * b:4 * b + 4], Z[:, 4 * b:4 * b + 4], ell)
        return out / nb
    sq = np.maximum(0, np.sum(X ** 2, 1)[:, None] + np.sum(Z ** 2, 1)[None, :] - 2 * X @ Z.T) / X.shape[1]
    r = np.sqrt(5 * sq) / ell
    return (1 + r + r ** 2 / 3) * np.exp(-r)


def gp_fit(X, y, scales=(0.15, 0.3, 0.6), nugget=1e-6, kern='iso'):
    off = np.mean(y); sc = np.std(y, ddof=1) if len(y) > 1 else 1.0
    if not np.isfinite(sc) or sc < 1e-8: sc = 1.0
    yn = (y - off) / sc; best = None
    if kern == 'pa':
        scales = [(a, b) for a in (0.05, 0.15, 0.3, 0.6) for b in scales]
    for ell in scales:
        Kx = kernel(X, X, ell, kern); jit = nugget
        for _ in range(8):
            try:
                Lc = np.linalg.cholesky(Kx + jit * np.eye(len(y))); break
            except np.linalg.LinAlgError:
                jit *= 10
        a = np.linalg.solve(Lc.T, np.linalg.solve(Lc, yn))
        obj = 0.5 * yn @ a + np.sum(np.log(np.diag(Lc)))
        if best is None or obj < best[0]:
            best = (obj, dict(X=X, off=off, sc=sc, ell=ell, L=Lc, a=a, kern=kern))
    return best[1]


def gp_fit_trend(X, y, H, scales=(0.15, 0.3, 0.6), nugget=1e-6, kern='iso'):
    """GP with the mean H @ beta (universal kriging): beta by generalized least squares and the signal variance by
    maximum likelihood, for every length scale; the length scale with the largest likelihood is kept."""
    off = np.mean(y); sc = np.std(y, ddof=1) if len(y) > 1 else 1.0
    if not np.isfinite(sc) or sc < 1e-8: sc = 1.0
    yn = (y - off) / sc; best = None; n = len(y)
    for ell in scales:
        Kx = kernel(X, X, ell, kern); jit = nugget
        for _ in range(8):
            try:
                Lc = np.linalg.cholesky(Kx + jit * np.eye(n)); break
            except np.linalg.LinAlgError:
                jit *= 10
        KiH = np.linalg.solve(Lc.T, np.linalg.solve(Lc, H)); Kiy = np.linalg.solve(Lc.T, np.linalg.solve(Lc, yn))
        Am = H.T @ KiH + 1e-10 * np.eye(H.shape[1]); beta = np.linalg.solve(Am, H.T @ Kiy)
        Kir = Kiy - KiH @ beta; s2 = max(float((yn - H @ beta) @ Kir) / n, 1e-12)
        obj = 0.5 * n * np.log(s2) + np.sum(np.log(np.diag(Lc)))
        if best is None or obj < best[0]:
            best = (obj, dict(X=X, off=off, sc=sc, s2=s2, ell=ell, L=Lc, Kir=Kir, KiH=KiH, beta=beta, Am=Am, kern=kern))
    return best[1]


def gp_predict_trend(m, Z, Hz):
    Kz = kernel(m['X'], Z, m['ell'], m['kern']); v = np.linalg.solve(m['L'], Kz)
    mu = m['off'] + m['sc'] * (Hz @ m['beta'] + Kz.T @ m['Kir'])
    u = Hz - Kz.T @ m['KiH']
    var = m['s2'] * np.maximum(1e-12, 1 - np.sum(v ** 2, 0) + np.sum(u * np.linalg.solve(m['Am'], u.T).T, 1))
    return mu, m['sc'] * np.sqrt(var)


def gp_predict(m, Z):
    Kz = kernel(m['X'], Z, m['ell'], m.get('kern', 'iso')); v = np.linalg.solve(m['L'], Kz)
    mu = m['off'] + m['sc'] * (Kz.T @ m['a'])
    sd = m['sc'] * np.sqrt(np.maximum(1e-12, 1 - np.sum(v ** 2, 0)))
    return mu, sd


def log_ei(z, s):
    val = np.zeros_like(z); tail = z < -20; mid = (z < 0) & ~tail; pos = z >= 0
    u = 1 / z[tail] ** 2
    corr = 1 - 3 * u + 15 * u ** 2 - 105 * u ** 3 + 945 * u ** 4 - 10395 * u ** 5 + 135135 * u ** 6
    val[tail] = np.log(s[tail]) - 0.5 * z[tail] ** 2 - 0.5 * math.log(2 * math.pi) - 2 * np.log(-z[tail]) + np.log(corr)
    fac = 1 + z[mid] * math.sqrt(math.pi / 2) * erfcx(-z[mid] / math.sqrt(2))
    val[mid] = np.log(s[mid]) - 0.5 * z[mid] ** 2 - 0.5 * math.log(2 * math.pi) + np.log(fac)
    term = np.exp(-0.5 * z[pos] ** 2) / math.sqrt(2 * math.pi) + z[pos] * 0.5 * erfc(-z[pos] / math.sqrt(2))
    val[pos] = np.log(s[pos]) + np.log(term)
    return val


def bayes_opt(evaluate, net, p, budget, seed, n_init=5, n_cand=256, log=None, time_limit=None):
    """Merit-based BO: every evaluation (feasible or not) trains the GP."""
    base = H_.pose_from(net['p0'], net['angles0'])
    rng0 = np.random.default_rng(seed + 200000)
    init = [base] + proposals(net, p, rng0, n_init - 1, base, features([base], p), False)
    X = []; y = []; recs = []
    t0 = time.time()
    for q in range(budget):
        if q < min(n_init, budget):
            pose = init[q]
        else:
            ib = int(np.argmax(y)); center = recs[ib]['pose']
            rng = np.random.default_rng([seed + 300000, q])
            cands = proposals(net, p, rng, n_cand, center, np.array(X), True)
            Z = features(cands, p)
            m = gp_fit(np.array(X), np.array(y))
            mu, sd = gp_predict(m, Z)
            z = (mu - np.max(y)) / sd
            pose = cands[int(np.argmax(log_ei(z, sd)))]
        r = evaluate(pose); r['pose'] = pose
        X.append(features([pose], p)[0]); y.append(r['merit']); recs.append(r)
        if log: log(f'  q{q + 1:2d}: merit {r["merit"]:9.4f} rate {r["rate"]:8.4f} def {r["deficit"]:.3g} '
                    f'calls {r["calls"]} ({r["seconds"]:.2f}s)')
    ib = int(np.argmax(y))
    return recs, ib, time.time() - t0


def init_base_extra(H, d, p):
    """Base streams (super-common + core private streams served) share the power equally with
    RZF (over the served core users) / dominant-eigenvector directions; every other served stream
    (group-common, non-core private) starts at the noise level, i.e. unit SNR at its weakest decoder."""
    BN = p['BN']; W = np.zeros((BN, d.nS), complex)
    core = [k for k in d.core if (1 + d.G + k) in d.col]
    base = {0} | {1 + d.G + k for k in core}
    if core:
        Hp = H[core]
        alpha = len(core) / (p['B'] * p['Pmax'])
        Vp = np.linalg.solve(Hp.conj().T @ Hp + alpha * np.eye(BN), Hp.conj().T)
        for i, k in enumerate(core):
            W[:, d.col[1 + d.G + k]] = Vp[:, i] / np.linalg.norm(Vp[:, i])
    for j in d.served:
        if j > d.G and (j in base): continue
        F = H[d.decoders[j]]
        if j <= d.G:
            Fn = F / np.linalg.norm(F, axis=1)[:, None]
            w_, U = np.linalg.eigh(Fn.conj().T @ Fn); v = U[:, -1]
        else:
            v = F[0].conj() / np.linalg.norm(F[0])
        W[:, d.col[j]] = v
    bcols = [d.col[j] for j in d.served if j in base]
    Wb = np.zeros_like(W); Wb[:, bcols] = W[:, bcols]
    Wb = H_.scale_to_power(Wb, p)
    for j in d.served:
        if j in base: continue
        c = d.col[j]; v = W[:, c]
        amp = np.min(np.abs(H[d.decoders[j]] @ v))
        Wb[:, c] = v / max(amp, 1e-300)
    return Wb


INITS['base_extra'] = init_base_extra


def init_layered(H, d, p):
    """rzf_eig directions; the per-BS power is split equally among the HRS layers present
    (super-common, group-common, private), and equally among the streams of a layer."""
    W = init_rzf_eig(H, d, p)
    layers = {'c': [j for j in d.served if j == 0], 'g': [j for j in d.served if 1 <= j <= d.G],
              'p': [j for j in d.served if j > d.G]}
    nl = sum(1 for v in layers.values() if v)
    for v in layers.values():
        for j in v:
            c = d.col[j]; W[:, c] = W[:, c] / np.linalg.norm(W[:, c]) * math.sqrt(1.0 / (nl * len(v)))
    return H_.scale_to_power(W, p)


INITS['layered'] = init_layered


def _init_generic(H, d, p, base, active_extra=()):
    """Directions: RZF over the private streams in `base`, dominant eigenvector for common streams,
    matched filter for other privates.  Streams in `base` + `active_extra` share the power equally;
    the remaining served streams start at the noise level (unit SNR at their weakest decoder)."""
    BN = p['BN']; W = np.zeros((BN, d.nS), complex)
    on = set(base) | set(active_extra)
    bpriv = [j - 1 - d.G for j in d.served if j > d.G and j in base]
    if bpriv:
        Hp = H[bpriv]; alpha = len(bpriv) / (p['B'] * p['Pmax'])
        Vp = np.linalg.solve(Hp.conj().T @ Hp + alpha * np.eye(BN), Hp.conj().T)
        for i, k in enumerate(bpriv):
            W[:, d.col[1 + d.G + k]] = Vp[:, i] / np.linalg.norm(Vp[:, i])
    for j in d.served:
        if j > d.G and j - 1 - d.G in bpriv: continue
        F = H[d.decoders[j]]
        if j <= d.G:
            Fn = F / np.linalg.norm(F, axis=1)[:, None]
            _, U = np.linalg.eigh(Fn.conj().T @ Fn); v = U[:, -1]
        else:
            v = F[0].conj() / np.linalg.norm(F[0])
        W[:, d.col[j]] = v
    cols = [d.col[j] for j in d.served if j in on]
    Wb = np.zeros_like(W); Wb[:, cols] = W[:, cols]
    Wb = H_.scale_to_power(Wb, p)
    for j in d.served:
        if j in on: continue
        c = d.col[j]; v = W[:, c]
        Wb[:, c] = v / max(np.min(np.abs(H[d.decoders[j]] @ v)), 1e-300)
    if getattr(d, 'bsmask', None) is not None:      # BS-specific stream association: zero the blocks not carried
        N = p['N']
        for b in range(p['B']):
            Wb[b * N:(b + 1) * N, ~d.bsmask[b]] = 0.0
    return H_.scale_to_power(Wb, p)


def init_bn(H, d, p):
    return _init_generic(H, d, p, d.served[:0] + [j for j in d.priority[:p['BN']] if j in d.col])


def init_zf(H, d, p, groups_on=False):
    priv = [j - 1 - d.G for j in d.served if j > d.G]
    sel = H_.zf_greedy(H[priv], p['B'] * p['Pmax']) if priv else []
    base = [0] + [1 + d.G + priv[i] for i in sel]
    extra = [j for j in d.served if 1 <= j <= d.G] if groups_on else []
    return _init_generic(H, d, p, base, extra)


def init_mf(H, d, p):
    """Simplified start: every served stream along the dominant eigenvector of the normalized channels of its decoders
    (the matched filter for a private stream), scaled to the full power of every BS."""
    W = np.zeros((p['BN'], d.nS), complex)
    for j in d.served:
        F = H[d.decoders[j]]
        Fn = F / np.linalg.norm(F, axis=1)[:, None]
        _, U = np.linalg.eigh(Fn.conj().T @ Fn)
        W[:, d.col[j]] = U[:, -1]
    if getattr(d, 'bsmask', None) is not None:
        N = p['N']
        for b in range(p['B']):
            W[b * N:(b + 1) * N, ~d.bsmask[b]] = 0.0
    return H_.scale_to_power(W, p)


INITS['mf'] = init_mf


def init_core(H, d, p, mf=False):
    """Simplified start without a separate user selection: the private streams of the core users of the discrete
    design use the columns of the regularized ZF matrix of their channels (matched filters with mf=True), the
    super-common stream the dominant eigenvector of the normalized channels; these streams share the power, and every
    other served stream starts at the noise level (as in init_zf)."""
    base = [0] + [1 + d.G + k for k in d.core if (1 + d.G + k) in d.col]
    if not mf:
        return _init_generic(H, d, p, base)
    W = np.zeros((p['BN'], d.nS), complex)
    for j in d.served:
        F = H[d.decoders[j]]; Fn = F / np.linalg.norm(F, axis=1)[:, None]
        _, U = np.linalg.eigh(Fn.conj().T @ Fn); W[:, d.col[j]] = U[:, -1]
    on = [d.col[j] for j in d.served if j in base]
    Wb = np.zeros_like(W); Wb[:, on] = W[:, on]; Wb = H_.scale_to_power(Wb, p)
    for j in d.served:
        if j in base: continue
        c = d.col[j]; v = W[:, c]; Wb[:, c] = v / max(np.min(np.abs(H[d.decoders[j]] @ v)), 1e-300)
    return H_.scale_to_power(Wb, p)


INITS['core'] = init_core
INITS['core_mf'] = lambda H, d, p: init_core(H, d, p, mf=True)
INITS['bn'] = init_bn
INITS['zf'] = init_zf
INITS['zfg'] = lambda H, d, p: init_zf(H, d, p, True)
INITS['eqc'] = lambda H, d, p: _init_generic(H, d, p, list(d.served))


def seed_logpow(ch, net, d, p, init=None):
    """Maximiser of the first-order expansion, at the direct channels, of
    sum_{j active} sum_{k in D_j} ln |h_k w_j|^2 for the precoder initialisation W0(Hd):
    field z = sum_{j,k} A_k w_j / (h_{d,k} w_j)."""
    init = init or INITS['core']
    Hd = net['Hd']; M, L, B, N = p['M'], p['L'], p['B'], p['N']
    W0 = init(Hd, d, p)
    pw = np.sum(np.abs(W0) ** 2, axis=0); act = pw > 1e-3 * np.max(pw)
    field = np.zeros((M, 4, L), complex)
    for j in d.served:
        c = d.col[j]
        if not act[c]: continue
        w = W0[:, c]
        for k in d.decoders[j]:
            a = Hd[k] @ w
            for l in range(L):
                for b in range(B):
                    o = ch['mode'][l, b, k]
                    av = ch['aBS'][b, l] @ w[b * N:(b + 1) * N]
                    field[:, o, l] += ch['s'][k, b, l] * av / a * ch['e'][k, b, l]
    return H_.theta_from_field(field)


SEEDS['logpow'] = seed_logpow
SEEDS['logpow_eq'] = lambda ch, net, d, p: seed_logpow(ch, net, d, p, INITS['eqc'])


# --------------------------------------------------------------------------- closed-form coefficient rule
def coef_rule(H, W, ch, d, p):
    """theta maximising sum_{(k,j)} Re{ conj(y_kj)/(I_kj+|y_kj|^2) * y_kj(theta) } over the coupled
    hardware set, i.e. the desired-signal part of the normalised QT surrogates of all rate constraints
    (first-order expansion of sum_{(k,j)} R_kj in the desired amplitudes). y_kj(theta) is affine in theta."""
    Y = H @ W
    K, B, L, M, N = p['K'], p['B'], p['L'], p['M'], p['N']
    beta = np.zeros((K, B, L), complex)
    for (k, j, I) in d.pairs:
        c = d.col[j]; ic = [d.col[i] for i in I]
        y = Y[k, c]; a = np.conj(y) / (1.0 + np.sum(np.abs(Y[k, ic]) ** 2) + abs(y) ** 2)
        for b in range(B):
            beta[k, b, :] += a * (ch['aBS'][b] @ W[b * N:(b + 1) * N, c])
    field = np.zeros((M, 4, L), complex)
    for l in range(L):
        for b in range(B):
            for k in range(K):
                field[:, ch['mode'][l, b, k], l] += beta[k, b, l] * ch['s'][k, b, l] * ch['e'][k, b, l]
    return H_.theta_from_field(field)


def seed_rule(ch, net, d, p):
    """Coefficient initialisation: the rule above at the direct channels with the precoder
    initialisation (core-user start) computed on the direct channels."""
    return coef_rule(net['Hd'], INITS['core'](net['Hd'], d, p), ch, d, p)


SEEDS['rule'] = seed_rule
SEEDS['rule_zf'] = lambda ch, net, d, p: coef_rule(net['Hd'], INITS['zf'](net['Hd'], d, p), ch, d, p)   # earlier start (greedy ZF)
SEEDS['rule_mf'] = lambda ch, net, d, p: coef_rule(net['Hd'], INITS['mf'](net['Hd'], d, p), ch, d, p)      # simplified start
SEEDS['rule_core'] = lambda ch, net, d, p: coef_rule(net['Hd'], INITS['core'](net['Hd'], d, p), ch, d, p)
SEEDS['rule_core_mf'] = lambda ch, net, d, p: coef_rule(net['Hd'], INITS['core_mf'](net['Hd'], d, p), ch, d, p)
SEEDS['rule_eqc'] = lambda ch, net, d, p: coef_rule(net['Hd'], INITS['eqc'](net['Hd'], d, p), ch, d, p)


def init_core_carried(H, d, p):
    """The core-user start (init_core) with every direction computed on the antennas of the BSs that carry the stream
    (d.bsmask): the private stream of a core user k carried by the BSs A uses column k of the regularized ZF matrix of
    the core channels restricted to the antennas of A (the same regularization as init_core), every common stream the
    dominant eigenvector of the normalized channels of its decoders restricted to A, every other private stream the
    matched filter on A. Base streams (super-common, core privates) share the power, every other served stream starts
    at the noise level of its weakest decoder, as in init_core. With joint transmission (d.bsmask None) it is
    init_core. Switch of the design study (BS-specific designs, harness/variants.py 'opt': init/seed), off by default."""
    if getattr(d, 'bsmask', None) is None:
        return init_core(H, d, p)
    BN, N, B = p['BN'], p['N'], p['B']
    W = np.zeros((BN, d.nS), complex)
    ant = lambda j: np.concatenate([np.arange(b * N, (b + 1) * N) for b in range(B) if d.bsmask[b, d.col[j]]])
    base = [0] + [1 + d.G + k for k in d.core if (1 + d.G + k) in d.col]
    bpriv = [j - 1 - d.G for j in base if j > d.G]
    alpha = len(bpriv) / (B * p['Pmax']) if bpriv else 1.0
    for j in d.served:
        A_ = ant(j)
        if j > d.G and (j - 1 - d.G) in bpriv:
            Hp = H[np.ix_(bpriv, A_)]
            V = np.linalg.solve(Hp.conj().T @ Hp + alpha * np.eye(len(A_)), Hp.conj().T)
            v = V[:, bpriv.index(j - 1 - d.G)]
        elif j <= d.G:
            F = H[np.ix_(d.decoders[j], A_)]; Fn = F / np.linalg.norm(F, axis=1)[:, None]
            _, U = np.linalg.eigh(Fn.conj().T @ Fn); v = U[:, -1]
        else:
            v = H[d.decoders[j][0], A_].conj()
        W[A_, d.col[j]] = v / max(np.linalg.norm(v), 1e-300)
    on = [d.col[j] for j in d.served if j in base]
    Wb = np.zeros_like(W); Wb[:, on] = W[:, on]; Wb = H_.scale_to_power(Wb, p)
    for j in d.served:
        if j in base: continue
        c = d.col[j]; v = W[:, c]
        Wb[:, c] = v / max(np.min(np.abs(H[d.decoders[j]] @ v)), 1e-300)
    return H_.scale_to_power(Wb, p)


INITS['core_carried'] = init_core_carried
SEEDS['rule_carried'] = lambda ch, net, d, p: coef_rule(net['Hd'], INITS['core_carried'](net['Hd'], d, p), ch, d, p)


def refine(theta, W, ch, net, d, p, qtp, lamQ=LAMQ, rounds=20, qt_iter=20, qt_tol=0.1, log=None, init='core'):
    """Refinement at the selected deployment. The QT iterations of its evaluation are continued to
    convergence; then, while F increases, the coefficients are updated in closed form at the current
    solution and the precoders are re-optimized by QT iterations (to convergence) from the closed-form start."""
    H = H_.effective_channel(theta, ch, net, p)
    W, cur, calls, _ = qt_precoder_loop(H, W, d, p, qtp, lamQ, qt_iter, qt_tol)
    if log: log(f'    refine 0 (QT to convergence): {cur[0]:.4f}')
    n = 0
    for n in range(1, rounds + 1):
        th2 = coef_rule(H, W, ch, d, p)
        H2 = H_.effective_channel(th2, ch, net, p)
        W2, c2, cl, _ = qt_precoder_loop(H2, INITS[init](H2, d, p), d, p, qtp, lamQ, qt_iter, qt_tol); calls += cl
        if log: log(f'    refine {n}: {cur[0]:.4f} -> {c2[0]:.4f}')
        if c2[0] > cur[0] + 1e-9:
            theta, W, H, cur = th2, W2, H2, c2
        else:
            break
    return dict(theta=theta, W=W, H=H, merit=cur[0], rate=cur[1], deficit=cur[2], iterations=n, calls=calls)




# --------------------------------------------------------------------------- deployment search (v17: scale-aware BO)
R0 = 1.0       # smallest displacement scale (m) of the candidates and of the GP mean
SEARCH = 'R'   # the deployment search of every caller of bayes_opt_fast without an explicit `design`: 'R' = design R
               # (physbo.DESIGN_R, the search of package v6); a dict of physbo.search options; None = designs T / P


def proposals_fast(net, p, rng, count, center, seen, use_local, local_scale=0.15, ang_scale=None, radial='volume', local_mode='all',
                   sampler='ball', fix_angles=False, scale_min=None, scale_local=False, u_lim=(0.0, 1.0), scale_mix=0.0):
    """Admissible candidate deployments, alternately random and local (perturbations of center) when use_local.

    scale_min=None: a random candidate places every surface uniformly in the admissible part of its ball of radius
    Dmax, and a local step has the standard deviation local_scale * Dmax ('ball') or local_scale * the half width of
    the admissible interval of the coordinate ('box').
    scale_min=r0 (scale-aware search): every candidate first draws a displacement scale R, log-uniformly from
    [r0, Dmax] (u_lim: the part [u0, u1] of the logarithmic range, for the initial design); a random candidate
    places every surface uniformly in the admissible part of its ball of radius R, and with scale_local the local
    steps are those of the admissible set with Dmax = R. Surfaces with Dmax <= r0 use R = Dmax.
    'ball' and 'box' give the same distribution of the random candidates; 'box' draws from the bounding box of the
    ball and rejects outside it, which keeps the acceptance high when the radius exceeds the height range."""
    L = p['L']; out = []; feats = []
    seen = np.zeros((0, (5 if PERIODIC else 4) * L)) if seen is None or len(seen) == 0 else np.asarray(seen)
    fixed = np.vstack([net['BS'], net['UE']])
    Dv = np.asarray(p['Dmax'], float)
    ms = scale_min is not None and bool(np.any(Dv > scale_min))
    attempt = 0
    while len(out) < count and attempt < max(5000, 200 * count):
        nb = 2 * count
        idx = attempt + 1 + np.arange(nb)
        local = use_local & (idx % 2 == 0)
        pos = np.empty((nb, L, 3)); ang = np.empty((nb, L))
        Dg = np.broadcast_to(Dv[None, :], (nb, L))
        if ms:      # one log-uniform scale per candidate, shared by its surfaces
            uu = u_lim[0] + (u_lim[1] - u_lim[0]) * rng.random((nb, 1))
            Dg = Dv[None, :] * np.minimum(1.0, scale_min / np.maximum(Dv, 1e-300))[None, :] ** (1.0 - uu)
            if scale_mix > 0:      # a share scale_mix of the candidates has the full scale Dmax (uniform over the admissible set)
                Dg = np.where(rng.random((nb, 1)) < scale_mix, Dv[None, :], Dg)
        dirn = rng.standard_normal((nb, L, 3)); dirn /= np.maximum(np.linalg.norm(dirn, axis=2, keepdims=True), 1e-300)
        u = rng.random((nb, L, 1))
        rad = Dg[:, :, None] * (u ** (1 / 3) if radial == 'volume' else u)   # 'radius': uniform distance
        gpos = net['p0'][None] + rad * dirn; gang = math.pi * rng.random((nb, L)) - math.pi / 2
        if sampler == 'box':
            lo = np.stack([np.maximum(0.0, net['p0'][:, 0] - Dv), np.maximum(0.0, net['p0'][:, 1] - Dv),
                           np.maximum(p['zmin'], net['p0'][:, 2] - Dv)], axis=1)
            hi = np.stack([np.minimum(p['X'], net['p0'][:, 0] + Dv), np.minimum(p['Y'], net['p0'][:, 1] + Dv),
                           np.minimum(p['zmax'], net['p0'][:, 2] + Dv)], axis=1)
            half = (hi - lo) / 2
            if ms:     # per-candidate boxes of radius R
                p0 = net['p0'][None]
                glo = np.stack([np.maximum(0.0, p0[..., 0] - Dg), np.maximum(0.0, p0[..., 1] - Dg), np.maximum(p['zmin'], p0[..., 2] - Dg)], axis=2)
                ghi = np.stack([np.minimum(p['X'], p0[..., 0] + Dg), np.minimum(p['Y'], p0[..., 1] + Dg), np.minimum(p['zmax'], p0[..., 2] + Dg)], axis=2)
                gpos = glo + (ghi - glo) * rng.random((nb, L, 3))
                if scale_local: half = (ghi - glo) / 2
            else:
                gpos = lo[None] + (hi - lo)[None] * rng.random((nb, L, 3))
            lpos = center['pos'][None] + local_scale * (half if half.ndim == 3 else half[None]) * rng.standard_normal((nb, L, 3))
        else:
            lsc = Dg[:, :, None] if (ms and scale_local) else Dv[None, :, None]
            lpos = center['pos'][None] + local_scale * lsc * rng.standard_normal((nb, L, 3))
        lang = center['angles'][None] + (local_scale if ang_scale is None else ang_scale) * math.pi * rng.standard_normal((nb, L))
        if PERIODIC: lang = wrap_angle(lang)
        if local_mode == 'one':      # local candidates move one randomly chosen surface only
            pick = rng.integers(0, L, nb)
            keep = np.arange(L)[None, :] != pick[:, None]
            lpos = np.where(keep[:, :, None], center['pos'][None], lpos); lang = np.where(keep, center['angles'][None], lang)
        pos[:] = np.where(local[:, None, None], lpos, gpos); ang[:] = np.where(local[:, None], lang, gang)
        if fix_angles:      # ablation without rotation: every azimuth stays at the prescribed one
            ang[:] = net['angles0'][None]
        attempt += nb
        ok = np.all(np.abs(ang) <= math.pi / 2, axis=1)
        ok &= np.all(np.linalg.norm(pos - net['p0'][None], axis=2) <= Dv[None] + 1e-7, axis=1)
        if ms:       # random candidates stay within their radius R
            ok &= np.all(local[:, None] | (np.linalg.norm(pos - net['p0'][None], axis=2) <= Dg + 1e-7), axis=1)
        ok &= np.all((pos[:, :, 0] >= -1e-7) & (pos[:, :, 0] <= p['X'] + 1e-7) & (pos[:, :, 1] >= -1e-7) &
                     (pos[:, :, 1] <= p['Y'] + 1e-7) & (pos[:, :, 2] >= p['zmin'] - 1e-7) & (pos[:, :, 2] <= p['zmax'] + 1e-7), axis=1)
        for l in range(L):
            for h in range(l + 1, L):
                ok &= np.linalg.norm(pos[:, l] - pos[:, h], axis=1) >= p['dmin'] - 1e-7
            ok &= np.min(np.linalg.norm(pos[:, l][:, None, :] - fixed[None], axis=2), axis=1) >= p['d0'] - 1e-7
        for i in np.nonzero(ok)[0]:
            pose = H_.pose_from(pos[i], ang[i]); pose["kind"] = "local" if local[i] else "random"
            f = features([pose], p)[0]
            allf = np.vstack([seen] + ([np.array(feats)] if feats else []))
            if allf.size and np.min(np.max(np.abs(allf - f), axis=1)) < 1e-10: continue
            out.append(pose); feats.append(f)
            if len(out) >= count: break
    return out


def gp_fit_trend(X, y, H, scales=(0.15, 0.3, 0.6), nugget=1e-6, kern='iso', fixed_amp=False):
    """GP with the prior mean H @ beta (universal kriging): beta by generalized least squares and the signal
    variance by maximum likelihood, for every length scale; the length scale with the largest likelihood is kept."""
    off = np.mean(y); sc = np.std(y, ddof=1) if len(y) > 1 else 1.0
    if not np.isfinite(sc) or sc < 1e-8: sc = 1.0
    yn = (y - off) / sc; best = None; n = len(y)
    for ell in scales:
        Kx = kernel(X, X, ell, kern); jit = nugget
        for _ in range(8):
            try:
                Lc = np.linalg.cholesky(Kx + jit * np.eye(n)); break
            except np.linalg.LinAlgError:
                jit *= 10
        KiH = np.linalg.solve(Lc.T, np.linalg.solve(Lc, H)); Kiy = np.linalg.solve(Lc.T, np.linalg.solve(Lc, yn))
        Am = H.T @ KiH + 1e-10 * np.eye(H.shape[1]); beta = np.linalg.solve(Am, H.T @ Kiy)
        Kir = Kiy - KiH @ beta; s2 = max(float((yn - H @ beta) @ Kir) / n, 1e-12)
        obj = 0.5 * n * np.log(s2) + np.sum(np.log(np.diag(Lc)))
        if fixed_amp:      # signal variance = sample variance of F (as the constant-mean GP)
            obj = 0.5 * float((yn - H @ beta) @ Kir) + np.sum(np.log(np.diag(Lc))); s2 = 1.0
        if best is None or obj < best[0]:
            best = (obj, dict(X=X, off=off, sc=sc, s2=s2, ell=ell, L=Lc, Kir=Kir, KiH=KiH, beta=beta, Am=Am, kern=kern))
    return best[1]


def gp_predict_trend(m, Z, Hz, beta_var=True):
    """Posterior mean and standard deviation, including the uncertainty of beta."""
    Kz = kernel(m['X'], Z, m['ell'], m['kern']); v = np.linalg.solve(m['L'], Kz)
    mu = m['off'] + m['sc'] * (Hz @ m['beta'] + Kz.T @ m['Kir'])
    u = Hz - Kz.T @ m['KiH']
    var = m['s2'] * np.maximum(1e-12, 1 - np.sum(v ** 2, 0) + (np.sum(u * np.linalg.solve(m['Am'], u.T).T, 1) if beta_var else 0.0))
    return mu, m['sc'] * np.sqrt(var)


def scale_basis(poses, net, r0=R0):
    """Regressors of the GP mean: 1, s and s^2, s = log(1 + largest displacement / r0) (normalized by log(1 + 100/r0))."""
    s = np.array([math.log1p(float(np.max(np.linalg.norm(ps['pos'] - net['p0'], axis=1))) / r0) for ps in poses]) / math.log1p(100.0 / r0)
    return np.stack([np.ones_like(s), s, s ** 2], 1)


def proposals_nested(net, p, rng, n_raw, center, seen, use_local, local_scale=0.15, ang_scale=None, fix_angles=False, r0=1.0, rmax=100.0):
    """Candidates drawn from a distribution that does not depend on Dmax, which only removes the inadmissible ones:
    n_raw raw draws (alternately random and local), each with a displacement scale R log-uniform on [r0, rmax]; a
    random draw places every surface uniformly in its ball of radius R (within the area), a local draw perturbs
    center with standard deviations local_scale times the half widths of the R-box. The candidates admissible for a
    smaller Dmax are a subset of those for a larger Dmax (same random numbers)."""
    L = p['L']; Dv = np.asarray(p['Dmax'], float); p0 = net['p0']
    seen = np.zeros((0, 4 * L)) if seen is None or len(seen) == 0 else np.asarray(seen)
    fixed = np.vstack([net['BS'], net['UE']])
    nb = n_raw; idx = 1 + np.arange(nb)
    local = use_local & (idx % 2 == 0)
    R = r0 * (rmax / r0) ** rng.random((nb, 1))
    Rg = np.broadcast_to(R, (nb, L))
    glo = np.stack([np.maximum(0.0, p0[None, :, 0] - Rg), np.maximum(0.0, p0[None, :, 1] - Rg), np.maximum(p['zmin'], p0[None, :, 2] - Rg)], axis=2)
    ghi = np.stack([np.minimum(p['X'], p0[None, :, 0] + Rg), np.minimum(p['Y'], p0[None, :, 1] + Rg), np.minimum(p['zmax'], p0[None, :, 2] + Rg)], axis=2)
    gpos = glo + (ghi - glo) * rng.random((nb, L, 3))
    lpos = center['pos'][None] + local_scale * (ghi - glo) / 2 * rng.standard_normal((nb, L, 3))
    gang = math.pi * rng.random((nb, L)) - math.pi / 2
    lang = center['angles'][None] + (local_scale if ang_scale is None else ang_scale) * math.pi * rng.standard_normal((nb, L))
    pos = np.where(local[:, None, None], lpos, gpos); ang = np.where(local[:, None], lang, gang)
    pos = np.where((Dv <= 0)[None, :, None], p0[None], pos)          # surfaces that may not move
    if fix_angles: ang = np.broadcast_to(net['angles0'][None], ang.shape).copy()
    dist = np.linalg.norm(pos - p0[None], axis=2)
    ok = np.all(np.abs(ang) <= math.pi / 2, axis=1)
    ok &= np.all(local[:, None] | (dist <= Rg + 1e-7) | (Dv <= 0)[None], axis=1)      # random draws stay within their radius
    ok &= np.all(dist <= Dv[None] + 1e-7, axis=1)
    ok &= np.all((pos[:, :, 0] >= -1e-7) & (pos[:, :, 0] <= p['X'] + 1e-7) & (pos[:, :, 1] >= -1e-7) &
                 (pos[:, :, 1] <= p['Y'] + 1e-7) & (pos[:, :, 2] >= p['zmin'] - 1e-7) & (pos[:, :, 2] <= p['zmax'] + 1e-7), axis=1)
    for l in range(L):
        for h in range(l + 1, L):
            ok &= np.linalg.norm(pos[:, l] - pos[:, h], axis=1) >= p['dmin'] - 1e-7
        ok &= np.min(np.linalg.norm(pos[:, l][:, None, :] - fixed[None], axis=2), axis=1) >= p['d0'] - 1e-7
    out = []; feats = []
    for i in np.nonzero(ok)[0]:
        pose = H_.pose_from(pos[i], ang[i])
        f = features([pose], p)[0]
        allf = np.vstack([seen] + ([np.array(feats)] if feats else []))
        if allf.size and np.min(np.max(np.abs(allf - f), axis=1)) < 1e-10: continue
        out.append(pose); feats.append(f)
    return out


def initial_octaves(net, p, seed, n_init, first_pose=None, fix_angles=False, r0=1.0):
    """delta0 and n_init - 1 random deployments at the scales r0, 2 r0, 4 r0, ... (capped at Dmax): every surface
    uniformly in the ball of that radius, uniform azimuths. Identical for every Dmax >= 2^(n_init - 2) r0."""
    base = H_.pose_from(net['p0'], net['angles0'])
    first = base if first_pose is None else first_pose
    rng0 = np.random.default_rng(seed + 200000); init = [first]
    Dv = np.asarray(p['Dmax'], float)
    for k in range(n_init - 1):
        q = dict(p); q['Dmax'] = np.minimum(Dv, r0 * 2.0 ** k)
        init += proposals_fast(net, q, rng0, 1, base, features(init, p), False, sampler='box', fix_angles=fix_angles)
    return init


def initial_rotations(net, p, seed, n_init, first_pose=None):
    """delta0 and n_init - 1 random rotations of it (its positions with uniform azimuths); the same for every Dmax."""
    base = H_.pose_from(net['p0'], net['angles0'])
    first = base if first_pose is None else first_pose
    rng0 = np.random.default_rng(seed + 200000)
    return [first] + [H_.pose_from(net['p0'].copy(), math.pi * rng0.random(p['L']) - math.pi / 2) for _ in range(n_init - 1)]


def initial_design(net, p, seed, n_init, first_pose=None, sampler='box', radial='volume', fix_angles=False, scale_min=R0,
                   init_strata=True):
    """The initial design of bayes_opt_fast: the first deployment (the prescribed one) and n_init - 1 random
    candidates; with init_strata, the k-th has its scale in the k-th of n_init - 1 equal parts of the logarithmic
    scale range."""
    base = H_.pose_from(net['p0'], net['angles0'])
    first = base if first_pose is None else first_pose      # first_pose: nested search (diagnostic only)
    rng0 = np.random.default_rng(seed + 200000)
    if init_strata and scale_min is not None:
        init = [first]
        for i in range(n_init - 1):
            init += proposals_fast(net, p, rng0, 1, base, features(init, p), False, radial=radial, sampler=sampler, fix_angles=fix_angles,
                                   scale_min=scale_min, u_lim=(i / (n_init - 1), (i + 1) / (n_init - 1)))
        return init
    return [first] + proposals_fast(net, p, rng0, n_init - 1, base, features([first], p), False, radial=radial, sampler=sampler,
                                    fix_angles=fix_angles, scale_min=scale_min)


def bayes_opt_fast(evaluate, net, p, budget, seed, n_init=5, n_cand=256, log=None, replay=None, local_scale=0.15, ang_scale=None,
                   radial='volume', first_pose=None, kern='iso', local_mode='all', sampler='box', fix_angles=False, scale_min=R0,
                   trend=R0, init_strata=True, scale_mix=0.0, init_scale_min='same', fixed_amp=False, nested=None, init_kind=None, beta_var=True, pa_grid=None, warp=None, info=None,
                   grow=None, gp_scales=(0.15, 0.3, 0.6), window=None, design=None):
    """The deployment search (v17, scale-aware): candidates with log-uniform displacement scales in [scale_min, Dmax]
    and local steps at the scale of the candidate, a GP whose prior mean is quadratic in log(1 + largest
    displacement / trend), and a stratified initial design. scale_min=None, trend=None, init_strata=False and
    sampler='ball' give the search of v16 (uniform candidates, constant prior mean).
    replay: merits of the first evaluations of an earlier run with the same seed; the search is deterministic, so
    these evaluations are taken from the record instead of being recomputed (the best one is re-evaluated if it is
    selected).

    Experimental switches of the design study (all off by default = design T; see FINDINGS.md and harness/variants.py):
      module flag PERIODIC   azimuth modulo pi (GP feature on a circle, wrapped local steps)          variant P
      window=w               candidates within w of the incumbent (D_max only removes); init within w   variant PW5
      grow=dict(rho0, stall) expanding search radius                                                   variant G2s5
      gp_scales=(...)        length-scale grid of the GP                                               variant PE
      nested=dict(n_raw)     D_max-independent candidate stream filtered by D_max                      variant Foct
      init_kind='octave'     D_max-independent initial design (scales r0, 2 r0, 4 r0, 8 r0)          variants Foct, PO
      pa_grid, warp, scale_mix, fixed_amp, beta_var, init_scale_min: earlier experiments (TPA, TW, M50, TA, ...)
      info=dict()            receives per-iteration diagnostics ('ell', 'rho')
      design=dict(...)       hands over to the search of physbo.py (design round v5) with these options
    Candidates carry pose['kind'] = 'local' or 'random' (diagnostic only)."""
    if design is None and SEARCH is not None:
        import physbo
        design = physbo.DESIGN_R if SEARCH == 'R' else SEARCH
    if design is not None:
        import physbo
        c_d = time.process_time()
        out = physbo.search(evaluate, net, p, budget, seed, n_init=n_init, log=log, fix_angles=fix_angles, sampler=sampler,
                            radial=radial, ang_scale=ang_scale, local_scale=local_scale, first_pose=first_pose, info=info,
                            **design)
        if replay is not None:      # replay is not supported by this search: every deployment is evaluated (all CPU is new)
            return out + (time.process_time() - c_d,)
        return out
    p_all = p; Dv = np.asarray(p['Dmax'], float); rho = None
    if grow is not None:      # expanding search radius: rho0, doubled after `stall` evaluations without improvement
        rho = grow.get('rho0', 2.0); fails = 0     # (or, with `every`, doubled every `every` evaluations)
        p = dict(p_all, Dmax=np.minimum(Dv, rho))
    if window is not None:      # the initial design within the window around the initial deployment
        init = initial_design(net, dict(p, Dmax=np.minimum(Dv, window)), seed, n_init, first_pose=first_pose, sampler=sampler, radial=radial,
                              fix_angles=fix_angles, scale_min=scale_min, init_strata=init_strata)
    elif init_kind == 'octave':
        init = initial_octaves(net, p, seed, n_init, first_pose=first_pose, fix_angles=fix_angles, r0=scale_min or R0)
    elif init_kind == 'rotate':
        init = initial_rotations(net, p, seed, n_init, first_pose=first_pose)
    else:
        init = initial_design(net, p, seed, n_init, first_pose=first_pose, sampler=sampler, radial=radial, fix_angles=fix_angles,
                              scale_min=scale_min if init_scale_min == 'same' else init_scale_min, init_strata=init_strata)
    X = []; y = []; recs = []
    nrep = len(replay) if replay is not None else 0
    t0 = time.time(); c_new = 0.0
    for q in range(budget):
        c_q = time.process_time()
        if q < min(len(init), budget):
            pose = init[q]
        else:
            ib = int(np.argmax(y)); center = recs[ib]['pose']
            rng = np.random.default_rng([seed + 300000, q])
            if nested is not None:      # candidates of a fixed distribution, of which Dmax only removes the inadmissible ones
                cands = proposals_nested(net, p, rng, nested.get('n_raw', 512), center, np.array(X), True, local_scale=local_scale,
                                         ang_scale=ang_scale, fix_angles=fix_angles, r0=scale_min or R0, rmax=nested.get('rmax', 100.0))
            elif window is not None:     # candidates around the incumbent, within `window` of it (D_max only removes)
                net_w = dict(net, p0=center['pos']); p_w = dict(p, Dmax=np.where(Dv > 0, window, 0.0))
                cands = proposals_fast(net_w, p_w, rng, n_cand, center, np.array(X), True, local_scale=local_scale, ang_scale=ang_scale,
                                       radial=radial, local_mode=local_mode, sampler=sampler, fix_angles=fix_angles, scale_min=scale_min,
                                       scale_local=scale_min is not None, scale_mix=scale_mix)
                cands = [c for c in cands if np.all(np.linalg.norm(c['pos'] - net['p0'], axis=1) <= Dv + 1e-7)]
            else:
              cands = proposals_fast(net, p, rng, n_cand, center, np.array(X), True, local_scale=local_scale, ang_scale=ang_scale, radial=radial,
                                   local_mode=local_mode, sampler=sampler, fix_angles=fix_angles, scale_min=scale_min,
                                   scale_local=scale_min is not None, scale_mix=scale_mix)
            Z = features(cands, p)
            if trend is not None:
                if warp is not None:         # features measured from the best deployment on a logarithmic scale
                    Z = features_w(cands, center, **warp)
                    m = gp_fit_trend(features_w([r['pose'] for r in recs], center, **warp), np.array(y), scale_basis([r['pose'] for r in recs], net, trend), kern=kern)
                elif pa_grid is not None:      # separate length scales for the positions and the azimuths
                    Z = features_m(cands, p)
                    m = gp_fit_trend(features_m([r['pose'] for r in recs], p), np.array(y), scale_basis([r['pose'] for r in recs], net, trend), kern='pa', fixed_amp=fixed_amp,
                                     scales=[(a, b) for a in pa_grid[0] for b in pa_grid[1]])
                else:
                    m = gp_fit_trend(np.array(X), np.array(y), scale_basis([r['pose'] for r in recs], net, trend), kern=kern, fixed_amp=fixed_amp, scales=gp_scales)
                mu, sd = gp_predict_trend(m, Z, scale_basis(cands, net, trend), beta_var=beta_var)
            else:
                m = gp_fit(np.array(X), np.array(y), kern=kern)
                mu, sd = gp_predict(m, Z)
            if info is not None: info.setdefault('ell', []).append(m['ell'])
            z = (mu - np.max(y)) / sd
            pose = cands[int(np.argmax(log_ei(z, sd)))]
        if q < nrep:
            r = dict(merit=replay[q], replayed=True)
        else:
            r = evaluate(pose)
        r['pose'] = pose
        X.append(features([pose], p)[0]); y.append(r['merit']); recs.append(r)
        if rho is not None and q >= n_init - 1:
            if grow.get('every'):
                rho = grow['rho0'] * 2.0 ** ((q - n_init + 2) / grow['every'])
            else:
                fails = 0 if (q >= n_init and y[-1] > max(y[:-1])) else fails + (q >= n_init)
                if fails >= grow['stall']: rho *= 2.0; fails = 0
            p = dict(p_all, Dmax=np.minimum(Dv, rho))
            if info is not None: info.setdefault('rho', []).append(float(rho))
        if q >= nrep: c_new += time.process_time() - c_q
        if log and q >= nrep: log(f'  q{q + 1:2d}: merit {r["merit"]:9.4f} rate {r["rate"]:8.4f} def {r["deficit"]:.3g} '
                                  f'calls {r["calls"]} ({r["seconds"]:.2f}s)')
    ib = int(np.argmax(y))
    if recs[ib].get('replayed'):
        r = evaluate(recs[ib]['pose']); r['pose'] = recs[ib]['pose']
        assert abs(r['merit'] - y[ib]) <= 1e-6 * max(1.0, abs(y[ib])), (r['merit'], y[ib])
        recs[ib] = r
    if replay is not None:
        return recs, ib, time.time() - t0, c_new
    return recs, ib, time.time() - t0
