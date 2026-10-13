"""Physics-informed Bayesian deployment search: design R of the design round v5 (DESIGN.md).

Rate score. For a deployment delta, the potential channel gain of UE k on BS b is
    G_kb(delta) = ||h_d,kb||^2 + N * (M * sum_l |s_kbl(delta)|)^2
(direct channel plus the cascades of all surfaces added coherently with all M elements aligned; |s_kbl| is the
amplitude of the single-hop LoS cascade: path loss of both hops and the element pattern |cos|^q of both angles, as in
hrs.channel_terms). Every served stream j is limited by its weakest decoder, so the score is the mean over the served
streams of the high-SNR capacity proxy of that decoder:
    S(delta) = (1/|J|) sum_{j in J} log2( min_{k in D_j} sum_{b carries j} G_kb(delta) ).
S needs no channel phases and no optimization; it is computed for thousands of candidates in milliseconds.

Search (design R = search(**DESIGN_R)): BO with a GP whose prior mean is beta0 + beta1 S (beta by GLS after every
evaluation), Matern-5/2 covariance with one length scale for x/y, z and the azimuth and a nugget (ML on grids,
coordinate-wise from the previous choice), candidates = random deployments + re-draws of one surface around its
current position, log EI refined by stochastic hill climbing (the trimmed design R; round v5's R also drew local
steps proportional to the length scales as candidates).

The other options of search() are the experimental switches of the design study (variants PB*, Rx, Rk, ... in
harness/variants.py); they are kept so that every recorded variant can be re-run.
"""
import math, time
import numpy as np
from scipy.linalg import solve_triangular
import hrs as H_
import algo as A

_TRI = False      # linalg='tri': triangular solves with the Cholesky factor (set by search(); development variants: general solves)


def _lsolve(Lc, B):
    """Solves Lc X = B for the lower triangular Cholesky factor Lc."""
    return solve_triangular(Lc, B, lower=True, check_finite=False) if _TRI else np.linalg.solve(Lc, B)


def _ltsolve(Lc, B):
    """Solves Lc^T X = B."""
    return solve_triangular(Lc, B, lower=True, trans='T', check_finite=False) if _TRI else np.linalg.solve(Lc.T, B)


# the options of design R, trimmed (2026-10-12): the search of every caller (algo.SEARCH = 'R'). Candidates: 512 random
# deployments and 512 one-surface re-draws around the best deployment; the expected improvement is refined by hill
# climbing from the 4 best candidates. (Design R of round v5 also had 512 local-step candidates, n_loc=512; removing
# them changed the rate by +0.04 +- 0.37 at Q = 50 and -0.33 +- 0.33 at Q = 100 on 44 paired pilot runs.)
DESIGN_R = dict(mean='phys', kern='ardcw', local='ard', n_glob=512, n_loc=0, n_block=512, block_center='inc',
                acq_opt=dict(starts=4, iters=10, n=32, s=(0.02, 0.5), batch=True), linalg='tri')


# --------------------------------------------------------------------------- rate score
def design_of(evaluate):
    """The stream design served by the evaluator (AOEvaluator, CachedEvaluator, FixedCoefEvaluator, ...)."""
    for obj in (evaluate, getattr(evaluate, 'ev', None)):
        d = getattr(obj, 'd', None)
        if d is not None:
            return d
    return None


class RateScore:
    """S(delta) for many deployments at once (positions (n, L, 3), azimuths (n, L))."""

    def __init__(self, net, p, d=None):
        self.net, self.p, self.d = net, p, d
        BS = np.asarray(net['BS'], float); UE = np.asarray(net['UE'], float)
        self.BS, self.UE = BS, UE
        dd = np.linalg.norm(UE[:, None, :] - BS[None, :, :], axis=-1)                    # (K, B)
        self.hd2 = p['N'] * p['PL0'] / p['noise'] * dd ** (-p['alpha'])                   # ||h_d,kb||^2
        self.c0 = p['gainScale'] * p['PL0'] / math.sqrt(p['noise'])
        K, B = len(UE), len(BS)
        if d is None:     # no design known: every UE decodes one stream from both BSs
            self.streams = [(np.array([k]), np.ones(B, bool)) for k in range(K)]
        else:
            self.streams = []
            for j in d.served:
                carry = np.ones(B, bool) if getattr(d, 'bsmask', None) is None else np.asarray(d.bsmask)[:, d.col[j]]
                self.streams.append((np.array(d.decoders[j]), carry))

    def gains(self, pos, ang):
        """Potential gains G (n, K, B)."""
        p = self.p; a = p['alpha'] / 2; q = p['q']
        nrm = np.stack([-np.sin(ang), np.cos(ang), np.zeros_like(ang)], axis=-1)             # (n, L, 3) normals
        rb = self.BS[None, :, None, :] - pos[:, None, :, :]                                     # (n, B, L, 3)
        db = np.linalg.norm(rb, axis=-1); cb = np.abs(np.sum(rb * nrm[:, None], -1)) / db
        ru = self.UE[None, :, None, :] - pos[:, None, :, :]                                     # (n, K, L, 3)
        du = np.linalg.norm(ru, axis=-1); cu = np.abs(np.sum(ru * nrm[:, None], -1)) / du
        amp = self.c0 * (db[:, None] ** -a) * (cb[:, None] ** q) * (du[:, :, None] ** -a) * (cu[:, :, None] ** q)   # (n, K, B, L)
        return self.hd2[None] + p['N'] * (p['M'] * np.sum(amp, axis=3)) ** 2

    def __call__(self, poses=None, pos=None, ang=None):
        if poses is not None:
            pos = np.array([x['pos'] for x in poses], float); ang = np.array([x['angles'] for x in poses], float)
        G = self.gains(pos, ang)
        S = np.zeros(len(pos))
        for dec, carry in self.streams:
            S += np.log2(np.min(np.sum(G[:, dec][:, :, carry], axis=2), axis=1))
        return S / len(self.streams)


# --------------------------------------------------------------------------- features and GP
def feats(pos, ang, p):
    """Per surface [x/X, y/Y, (z - zmin)/(zmax - zmin), cos(2 phi)/(2 pi), sin(2 phi)/(2 pi)] (design P's features)."""
    a = 2.0 * ang
    blk = np.stack([pos[..., 0] / p['X'], pos[..., 1] / p['Y'], (pos[..., 2] - p['zmin']) / (p['zmax'] - p['zmin']),
                    np.cos(a) / (2 * math.pi), np.sin(a) / (2 * math.pi)], axis=-1)
    return blk.reshape(len(pos), -1)


def matern(X, Z, w):
    Xs, Zs = X * w, Z * w
    sq = np.maximum(0, np.sum(Xs ** 2, 1)[:, None] + np.sum(Zs ** 2, 1)[None, :] - 2 * Xs @ Zs.T) / X.shape[1]
    r = np.sqrt(5 * sq)
    return (1 + r + r ** 2 / 3) * np.exp(-r)


def ard_weights(ell, L):
    """ell = (l_xy, l_z, l_az) -> feature weights 1/l per feature."""
    lxy, lz, la = ell
    return np.tile([1 / lxy, 1 / lxy, 1 / lz, 1 / la, 1 / la], L)


def gp_fit(X, y, H, grid, L, with_obj=False, extra=None):
    """GP with prior mean H beta (GLS) and Matern-5/2 kernel with feature weights; (ell, nugget) from the grid by
    maximum likelihood (signal variance profiled out). extra: additional noise variance of single evaluations."""
    off = float(np.mean(y)); sc = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0
    if not np.isfinite(sc) or sc < 1e-8: sc = 1.0
    yn = (y - off) / sc; n = len(y); best = None
    for ell, nug in grid:
        w = ard_weights(ell, L)
        Kx = matern(X, X, w) + nug * np.eye(n); jit = 1e-8
        if extra is not None: Kx = Kx + np.diag(extra)
        for _ in range(8):
            try:
                Lc = np.linalg.cholesky(Kx + jit * np.eye(n)); break
            except np.linalg.LinAlgError:
                jit *= 10
        KiH = _ltsolve(Lc, _lsolve(Lc, H)); Kiy = _ltsolve(Lc, _lsolve(Lc, yn))
        Am = H.T @ KiH + 1e-10 * np.eye(H.shape[1]); beta = np.linalg.solve(Am, H.T @ Kiy)
        Kir = Kiy - KiH @ beta; s2 = max(float((yn - H @ beta) @ Kir) / n, 1e-12)
        obj = 0.5 * n * np.log(s2) + np.sum(np.log(np.diag(Lc)))
        if best is None or obj < best[0]:
            best = (obj, dict(X=X, off=off, sc=sc, s2=s2, ell=ell, nug=nug, w=w, L=Lc, Kir=Kir, KiH=KiH, beta=beta, Am=Am))
    return best if with_obj else best[1]


def gp_predict(m, Z, Hz):
    """Posterior mean and standard deviation of the latent merit (uncertainty of beta included)."""
    Kz = matern(m['X'], Z, m['w']); v = _lsolve(m['L'], Kz)
    mu = m['off'] + m['sc'] * (Hz @ m['beta'] + Kz.T @ m['Kir'])
    u = Hz - Kz.T @ m['KiH']
    var = m['s2'] * np.maximum(1e-12, 1 - np.sum(v ** 2, 0) + np.sum(u * np.linalg.solve(m['Am'], u.T).T, 1))
    return mu, m['sc'] * np.sqrt(var)


# ---- variant with the score in the covariance (mean='physk'): k = Matern + tau2 s~ s~' + nugget, s~ the standardized
# score; tau2 (the prior variance of the score's slope, relative to the GP variance) by ML like the length scales
TAU2 = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def gp_fit_s(X, y, s, ell, nug, tau2, L):
    """GP with constant mean (GLS) and covariance Matern + tau2 s s' + nug I; returns (neg. profile log-lik, model)."""
    off = float(np.mean(y)); sc = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0
    if not np.isfinite(sc) or sc < 1e-8: sc = 1.0
    yn = (y - off) / sc; n = len(y); H = np.ones((n, 1))
    w = ard_weights(ell, L)
    Kx = matern(X, X, w) + nug * np.eye(n) + tau2 * np.outer(s, s); jit = 1e-8
    for _ in range(8):
        try:
            Lc = np.linalg.cholesky(Kx + jit * np.eye(n)); break
        except np.linalg.LinAlgError:
            jit *= 10
    KiH = _ltsolve(Lc, _lsolve(Lc, H)); Kiy = _ltsolve(Lc, _lsolve(Lc, yn))
    Am = H.T @ KiH + 1e-10 * np.eye(1); beta = np.linalg.solve(Am, H.T @ Kiy)
    Kir = Kiy - KiH @ beta; s2 = max(float((yn - H @ beta) @ Kir) / n, 1e-12)
    obj = 0.5 * n * np.log(s2) + np.sum(np.log(np.diag(Lc)))
    return obj, dict(X=X, off=off, sc=sc, s2=s2, ell=ell, nug=nug, tau2=tau2, w=w, L=Lc, Kir=Kir, KiH=KiH, beta=beta, Am=Am, s=s)


def gp_predict_s(m, Z, sz):
    """Posterior mean and sd of the latent merit at Z with standardized scores sz."""
    Kz = matern(m['X'], Z, m['w']) + m['tau2'] * np.outer(m['s'], sz); v = _lsolve(m['L'], Kz)
    mu = m['off'] + m['sc'] * (m['beta'][0] + Kz.T @ m['Kir'])
    u = 1.0 - Kz.T @ m['KiH'][:, 0]
    var = m['s2'] * np.maximum(1e-12, 1 + m['tau2'] * sz ** 2 - np.sum(v ** 2, 0) + u ** 2 / m['Am'][0, 0])
    return mu, m['sc'] * np.sqrt(var)


def gp_fit_cw_s(X, y, s, L, start=None, axes=None):
    """gp_fit_cw with the fifth hyperparameter tau2."""
    axes = AXES + (TAU2,) if axes is None else axes
    cache = {}

    def fit(idx):
        if idx not in cache:
            cache[idx] = gp_fit_s(X, y, s, (axes[0][idx[0]], axes[1][idx[1]], axes[2][idx[2]]), axes[3][idx[3]], axes[4][idx[4]], L)
        return cache[idx]
    if start is None:
        import itertools
        idx = min(itertools.product(*[range(len(a)) for a in axes]), key=lambda i: fit(i)[0])
    else:
        idx = tuple(start)
        for _ in range(20):
            moved = False
            for a in range(len(axes)):
                for st in (-1, 1):
                    j = list(idx); j[a] += st
                    if 0 <= j[a] < len(axes[a]) and fit(tuple(j))[0] < fit(idx)[0] - 1e-12:
                        idx = tuple(j); moved = True
            if not moved:
                break
    return fit(idx)[1], idx


ISO_GRID = [((e, e, e), 1e-6) for e in (0.15, 0.3, 0.6)]          # design P's kernel

# grids of the ARD length scales (x/y, z, azimuth features) and of the nugget (relative to the signal variance)
AXES = ((0.0125, 0.025, 0.05, 0.1, 0.2, 0.4), (0.05, 0.1, 0.2, 0.4, 0.8), (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4),
        (1e-6, 1e-3, 1e-2, 0.03, 0.1, 0.3))


def loo_z(m, H):
    """Standardized leave-one-out residuals of the evaluations under the fitted GP (trend coefficients fixed)."""
    Ki = _ltsolve(m['L'], _lsolve(m['L'], np.eye(len(m['L']))))
    d = np.diag(Ki)
    return m['Kir'] / d / np.sqrt(m['s2'] / d)


def gp_fit_cw(X, y, H, L, start=None, axes=AXES, extra=None):
    """Maximum likelihood over the grid AXES: the full grid when start is None, else coordinate-wise moves to the
    neighbouring grid values, from start, while the likelihood increases. Returns (model, grid index)."""
    cache = {}

    def fit(idx):
        if idx not in cache:
            cache[idx] = gp_fit(X, y, H, [((axes[0][idx[0]], axes[1][idx[1]], axes[2][idx[2]]), axes[3][idx[3]])], L, with_obj=True, extra=extra)
        return cache[idx]
    if start is None:
        import itertools
        idx = min(itertools.product(*[range(len(a)) for a in axes]), key=lambda i: fit(i)[0])
    else:
        idx = tuple(start)
        for _ in range(20):
            moved = False
            for a in range(len(axes)):
                for st in (-1, 1):
                    j = list(idx); j[a] += st
                    if 0 <= j[a] < len(axes[a]) and fit(tuple(j))[0] < fit(idx)[0] - 1e-12:
                        idx = tuple(j); moved = True
            if not moved:
                break
    return fit(idx)[1], idx


def ard_grid(nugs=(1e-6,)):
    return [((a, b, c), g) for a in (0.03, 0.06, 0.12, 0.25) for b in (0.1, 0.2, 0.4) for c in (0.05, 0.1, 0.2, 0.4) for g in nugs]


# --------------------------------------------------------------------------- candidates
def admissible(pos, ang, net, p, local=None, R=None):
    """Constraint mask (the checks of algo.proposals_fast)."""
    Dv = np.asarray(p['Dmax'], float); L = p['L']
    fixed = np.vstack([net['BS'], net['UE']])
    ok = np.all(np.abs(ang) <= math.pi / 2, axis=1)
    dist = np.linalg.norm(pos - net['p0'][None], axis=2)
    ok &= np.all(dist <= Dv[None] + 1e-7, axis=1)
    if R is not None:
        ok &= np.all(local[:, None] | (dist <= R + 1e-7), axis=1)
    ok &= np.all((pos[:, :, 0] >= -1e-7) & (pos[:, :, 0] <= p['X'] + 1e-7) & (pos[:, :, 1] >= -1e-7) &
                 (pos[:, :, 1] <= p['Y'] + 1e-7) & (pos[:, :, 2] >= p['zmin'] - 1e-7) & (pos[:, :, 2] <= p['zmax'] + 1e-7), axis=1)
    for l in range(L):
        for h in range(l + 1, L):
            ok &= np.linalg.norm(pos[:, l] - pos[:, h], axis=1) >= p['dmin'] - 1e-7
        ok &= np.min(np.linalg.norm(pos[:, l][:, None, :] - fixed[None], axis=2), axis=1) >= p['d0'] - 1e-7
    return ok


def global_cands(net, p, rng, count, r0=A.R0, fix_angles=False):
    """Random admissible deployments with a log-uniform displacement scale R in [r0, Dmax] shared by the surfaces;
    every surface uniform in the admissible part of its R-box (rejecting outside the R-ball), uniform azimuth
    (design T's random candidates)."""
    L = p['L']; Dv = np.asarray(p['Dmax'], float); p0 = net['p0']
    outp, outa = [], []; got = 0; tries = 0
    while got < count and tries < 50:
        tries += 1; nb = 4 * count
        uu = rng.random((nb, 1))
        Rg = Dv[None, :] * np.minimum(1.0, r0 / np.maximum(Dv, 1e-300))[None, :] ** (1.0 - uu)
        glo = np.stack([np.maximum(0.0, p0[None, :, 0] - Rg), np.maximum(0.0, p0[None, :, 1] - Rg), np.maximum(p['zmin'], p0[None, :, 2] - Rg)], axis=2)
        ghi = np.stack([np.minimum(p['X'], p0[None, :, 0] + Rg), np.minimum(p['Y'], p0[None, :, 1] + Rg), np.minimum(p['zmax'], p0[None, :, 2] + Rg)], axis=2)
        pos = glo + (ghi - glo) * rng.random((nb, L, 3))
        ang = math.pi * rng.random((nb, L)) - math.pi / 2
        if fix_angles: ang[:] = net['angles0'][None]
        ok = admissible(pos, ang, net, p, np.zeros(nb, bool), Rg)
        outp.append(pos[ok]); outa.append(ang[ok]); got += int(ok.sum())
    return np.concatenate(outp)[:count], np.concatenate(outa)[:count]


def block_cands(net, p, rng, count, center, r0=A.R0, fix_angles=False, around='p0'):
    """Re-draws of one surface: a randomly chosen surface is placed as in a random candidate (log-uniform scale R in
    [r0, Dmax], uniform in the admissible part of its R-ball, uniform azimuth), around its initial position
    (around='p0') or around its position in center (around='inc', design R); the other surfaces stay at center."""
    L = p['L']; Dv = np.asarray(p['Dmax'], float); p0 = net['p0']
    c0 = p0 if around == 'p0' else center['pos']
    outp, outa = [], []; got = 0; tries = 0
    while got < count and tries < 50:
        tries += 1; nb = 4 * count
        uu = rng.random((nb, 1))
        Rg = Dv[None, :] * np.minimum(1.0, r0 / np.maximum(Dv, 1e-300))[None, :] ** (1.0 - uu)
        glo = np.stack([np.maximum(0.0, c0[None, :, 0] - Rg), np.maximum(0.0, c0[None, :, 1] - Rg), np.maximum(p['zmin'], c0[None, :, 2] - Rg)], axis=2)
        ghi = np.stack([np.minimum(p['X'], c0[None, :, 0] + Rg), np.minimum(p['Y'], c0[None, :, 1] + Rg), np.minimum(p['zmax'], c0[None, :, 2] + Rg)], axis=2)
        gpos = glo + (ghi - glo) * rng.random((nb, L, 3))
        gang = math.pi * rng.random((nb, L)) - math.pi / 2
        pick = rng.integers(0, L, nb); sel = np.arange(L)[None, :] == pick[:, None]
        pos = np.where(sel[:, :, None], gpos, center['pos'][None]); ang = np.where(sel, gang, center['angles'][None])
        if fix_angles: ang[:] = net['angles0'][None]
        dist = np.linalg.norm(pos - c0[None], axis=2)
        ok = admissible(pos, ang, net, p) & np.all(~sel | (dist <= Rg + 1e-7), axis=1)
        outp.append(pos[ok]); outa.append(ang[ok]); got += int(ok.sum())
    return np.concatenate(outp)[:count], np.concatenate(outa)[:count]


def project(pos, net, p):
    """Positions projected onto the box (area, heights) and then radially onto the D_max ball around the initial
    position (which keeps them in the box, since the box is convex and contains the initial position)."""
    lo = np.array([0.0, 0.0, p['zmin']]); hi = np.array([p['X'], p['Y'], p['zmax']])
    pos = np.clip(pos, lo, hi)
    d = pos - net['p0'][None]; r = np.linalg.norm(d, axis=2, keepdims=True)
    Dv = np.asarray(p['Dmax'], float)[None, :, None]
    return net['p0'][None] + d * np.minimum(1.0, Dv / np.maximum(r, 1e-300))


def local_cands(net, p, rng, count, center, ell, s_range=(0.05, 1.0), one_surface=0.5, fix_angles=False, proj=False):
    """Perturbations of center: every coordinate moves by a Gaussian step whose standard deviation is s times the
    GP length scale of its coordinate type (s log-uniform in s_range, one per candidate); with probability one_surface
    only one randomly chosen surface moves."""
    L = p['L']; outp, outa = [], []; got = 0; tries = 0
    sx = ell[0] * p['X']; sy = ell[0] * p['Y']; sz = ell[1] * (p['zmax'] - p['zmin']); sa = ell[2] * math.pi
    while got < count and tries < 50:
        tries += 1; nb = 3 * count
        s = s_range[0] * (s_range[1] / s_range[0]) ** rng.random((nb, 1, 1))
        step = rng.standard_normal((nb, L, 4)) * s * np.array([sx, sy, sz, sa])[None, None, :]
        one = rng.random(nb) < one_surface; pick = rng.integers(0, L, nb)
        mask = np.where(one[:, None], np.arange(L)[None, :] == pick[:, None], True)
        step = step * mask[:, :, None]
        step[:, :, :3] *= (np.asarray(p['Dmax'], float) > 0)[None, :, None]      # surfaces that may not move keep their position
        pos = center['pos'][None] + step[:, :, :3]
        if proj:      # steps that leave the box or the D_max ball end on its boundary
            pos = project(pos, net, p)
        ang = A.wrap_angle(center['angles'][None] + step[:, :, 3])
        if fix_angles: ang[:] = net['angles0'][None]
        ok = admissible(pos, ang, net, p)
        outp.append(pos[ok]); outa.append(ang[ok]); got += int(ok.sum())
    return np.concatenate(outp)[:count], np.concatenate(outa)[:count]


def local_steps(net, p, rng, cpos, cang, n_each, ell, s_range, one_surface=0.5, proj=False, fix_angles=False):
    """n_each perturbations of each of the centers (cpos (k, L, 3), cang (k, L)), as in local_cands, in one batch;
    returns admissible positions, azimuths and the index of their center."""
    L = p['L']; k = len(cpos); nb = n_each * k
    sx = ell[0] * p['X']; sy = ell[0] * p['Y']; sz = ell[1] * (p['zmax'] - p['zmin']); sa = ell[2] * math.pi
    own = np.repeat(np.arange(k), n_each)
    s = s_range[0] * (s_range[1] / s_range[0]) ** rng.random((nb, 1, 1))
    step = rng.standard_normal((nb, L, 4)) * s * np.array([sx, sy, sz, sa])[None, None, :]
    one = rng.random(nb) < one_surface; pick = rng.integers(0, L, nb)
    step = step * np.where(one[:, None], np.arange(L)[None, :] == pick[:, None], True)[:, :, None]
    step[:, :, :3] *= (np.asarray(p['Dmax'], float) > 0)[None, :, None]
    pos = cpos[own] + step[:, :, :3]
    if proj:
        pos = project(pos, net, p)
    ang = A.wrap_angle(cang[own] + step[:, :, 3])
    if fix_angles: ang[:] = net['angles0'][None]
    ok = admissible(pos, ang, net, p)
    return pos[ok], ang[ok], own[ok]


def climb_acq_batch(net, p, rng, m, cpos, cang, acq, Xa, ymax, basis, o, fix_angles=False, predict=None):
    """Stochastic hill climbing of log EI from the o['starts'] best candidates, all starts in one batch: o['iters']
    rounds of o['n'] perturbations per start (steps of o['s'] times the GP length scales); a start moves to its best
    perturbation when that raises log EI. The climbed points are appended to the candidates."""
    order = [j for j in np.argsort(-acq)[:o.get('starts', 4)] if np.isfinite(acq[j])]
    if not order:
        return cpos, cang, acq
    P_, A_, V_ = cpos[order].copy(), cang[order].copy(), acq[order].copy()
    for _ in range(o.get('iters', 10)):
        lp, la, own = local_steps(net, p, rng, P_, A_, o.get('n', 32), m['ell'], o.get('s', (0.02, 0.5)), proj=o.get('proj', False),
                                  fix_angles=fix_angles)
        if len(lp) == 0: continue
        Zl = feats(lp, la, p)
        mu_, sd_ = gp_predict(m, Zl, basis(lp, la)) if predict is None else predict(m, Zl, lp, la)
        fr = not_evaluated(Zl, Xa)
        a_ = np.where(fr, A.log_ei((mu_ - ymax) / sd_, sd_), -np.inf)
        for k in range(len(order)):
            idx = np.nonzero(own == k)[0]
            if len(idx) == 0: continue
            j = idx[int(np.argmax(a_[idx]))]
            if a_[j] > V_[k]:
                P_[k], A_[k], V_[k] = lp[j], la[j], a_[j]
    return np.concatenate([cpos, P_]), np.concatenate([cang, A_]), np.concatenate([acq, V_])


def climb_acq(net, p, rng, m, cpos, cang, acq, Xa, ymax, basis, o):
    """Stochastic hill climbing of log EI from the o['starts'] best candidates: o['iters'] rounds of o['n'] local
    perturbations (local_cands with steps of o['s'] times the GP length scales); a start moves to its best
    perturbation when that raises log EI. The climbed points are appended to the candidates."""
    order = np.argsort(-acq)[:o.get('starts', 4)]
    npos, nang, nacq = [], [], []
    for j in order:
        if not np.isfinite(acq[j]): continue
        cp_, ca_, cv = cpos[j], cang[j], acq[j]
        for _ in range(o.get('iters', 10)):
            lp, la = local_cands(net, p, rng, o.get('n', 32), dict(pos=cp_, angles=ca_), m['ell'], s_range=o.get('s', (0.02, 0.5)), one_surface=0.5,
                                 proj=o.get('proj', False))
            if len(lp) == 0: continue
            Zl = feats(lp, la, p)
            mu_, sd_ = gp_predict(m, Zl, basis(lp, la))
            fr = not_evaluated(Zl, Xa)
            a_ = np.where(fr, A.log_ei((mu_ - ymax) / sd_, sd_), -np.inf)
            k = int(np.argmax(a_))
            if a_[k] > cv:
                cp_, ca_, cv = lp[k], la[k], a_[k]
        npos.append(cp_); nang.append(ca_); nacq.append(cv)
    if not npos:
        return cpos, cang, acq
    return np.concatenate([cpos, np.array(npos)]), np.concatenate([cang, np.array(nang)]), np.concatenate([acq, np.array(nacq)])


_HV = np.random.default_rng(12345).standard_normal(64)


def not_evaluated(Z, Xa):
    """Mask of the candidates whose features differ from those of every evaluated deployment (max |diff| > 1e-10).
    A random projection screens for possible duplicates, which are then checked exactly."""
    v = _HV[:Z.shape[1]]
    hz, hx = Z @ v, Xa @ v
    near = np.min(np.abs(hz[:, None] - hx[None, :]), axis=1) <= 1e-9 * (1.0 + np.abs(hz)) + 1e-9
    out = np.ones(len(Z), bool)
    for i in np.nonzero(near)[0]:
        out[i] = np.min(np.max(np.abs(Xa - Z[i]), axis=1)) > 1e-10
    return out


def basin_centers(X, y, m, k, rho_sep):
    """Indices of up to k evaluated deployments, in decreasing merit, each with GP correlation below rho_sep to every
    better one already chosen (the best evaluated deployments of distinct basins)."""
    order = np.argsort(-y, kind='stable'); out = []
    for i in order:
        if all(matern(X[i:i + 1], X[j:j + 1], m['w'])[0, 0] < rho_sep for j in out):
            out.append(int(i))
            if len(out) >= k: break
    return out


# --------------------------------------------------------------------------- search
def search(evaluate, net, p, budget, seed, n_init=5, log=None, fix_angles=False, sampler='box', radial='volume', ang_scale=None,
           local_scale=0.15, first_pose=None, info=None, mean='phys', kern='iso', nugs=(1e-6,), n_cand=256, local='T',
           n_glob=128, n_loc=128, s_range=(0.05, 1.0), one_surface=0.5, local_mode='all', scale_min=A.R0, n_block=0, block_center='p0',
           stop=None, clip=None, n_centers=1, rho_sep=0.5, acq_opt=None, trim=None, explore_every=0, proj=False, linalg='gen', **ignored):
    """The deployment search; same interface and return value as algo.bayes_opt_fast (recs, ib, seconds).
    Design R is search(..., **DESIGN_R). Options (defaults: development settings; see harness/variants.py):
      mean        'phys': prior mean beta0 + beta1 S (GLS); 'chi': design P's displacement trend; 'phys+chi': both;
                  'physk': the score in the covariance with an ML weight (variant Rk)
      kern        'iso': design P's kernel; 'ard': ARD grid with nugs; 'ardcw': ARD + nugget, coordinate-wise ML (R)
      local       'T': design P's candidates (n_cand); 'ard': n_glob random + n_block re-draws + n_loc local steps
      s_range, one_surface   local steps: s log-uniform in s_range (length scales), one surface w.p. one_surface
      block_center 'p0' / 'inc': re-drawn surface around its initial / current position
      acq_opt     hill climbing of log EI (starts, iters, n, s, batch, proj)
      linalg      'gen' / 'tri': general / triangular solves (identical decisions, 'tri' is faster)
      clip, n_centers, rho_sep, explore_every, proj: further experiments (PB5, PB6, PB9, PB10);
      trim, stop: implemented but not run in the pilots (robust GP, stopping rule)
    The initial design is design T's (algo.initial_design), so the stream-set test (design_select) is unchanged."""
    global _TRI
    _TRI = (linalg == 'tri')
    d = design_of(evaluate)
    score = RateScore(net, p, d)
    L = p['L']
    init = A.initial_design(net, p, seed, n_init, first_pose=first_pose, sampler=sampler, radial=radial, fix_angles=fix_angles,
                            scale_min=scale_min, init_strata=True)
    grid = ISO_GRID if kern == 'iso' else ard_grid(nugs) if kern == 'ard' else None

    def basis_pa(pos, ang, poses=None):
        if mean == 'chi':
            return A.scale_basis(poses if poses is not None else [H_.pose_from(a, b) for a, b in zip(pos, ang)], net)
        S = score(pos=pos, ang=ang)[:, None] / 10.0
        one = np.ones((len(pos), 1))
        if mean in ('phys', 'physk'):
            return np.hstack([one, S])
        if mean == 'phys+chi':
            mv = np.max(np.linalg.norm(pos - net['p0'][None], axis=2), axis=1)
            s = np.log1p(mv / A.R0) / math.log1p(100.0 / A.R0)
            return np.stack([one[:, 0], s, s ** 2, S[:, 0]], 1)
        raise ValueError(mean)

    if kern == 'iso' and local == 'T':      # round-1 variants PB1, PB1c: design P's loop with the rate-score mean
        return _search_v1(evaluate, net, p, budget, seed, init, score, mean, n_cand, local_scale, ang_scale, radial, local_mode, sampler,
                          fix_angles, scale_min, info, log)
    X = []; y = []; recs = []; H = []
    t0 = time.time(); m = None; hidx = None
    for q in range(budget):
        if q < min(len(init), budget):
            pose = init[q]
        else:
            ib = int(np.argmax(y)); center = recs[ib]['pose']
            rng = np.random.default_rng([seed + 300000, q])
            yt = np.array(y)
            if clip is not None:      # the GP models the merit clipped from below at a quantile of the evaluated merits
                yt = np.maximum(yt, np.quantile(yt, clip))
            if mean == 'physk':      # the score in the covariance with an ML weight, constant mean
                Sx = np.array(H)[:, 1] * 10.0; s_mu = float(np.mean(Sx)); s_sd = float(np.std(Sx)) or 1.0
                m, hidx = gp_fit_cw_s(np.array(X), yt, (Sx - s_mu) / s_sd, L, start=hidx)
                m['s_mu'], m['s_sd'] = s_mu, s_sd
            elif kern == 'ardcw':
                m, hidx = gp_fit_cw(np.array(X), yt, np.array(H), L, start=hidx)
                if trim is not None:      # evaluations far below the leave-one-out prediction: failed AO runs, large noise
                    out = loo_z(m, np.array(H)) < -trim
                    if out.any():
                        m, hidx = gp_fit_cw(np.array(X), yt, np.array(H), L, start=hidx, extra=np.where(out, 1e3, 0.0))
            else:
                m = gp_fit(np.array(X), yt, np.array(H), grid, L)
            if local == 'T':
                cands = A.proposals_fast(net, p, rng, n_cand, center, np.array(X), True, local_scale=local_scale, ang_scale=ang_scale,
                                         radial=radial, local_mode=local_mode, sampler=sampler, fix_angles=fix_angles, scale_min=scale_min,
                                         scale_local=scale_min is not None)
                cpos = np.array([c['pos'] for c in cands]); cang = np.array([c['angles'] for c in cands])
            else:
                gp_, ga_ = global_cands(net, p, rng, n_glob, fix_angles=fix_angles)
                if n_centers > 1:     # local and block candidates around the best evaluated deployments of distinct basins
                    cs = basin_centers(np.array(X), np.array(y), m, n_centers, rho_sep)
                    lps, las, bps, bas = [], [], [], []
                    for c in cs:
                        cc = recs[c]['pose']
                        a_, b_ = local_cands(net, p, rng, n_loc // len(cs), cc, m['ell'], s_range=s_range, one_surface=one_surface, fix_angles=fix_angles)
                        lps.append(a_); las.append(b_)
                        if n_block:
                            a_, b_ = block_cands(net, p, rng, n_block // len(cs), cc, fix_angles=fix_angles, around=block_center)
                            bps.append(a_); bas.append(b_)
                    lp_, la_ = np.concatenate(lps), np.concatenate(las); nbk = 0
                    if n_block:
                        bp_, ba_ = np.concatenate(bps), np.concatenate(bas); nbk = len(bp_)
                        lp_, la_ = np.concatenate([bp_, lp_]), np.concatenate([ba_, la_])
                else:
                  if n_loc:
                    lp_, la_ = local_cands(net, p, rng, n_loc, center, m['ell'], s_range=s_range, one_surface=one_surface, fix_angles=fix_angles, proj=proj)
                  else:      # no local-step candidates (trimmed design R)
                    lp_, la_ = np.zeros((0, L, 3)), np.zeros((0, L))
                  nbk = 0
                  if n_block:
                    bp_, ba_ = block_cands(net, p, rng, n_block, center, fix_angles=fix_angles, around=block_center); nbk = len(bp_)
                    lp_, la_ = np.concatenate([bp_, lp_]), np.concatenate([ba_, la_])
                cpos = np.concatenate([gp_, lp_]); cang = np.concatenate([ga_, la_])
            if len(cpos) == 0:      # no admissible candidate (never happened in the study): stop the search
                break
            Z = feats(cpos, cang, p)
            Xa = np.array(X)
            fresh = not_evaluated(Z, Xa)      # not evaluated before
            if mean == 'physk':
                mu, sd = gp_predict_s(m, Z, (score(pos=cpos, ang=cang) - m['s_mu']) / m['s_sd'])
            else:
                mu, sd = gp_predict(m, Z, basis_pa(cpos, cang))
            acq = np.where(fresh, A.log_ei((mu - np.max(y)) / sd, sd), -np.inf)
            if explore_every and (q - len(init)) % explore_every == explore_every - 1 and local != 'T':
                # an exploration step: only the random and the block candidates (new deployments or a new role of one surface)
                acq = np.where(np.arange(len(acq)) < len(gp_) + nbk, acq, -np.inf)
            if acq_opt is not None:      # refine the acquisition maximum: stochastic hill climbing from the best candidates
                if acq_opt.get('batch'):
                    prd = None if mean != 'physk' else (lambda m_, Z_, P_, A_: gp_predict_s(m_, Z_, (score(pos=P_, ang=A_) - m_['s_mu']) / m_['s_sd']))
                    cpos, cang, acq = climb_acq_batch(net, p, rng, m, cpos, cang, acq, Xa, np.max(y), basis_pa, acq_opt, fix_angles=fix_angles,
                                                      predict=prd)
                else:
                    cpos, cang, acq = climb_acq(net, p, rng, m, cpos, cang, acq, Xa, np.max(y), basis_pa, acq_opt)
            i = int(np.argmax(acq))
            if info is not None and info.get('debug'):
                info.setdefault('models', []).append(dict(q=q, m=m, cpos=cpos, cang=cang, mu=mu, sd=sd, acq=acq, i=i, n_glob=len(gp_) if local != 'T' else 0,
                                                          H=basis_pa(cpos, cang), y=list(y)))
            pose = H_.pose_from(cpos[i], cang[i])
            if info is not None:
                if i >= len(mu):     # a climbed point
                    if mean == 'physk':
                        mu_i, sd_i = gp_predict_s(m, feats(cpos[i:i + 1], cang[i:i + 1], p), (score(pos=cpos[i:i + 1], ang=cang[i:i + 1]) - m['s_mu']) / m['s_sd'])
                    else:
                        mu_i, sd_i = gp_predict(m, feats(cpos[i:i + 1], cang[i:i + 1], p), basis_pa(cpos[i:i + 1], cang[i:i + 1]))
                    mu = np.concatenate([mu, np.full(i + 1 - len(mu), mu_i[0])]); sd = np.concatenate([sd, np.full(i + 1 - len(sd), sd_i[0])])
                info.setdefault('ell', []).append(m['ell']); info.setdefault('nug', []).append(m['nug'])
                info.setdefault('beta', []).append(m['beta'].tolist())
                info.setdefault('pick', []).append(dict(q=q, kind=('local' if local == 'T' else 'global' if i < len(gp_) else 'block' if i < len(gp_) + nbk else 'local' if i < len(Z) else 'climbed'), mu=float(mu[i]), sd=float(sd[i]),
                                                        ymax=float(np.max(y)), move=float(np.max(np.linalg.norm(cpos[i] - net['p0'], axis=1))),
                                                        dist=float(np.max(np.linalg.norm(cpos[i] - center['pos'], axis=1)))))
        r = evaluate(pose); r['pose'] = pose
        X.append(feats(pose['pos'][None], pose['angles'][None], p)[0]); y.append(r['merit']); recs.append(r)
        H.append(basis_pa(pose['pos'][None], pose['angles'][None], [pose])[0])
        if log: log(f'  q{q + 1:2d}: merit {r["merit"]:9.4f} rate {r["rate"]:8.4f} def {r["deficit"]:.3g}')
        if stop is not None and q + 1 >= max(stop.get('min', 0), stop['window'] + 1):
            # stop when the last `window` evaluations raised the best merit by at most `tol`
            if max(y) - max(y[:-stop['window']]) <= stop['tol']:
                break
    ib = int(np.argmax(y))
    return recs, ib, time.time() - t0


def _search_v1(evaluate, net, p, budget, seed, init, score, mean, n_cand, local_scale, ang_scale, radial, local_mode, sampler, fix_angles,
               scale_min, info, log):
    """Design P with the rate-score prior mean (variants PB1, PB1c)."""
    def basis(poses):
        if mean == 'chi':
            return A.scale_basis(poses, net)
        S = score(poses)[:, None] / 10.0
        one = np.ones((len(poses), 1))
        if mean == 'phys':
            return np.hstack([one, S])
        if mean == 'phys+chi':
            return np.hstack([A.scale_basis(poses, net), S])
        raise ValueError(mean)

    X = []; y = []; recs = []; H = []
    t0 = time.time()
    for q in range(budget):
        if q < min(len(init), budget):
            pose = init[q]
        else:
            ib = int(np.argmax(y)); center = recs[ib]['pose']
            rng = np.random.default_rng([seed + 300000, q])
            cands = A.proposals_fast(net, p, rng, n_cand, center, np.array(X), True, local_scale=local_scale, ang_scale=ang_scale,
                                     radial=radial, local_mode=local_mode, sampler=sampler, fix_angles=fix_angles, scale_min=scale_min,
                                     scale_local=scale_min is not None)
            Z = A.features(cands, p)
            m = A.gp_fit_trend(np.array(X), np.array(y), np.array(H))
            mu, sd = A.gp_predict_trend(m, Z, basis(cands))
            if info is not None:
                info.setdefault('ell', []).append(m['ell']); info.setdefault('beta', []).append(m['beta'].tolist())
            pose = cands[int(np.argmax(A.log_ei((mu - np.max(y)) / sd, sd)))]
        r = evaluate(pose); r['pose'] = pose
        X.append(A.features([pose], p)[0]); y.append(r['merit']); recs.append(r); H.append(basis([pose])[0])
        if log: log(f'  q{q + 1:2d}: merit {r["merit"]:9.4f} rate {r["rate"]:8.4f} def {r["deficit"]:.3g}')
    ib = int(np.argmax(y))
    return recs, ib, time.time() - t0
