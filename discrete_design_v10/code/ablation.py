"""Ablation study: the proposed framework against variants that replace one of its components.

Every variant keeps everything else of the proposed framework (BO over the deployment with the AO evaluation at
the loose tolerance, refinement with the tight tolerance at the selected deployment, penalized merit, same
initial design and random streams of the BO):
  proposed        joint-transmission design: SUS core users, nearest-core groups, fixed stream order
  rand_group      the UEs are partitioned uniformly at random into groups of the proposed sizes (core users, stream
                  order rule unchanged)
  rand_fronthaul  groups as proposed, every BS carries min(Smax, J) streams drawn uniformly from all J streams
                  (BS-specific association; a stream carried by one BS uses only its N antennas)
  rsma1           1-layer RSMA: super-common and private streams, no group-common streams
  sdma            private streams only (no common streams)
  noma2           2-UE NOMA: each core user is paired with at most one non-core UE (greedy matching by channel
                  correlation); the weak UE's message is a stream decoded by both UEs (SIC at the core user),
                  the core user has a private stream, unpaired UEs have private streams; no super-common stream
  rand_coef       DS-STAR coefficients drawn uniformly on the coupled set and kept fixed; precoders optimized
  rand_deploy     one random admissible deployment (no BO), AO evaluation and refinement there
  no_star         no DS-STAR (direct channels only), precoders optimized to the tight tolerance
  rand_wstart     random precoder start: every evaluation of the BO starts the AO from random precoders (i.i.d.
                  complex Gaussian entries scaled to the full power of every BS) instead of the closed-form start
  rand_cstart     random coefficient start: every evaluation starts from coefficients drawn uniformly on the coupled
                  set (the starting precoders then follow from the effective channels as usual)
  start_mf        simplified start: every served stream along the dominant eigenvector of the normalized channels of its
                  decoders (matched filter for a private stream), full BS power; coefficients by the closed-form rule at it
  start_rzf       simplified start: regularized ZF over all served private streams and the dominant eigenvector for the
                  common streams, full BS power; coefficients by the closed-form rule at it
  start_core      simplified start: the start of the proposed framework with the core users of the discrete design in
                  place of the greedy ZF user selection (regularized ZF for their private streams, dominant eigenvector
                  for the super-common stream, every other served stream at the noise level)
The random starts are drawn from a generator keyed by the scheme seed and the deployment, so that an evaluation stays
a deterministic function of the deployment; the refinement continues from the evaluation, as in the proposed framework.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time, math, zlib
import numpy as np
import hrs as H_, algo as A, run as R, newprop as NP
import composition as CM

SCHEMES = ('proposed', 'rand_group', 'rand_fronthaul', 'rsma1', 'sdma', 'noma2', 'rand_coef', 'rand_deploy', 'no_star',
           'rand_wstart', 'rand_cstart', 'start_mf', 'start_rzf', 'start_core', 'start_zf', 'no_rot', 'rand_start')
START_SCHEMES = ('rand_wstart', 'rand_cstart', 'rand_start')
START_KIND = {'rand_wstart': 'w', 'rand_cstart': 'c', 'rand_start': 'wc'}    # random precoders, coefficients, or both


def setup(cfg):
    p = H_.params(K=cfg['K'], L=cfg.get('L', 3), N=cfg['N'], M=cfg['M'], Smax=cfg['Smax'], Pmax=cfg.get('Pmax', 1.0),
                  Dmax=cfg.get('Dmax', 10.0), Rth=cfg['Rth'], bank=R.BANK)
    net = H_.make_network(p, R.BANK, drop=cfg['drop'], setup=cfg.get('setup', 1), kind=cfg.get('kind', 'random'))
    d = H_.joint_design(net, p, 'private_first', 'sus')
    seed = R.seed_of(cfg['K'], cfg['drop'], p['L'], cfg.get('setup', 1))
    return p, net, d, seed


def _rng(seed, scheme):
    return np.random.default_rng([seed + 700000, SCHEMES.index(scheme)])


def design(scheme, net, p, d, rng):
    K, S = p['K'], p['Smax']
    order, groups, energy, corr = H_.joint_structure(net['Hd'], p, 'sus')
    others = [k for k in range(K) if k not in order]
    if scheme in ('proposed', 'rand_coef', 'rand_deploy', 'no_star', 'start_mf', 'start_rzf', 'start_core', 'start_zf', 'no_rot') + START_SCHEMES:
        return d
    if scheme == 'rand_group':
        # the UEs are partitioned uniformly at random into groups with the sizes of the proposed grouping;
        # core users, private streams and the stream order rule are unchanged
        perm = rng.permutation(K).tolist(); gr = []; i = 0
        for m in groups:
            gr.append(sorted(perm[i:i + len(m)])); i += len(m)
        pr = H_.priority_list(order, gr, energy, K, 'private_first')
        dn = H_.Design(gr, pr[:min(S, len(pr))], K)
        dn.core = order; dn.priority = pr; dn.rule = 'random_groups'
        return dn
    if scheme == 'rand_fronthaul':
        J = d.J; sets = [set(rng.choice(J, size=min(S, J), replace=False).tolist()) for _ in range(p['B'])]
        served = sorted(set().union(*sets))
        dn = H_.Design(d.groups, served, K)
        dn.core = d.core; dn.priority = d.priority; dn.rule = 'random_fronthaul'
        dn.bsmask = np.array([[j in sets[b] for j in dn.served] for b in range(p['B'])])
        dn.bs_sets = [sorted(s) for s in sets]
        return dn
    if scheme == 'rsma1':
        return CM.nogroup_design(d, p)
    if scheme == 'sdma':
        rest = sorted(others, key=lambda k: (-energy[k], k))
        pr = [1 + k for k in order] + [1 + k for k in rest]
        return _assoc_design([], pr, K, p, net, order, 'sdma')
    if scheme == 'noma2':
        # greedy one-to-one matching of core users and non-core UEs by channel correlation
        cand = sorted(((corr[c, w], c, w) for c in order for w in others), key=lambda t: (-t[0], t[1], t[2]))
        mate = {}; used = set()
        for _, c, w in cand:
            if c in mate or w in used: continue
            mate[c] = w; used.add(w)
        pairs = [[c, mate[c]] for c in order if c in mate]
        G = len(pairs)
        unpaired = sorted([k for k in others if k not in used], key=lambda k: (-energy[k], k))
        pr = [1 + G + c for c in order] + [1 + g for g in range(G)] + [1 + G + k for k in unpaired]
        dn = _assoc_design(pairs, pr, K, p, net, order, 'noma2')
        dn.carriers = [[w] for _, w in pairs]          # the pair stream carries the weak UE's message only
        return dn
    raise ValueError(scheme)


def _assoc_design(groups, pr, K, p, net, order, rule):
    """BS-specific stream association for the schemes without common streams (SDMA, NOMA): every stream is
    carried by the BS with the larger direct-channel energy to its weakest decoder (the other BS if that one is
    full), and each BS then fills its remaining slots with the other streams in decreasing order of that energy,
    which it transmits jointly with the first BS. With at most S^max streams every BS carries all of them (joint
    transmission); otherwise up to B S^max streams are served."""
    B, N, S = p['B'], p['N'], p['Smax']
    tmp = H_.Design(groups, pr, K)
    gain = {j: [min(np.sum(np.abs(net['Hd'][k, b * N:(b + 1) * N]) ** 2) for k in tmp.decoders[j]) for b in range(B)] for j in pr}
    sets = [[] for _ in range(B)]
    for j in pr:
        for b in sorted(range(B), key=lambda b: -gain[j][b]):
            if len(sets[b]) < S:
                sets[b].append(j); break
    carried = set().union(*map(set, sets))
    for b in range(B):
        for j in sorted(carried - set(sets[b]), key=lambda j: -gain[j][b]):
            if len(sets[b]) >= S: break
            sets[b].append(j)
    served = [j for j in pr if j in carried]
    dn = H_.Design(groups, served, K)
    dn.core = order; dn.priority = pr; dn.rule = rule
    if any(len(sets[b]) < len(served) for b in range(B)):
        dn.bsmask = np.array([[j in sets[b] for j in dn.served] for b in range(B)])
        dn.bs_sets = [sorted(sets[b]) for b in range(B)]
    return dn


def random_theta(p, rng):
    """Uniform on the coupled set: u = t + r and v = t - r with independent uniform unit-modulus entries."""
    M, L = p['M'], p['L']; th = np.zeros((M, 4, L), complex)
    for q in (0, 2):
        u = np.exp(2j * math.pi * rng.random((M, L))); v = np.exp(2j * math.pi * rng.random((M, L)))
        th[:, q, :] = (u + v) / 2; th[:, q + 1, :] = (u - v) / 2
    return th


def prec_ao(H, W, d, p, qtp, lamQ, tol, max_iter=20, steps=3):
    """The AO with the coefficient block removed: iterations of `steps` QT precoder steps, stopped after the first
    iteration that raises F by less than tol."""
    cur = H_.merit(W, H, d, p, lamQ); calls = 0; it = 0
    for it in range(1, max_iter + 1):
        old = cur[0]; fail = False
        for _ in range(steps):
            try:
                Wn = qtp.solve(H, W); calls += 1
            except Exception:
                calls += 1; fail = True; break
            mn = H_.merit(Wn, H, d, p, lamQ)
            if mn[0] >= cur[0] - 1e-9: W, cur = Wn, mn
        if fail or cur[0] - old < tol: break
    return W, cur, calls, it


def pose_key(pose):
    """Integer key of a deployment (positions and azimuths rounded to 1e-9)."""
    v = np.round(np.concatenate([np.ravel(pose['pos']), np.ravel(pose['angles'])]), 9) + 0.0
    return zlib.crc32(v.tobytes())


def random_precoders(d, p, rng):
    """I.i.d. complex Gaussian precoders of the served streams, scaled to the full power of every BS."""
    W = (rng.standard_normal((p['BN'], d.nS)) + 1j * rng.standard_normal((p['BN'], d.nS))) / math.sqrt(2)
    if getattr(d, 'bsmask', None) is not None:
        N = p['N']
        for b in range(p['B']):
            W[b * N:(b + 1) * N, ~d.bsmask[b]] = 0.0
    return H_.scale_to_power(W, p)


class RandomStartEvaluator(A.AOEvaluator):
    """The AO evaluation of the proposed framework from a random start: random precoders ('w') or random coefficients
    ('c'), drawn from a generator keyed by key and the deployment."""
    def __init__(self, net, d, p, kind, key, **kw):
        super().__init__(net, d, p, **kw); self.kind, self.key = kind, key

    def __call__(self, pose):
        t0 = time.time(); c0 = time.process_time()
        rng = np.random.default_rng([self.key, pose_key(pose)])
        ch = H_.channel_terms(pose, self.net, self.p)
        theta = random_theta(self.p, rng) if 'c' in self.kind else A.SEEDS[self.seed](ch, self.net, self.d, self.p)
        H = H_.effective_channel(theta, ch, self.net, self.p)
        W0 = random_precoders(self.d, self.p, rng) if 'w' in self.kind else A.INITS[self.init](H, self.d, self.p)
        m0 = H_.merit(W0, H, self.d, self.p, self.lamQ)
        out = H_.ao(theta, W0, ch, self.net, self.d, self.p, self.lamQ, qtp=self.qtp, qtc=self.qtc, **self.aokw)
        return dict(merit=out['merit'], rate=out['rate'], deficit=out['deficit'], theta=out['theta'], W=out['W'],
                    calls=out['calls'], iters=out['iterations'], init_merit=m0[0], seconds=time.time() - t0,
                    cpu=time.process_time() - c0, hist=out['hist'])


class FixedCoefEvaluator:
    def __init__(self, net, d, p, theta, o):
        self.net, self.d, self.p, self.theta, self.o = net, d, p, theta, o
        self.qtp = H_.QTPrecoderDirect(d, p, o['lamQ'])

    def __call__(self, pose, tol=None):
        t0 = time.time(); c0 = time.process_time()
        ch = H_.channel_terms(pose, self.net, self.p)
        H = H_.effective_channel(self.theta, ch, self.net, self.p)
        W0 = A.INITS[self.o['init']](H, self.d, self.p)
        W, cur, calls, it = prec_ao(H, W0, self.d, self.p, self.qtp, self.o['lamQ'], self.o['tol_bo'] if tol is None else tol,
                                    self.o['max_iter'], self.o['prec_steps'])
        return dict(merit=cur[0], rate=cur[1], deficit=cur[2], theta=self.theta, W=W, H=H, calls=calls, iters=it,
                    seconds=time.time() - t0, cpu=time.process_time() - c0)


# Starting precoders: SDMA has no common stream that could carry the QoS of the UEs left at the noise level by the
# core-user start, whose private streams then stay switched off; all its private streams therefore start with
# regularized ZF at equal power. All other schemes use the start of the proposed framework (core-user start).
# start_zf: the earlier start (greedy ZF user selection); start_mf: every stream at full power; start_rzf: RZF on
# every private stream; start_core: the core-user start (the default, kept as an explicit variant).
SCHEME_INIT = {'sdma': 'eqc', 'start_mf': 'mf', 'start_rzf': 'eqc', 'start_core': 'core', 'start_zf': 'zf'}
# the coefficients start from the closed-form rule at the respective precoders
SCHEME_SEED = {'start_mf': 'rule_mf', 'start_rzf': 'rule_eqc', 'start_core': 'rule_core', 'start_zf': 'rule_zf'}
SELECT_SCHEMES = ('proposed', 'rand_group', 'rand_coef', 'rand_deploy', 'no_star', 'start_mf', 'start_rzf', 'start_core', 'start_zf', 'no_rot') + START_SCHEMES


def run_scheme(cfg, scheme, o=None):
    o = dict(NP.NEW, **(o or {}))
    o['init'] = SCHEME_INIT.get(scheme, o['init']); o['seed'] = SCHEME_SEED.get(scheme, o['seed'])
    p, net, d0, seed = setup(cfg)
    rng = _rng(seed, scheme)
    c0 = time.process_time()
    d = design(scheme, net, p, d0, rng)
    out = dict(cfg=cfg, scheme=scheme, nS=d.nS, G=d.G, served=[d.name(j) for j in d.served])
    if d.bsmask is not None:
        out['bs_streams'] = [[d.name(j) for j in s] for s in d.bs_sets]
    # stream-set test of the new discrete design (o['select']): the variants that keep the stream order of the proposed
    # framework (proposed, rand_group, rand_coef, rand_deploy, no_star) compare the stream sets with and without the
    # group-common streams with their own evaluations, at the initial design of the BO or at their single deployment.
    # rand_fronthaul replaces the served-stream selection itself, and rsma1, sdma and noma2 serve no group-common stream.
    sel = bool(o.get('select')) and scheme in SELECT_SCHEMES
    if scheme == 'no_star':
        def evaluate(dd):
            # evaluation: the precoder iterations of no_star with the loose tolerance
            t = time.process_time(); qtp = H_.QTPrecoderDirect(dd, p, o['lamQ'])
            W0 = A.INITS[o['init']](net['Hd'], dd, p)
            W, cur, calls, it = prec_ao(net['Hd'], W0, dd, p, qtp, o['lamQ'], o['tol_bo'], o['max_iter'], o['prec_steps'])
            return dict(merit=cur[0], W=W, qtp=qtp, calls=calls, cpu=time.process_time() - t)
        if sel:
            import design_select as DSel
            d, e, out['select'] = DSel.select_at(d, p, evaluate)
            out.update(nS=d.nS, G=d.G, served=[d.name(j) for j in d.served])
            W, cur, calls, it = prec_ao(net['Hd'], e['W'], d, p, e['qtp'], o['lamQ'], o['tol_ref'], o['max_iter'], o['prec_steps'])
            calls += e['calls'] + out['select'].get('other_init_calls', 0)
        else:
            qtp = H_.QTPrecoderDirect(d, p, o['lamQ'])
            W0 = A.INITS[o['init']](net['Hd'], d, p)
            W, cur, calls, it = prec_ao(net['Hd'], W0, d, p, qtp, o['lamQ'], o['tol_ref'], o['max_iter'], o['prec_steps'])
        out.update(rate=cur[1], merit=cur[0], deficit=cur[2], cpu=time.process_time() - c0, calls=calls)
    elif scheme == 'rand_coef':
        theta_r = random_theta(p, rng)
        if sel:
            import design_select as DSel
            d, ev, out['select'] = DSel.select(net, p, d, dict(n_init=o['n_init']), seed, lambda dd: FixedCoefEvaluator(net, dd, p, theta_r, o),
                                               hrs_init=o.get('hrs_init'))
            out.update(nS=d.nS, G=d.G, served=[d.name(j) for j in d.served])
            if ev is None:
                out['reuse'] = 'hrs'; return out
        else:
            ev = FixedCoefEvaluator(net, d, p, theta_r, o)
        recs, ib, _ = A.bayes_opt_fast(ev, net, p, o['budget'], seed, n_init=o['n_init'], ang_scale=0.15)
        best = recs[ib]
        W, cur, calls, it = prec_ao(best['H'], best['W'], d, p, ev.qtp, o['lamQ'], o['tol_ref'], o['max_iter'], o['prec_steps'])
        out.update(rate=cur[1], merit=cur[0], deficit=cur[2], cpu=time.process_time() - c0, best_q=ib + 1,
                   bo_rate=best['rate'], trace=[r['merit'] for r in recs], pose=_pose(best['pose']))
    else:
        aokw = dict(seed=o['seed'], init=o['init'], lamQ=o['lamQ'], max_iter=o['max_iter'], prec_steps=o['prec_steps'],
                    coef_steps=o['coef_steps'], tol=o['tol_bo'], stable_req=o['stable'])
        if scheme in START_SCHEMES:
            key = seed + 800000 + 1000 * SCHEMES.index(scheme)
            mk = lambda dd: RandomStartEvaluator(net, dd, p, START_KIND[scheme], key, **aokw)
        else:
            mk = lambda dd: A.AOEvaluator(net, dd, p, **aokw)
        if scheme == 'rand_deploy':
            base = H_.pose_from(net['p0'], net['angles0'])
            pose = A.proposals_fast(net, p, rng, 1, base, None, False)[0]
            if sel:
                import design_select as DSel
                evs = {}
                def evaluate(dd):
                    evs[id(dd)] = mk(dd); return evs[id(dd)](pose)
                d, best, out['select'] = DSel.select_at(d, p, evaluate, hrs_merit=o.get('hrs_merit'))
                out.update(nS=d.nS, G=d.G, served=[d.name(j) for j in d.served])
                if best is None:
                    out['reuse'] = 'hrs'; return out
                ev = evs[id(d)]
            else:
                ev = mk(d); best = ev(pose)
            best['pose'] = pose; recs = [best]; ib = 0
        else:
            if sel:
                import design_select as DSel
                d, ev, out['select'] = DSel.select(net, p, d, dict(n_init=o['n_init'], fix_angles=scheme == 'no_rot'), seed, mk, hrs_init=o.get('hrs_init'))
                out.update(nS=d.nS, G=d.G, served=[d.name(j) for j in d.served])
                if ev is None:
                    out['reuse'] = 'hrs'; return out
            else:
                ev = mk(d)
            recs, ib, _ = A.bayes_opt_fast(ev, net, p, o['budget'], seed, n_init=o['n_init'], ang_scale=0.15, fix_angles=scheme == 'no_rot')
            best = recs[ib]
        ch = H_.channel_terms(best['pose'], net, p)
        ref = NP._ao(best['theta'], best['W'], ch, net, d, p, o, o['tol_ref'], ev.qtp, ev.qtc)
        out.update(rate=ref['rate'], merit=ref['merit'], deficit=ref['deficit'], cpu=time.process_time() - c0, best_q=ib + 1,
                   bo_rate=best['rate'], trace=[r['merit'] for r in recs], pose=_pose(best['pose']))
        if scheme in ('proposed', 'rsma1'):
            out['comp'] = CM.composition(ref['W'], ref['H'], d, p)
            out['detail'] = dict(alloc=CM.ue_allocation(ref['W'], ref['H'], d, p), groups=[list(map(int, g)) for g in d.groups],
                                 carriers=[list(map(int, g)) for g in d.carriers], core=[int(k) for k in d.core],
                                 p0=np.asarray(net['p0']).tolist(), angles0=np.asarray(net['angles0']).tolist(),
                                 UE=np.asarray(net['UE']).tolist(), BS=np.asarray(net['BS']).tolist())
    out['feasible'] = out['deficit'] <= 1e-6
    return out


def _pose(pose):
    return dict(pos=np.asarray(pose['pos']).tolist(), angles=np.asarray(pose['angles']).tolist())
