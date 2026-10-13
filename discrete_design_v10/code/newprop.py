"""Proposed method: Bayesian optimization of the deployment with an AO evaluation at a loose convergence
threshold, followed by AO at the best deployment with a tight threshold.

AO = alternating QT precoder steps and QT steps of the relaxed DS-STAR coefficients (proximal penalty, exact
reduced form over the cascade gains), each coefficient step followed by the projection onto the coupled
transmission/reflection set. The AO stops when an iteration increases the penalized sum rate F by less than
the threshold (tol_bo during BO, tol_ref in the refinement) or after max_iter iterations."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np
import hrs as H_, algo as A, run as R
import composition as CM
import turbo as TB

NEW = dict(budget=50, n_init=5, tol_bo=3.0, tol_ref=0.1, stable=1, max_iter=20, prec_steps=3, coef_steps=3, lamQ=200.0,
           seed='rule', init='core', search='bo')   # search: 'bo' (EI over random/local candidates) | 'turbo' (TuRBO-1)


def _ao(theta, W, ch, net, d, p, o, tol, qtp, qtc):
    return H_.ao(theta, W, ch, net, d, p, o['lamQ'], max_iter=o['max_iter'], prec_steps=o['prec_steps'],
                 coef_steps=o['coef_steps'], tol=tol, stable_req=o['stable'], qtp=qtp, qtc=qtc)


def run_new(cfg, opt=None, composition=True, ablation=True, start_pose=None):
    o = dict(NEW); o.update(opt or {})
    K, N, M, Smax, Rth = cfg['K'], cfg['N'], cfg['M'], cfg['Smax'], cfg['Rth']
    L = cfg.get('L', 3); Dmax = cfg.get('Dmax', 10.0); Pmax = cfg.get('Pmax', 1.0); drop = cfg['drop']
    p = H_.params(K=K, L=L, N=N, M=M, Smax=Smax, Pmax=Pmax, Dmax=Dmax, Rth=Rth, bank=R.BANK)
    net = H_.make_network(p, R.BANK, drop=drop, setup=cfg.get('setup', 1), kind=cfg.get('kind', 'random'))
    d = H_.joint_design(net, p, 'private_first', 'sus')
    seed = R.seed_of(K, drop, L, cfg.get('setup', 1))
    out = dict(cfg=cfg, method='proposed_ao' + ('' if o['search'] == 'bo' else '_' + o['search']), settings=o, nS=d.nS, G=d.G,
               served=[d.name(j) for j in d.served])
    c1 = time.process_time()
    make_ev = lambda dd: A.AOEvaluator(net, dd, p, seed=o['seed'], init=o['init'], lamQ=o['lamQ'], max_iter=o['max_iter'],
                                       prec_steps=o['prec_steps'], coef_steps=o['coef_steps'], tol=o['tol_bo'], stable_req=o['stable'])
    if o.get('select'):
        # new discrete design: keep HRS or the same order without group-common streams, by the initial design of BO
        import design_select as DSel
        d, ev, sel_info = DSel.select(net, p, d, o, seed, make_ev, hrs_init=o.get('hrs_init'))
        out.update(select=sel_info, nS=d.nS, G=d.G, served=[d.name(j) for j in d.served])
        if ev is None:      # HRS kept: the earlier run of the HRS design is the result
            out['reuse'] = 'hrs'; out['settings'] = {k: v for k, v in o.items() if k != 'hrs_init'}
            return out
    else:
        ev = make_ev(d)
    if o['search'] == 'turbo':
        recs, ib, tbo, tinfo = TB.turbo(ev, net, p, o['budget'], seed, n_init=o['n_init'], **o.get('turbo_kw', {}))
        out['turbo'] = dict(restarts=tinfo['restarts'], d=tinfo['d'], fail_tol=tinfo['fail_tol'], n_cand=tinfo['n_cand'], L=tinfo['L'])
    else:
        ls = 0.15 if not o.get('local_m') or p['Dmax'][0] <= 0 else o['local_m'] / p['Dmax'][0]
        fp = None if start_pose is None else H_.pose_from(start_pose['pos'], start_pose['angles'])
        recs, ib, tbo = A.bayes_opt_fast(ev, net, p, o['budget'], seed, n_init=o['n_init'], n_cand=o.get('n_cand', 256), local_scale=ls, ang_scale=0.15,
                                         radial=o.get('radial', 'volume'), first_pose=fp, kern=o.get('kern', 'iso'),
                                         local_mode=o.get('local_mode', 'all'), **o.get('bo', {}))
    bo_cpu = time.process_time() - c1
    best = recs[ib]
    ch = H_.channel_terms(best['pose'], net, p)
    c2 = time.process_time()
    ref = _ao(best['theta'], best['W'], ch, net, d, p, o, o['tol_ref'], ev.qtp, ev.qtc)
    ref_cpu = time.process_time() - c2
    out['prop'] = dict(bo_merit=best['merit'], bo_rate=best['rate'], bo_deficit=best['deficit'], bo_feasible=best['deficit'] <= 1e-6,
                       merit=ref['merit'], rate=ref['rate'], deficit=ref['deficit'], feasible=ref['deficit'] <= 1e-6,
                       cpu=bo_cpu + ref_cpu, bo_cpu=bo_cpu, ref_cpu=ref_cpu, ref_iters=ref['iterations'], ref_calls=ref['calls'],
                       best_q=ib + 1, init_merit=recs[0]['merit'], init_rate=recs[0]['rate'],
                       trace=[r['merit'] for r in recs], ev_cpu=[r['cpu'] for r in recs], ev_calls=[r['calls'] for r in recs],
                       ev_iters=[r['iters'] for r in recs], hw=H_.hardware_residual(ref['theta']),
                       pose=dict(pos=best['pose']['pos'].tolist(), angles=best['pose']['angles'].tolist()))
    if composition:
        out['comp'] = CM.composition(ref['W'], ref['H'], d, p)
        if d.served_groups and ablation:
            # same method without group-common streams at the selected deployment (no-group ablation)
            dn = CM.nogroup_design(d, p)
            evn = A.AOEvaluator(net, dn, p, seed=o['seed'], init=o['init'], lamQ=o['lamQ'], max_iter=o['max_iter'],
                                prec_steps=o['prec_steps'], coef_steps=o['coef_steps'], tol=o['tol_bo'], stable_req=o['stable'])
            rn = evn(best['pose'])
            refn = _ao(rn['theta'], rn['W'], ch, net, dn, p, o, o['tol_ref'], evn.qtp, evn.qtc)
            out['nogroup'] = dict(rate=refn['rate'], merit=refn['merit'], deficit=refn['deficit'], feasible=refn['deficit'] <= 1e-6,
                                  comp=CM.composition(refn['W'], refn['H'], dn, p))
    return out


def run_prescribed(cfg, select=False):
    """Prescribed-deployment reference (as for the thesis): the evaluation at delta_0, the refinement and QT
    precoder iterations to convergence; at delta_0 the cascaded channels vanish. With select, the stream-set test is
    made at delta_0 (both stream sets evaluated there, the larger F kept)."""
    K, N, M, Smax, Rth = cfg['K'], cfg['N'], cfg['M'], cfg['Smax'], cfg['Rth']
    p = H_.params(K=K, N=N, M=M, Smax=Smax, Rth=Rth, Dmax=cfg.get('Dmax', 10.0), L=cfg.get('L', 3),
                  Pmax=cfg.get('Pmax', 1.0), bank=R.BANK)
    net = H_.make_network(p, R.BANK, drop=cfg['drop'], setup=cfg.get('setup', 1), kind=cfg.get('kind', 'random'))
    d = H_.joint_design(net, p, 'private_first', 'sus')
    pose = H_.pose_from(net['p0'], net['angles0'])
    c0 = time.process_time()
    evs = {}
    def evaluate(dd):
        evs[id(dd)] = A.Evaluator(net, dd, p, seed='rule', init='core', max_iter=3, tol=-1e9)
        return evs[id(dd)](pose)
    info = None
    if select:
        import design_select as DSel
        d, r, info = DSel.select_at(d, p, evaluate)
    else:
        r = evaluate(d)
    ev = evs[id(d)]
    ch = H_.channel_terms(pose, net, p)
    ref = A.refine(r['theta'], r['W'], ch, net, d, p, ev.qtp, 200.0, qt_iter=20, qt_tol=0.1)
    W, cur, _, _ = A.qt_precoder_loop(ref['H'], ref['W'], d, p, ev.qtp, 200.0, 20, 0.1)
    out = dict(cfg=cfg, method='prescribed', rate=cur[1], merit=cur[0], deficit=cur[2], feasible=cur[2] <= 1e-6,
               nS=d.nS, G=d.G, cpu=time.process_time() - c0)
    if info is not None: out['select'] = info
    return out
