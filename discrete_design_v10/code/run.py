"""Full runs: proposed (QT-precoder BO + AO refinement) and AO-BO benchmark."""
import os, sys, time, json, math
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import hrs as H_, algo as A
import composition as CM

BANK = H_.load_bank(os.path.join(os.path.dirname(__file__), '..', 'rsma_scenarios.json'))
OPT = dict(rule='private_first', core='sus', seed='rule', init='core', refine='rule', lamQ=200.0,
           ev_iter=3, ev_tol=-1e9, budget_prop=50, budget_ao=10, n_init=5,
           ao_iter=20, ao_prec=3, ao_coef=3, ao_tol=0.1)


def seed_of(K, drop, L=3, setup=1, base=43003):
    kidx = [8, 12, 16, 20, 10, 14, 18].index(K) + 1
    sid = (drop - 1) + 1000 * ((0) + 4 * ((kidx - 1) + 4 * ((L - 2) + 3 * (setup - 1))))
    return base + sid


def run_config(cfg, methods=('prop', 'ao'), opt=None, log=None):
    o = dict(OPT); o.update(opt or {})
    K, N, M, Smax, Rth = cfg['K'], cfg['N'], cfg['M'], cfg['Smax'], cfg['Rth']
    L = cfg.get('L', 3); Dmax = cfg.get('Dmax', 10.0); Pmax = cfg.get('Pmax', 1.0); drop = cfg['drop']
    p = H_.params(K=K, L=L, N=N, M=M, Smax=Smax, Pmax=Pmax, Dmax=Dmax, Rth=Rth, bank=BANK)
    net = H_.make_network(p, BANK, drop=drop, setup=cfg.get('setup', 1), kind=cfg.get('kind', 'random'))
    t0 = time.time()
    d = H_.joint_design(net, p, o['rule'], o['core'])
    tdesign = time.time() - t0
    seed = seed_of(K, drop, L, cfg.get('setup', 1))
    out = dict(cfg=cfg, nS=d.nS, G=d.G, served=[d.name(j) for j in d.served], design_sec=tdesign)
    lamQ = o['lamQ']
    base = H_.pose_from(net['p0'], net['angles0'])
    if 'prop' in methods:
        t1 = time.time(); c1 = time.process_time()
        make_ev = lambda dd: A.Evaluator(net, dd, p, seed=o['seed'], init=o['init'], lamQ=lamQ, max_iter=o['ev_iter'], tol=o['ev_tol'])
        if o.get('select'):
            # stream-set test of the new discrete design, with the evaluations of this scheme at its initial design
            import design_select as DSel
            d, ev, sel_info = DSel.select(net, p, d, dict(n_init=o['n_init'], bo=o.get('bo', {})), seed, make_ev, hrs_init=o.get('hrs_init'))
            out.update(select=sel_info, nS=d.nS, G=d.G, served=[d.name(j) for j in d.served])
            if ev is None:      # HRS kept: the earlier run of the HRS design is the result
                out['reuse'] = 'hrs'
                return out
        else:
            ev = make_ev(d)
        recs, ib, tbo = A.bayes_opt_fast(ev, net, p, o['budget_prop'], seed, n_init=o['n_init'], log=log, **o.get('bo', {}))
        best = recs[ib]
        # AO refinement at the selected deployment, continuing from the evaluator's solution
        ch = H_.channel_terms(best['pose'], net, p)
        t2 = time.time(); c2 = time.process_time()
        if o['refine'] == 'rule':
            ref = A.refine(best['theta'], best['W'], ch, net, d, p, ev.qtp, lamQ, qt_iter=20, qt_tol=0.1, log=log, init=o['init'])
        else:
            ref = H_.ao(best['theta'], best['W'], ch, net, d, p, lamQ, max_iter=o['ao_iter'], prec_steps=o['ao_prec'],
                        coef_steps=o['ao_coef'], tol=o['ao_tol'], log=log)
        tref = time.time() - t2; cref = time.process_time() - c2
        out['prop'] = dict(bo_merit=best['merit'], bo_rate=best['rate'], bo_deficit=best['deficit'],
                           merit=ref['merit'], rate=ref['rate'], deficit=ref['deficit'],
                           feasible=ref['deficit'] <= 1e-6, bo_sec=tbo, ref_sec=tref, sec=time.time() - t1,
                           ev_calls=sum(r['calls'] for r in recs), ref_calls=ref['calls'], ref_iters=ref['iterations'],
                           init_merit=recs[0]['merit'], init_rate=recs[0]['rate'], best_q=ib + 1,
                           trace=[r['merit'] for r in recs], ev_sec=[r['seconds'] for r in recs],
                           hw=H_.hardware_residual(ref['theta']), ref_gain=ref['merit'] - best['merit'],
                           cpu=time.process_time() - c1, ref_cpu=cref, ev_cpu=[r['cpu'] for r in recs])
        if o.get('composition'):
            out['prop']['pose'] = dict(pos=best['pose']['pos'].tolist(), angles=best['pose']['angles'].tolist())
            out['comp'] = CM.composition(ref['W'], ref['H'], d, p)
            if d.served_groups:
                # ablation at the selected deployment: same algorithm without group-common streams
                dn = CM.nogroup_design(d, p)
                evn = A.Evaluator(net, dn, p, seed=o['seed'], init=o['init'], lamQ=lamQ, max_iter=o['ev_iter'], tol=o['ev_tol'])
                rn = evn(best['pose'])
                refn = A.refine(rn['theta'], rn['W'], ch, net, dn, p, evn.qtp, lamQ, qt_iter=20, qt_tol=0.1, init=o['init'])
                out['nogroup'] = dict(rate=refn['rate'], merit=refn['merit'], deficit=refn['deficit'], feasible=refn['deficit'] <= 1e-6,
                                      comp=CM.composition(refn['W'], refn['H'], dn, p))
    if 'ao' in methods:
        t1 = time.time(); c1 = time.process_time()
        ev = A.AOEvaluator(net, d, p, seed=o['seed'], init=o['init'], lamQ=lamQ, max_iter=o['ao_iter'],
                           prec_steps=o['ao_prec'], coef_steps=o['ao_coef'], tol=o['ao_tol'])
        recs, ib, tbo = A.bayes_opt_fast(ev, net, p, o['budget_ao'], seed, n_init=o['n_init'], log=log, **o.get('bo', {}))
        best = recs[ib]
        out['ao'] = dict(merit=best['merit'], rate=best['rate'], deficit=best['deficit'], feasible=best['deficit'] <= 1e-6,
                         sec=time.time() - t1, calls=sum(r['calls'] for r in recs), best_q=ib + 1,
                         trace=[r['merit'] for r in recs], ev_sec=[r['seconds'] for r in recs],
                         iters=[r['iters'] for r in recs], cpu=time.process_time() - c1, ev_cpu=[r['cpu'] for r in recs])
    return out


if __name__ == '__main__':
    cfg = dict(K=12, N=3, M=300, Smax=6, Rth=0.5, drop=801)
    r = run_config(cfg, log=print)
    print(json.dumps({k: v for k, v in r.items() if k != 'cfg'}, indent=1, default=float)[:3000])


def run_prop_replay(cfg, old, opt=None):
    """Proposed framework with budget o['budget_prop'], reusing the first len(old['trace']) evaluations of an
    earlier run with the same seed (deterministic BO). CPU accounting: earlier BO-stage CPU + new evaluations."""
    o = dict(OPT); o.update(opt or {})
    K, N, M, Smax, Rth = cfg['K'], cfg['N'], cfg['M'], cfg['Smax'], cfg['Rth']
    L = cfg.get('L', 3); Dmax = cfg.get('Dmax', 10.0); Pmax = cfg.get('Pmax', 1.0); drop = cfg['drop']
    p = H_.params(K=K, L=L, N=N, M=M, Smax=Smax, Pmax=Pmax, Dmax=Dmax, Rth=Rth, bank=BANK)
    net = H_.make_network(p, BANK, drop=drop, setup=cfg.get('setup', 1), kind=cfg.get('kind', 'random'))
    d = H_.joint_design(net, p, o['rule'], o['core'])
    seed = seed_of(K, drop, L, cfg.get('setup', 1))
    out = dict(cfg=cfg, nS=d.nS, G=d.G, served=[d.name(j) for j in d.served])
    lamQ = o['lamQ']
    ev = A.Evaluator(net, d, p, seed=o['seed'], init=o['init'], lamQ=lamQ, max_iter=o['ev_iter'], tol=o['ev_tol'])
    nrep = len(old['trace'])
    recs, ib, tbo, c_new = A.bayes_opt_fast(ev, net, p, o['budget_prop'], seed, n_init=o['n_init'], replay=old['trace'])
    best = recs[ib]
    ch = H_.channel_terms(best['pose'], net, p)
    t2 = time.time(); c2 = time.process_time()
    ref = A.refine(best['theta'], best['W'], ch, net, d, p, ev.qtp, lamQ, qt_iter=20, qt_tol=0.1, init=o['init'])
    tref = time.time() - t2; cref = time.process_time() - c2
    new = recs[nrep:]
    bo_cpu_old = old['cpu'] - old['ref_cpu']
    out['prop'] = dict(bo_merit=best['merit'], bo_rate=best['rate'], bo_deficit=best['deficit'],
                       merit=ref['merit'], rate=ref['rate'], deficit=ref['deficit'], feasible=ref['deficit'] <= 1e-6,
                       ref_sec=tref, ev_calls=old['ev_calls'] + sum(r['calls'] for r in new), ref_calls=ref['calls'],
                       ref_iters=ref['iterations'], init_merit=old['init_merit'], init_rate=old['init_rate'], best_q=ib + 1,
                       trace=list(old['trace']) + [r['merit'] for r in new],
                       ev_sec=list(old['ev_sec']) + [r['seconds'] for r in new],
                       ev_cpu=list(old['ev_cpu']) + [r['cpu'] for r in new],
                       hw=H_.hardware_residual(ref['theta']), ref_gain=ref['merit'] - best['merit'],
                       cpu=bo_cpu_old + c_new + cref, ref_cpu=cref, bo_cpu=bo_cpu_old + c_new,
                       sec=bo_cpu_old + c_new + cref, bo_sec=bo_cpu_old + c_new, replayed=nrep)
    if o.get('composition'):
        out['prop']['pose'] = dict(pos=best['pose']['pos'].tolist(), angles=best['pose']['angles'].tolist())
        out['comp'] = CM.composition(ref['W'], ref['H'], d, p)
        if d.served_groups:
            dn = CM.nogroup_design(d, p)
            evn = A.Evaluator(net, dn, p, seed=o['seed'], init=o['init'], lamQ=lamQ, max_iter=o['ev_iter'], tol=o['ev_tol'])
            rn = evn(best['pose'])
            refn = A.refine(rn['theta'], rn['W'], ch, net, dn, p, evn.qtp, lamQ, qt_iter=20, qt_tol=0.1, init=o['init'])
            out['nogroup'] = dict(rate=refn['rate'], merit=refn['merit'], deficit=refn['deficit'], feasible=refn['deficit'] <= 1e-6,
                                  comp=CM.composition(refn['W'], refn['H'], dn, p))
    return out
