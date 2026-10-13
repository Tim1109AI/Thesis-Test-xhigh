"""Choice of the served stream set (new discrete design).

The group-common streams take fronthaul slots that could carry private streams of non-core UEs. Whether they pay off
depends on the QoS load relative to the channel quality, which a channel-only rule cannot see. Two stream sets are
therefore considered when the order of the joint-transmission design serves group-common streams:
    HRS    the order of the joint-transmission design (core privates, group-common streams, non-core privates);
    1-L    the same order without the group-common streams (1-layer RSMA, i.e., HRS with no group-common stream).
Both are evaluated by AO_{eps_E} at the deployments of the initial design of BO, and the set with the larger best
value of F is kept for the rest of the search (HRS on ties). When the order serves no group-common stream, both sets
coincide and nothing is tested. The evaluations of the kept set are the first evaluations of its BO run.
Schemes without deployment search (prescribed deployment, random deployment, no DS-STAR) make the same test at their
single deployment (select_at): both sets are evaluated there by the scheme's own evaluation and the larger F is kept."""
import numpy as np
import hrs as H_, algo as A
import composition as CM


def initial_design(net, p, seed, n_init, radial='volume', sampler='box', fix_angles=False):
    """The deployments of the initial design of bayes_opt_fast (same generator and seed)."""
    return A.initial_design(net, p, seed, n_init, sampler=sampler, radial=radial, fix_angles=fix_angles)


class CachedEvaluator:
    """Returns the stored evaluations for the first deployments of the search, then evaluates."""
    def __init__(self, ev, recs):
        self.ev = ev; self.recs = list(recs); self.qtp = ev.qtp; self.qtc = getattr(ev, 'qtc', None)

    def __call__(self, pose):
        if self.recs:
            r = self.recs.pop(0)
            assert np.allclose(r['pose']['pos'], pose['pos']) and np.allclose(r['pose']['angles'], pose['angles']), 'initial design mismatch'
            return dict(r)
        return self.ev(pose)


def select(net, p, d, o, seed, make_ev, hrs_init=None):
    """Returns (design, evaluator for its BO run, info). make_ev(design) builds an AO evaluator.

    hrs_init: the values of F of the HRS set at the initial design, taken from an earlier run of the HRS design with
    the same seed (deterministic, so identical to evaluating them again). Then only the set without group-common
    streams is evaluated; if HRS is kept, the evaluator returned is None and the earlier HRS run is the result."""
    if not d.served_groups:
        return d, make_ev(d), dict(tested=False, chosen='hrs')
    kw = dict(radial=o.get('radial', 'volume'), sampler=o.get('sampler', 'box'), fix_angles=o.get('fix_angles', False))
    kw.update({k: v for k, v in o.get('bo', {}).items() if k in ('sampler', 'scale_min', 'init_strata')})
    init = A.initial_design(net, p, seed, o['n_init'], **kw)
    if hrs_init is not None:
        assert len(hrs_init) == len(init), 'hrs_init must hold the values at the initial design'
        dn = CM.nogroup_design(d, p); ev = make_ev(dn); recs = []
        for pose in init:
            r = ev(pose); r['pose'] = pose; recs.append(r)
        bh, bn = float(max(hrs_init)), float(max(r['merit'] for r in recs))
        info = dict(tested=True, hrs_init_reused=True, chosen='hrs' if bh >= bn else 'one_layer', init_best=dict(hrs=bh, one_layer=bn),
                    one_layer_init_trace=[float(r['merit']) for r in recs], one_layer_init_cpu=float(sum(r['cpu'] for r in recs)),
                    one_layer_init_calls=int(sum(r['calls'] for r in recs)))
        if bh >= bn:
            return d, None, info
        return dn, CachedEvaluator(ev, recs), info
    cands = [('hrs', d), ('one_layer', CM.nogroup_design(d, p))]
    res = []
    for name, dd in cands:
        ev = make_ev(dd); recs = []
        for pose in init:
            r = ev(pose); r['pose'] = pose; recs.append(r)
        res.append((name, dd, ev, recs, max(r['merit'] for r in recs)))
    k = 0 if res[0][4] >= res[1][4] else 1
    name, dd, ev, recs, _ = res[k]
    other = res[1 - k]
    info = dict(tested=True, chosen=name, init_best={r[0]: float(r[4]) for r in res},
                other_init_cpu=float(sum(r['cpu'] for r in other[3])), other_init_trace=[float(r['merit']) for r in other[3]],
                other_init_calls=int(sum(r['calls'] for r in other[3])))
    return dd, CachedEvaluator(ev, recs), info


def select_at(d, p, evaluate, dn=None, hrs_merit=None):
    """The test for a scheme without deployment search: both stream sets are evaluated at its single deployment.
    evaluate(design) returns an evaluation dict with 'merit'; dn is the design without group-common streams (built
    if not given). Returns (design, its evaluation, info). With hrs_merit (the evaluation of the HRS set from an
    earlier run), only the other set is evaluated, and the evaluation returned is None if HRS is kept."""
    if not d.served_groups:
        return d, evaluate(d), dict(tested=False, chosen='hrs')
    dn = CM.nogroup_design(d, p) if dn is None else dn
    if hrs_merit is not None:
        en = evaluate(dn)
        k = 0 if hrs_merit >= en['merit'] else 1
        info = dict(tested=True, hrs_init_reused=True, chosen=('hrs', 'one_layer')[k], init_best=dict(hrs=float(hrs_merit), one_layer=float(en['merit'])),
                    one_layer_init_cpu=float(en.get('cpu', 0.0)), one_layer_init_calls=int(en.get('calls', 0)))
        return (d, dn)[k], (None, en)[k], info
    eh = evaluate(d); en = evaluate(dn)
    k = 0 if eh['merit'] >= en['merit'] else 1
    info = dict(tested=True, chosen=('hrs', 'one_layer')[k], init_best=dict(hrs=float(eh['merit']), one_layer=float(en['merit'])),
                other_init_cpu=float((en, eh)[k].get('cpu', 0.0)), other_init_calls=int((en, eh)[k].get('calls', 0)))
    return (d, dn)[k], (eh, en)[k], info
