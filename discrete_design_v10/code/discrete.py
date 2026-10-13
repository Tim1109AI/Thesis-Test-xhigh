"""Discrete designs (grouping, served streams, stream-to-BS association) and the run of the framework with a design.

A discrete design is a function   design(net, p) -> hrs.Design   computed once per drop and setting, before the
deployment search. The Design (hrs.Design) holds:
  groups     list of disjoint UE lists (each of size 2 .. K-1, at most floor(K/2) groups); group g has global id 1 + g
  served     the served global stream ids: 0 = super-common, 1..G = group-common, G + 1 + k = private stream of UE k
  bsmask     None (every BS carries every served stream: joint transmission) or a (B, nS) bool array: BS b carries
             served column i (columns in the order of d.served); set d.bs_sets = [sorted global ids carried by BS b]
  core       UEs whose private streams start the AO (algo.init_core: regularized ZF over their served private streams
             on the stacked channel, dominant eigenvector for the super-common stream, every other served stream at the
             noise level; the blocks of a BS that does not carry a stream are zeroed)
  priority   a stream order (used by composition.nogroup_design and some starts); rule: a name
check(d, p) tests the constraints of the problem (disjoint groups of admissible size, at most S_max streams per BS; a
stream counts only at the BSs that carry it) and the requirement of this round (BRIEF.md) that the super-common
stream is served and carried by every BS.

Designs of the current thesis (the baselines of the harness):
  current          joint transmission, SUS core users, nearest-core groups, fixed stream order (hrs.joint_design)
  current_1layer   1-layer RSMA of it: the group-common streams removed, their slots to the next private streams
The thesis framework ("J") also makes the stream-set test (design_select.select) between the two; tools/report.py
derives J from the runs of 'current' and 'current_1layer' (the search is deterministic, so J's run is the run of the
set the test keeps).

run(cfg, design_fn, budget) is the proposed framework with a given design: AO evaluator (core-user start, tolerance
3), deployment search (algo.bayes_opt_fast = design R, trimmed), refinement at the selected deployment (tolerance 0.1).
The network of a run comes from code/scenarios.py (T2: B = 2, E3: B = 3).
"""
import time
import numpy as np
import hrs as H_, algo as A, run as R, newprop as NP
import composition as CM
import scenarios as SC


# --------------------------------------------------------------------------- baselines
def current(net, p):
    """The thesis design (HRS order, joint transmission)."""
    return H_.joint_design(net, p, 'private_first', 'sus')


def current_1layer(net, p):
    """1-layer RSMA of the thesis design (super-common and private streams, joint transmission)."""
    return CM.nogroup_design(current(net, p), p)


# --------------------------------------------------------------------------- constraints
def check(d, p, framing=True):
    """Raises AssertionError if the design violates a constraint of the problem or (framing=True) leaves the
    super-common stream unserved or to some BSs only."""
    K, B, S = p['K'], p['B'], p['Smax']
    seen = set()
    for m in d.groups:
        assert 2 <= len(m) <= K - 1, f'group size {len(m)} not in 2..K-1'
        assert not (set(m) & seen), 'groups overlap'; seen |= set(m)
        assert all(0 <= k < K for k in m)
    assert d.G <= K // 2, 'more than floor(K/2) groups'
    assert len(set(d.served)) == len(d.served) and all(0 <= j < d.J for j in d.served), 'invalid served streams'
    if d.bsmask is None:
        assert d.nS <= S, f'{d.nS} streams > S_max = {S}'
    else:
        bm = np.asarray(d.bsmask, bool)
        assert bm.shape == (B, d.nS), 'bsmask must be (B, nS)'
        assert np.all(bm.sum(1) <= S), f'a BS carries more than S_max = {S} streams'
        assert np.all(bm.any(0)), 'a served stream is carried by no BS'
    assert hasattr(d, 'core') and all(0 <= k < K for k in d.core), 'd.core (UE list for the AO start) missing'
    if framing:
        assert 0 in d.col, 'the super-common stream must be served'
        if d.bsmask is not None:
            assert np.asarray(d.bsmask, bool)[:, d.col[0]].all(), 'the super-common stream must be carried by every BS'


def streams_of(d):
    """Summary of a design for the records."""
    out = dict(nS=d.nS, G=d.G, served=[d.name(j) for j in d.served], n_groups_served=len(d.served_groups),
               group_sizes=[len(d.groups[g]) for g in d.served_groups], n_private=len(d.private_served),
               super_common=0 in d.col, bs_specific=d.bsmask is not None)
    if d.bsmask is not None:
        out['bs_streams'] = [[d.name(d.served[i]) for i in np.flatnonzero(row)] for row in np.asarray(d.bsmask, bool)]
    return out


# --------------------------------------------------------------------------- one run of the framework
def setup(cfg):
    """Settings, network and seed of a run: cfg holds scenario (code/scenarios.py), K, N, M, Smax, Rth, drop and
    optionally L, Pmax, Dmax."""
    p, net = SC.network(cfg)
    seed = R.seed_of(cfg['K'], cfg['drop'], p['L'], cfg.get('setup', 1))
    return p, net, seed


def run(cfg, design_fn, budget=50, opt=None, framing=True):
    """The framework with the discrete design design_fn(net, p). Returns a record (rate counts as 0 if the QoS
    shortfall exceeds 1e-2 bit/s/Hz). framing=False skips the super-common check (the benchmarks of
    code/benchmarks.py and the self-test)."""
    o = dict(NP.NEW, budget=budget, **(opt or {}))
    p, net, seed = setup(cfg)
    c0 = time.process_time()
    d = design_fn(net, p)
    cpu_design = time.process_time() - c0
    check(d, p, framing)
    ev = A.AOEvaluator(net, d, p, seed=o['seed'], init=o['init'], lamQ=o['lamQ'], max_iter=o['max_iter'], prec_steps=o['prec_steps'],
                       coef_steps=o['coef_steps'], tol=o['tol_bo'], stable_req=o['stable'])
    recs, ib, _ = A.bayes_opt_fast(ev, net, p, o['budget'], seed, n_init=o['n_init'], ang_scale=0.15)
    best = recs[ib]
    ch = H_.channel_terms(best['pose'], net, p)
    ref = NP._ao(best['theta'], best['W'], ch, net, d, p, o, o['tol_ref'], ev.qtp, ev.qtc)
    ok = ref['deficit'] <= 1e-2
    out = dict(rate=float(ref['rate']) if ok else 0.0, rate_raw=float(ref['rate']), deficit=float(ref['deficit']), qos_met=bool(ok),
               merit=float(ref['merit']), bo_rate=float(best['rate']), best_q=ib + 1, n_evals=len(recs),
               trace=[float(r['merit']) for r in recs], ev_cpu=[float(r['cpu']) for r in recs], cpu=time.process_time() - c0,
               cpu_design=cpu_design,
               pose=dict(pos=np.asarray(best['pose']['pos']).tolist(), angles=np.asarray(best['pose']['angles']).tolist()))
    out.update(streams_of(d))
    out['comp'] = CM.composition(ref['W'], ref['H'], d, p)
    return out
