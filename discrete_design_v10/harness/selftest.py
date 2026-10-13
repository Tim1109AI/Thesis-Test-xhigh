"""Self-test: reproduce reference runs of the framework (environment check).

    python3 harness/selftest.py          (about 2 minutes)
    python3 harness/selftest.py --write  (coordinating session only: rewrite reference/selftest.jsonl)

Cases: the current design (JH) and its 1-layer RSMA (J1) in the thesis scenario T2 with the full budget, J derived from
them (it must equal the run of the thesis framework with the stream-set test, ablation.run_scheme(..., select=True)),
JH in scenario E3 (3 BSs), and a BS-specific association (the random fronthaul ablation of the thesis, seeded; it may
leave the super-common stream to some BSs, so that check is off for it) in scenario E3 for the bsmask code path, the
last two with a short budget; and sparse beamforming (code/benchmarks.py) in the benchmark setting (drop 801, M = 100),
whose trace must equal the stored run of reference/bench_fig510.jsonl.
"""
import os, sys, json, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code')); sys.path.insert(0, os.path.join(ROOT, 'harness'))
import multiprocessing as mp

BASE = dict(Dmax=10.0, L=3, Pmax=1.0)
C970 = dict(BASE, scenario='T2', N=3, K=8, M=500, Smax=12, Rth=2.0, drop=970)
CE = dict(BASE, scenario='E3', N=5, K=16, M=500, Smax=10, Rth=1.0, drop=901)
CB = dict(BASE, scenario='E3', N=3, K=12, M=300, Smax=6, Rth=1.0, drop=902)
CS = dict(BASE, scenario='T2', N=3, K=16, M=100, Smax=12, Rth=2.0, drop=801)
# name, design, cfg, budget
CASES = [('JH_T2', 'current', C970, 50), ('J1_T2', 'current_1layer', C970, 50),
         ('JH_E3', 'current', CE, 10), ('rand_fronthaul_E3', 'rand_fronthaul', CB, 10), ('sparse_bf_T2', 'sparse_bf', CS, 50)]


def one(case):
    import discrete as DC
    name, fn, cfg, budget = case
    t = time.process_time()
    if fn == 'thesis_select':      # the thesis framework with the stream-set test (ablation.run_scheme), T2 only
        import ablation as AB
        c = {k: v for k, v in cfg.items() if k != 'scenario'}
        r = AB.run_scheme(c, 'proposed', dict(budget=budget, select=True))
        return name, dict(name=name, cfg=cfg, budget=budget, trace=r['trace'], rate=r['rate'], select=r.get('select')), time.process_time() - t
    if fn == 'rand_fronthaul':      # the random fronthaul ablation of the thesis (seeded as there)
        import ablation as AB, run as R
        def f(net, p):
            seed = R.seed_of(p['K'], cfg['drop'], p['L'], 1)
            return AB.design('rand_fronthaul', net, p, DC.current(net, p), AB._rng(seed, 'rand_fronthaul'))
    elif fn == 'sparse_bf':          # the benchmark of requirement R2 (stored run: reference/bench_fig510.jsonl)
        import benchmarks as BM
        f = lambda net, p: BM.sparse_bf_design(net, p, None, 200.0)
    else:
        f = getattr(DC, fn)
    r = DC.run(cfg, f, budget=budget, framing=fn not in ('rand_fronthaul', 'sparse_bf'))
    return name, dict(name=name, cfg=cfg, budget=budget, trace=r['trace'], rate=r['rate'], deficit=r['deficit'], served=r['served'],
                      bs_streams=r.get('bs_streams')), time.process_time() - t


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--write', action='store_true'); a = ap.parse_args()
    fn = os.path.join(ROOT, 'reference', 'selftest.jsonl')
    cases = CASES + ([('J_T2_select', 'thesis_select', C970, 50)] if a.write else [])
    ref = {} if a.write else {json.loads(l)['name']: json.loads(l) for l in open(fn)}
    ok = True; got = {}
    with mp.get_context('spawn').Pool(min(len(cases), os.cpu_count()), maxtasksperchild=1) as pool:
        for name, r, cpu in pool.imap(one, cases):
            got[name] = r
            if a.write:
                print(f'{name}: rate {r["rate"]:.4f} [{cpu:.0f} CPU-s]', flush=True); continue
            tr, rt = r['trace'], ref[name]['trace']
            same = len(tr) == len(rt) and all(x == y for x, y in zip(tr, rt))
            diff = max(abs(x - y) for x, y in zip(tr, rt)) if len(tr) == len(rt) else float('inf')
            print(f'{name:20s}: identical trace: {same}' + ('' if same else f'  (max |diff| {diff:.3g})') + f'  rate {r["rate"]:.4f}  [{cpu:.0f} CPU-s]', flush=True)
            ok &= bool(diff < 1e-4)
    # J derived from JH and J1 (the stream-set test at the first 5 evaluations) = the thesis framework's run
    h, o = got['JH_T2'], got['J1_T2']
    j = h if max(h['trace'][:5]) >= max(o['trace'][:5]) else o
    jref = got['J_T2_select'] if a.write else ref['J_T2_select']
    same = j['trace'] == jref['trace']
    print(f'{"J_T2 (derived)":20s}: equals the thesis framework run with the stream-set test: {same}', flush=True)
    if a.write:
        with open(fn, 'w') as f:
            for name in got: f.write(json.dumps(got[name]) + '\n')
        print('written', fn)
    else:
        ok &= same or max(abs(x - y) for x, y in zip(j['trace'], jref['trace'])) < 1e-4
        print('SELFTEST PASSED' if ok else 'SELFTEST FAILED (traces differ beyond rounding; report this)')
