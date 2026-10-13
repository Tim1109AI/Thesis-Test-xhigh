"""Run a discrete-design variant on the test grid.

    python3 harness/grid.py VARIANT --tier screen     (Q = 5: both scenarios, all settings, 4 drops)
    python3 harness/grid.py VARIANT --tier confirm    (Q = 50: both scenarios, all settings, 3 drops)
    python3 harness/grid.py VARIANT --tier bench      (Q = 50: the setting of the benchmark figure, Fig. 5.10, T2)
    python3 harness/grid.py VARIANT@SC --tier bench   (stream clustering on the streams of VARIANT's design, same runs)
    python3 harness/grid.py VARIANT --tier smax       (Q = 50: sum rate versus S_max, T2)
    python3 harness/grid.py VARIANT [--tier ...] [--scenarios T2,E3] [--settings a,...,f] [--rth 0,...,2.5]
                                    [--drops n] [--budget Q] [--workers N]      (without --tier: Q = 50, 4 drops)
    python3 harness/grid.py VARIANT --smoke          (harness check: 2 short runs, about 1 minute)

VARIANT: a key of harness/variants.py, or KEY@SC for the stream-clustering benchmark (code/benchmarks.py) built on the
streams of KEY's design. Writes results/<VARIANT>__<scenario>.jsonl (Q = 50), results/<VARIANT>__<scenario>__Q<budget>.jsonl
(other budgets) or results/<VARIANT>__T2__bench.jsonl / __smax.jsonl, one line per run (resumable:
finished runs are skipped). Jobs are ordered drop by drop, so a partial or --drops-limited grid is balanced over
scenarios, settings and R_th values; a later call with more drops completes the same files.

Scenarios (code/scenarios.py; area 100 m x 100 m): T2 (the thesis: 2 BSs, uniform UEs), E3 (3 BSs far apart near the
edge, uniform UEs).
Settings (L = 3, P_max = 1 W per BS, D_max = 10 m, search R with budget Q = 50 and n_init = 5, as in the main runs of
the thesis; S_max per BS):
"""
import os, sys, json, argparse, importlib, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code')); sys.path.insert(0, os.path.join(ROOT, 'harness'))
import multiprocessing as mp

SETTINGS = {
    'a': dict(N=3, K=8, M=300, Smax=12),      # S_max = 4N
    'b': dict(N=3, K=12, M=300, Smax=4),      # S_max = N + 1
    'c': dict(N=3, K=16, M=300, Smax=6),      # S_max = 2N
    'd': dict(N=5, K=12, M=500, Smax=6),      # S_max = N + 1
    'e': dict(N=5, K=16, M=500, Smax=10),     # S_max = 2N
    'f': dict(N=5, K=16, M=300, Smax=20),     # S_max = 4N
}
RTH = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
DROPS = list(range(901, 917))       # 16 UE drops; --drops n takes the first n (default 4)
SCEN = ('T2', 'E3')
# preset blocks of runs: (scenarios, settings, number of drops); README.md, "Compute plan"
TIERS = {
    'screen': dict(budget=5, blocks=[(SCEN, tuple(SETTINGS), 4)]),
    'confirm': dict(budget=50, blocks=[(SCEN, tuple(SETTINGS), 3)]),
}
# the setting of the benchmark figure (Fig. 5.10 of the thesis; the shares of the common streams in Fig. 5.7 are those of
# the same runs): T2, K = 16, N = 3, S_max = 12, R_th = 2, M in {100, 300, 500}, drops 801-805
BENCH = [dict(N=3, K=16, M=M, Smax=12, setting=f'bench_M{M}', Rth=2.0, drop=dr, scenario='T2')
         for dr in range(801, 806) for M in (100, 300, 500)]
# sum rate versus S_max (Fig. 5.5 of the thesis: N = 3, K in {8, 12, 16}): T2, M = 300, R_th in {0, 1, 2}, drops 901-902
SMAX = [dict(N=3, K=K, M=300, Smax=S, setting=f'smax_K{K}_S{S}', Rth=R, drop=dr, scenario='T2')
        for dr in (901, 902) for K in (8, 12, 16) for S in (4, 6, 8, 10, 12, 14, 16) for R in (0.0, 1.0, 2.0)]
SPECIAL = {'bench': BENCH, 'smax': SMAX}
__doc__ += ''.join(f"  {k}: N={v['N']}, K={v['K']}, M={v['M']}, S_max={v['Smax']}\n" for k, v in SETTINGS.items())
__doc__ += f"R_th in {RTH} bit/s/Hz; drops {DROPS[0]}..{DROPS[-1]} (the first 4 in the screen, 3 in the confirm tier).\n"


def base_of(variant):
    return variant[:-3] if variant.endswith('@SC') else variant


def design_fn(variant):
    import variants as V
    if variant.endswith('@SC'):         # the stream-clustering benchmark on the streams of the base design
        import benchmarks as BM
        base = design_fn(base_of(variant))
        return lambda net, p: BM.stream_cluster_design(net, p, base(net, p))
    mod, fn = V.VARIANTS[variant]['design'].split(':')
    return getattr(importlib.import_module(mod), fn)


def run(job):
    variant, cfg, budget, tag = job
    import variants as V, discrete as DC
    t = time.process_time()
    r = DC.run(cfg, design_fn(variant), budget=budget, opt=V.VARIANTS[base_of(variant)].get('opt'),
               framing=not variant.endswith('@SC'))
    rec = dict(variant=variant, file=tag, cfg=cfg, budget=budget)
    rec.update(r)
    rec['cpu'] = time.process_time() - t
    return rec


def key(cfg):
    return (cfg['scenario'], cfg['setting'], cfg['drop'], float(cfg['Rth']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant')
    ap.add_argument('--tier', choices=sorted(TIERS) + sorted(SPECIAL), default=None, help='a preset block of runs (README.md)')
    ap.add_argument('--scenarios', default=None)
    ap.add_argument('--rth', default=None)
    ap.add_argument('--settings', default=None)
    ap.add_argument('--drops', type=int, default=None)
    ap.add_argument('--budget', type=int, default=None, help='search budget Q (default 50; the tier sets its own)')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    import variants as V
    assert base_of(a.variant) in V.VARIANTS, f'unknown variant {a.variant}; add it to harness/variants.py'
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    base = dict(Dmax=10.0, L=3, Pmax=1.0)
    if a.tier in SPECIAL:
        budget = a.budget or 50
        rths = [float(r) for r in a.rth.split(',')] if a.rth else None
        cfgs = [dict(c, **base) for c in SPECIAL[a.tier] if (rths is None or c['Rth'] in rths)
                and (not a.settings or c['setting'] in a.settings.split(','))]      # settings: e.g. bench_M100, smax_K16_S8
        if a.drops:
            keep = sorted({c['drop'] for c in cfgs})[:a.drops]; cfgs = [c for c in cfgs if c['drop'] in keep]
        files = {f'{a.variant}__T2__{a.tier}' + ('' if budget == 50 else f'__Q{budget}'): cfgs}
    elif a.smoke:
        files = {f'{a.variant}__smoke': [dict(SETTINGS['a'], setting='a', Rth=1.0, drop=901, scenario=sc, **base) for sc in SCEN]}
        budget = 8
    else:
        tier = TIERS[a.tier] if a.tier else dict(budget=50, blocks=[(SCEN, tuple(SETTINGS), 4)])
        budget = a.budget or tier['budget']
        rths = [float(r) for r in a.rth.split(',')] if a.rth else list(RTH)
        files = {}
        for scens, sets, nd in tier['blocks']:
            scens = a.scenarios.split(',') if a.scenarios else scens
            sets = a.settings.split(',') if a.settings else sets
            drops = DROPS[:a.drops or nd]
            for sc in scens:
                assert sc in SCEN, f'unknown scenario {sc}'
                tag = f'{a.variant}__{sc}' + ('' if budget == 50 else f'__Q{budget}')
                files.setdefault(tag, [])
                files[tag] += [dict(SETTINGS[st], setting=st, Rth=r, drop=dr, scenario=sc, **base) for dr in drops for st in sets for r in rths]
        files = {tag: list({key(c): c for c in cfgs}.values()) for tag, cfgs in files.items()}      # no duplicates
    jobs = []
    for tag, cfgs in files.items():
        outp = os.path.join(ROOT, 'results', tag + '.jsonl')
        done = set()
        if os.path.exists(outp):
            for l in open(outp):
                done.add(key(json.loads(l)['cfg']))
        todo = [c for c in cfgs if key(c) not in done]
        print(f'{tag}: {len(todo)} runs to do ({len(cfgs) - len(todo)} done) -> {outp}', flush=True)
        jobs += [(a.variant, c, budget, tag) for c in todo]
    # drop by drop over all files, so that partial results are balanced
    order = {d: i for i, d in enumerate(DROPS)}
    jobs.sort(key=lambda j: (order.get(j[1]['drop'], 0), SCEN.index(j[1]['scenario']), j[1]['setting'], j[1]['Rth']))
    if not jobs:
        print('DONE', flush=True); return
    t0 = time.time(); fh = {}
    # 'spawn': fresh worker processes (forked workers occasionally hung after recycling)
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=10) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs)):
            tag = r['file']
            if tag not in fh: fh[tag] = open(os.path.join(ROOT, 'results', tag + '.jsonl'), 'a')
            fh[tag].write(json.dumps(r) + '\n'); fh[tag].flush()
            c = r['cfg']
            print(f"  [{i + 1}/{len(jobs)}] {c['scenario']} {c['setting']} drop {c['drop']} R_th {c['Rth']:g}: rate {r['rate']:.2f}"
                  f"{'' if r['qos_met'] else ' (QoS missed)'}, {r['nS']} streams{' BS-specific' if r['bs_specific'] else ''}"
                  f"  ({r['cpu']:.0f} CPU-s, {time.time() - t0:.0f} s elapsed)", flush=True)
    for f in fh.values(): f.close()
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
