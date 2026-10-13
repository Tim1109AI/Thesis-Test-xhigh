"""Score a discrete-design variant against the requirements of BRIEF.md.

    python3 tools/report.py VARIANT [--budget 5] [--scenarios T2,E3] [--by-setting] [--json]    the test grid
    python3 tools/report.py VARIANT --bench [--json]                                           the benchmark setting
    python3 tools/report.py VARIANT --smax [--json]                                            sum rate versus S_max

Test grid (requirement 1): per scenario, on the runs common to the schemes that have results (same setting, drop and
R_th):
  VARIANT        your design, and its 1-layer RSMA counterpart (the variant that harness/variants.py names with
                 'one_layer': the same selection logic without group-common streams); VARIANT - one_layer is the
                 comparison of requirement 1
  J, JH, J1      for reference, when their runs exist: the current thesis framework (JH or J1 by the stream-set test at
                 the initial design of the search, derived exactly from the runs of JH and J1), the current joint design
                 without the test, and its 1-layer RSMA
  and, per run, the group-common and private streams that only some of the BSs carry (the design requirement).
--bench (requirement 2 and the shares): the setting of the benchmark figure (results/VARIANT__T2__bench.jsonl):
  VARIANT against sparse beamforming (stored runs, reference/bench_fig510.jsonl) and stream clustering on the streams of
  VARIANT's design (results/VARIANT@SC__T2__bench.jsonl); for reference, the stored runs of the current framework (J)
  and of stream clustering on its streams; and the shares of the super-common and group-common streams in the sum rate
  of VARIANT (ratio of the summed rates over its runs that meet the QoS constraints, as in Fig. 5.7 of the thesis).
--smax (sum rate versus S_max): results/VARIANT__T2__smax.jsonl, mean over the drops per K, R_th and S_max, and the
  paired change between consecutive values of S_max.
Rates in bit/s/Hz, zero when the QoS constraints are not met (shortfall > 1e-2). Differences are paired: mean and
standard error. QoS: number of runs meeting the QoS constraints. Works on partial results. --budget Q reads the grid
runs with search budget Q (default 50). --json writes results/VARIANT__report[_Q<budget>|_bench|_smax].json.
"""
import os, sys, json, argparse
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'harness'))
N_INIT = 5
SCEN = ('T2', 'E3')


def read(fn, key):
    out = {}
    if os.path.exists(fn):
        for l in open(fn):
            r = json.loads(l); out[key(r['cfg'])] = r
    return out


def load(variant, budget=50):
    """{scenario: {(setting, drop, R_th): record}}"""
    out = {}
    for sc in SCEN:
        fn = os.path.join(ROOT, 'results', f'{variant}__{sc}' + ('' if budget == 50 else f'__Q{budget}') + '.jsonl')
        d = read(fn, lambda c: (c['setting'], c['drop'], float(c['Rth'])))
        if d: out[sc] = d
    return out


def derive_j(h, o):
    """The run of the thesis framework from the runs of JH (h) and J1 (o)."""
    tested = h['n_groups_served'] > 0
    keep_h = (not tested) or max(h['trace'][:N_INIT]) >= max(o['trace'][:N_INIT])
    r = dict(h if keep_h else o)
    r['cpu'] = r['cpu'] + (sum((o if keep_h else h)['ev_cpu'][:N_INIT]) if tested else 0.0)
    r['j_tested'], r['j_one_layer'] = tested, not keep_h
    return r


def stats(d):
    d = np.asarray(d, float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan')


def table(rows, names, ref_pairs, title, out, group=lambda k: k[2], glabel='R_th'):
    """rows: {key: {name: record}}; one line per group value and one for all."""
    print(title)
    print(f"{glabel:>7} {'n':>4} " + ' '.join(f'{n:>9}' for n in names) + ' ' + ' '.join(f"{a + '-' + b:>17}" for a, b in ref_pairs)
          + '   QoS met ' + '/'.join(names))
    res = []
    for gv in sorted({group(k) for k in rows}) + ['all']:
        ks = [k for k in rows if gv == 'all' or group(k) == gv]
        if not ks: continue
        mean = {n: float(np.mean([rows[k][n]['rate'] for k in ks])) for n in names}
        diffs = {f'{a}-{b}': stats([rows[k][a]['rate'] - rows[k][b]['rate'] for k in ks]) for a, b in ref_pairs}
        qos = {n: int(sum(rows[k][n]['qos_met'] for k in ks)) for n in names}
        lab = f'{gv:g}' if isinstance(gv, float) else str(gv)
        print(f"{lab:>7} {len(ks):>4} " + ' '.join(f'{mean[n]:9.2f}' for n in names) + ' '
              + ' '.join(f'{diffs[f"{a}-{b}"][0]:+8.2f}±{diffs[f"{a}-{b}"][1]:5.2f}   ' for a, b in ref_pairs)
              + '   ' + '/'.join(str(qos[n]) for n in names))
        res.append(dict(group=lab, n=len(ks), mean=mean, diff=diffs, qos=qos))
    out.append(dict(title=title, rows=res))
    return res


def on_subset(r, c):
    """Served streams of type c ('g' group-common, 'p' private) that only some of the BSs carry."""
    bs = r.get('bs_streams') if r.get('bs_specific') else None
    if not bs: return 0
    names = [n for row in bs for n in row if n[0] == c]
    return sum(1 for n in set(names) if names.count(n) < len(bs))


def streams_line(X, xs):
    print(f'-- {X}: mean served streams {np.mean([r["nS"] for r in xs]):.2f}, group-common served {np.mean([r["n_groups_served"] for r in xs]):.2f}, '
          f'private served {np.mean([r["n_private"] for r in xs]):.2f}; BS-specific in {sum(bool(r["bs_specific"]) for r in xs)}/{len(xs)} runs; '
          f'per run, carried by a subset of the BSs: group-common {np.mean([on_subset(r, "g") for r in xs]):.2f}, '
          f'private {np.mean([on_subset(r, "p") for r in xs]):.2f}')


def grid(a, V, X, X1):
    data = {n: load(n, a.budget) for n in dict.fromkeys([X] + ([X1] if X1 else []) + ['JH', 'J1'])}
    scens = [sc for sc in SCEN if sc in data[X]] if a.scenarios is None else a.scenarios.split(',')
    out = dict(variant=X, one_layer=X1, budget=a.budget, scenarios={})
    for s in scens:
        have = {n: data[n][s] for n in data if s in data[n]}
        if X not in have: continue
        keys = set(have[X])
        for n in have: keys &= set(have[n])
        print(f'\n===== scenario {s}, Q = {a.budget}: {len(keys)} common runs (' + ', '.join(f'{n}: {len(have[n])}' for n in have) + ')')
        if not keys:
            print('  (run the counterpart on the same runs)'); continue
        refj = 'JH' in have and 'J1' in have
        rows = {}
        for k in sorted(keys):
            rows[k] = {n: have[n][k] for n in have}
            if refj: rows[k]['J'] = derive_j(have['JH'][k], have['J1'][k])
        names = list(dict.fromkeys([X] + ([X1] if X1 in have else []) + (['J'] if refj else []) + [n for n in ('JH', 'J1') if n in have]))
        pairs = ([(X, X1)] if X1 in have else []) + ([(X, 'J')] if refj and X != 'JH' else [])
        if X in ('JH', 'J1'): pairs = [('JH', 'J1'), ('J', 'J1')] if refj else []
        so = []
        table(rows, names, pairs, f'-- {s}, all settings', so)
        if a.by_setting:
            for st in sorted({k[0] for k in rows}):
                table({k: v for k, v in rows.items() if k[0] == st}, names, pairs, f'-- {s}, setting {st}', so)
        xs = [rows[k][X] for k in rows]
        streams_line(X, xs)
        if refj:
            js = [rows[k]['J'] for k in rows]; jt = [r for r in js if r['j_tested']]
            print(f'-- J: stream-set test made in {len(jt)}/{len(js)} runs, 1-layer kept in {sum(r["j_one_layer"] for r in jt)}')
        print(f'-- CPU per run (s): ' + ', '.join(f'{n} {np.mean([rows[k][n]["cpu"] for k in rows]):.1f}' for n in names)
              + f'; {X} discrete design {np.mean([r["cpu_design"] for r in xs]):.3f}')
        out['scenarios'][s] = dict(n=len(keys), tables=so)
    return out, f'{X}__report' + ('' if a.budget == 50 else f'_Q{a.budget}') + '.json'


def bench(a, X):
    key = lambda c: (int(c['M']), int(c['drop']))
    res = os.path.join(ROOT, 'results')
    have = {X: read(os.path.join(res, f'{X}__T2__bench.jsonl'), key), f'{X}@SC': read(os.path.join(res, f'{X}@SC__T2__bench.jsonl'), key)}
    ref = {}
    for l in open(os.path.join(ROOT, 'reference', 'bench_fig510.jsonl')):
        r = json.loads(l); ref.setdefault(r['scheme'], {})[key(r['cfg'])] = r
    have['sparse_bf'] = ref['sparse_bf']; have['J'] = ref['J']; have['J@SC'] = ref['stream_cluster']
    for n in have:
        for r in have[n].values():
            r.setdefault('qos_met', bool(r['deficit'] <= 1e-2)); r['rate'] = float(r['rate']) if r['qos_met'] else 0.0
    names = [n for n in (X, 'sparse_bf', f'{X}@SC', 'J', 'J@SC') if have[n]]
    keys = set(have[X])
    for n in names: keys &= set(have[n])
    print(f'\n===== benchmark setting (T2, K = 16, N = 3, S_max = 12, R_th = 2, Q = 50): {len(keys)} common runs ('
          + ', '.join(f'{n}: {len(have[n])}' for n in names) + ')')
    print('sparse_bf, J (the current framework) and J@SC (stream clustering on its streams): stored runs of Fig. 5.10')
    out = dict(variant=X, n=len(keys))
    if not keys: return out, f'{X}__report_bench.json'
    rows = {k: {n: have[n][k] for n in names} for k in sorted(keys)}
    pairs = [(X, 'sparse_bf')] + ([(X, f'{X}@SC')] if f'{X}@SC' in names else [])
    so = []
    table(rows, names, pairs, '-- by M', so, group=lambda k: k[0], glabel='M')
    out['tables'] = so
    xs = [rows[k][X] for k in rows]
    streams_line(X, xs)
    ok = [r for r in xs if r['qos_met']]
    if ok:
        tot = sum(r['comp']['rate'] for r in ok)
        sh = dict(super_common=100 * sum(r['comp']['r_c'] for r in ok) / tot, group_common=100 * sum(r['comp']['r_g'] for r in ok) / tot,
                  private=100 * sum(r['comp']['r_pc'] + r['comp']['r_pn'] for r in ok) / tot, n=len(ok))
        print(f"-- {X}: shares of the sum rate over its {len(ok)} runs meeting the QoS: super-common {sh['super_common']:.1f}%, "
              f"group-common {sh['group_common']:.1f}%, private {sh['private']:.1f}%")
        out['shares'] = sh
    return out, f'{X}__report_bench.json'


def smax(a, X):
    rs = read(os.path.join(ROOT, 'results', f'{X}__T2__smax.jsonl'), lambda c: (int(c['K']), int(c['Smax']), float(c['Rth']), int(c['drop'])))
    print(f'\n===== sum rate versus S_max (T2, N = 3, M = 300, Q = 50): {len(rs)} runs of {X}')
    out = dict(variant=X, K={})
    for r in rs.values():
        r['rate0'] = float(r['rate']) if r['qos_met'] else 0.0
    for K in sorted({k[0] for k in rs}):
        Ss = sorted({k[1] for k in rs if k[0] == K}); Rs = sorted({k[2] for k in rs if k[0] == K})
        print(f'-- K = {K}: mean sum rate over the drops (QoS met / runs)')
        print(f"{'R_th':>6} " + ' '.join(f'{"S=" + str(S):>12}' for S in Ss))
        tab = {}
        for R in Rs + ['all']:
            cells = []
            for S in Ss:
                v = [rs[k] for k in rs if k[0] == K and k[1] == S and (R == 'all' or k[2] == R)]
                cells.append((float(np.mean([x['rate0'] for x in v])) if v else float('nan'), sum(x['qos_met'] for x in v), len(v)))
            tab[f'{R:g}' if R != 'all' else 'all'] = cells
            print(f"{(f'{R:g}' if R != 'all' else 'all'):>6} " + ' '.join(f'{m:7.2f} {q:>2}/{n:<2}' for m, q, n in cells))
        steps = []
        for S0, S1 in zip(Ss, Ss[1:]):
            common = [(k[2], k[3]) for k in rs if k[0] == K and k[1] == S0 and (K, S1, k[2], k[3]) in rs]
            d = [rs[(K, S1) + c]['rate0'] - rs[(K, S0) + c]['rate0'] for c in common]
            if d: steps.append((S0, S1) + stats(d) + (len(d),))
        print('   change to the next S_max (paired over R_th and drops): '
              + ', '.join(f'{s0}->{s1} {m:+.2f}±{e:.2f}' for s0, s1, m, e, n in steps))
        out['K'][str(K)] = dict(S=Ss, R=[f'{R:g}' for R in Rs], table=tab, steps=[dict(S0=s0, S1=s1, mean=m, se=e, n=n) for s0, s1, m, e, n in steps])
    return out, f'{X}__report_smax.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant'); ap.add_argument('--scenarios', default=None); ap.add_argument('--budget', type=int, default=50)
    ap.add_argument('--by-setting', action='store_true'); ap.add_argument('--json', action='store_true')
    ap.add_argument('--bench', action='store_true'); ap.add_argument('--smax', action='store_true')
    a = ap.parse_args()
    import variants as V
    X = a.variant; X1 = V.VARIANTS.get(X, {}).get('one_layer') if X not in ('JH', 'J1') else None
    out, name = bench(a, X) if a.bench else smax(a, X) if a.smax else grid(a, V, X, X1)
    if a.json:
        fn = os.path.join(ROOT, 'results', name); json.dump(out, open(fn, 'w'), indent=1); print(f'\nwritten {fn}')


if __name__ == '__main__':
    main()
