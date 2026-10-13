"""Behaviour of the design for many BSs scattered over a large area (design only; no evaluation)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
import numpy as np, math
import hrs as H_, run as R, bsd, discrete as DC

def demo(name, BS, K, N, S, side, seed=1):
    rng = np.random.default_rng(seed)
    UE = np.column_stack([rng.uniform(0, side, K), rng.uniform(0, side, K), np.full(K, 1.5)])
    p = H_.params(K=K, L=3, N=N, M=100, Smax=S, Rth=1.0, bank=R.BANK, B=len(BS))
    net = H_.make_network(p, R.BANK, drop=801, BS=BS, UE=UE) if K in (8, 12, 16, 20, 10, 14, 18) else None
    if net is None:
        # make_network needs a bank drop of this K only for the default UEs; build the channels directly
        p['K'] = K
        net = dict(BS=np.array(BS, float), UE=UE)
        k0 = 2 * math.pi / p['lam']; Hd = np.zeros((K, len(BS) * N), complex)
        for k in range(K):
            for b in range(len(BS)):
                v = UE[k] - net['BS'][b]; d = np.linalg.norm(v); a = np.exp(1j * k0 * p['uBS'] @ (v / d))
                Hd[k, b * N:(b + 1) * N] = math.sqrt(p['PL0'] / p['noise']) * d ** (-p['alpha'] / 2) * np.exp(-1j * k0 * d) * a
        net['Hd'] = Hd
    d = bsd.design(net, p); DC.check(d, p)
    B = len(BS); bm = np.ones((B, d.nS), bool) if d.bsmask is None else d.bsmask
    n = len(d.core); m = math.ceil(n / N)
    priv = [j for j in d.served if j > d.G]; grp = [j for j in d.served if 1 <= j <= d.G]
    nb_priv = [int(bm[:, d.col[j]].sum()) for j in priv]; nb_grp = [int(bm[:, d.col[j]].sum()) for j in grp]
    # distance from each core UE to its carrying BSs vs. to the farthest BS
    dist = np.linalg.norm(net['UE'][:, None, :2] - net['BS'][None, :, :2], axis=-1)
    far = [float(dist[j - 1 - d.G][bm[:, d.col[j]]].max()) for j in priv]
    print(f"{name}: B={B}, N={N}, K={K}, S_max={S}: core n={n}, m={m}; private streams {len(priv)} carried by {np.mean(nb_priv):.1f} BSs on average "
          f"(max distance to a carrying BS {np.mean(far):.0f} m, farthest BS {np.mean(dist.max(1)):.0f} m); groups {d.G}, served {len(grp)}, "
          f"carried by {np.mean(nb_grp) if grp else 0:.1f} BSs; streams per BS {bm.sum(1).tolist()}")

side = 100.0
demo('T2', [(25, 50, 15), (75, 50, 15)], 16, 3, 12, side)
demo('T2 S=4', [(25, 50, 15), (75, 50, 15)], 16, 3, 4, side)
demo('E3 S=6', [(10, 10, 15), (90, 10, 15), (50, 90, 15)], 16, 3, 6, side)
g = [(x, y, 15.0) for x in (50, 150, 250, 350) for y in (50, 150, 250, 350)]
demo('16 BSs, 400 m', g, 64, 3, 12, 400.0)
demo('16 BSs, 400 m, K=24', g, 24, 3, 12, 400.0)
g2 = [(x, y, 15.0) for x in np.linspace(50, 950, 6) for y in np.linspace(50, 950, 6)]
demo('36 BSs, 1 km', g2, 120, 3, 12, 1000.0)
demo('4 BSs, 100 m, K=8, S=12', [(25, 25, 15), (75, 25, 15), (25, 75, 15), (75, 75, 15)], 8, 3, 12, 100.0)
