"""Test scenarios of the discrete design: one common cell edge, B = 2 or 3. The area is 100 m x 100 m in both.

  T2  the thesis scenario: B = 2 BSs at (25, 50, 15) and (75, 50, 15) m; UEs uniform over the area (the bank's drops)
  E3  B = 3 BSs far apart near the edge of the area, at (10, 10, 15), (90, 10, 15) and (50, 90, 15) m; UEs uniform
      over the area (the same drops as T2)

The DS-STARs start at the bank's initial deployment, (33, 50, 7.5), (50, 50, 7.5) and (67, 50, 7.5) m with azimuth 0,
in both scenarios (in T2 this is on the line between the BSs, where every cascaded channel vanishes).
"""
import hrs as H_, run as R

BS_T2 = [(25.0, 50.0, 15.0), (75.0, 50.0, 15.0)]
BS_E3 = [(10.0, 10.0, 15.0), (90.0, 10.0, 15.0), (50.0, 90.0, 15.0)]
SCENARIOS = {
    'T2': dict(BS=BS_T2),
    'E3': dict(BS=BS_E3),
}


def network(cfg):
    """(p, net) of a run: cfg holds scenario, K, N, M, Smax, Rth, drop and optionally L, Pmax, Dmax, setup."""
    BS = SCENARIOS[cfg.get('scenario', 'T2')]['BS']
    p = H_.params(K=cfg['K'], L=cfg.get('L', 3), N=cfg['N'], M=cfg['M'], Smax=cfg['Smax'], Pmax=cfg.get('Pmax', 1.0),
                  Dmax=cfg.get('Dmax', 10.0), Rth=cfg['Rth'], bank=R.BANK, B=len(BS))
    net = H_.make_network(p, R.BANK, drop=cfg['drop'], setup=cfg.get('setup', 1), kind='random', BS=BS)
    return p, net
