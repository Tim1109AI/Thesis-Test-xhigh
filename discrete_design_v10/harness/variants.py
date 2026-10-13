"""Discrete-design variants for the harness: name -> entry.

Entry keys:
  design      'module:function' in code/ ; the function design(net, p) -> hrs.Design (see code/discrete.py)
  one_layer   the name of the variant that is this design's 1-layer RSMA counterpart (requirement R1 of BRIEF.md)
  opt         (optional) dict of options passed to discrete.run (keys of newprop.NEW); leave it out unless your design
              needs a changed evaluation, and then explain why in DESIGN.md
Use a new name instead of editing a variant that already has results.
KEY@SC (no entry needed; harness/grid.py and tools/report.py) is the stream-clustering benchmark (code/benchmarks.py)
on the streams of KEY's design.

The baselines (the current thesis):
  JH   current design (joint transmission, HRS order)           discrete.current
  J1   its 1-layer RSMA (group-common streams removed)          discrete.current_1layer
  J    the thesis framework = JH or J1 by the stream-set test at the initial design of the search; derived by
       tools/report.py from the runs of JH and J1 (not run separately)
"""

VARIANTS = {
    'JH': dict(design='discrete:current', one_layer='J1'),
    'J1': dict(design='discrete:current_1layer'),
}

# ---- design study (bsx.py: experimental BS-specific designs)
VARIANTS.update({
    'XA': dict(design='bsx:A', one_layer='XA1'), 'XA1': dict(design='bsx:A1'),
    'XC': dict(design='bsx:C', one_layer='XC1'), 'XC1': dict(design='bsx:C1'),
})
VARIANTS.update({
    'XM': dict(design='bsx:M', one_layer='XM1'), 'XM1': dict(design='bsx:M1'),
    'XAc': dict(design='bsx:A', one_layer='XA1c', opt=dict(init='core_carried', seed='rule_carried')),
    'XA1c': dict(design='bsx:A1', opt=dict(init='core_carried', seed='rule_carried')),
})
_C = dict(init='core_carried', seed='rule_carried')
VARIANTS.update({
    'XZ': dict(design='bsx:Z', one_layer='XZ1'), 'XZ1': dict(design='bsx:Z1'),
    'XZc': dict(design='bsx:Z', one_layer='XZ1c', opt=_C), 'XZ1c': dict(design='bsx:Z1', opt=_C),
    'XZD': dict(design='bsx:ZD', one_layer='XZD1'), 'XZD1': dict(design='bsx:ZD1'),
    'XZDc': dict(design='bsx:ZD', one_layer='XZD1c', opt=_C), 'XZD1c': dict(design='bsx:ZD1', opt=_C),
})
VARIANTS.update({
    'XZD': dict(design='bsx:ZD', one_layer='XZD1'), 'XZD1': dict(design='bsx:ZD1'),
})
VARIANTS.update({
    'XZDn': dict(design='bsx:ZDn', one_layer='XZDn1'), 'XZDn1': dict(design='bsx:ZDn1'),
    'XZn': dict(design='bsx:Zn', one_layer='XZn1'), 'XZn1': dict(design='bsx:Zn1'),
})

# ---- the final design of this round (code/bsd.py): zero-forcing-minimal cooperation; AO start on the carried antennas
# (algo.init_core_carried / SEEDS['rule_carried']: a switch, off by default; identical to the default start under
# joint transmission). Its 1-layer counterpart: the same core users and BSs, no group-common streams.
VARIANTS.update({
    'BSD': dict(design='bsd:design_dof', one_layer='BSD1', opt=dict(init='core_carried', seed='rule_carried')),
    'BSD1': dict(design='bsd:design_dof_1layer', opt=dict(init='core_carried', seed='rule_carried')),
})
VARIANTS.update({
    # sensitivity check of the counterpart: the free slots carry private streams of non-core UEs (J1 style)
    'BSD1f': dict(design='bsd:design_dof_1layer_fill', opt=dict(init='core_carried', seed='rule_carried')),
})
VARIANTS.update({
    # study: the current design's number of private streams n = min(K, BN, S_max - 1), zero-forcing-minimal carriage,
    # private streams for the core users only (free slots: group-common streams / nothing); carried-antenna start
    'XZnc': dict(design='bsd:design', one_layer='XZn1c', opt=dict(init='core_carried', seed='rule_carried')),
    'XZn1c': dict(design='bsd:design_1layer', opt=dict(init='core_carried', seed='rule_carried')),
    'XZD1gc': dict(design='bsx:ZD1g', one_layer='BSD1', opt=dict(init='core_carried', seed='rule_carried')),
})
