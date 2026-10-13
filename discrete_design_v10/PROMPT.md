# Prompt for a design session

Attach `discrete_design_v10.zip` and paste the text below, replacing `<LABEL>` by a short name for the session.

---

I'm attaching discrete_design_v10.zip, a design task for the discrete design used in my thesis simulations.

1. Unzip it into a new folder and read BRIEF.md, FINDINGS.md and README.md in full. BRIEF.md quotes my request in my
   own words and lists the requirements I confirmed; make sure you get them: one central discrete design, settled
   before the deployment search, in which each BS has its own stream set (every BS still carries the super-common
   stream), designed for many BSs scattered over a large area (so it can associate a core UE with only some BSs, and
   with all BSs when the scenario calls for it), and which assigns different group-common or private streams to
   different BSs in some scenarios. It must beat its own 1-layer RSMA at every R_th from 0 to 2.5 (a tie at 0 and
   0.5 is acceptable), clearly beat sparse beamforming and stream clustering in the benchmark setting, keep both the
   super-common and the group-common streams around or over 10% of the sum rate at K = 16 and R_th = 2, and its rate
   must rise monotonically with S_max. It need not match the current joint design. A straightforward design, please;
   no AO tests, no empirically tuned parameters. Simplicity and elegance are vital.
2. Install the requirements and run `python3 harness/selftest.py`.
3. Design freely within the ground rules of BRIEF.md. Implement your ideas, run them with the harness, measure them
   with `tools/report.py`, and iterate until you have one design that satisfies my requirements and that you can
   justify in the thesis. Follow the compute plan in README.md: screen every idea with `--tier screen` (Q = 5), check
   the ones you shortlist at Q = 50 on subsets and with `--tier bench` and `--tier smax`, and confirm your final design
   and its 1-layer counterpart with `--tier confirm` (Q = 50). Take the time this needs.
4. When done, give me `design_return_<LABEL>.zip` as described in BRIEF.md ("What to return"), and summarize your
   design and its results in a few lines.

Work autonomously: don't stop to ask me questions; make reasonable choices and note them in DESIGN.md. Run long jobs
in the background and keep checking on them.
