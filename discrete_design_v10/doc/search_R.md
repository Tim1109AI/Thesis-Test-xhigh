# The deployment search R, trimmed (fixed in this package)

Thesis-ready description by the design session that proposed it (design round v5, return "max_2"); code: code/physbo.py (search(**DESIGN_R)), called by algo.bayes_opt_fast for every caller (algo.SEARCH = 'R'). The candidate paragraph is updated to the trimmed design.


A deployment $\boldsymbol\delta$ is evaluated as before: $\mathrm{AO}_{\epsilon_E}$ from the core-user start returns
$F(\boldsymbol\delta)$. The search is a Bayesian optimization whose surrogate combines a physical **rate score** with a
Gaussian process, and whose candidates follow the structure of a deployment of $L$ separate surfaces.

**Rate score.** In the LoS model every cascade is rank one, $\mathbf A_{k,b,l}=s_{k,b,l}\,\mathbf e_{k,b,l}\mathbf a_{b,l}^T$,
and its amplitude
$$|s_{k,b,l}(\boldsymbol\delta)| = c\, d_{b,l}^{-\alpha/2}\, d_{l,k}^{-\alpha/2}\,|\cos\vartheta_{b,l}|^{q}\,|\cos\vartheta_{l,k}|^{q}$$
depends only on the positions and azimuths (hop distances $d$, angles $\vartheta$ to the surface normal, $c$ the
gain constant of \eqref{eq:channel}). If the $M$ elements of every surface added the cascades of UE $k$ coherently, its
channel gain on BS $b$ would be
$$G_{k,b}(\boldsymbol\delta)=\|\mathbf h_{d,k,b}\|^2+N\Bigl(M\sum_{l}|s_{k,b,l}(\boldsymbol\delta)|\Bigr)^2 .$$
A stream $j$ is decoded by the UEs $\mathcal D_j$ and its rate is limited by the weakest of them, so the score of a
deployment is the average over the served streams $\mathcal J$ of the high-SNR capacity of the weakest decoder,
$$S(\boldsymbol\delta)=\frac1{|\mathcal J|}\sum_{j\in\mathcal J}\log_2\min_{k\in\mathcal D_j}\sum_{b\in\mathcal B_j}G_{k,b}(\boldsymbol\delta)$$
($\mathcal B_j$: the BSs that carry stream $j$). $S$ needs no channel phases and no optimization (a few milliseconds
for thousands of deployments), and it captures what the deployment does to the UEs that matter for $F$: how close
the surfaces are to the BSs and to the decoders of every stream, and whether they face them.

**Surrogate.** $F$ is modeled as a GP with prior mean $m(\boldsymbol\delta)=\beta_0+\beta_1S(\boldsymbol\delta)$ and a
Matérn-5/2 covariance with automatic relevance determination over the three kinds of coordinates,
$$k(\mathbf x,\mathbf x')=\sigma^2\bigl[(1+u+u^2/3)e^{-u}+\eta\,\mathbb 1(\mathbf x=\mathbf x')\bigr],\quad
u=\sqrt{\tfrac{5}{5L}\sum_{i}\Bigl(\tfrac{x_i-x_i'}{\ell_{t(i)}}\Bigr)^2},$$
on design P's features ($x/X$, $y/Y$, $(z-z_{\min})/(z_{\max}-z_{\min})$ and the azimuth on a circle,
$(\cos2\phi,\sin2\phi)/2\pi$), with one length scale $\ell_t$ for the horizontal positions, one for the heights and one
for the azimuths, and a nugget $\eta$. After every evaluation, $\beta_0,\beta_1$ are re-estimated by generalized least
squares and $\sigma^2$, $(\ell_{xy},\ell_z,\ell_\phi)$, $\eta$ by maximum likelihood (logarithmic grids, coordinate-wise
search starting from the previous choice); the predictive variance includes the uncertainty of $\beta$. The rate
score is thus **calibrated on the run's own evaluations and keeps adapting** as the search proceeds ($\beta_1$ converts
bits of score into merit; over the deployments a run evaluates, the least-squares slope of F on S is 3–5 bit/s/Hz per
bit of score, i.e. most of a bit of rate per stream), while the GP learns what the score misses: interference between
the streams, the AO, the interaction of the surfaces.

**Candidates.** Every iteration draws two sets of admissible candidates (512 each):
(i) *random deployments* as in design T: a displacement scale $R$ log-uniform in $[r_0,D^{\max}]$ ($r_0=1$ m), every
surface uniform in the admissible part of its $R$-ball around its initial position, uniform azimuths;
(ii) *one-surface re-draws*: a randomly chosen surface of the best deployment is placed anew, uniformly in the
admissible part of the ball of radius $R$ (log-uniform in $[r_0,D^{\max}]$) around its current position, with a
uniform azimuth, while the other surfaces keep their deployment.
Candidates that violate a constraint are discarded. (Design R as returned in round v5 also drew 512 *local steps*, Gaussian
perturbations of the best deployment proportional to the length scales; the trimmed design R adopted on 2026-10-12
drops them: the hill climb below does the local refinement, at no measurable loss on 44 paired pilot runs.)

**Acquisition.** The 4 candidates with the largest (log) expected improvement are refined by a short stochastic hill
climb of the expected improvement (10 rounds of 32 local steps per start: Gaussian perturbations whose standard
deviation is $s\,\ell_t$ in the feature of type $t$, with $s$ log-uniform in $[0.02,0.5]$, half of them moving a single
surface; a start moves to its best step when that raises the expected improvement), and the deployment with the largest expected
improvement among all candidates and climbed points is evaluated (deployments already evaluated are excluded). The initial design is design T's ($\boldsymbol\delta_0$ and 4 random candidates with stratified scales), so
the stream-set test of the main runs is unchanged. The search returns the best evaluated deployment, which is refined
as before.

