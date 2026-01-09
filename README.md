# Model for evaporation of water and with three phases (liquid, crystal, vapor) and a solute concentration

**Description**: Supplemental code for the paper *Combined effects of evaporation, sedimentation and  solute crystallization on the 
dynamics of aerosol size distributions on multiple length and time scales* by Sina Zendehroud, Ole Kleinjung, Philip Loche, Lyderic Bocquet,
Roland R. Netz, Erica Ipocoana, Dirk Peschka, and Marita Thomas

**Code/Model authors**: Erica Ipocoana, Dirk Peschka, Marita Thomas


## Variables
- $\varphi_l, \varphi_c, \varphi_v$ : phase fields (liquid, crystal, vapor) with $\varphi_l+\varphi_c+\varphi_v=1$
- $s$ : solute concentration
- $\mu_l, \mu_c, \mu_v, \mu_s$ : chemical potentials
- $\kappa$ : Lagrange multiplier for constraint

## Energy
$$
\mathscr{E}(\varphi,s)=\int_\Omega \sum_{i\in\{l,c,v\}} \gamma_i\left[\frac{\varepsilon}{2}|\nabla\varphi_i|^2 + \frac{1}{\varepsilon}W(\varphi_i)\right] + s\ln s + (1-s)\ln(1-s) + \beta\varphi_c(s-s_{\mathrm{sat}}) + \lambda\varphi_v s \,\mathrm{d}x
$$

## Lagrangian
$$
\mathscr{L}(\varphi,s,\kappa) = \mathscr{E}(\varphi,s) + \int_\Omega \kappa(\varphi_l+\varphi_c+\varphi_v-1)\,\mathrm{d}x
$$

with double-well potential $W(\varphi)=18\varphi^2(1-\varphi)^2$ (penalized outside $[0,1]$).

## Dynamics
| Variable | Evolution |
|----------|-----------|
| $\varphi_l$ | $\partial_t\varphi_l = \nabla\cdot(m_l\nabla\mu_l) + h_e(\mu_v-\mu_l) + h_c(\mu_c-\mu_l)$ |
| $\varphi_c$ | $\partial_t\varphi_c = -h_c(\mu_c-\mu_l)$ |
| $\varphi_v$ | $\partial_t\varphi_v = \nabla\cdot(m_v\nabla\mu_v) - h_e(\mu_v-\mu_l)$ |
| $s$ | $\partial_t s = \nabla\cdot(m_s\nabla\mu_s)$ |

where $\mu_i = \delta\mathscr{E}/\delta\varphi_i$, $\mu_s = \delta\mathscr{E}/\delta s$, and mobilities $m_i$ depend on local phase composition.
