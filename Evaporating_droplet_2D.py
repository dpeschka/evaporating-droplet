from dolfin import *
import logging
import numpy as np
from tqdm import tqdm
from ufl import ln


logging.getLogger("FFC").setLevel(logging.ERROR)
logging.getLogger("UFL_LEGACY").setLevel(logging.ERROR)
logging.getLogger("UFL").setLevel(logging.ERROR)
set_log_level(LogLevel.ERROR)

#parameters['linear_algebra_backend'] = 'mumps'
parameters['reorder_dofs_serial'] = True
parameters['form_compiler']['quadrature_degree'] = 3
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True

# =============================================================================
# Parameter sets (compact format)
# =============================================================================
P_basic = dict(
    # Surface tensions: [gamma_l, gamma_c, gamma_v]
    gamma = [1/8, 1/8, 2.0],
    # Energy: [eps, s_sat, Lambda]
    FP = [0.2*2,0.3,100],
    # Mobilities: [[m_ll,m_lv,m_lc], [m_vl,m_vv,m_vc], [m_sl,m_sv,m_sc]]
    m = [[1.0, 1.0, 0.01], [0.01, 1.0, 0.01], [0.01, 1.0, 0.01]],
    # Rates: [evaporation, crystallization]
    h = [1.0, 1.0],
    # Energy: [lam, beta,c0]
    E = [10.0, -10.0, 0.01],
    # Geometry: [r0, L]
    G = [3.0, 4.0],
    # Simulation: [RADI, T]
    S = [True, 50],
    NAME = 'BASIC'
)

P1 = {**P_basic, 'E': [10,-10,0.01],'NAME': 'P1'} # Droplet evaporates and crystal remains
P2 = {**P_basic, 'E': [10,-10,0.10],'NAME': 'P2'} # Droplet evaporates a bit and crystal shell forms
P3 = {**P_basic, 'E': [1.,-1.,0.01],'NAME': 'P3'} # Droplet fully evaporates and no crystal remains
P4 = {**P_basic, 'E': [10,-1,0.1],'NAME': 'P4', 'S': [True, 100]} # Droplet evaporates a bit and stablizes with saline solution

P5 = {**P_basic, 'E': [1.,-1.,0.01],'NAME': 'P5', 'm': [[1e-3,1e-3,1e-3], [0.01, 1.0, 0.01], [0.01, 1.0, 0.01]]} # Droplet fully evaporates and no crystal remains



PARS = P1
variant = 'P1'

# Unpack
gamma_l, gamma_c, gamma_v = PARS['gamma']
(m_ll, m_lv, m_lc), (m_vl, m_vv, m_vc), (m_sl, m_sv, m_sc) = PARS['m']
he0, hc0 = PARS['h']
eps,s_sat,Lambda = PARS['FP']
lam,beta,c_0 = PARS['E']
r0, L = PARS['G']
RADI, T = PARS['S']
FNAME = PARS['NAME']

# FE definitions
# mesh = IntervalMesh(128,0,L)
mesh = RectangleMesh(Point(0,-4.0), Point(L,6.0), 96,3*96)
FE   = FiniteElement("P", mesh.ufl_cell(), 1)   # scalar element
Q    = FunctionSpace(mesh,MixedElement([FE,FE,FE,FE,FE,FE,FE,FE,FE]))# mixed space

if RADI:
  r = SpatialCoordinate(mesh)[0] #**2
else:
  r = Constant(1.0)

def W(phi):
  W1 = conditional(lt(phi,0),Lambda*phi**2,18*(phi*(1-phi))**2)
  W2 = conditional(gt(phi,1),Lambda*(1-phi)**2,W1)
  return W2

# energy
def energy(q):
    phi_l,phi_c,phi_v,s,mu_l,mu_c,mu_v,mu_s,kappa = split(q)
    
    E  = gamma_l * eps/2 * inner(grad(phi_l),grad(phi_l))*r*dx
    E += gamma_c * eps/2 * inner(grad(phi_c),grad(phi_c))*r*dx
    E += gamma_v * eps/2 * inner(grad(phi_v),grad(phi_v))*r*dx
    E += gamma_l/eps * W(phi_l)*r*dx
    E += gamma_c/eps * W(phi_c)*r*dx
    E += gamma_v/eps * W(phi_v)*r*dx
    # E += delta * inner(grad(s),grad(s))*dx

    E += (s*ln(s) + (1-s)*ln(1-s) + beta*phi_c*(s-s_sat))*r*dx
    E += lam * phi_v * s *r*dx
    E += kappa*(phi_l + phi_c + phi_v - 1)*r*dx
    # E += mu_inf * phi_v*r*dx
    return E


# single time step
def evolve(old_q, tau):
    # set up function spaces
    q,v = Function(Q),TestFunction(Q)
    phi_l,phi_c,phi_v,s,mu_l,mu_c,mu_v,mu_s,kappa = split(q)
    vphi_l,vphi_c,vphi_v,vs,vmu_l,vmu_c,vmu_v,vmu_s,vkappa = split(v)
    old_phi_l,old_phi_c,old_phi_v,old_s,old_mu_l,old_mu_c,old_mu_v,old_mu_s,old_kappa = split(old_q)

    # define energy
    E = energy(q)
    m_s = m_sl*abs(old_phi_l) + m_sv*abs(old_phi_v) + m_sc*abs(old_phi_c)
    m_l = m_ll*abs(old_phi_l) + m_lv*abs(old_phi_v) + m_lc*abs(old_phi_c)
    m_v = m_vl*abs(old_phi_l) + m_vv*abs(old_phi_v) + m_vc*abs(old_phi_c)

    h_eva = he0*abs(old_phi_l)*abs(old_phi_v)
    h_cry = hc0*abs(old_phi_l) 

    # define weak form
    Res  = (mu_l*vphi_l + mu_c*vphi_c + mu_v*vphi_v + mu_s*vs)*r*dx - derivative(E, q, v)
    Res += m_l*tau*inner(grad(mu_l),grad(vmu_l))*r*dx + vmu_l*(phi_l-old_phi_l)*r*dx
    Res += vmu_c*(phi_c-old_phi_c)*r*dx
    Res += m_v*tau*inner(grad(mu_v),grad(vmu_v))*r*dx + vmu_v*(phi_v-old_phi_v)*r*dx
    
    Res += abs(old_s)*abs(1-old_s)*m_s*tau*inner(grad(mu_s),grad(vmu_s))*r*dx + vmu_s*(s-old_s)*r*dx

    Res +=  tau*(-h_eva*(mu_v-mu_l) - h_cry*(mu_c-mu_l) )*vmu_l*r*dx
    Res +=  tau*(                   h_cry*(mu_c-mu_l) )*vmu_c*r*dx
    Res +=  tau*( h_eva*(mu_v-mu_l)                   )*vmu_v*r*dx

    bc = []

    Jac     = derivative(Res, q)
    problem = NonlinearVariationalProblem(Res, q, bc, Jac)
    solver  = NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm['newton_solver']['linear_solver'] = 'mumps'

    prm['newton_solver']['error_on_nonconvergence'] = False
    prm['newton_solver']['report'] = False
    prm['newton_solver']['absolute_tolerance'] = 1e-6
    prm['newton_solver']['relative_tolerance'] = 1e-6

    q.assign(old_q)
    iterations, converged = solver.solve()
    
    dissi_ml = assemble(m_l*inner(grad(mu_l),grad(mu_l))*r*dx)
    dissi_mc = 0.0
    dissi_mv = assemble(m_v*inner(grad(mu_v),grad(mu_v))*r*dx)
    dissi_ms = assemble(abs(old_s)*abs(1-old_s)*m_s*inner(grad(mu_s),grad(mu_s))*r*dx)
    dissi_eva = assemble(h_eva*abs(mu_v-mu_l)**2*r*dx)
    dissi_cry = assemble(h_cry*abs(mu_c-mu_l)**2*r*dx)
    dissi = [dissi_ml,dissi_mc,dissi_mv,dissi_ms,dissi_eva,dissi_cry]
    e = assemble(E)

    return q,e,iterations,converged,dissi

# initial data
#func = '1+tanh(3*(x[0]-r0)/eps)'
func = '1+tanh(3*(pow(pow(x[0],2)+pow(x[1],2),0.5)-r0)/eps)'
func = 'tanh(3*(pow(pow(x[0],2)+pow(x[1],2),0.5)-r0)/eps) + tanh(3*(pow(pow(x[0],2)+pow(x[1]-4.2,2),0.5)-1.0)/eps)'
#func = '1+tanh(3*(pow(pow(x[0],2)+pow(x[1],2),0.5)-r0)/eps) + 1+tanh(3*(pow(pow(x[0],2)+pow((x[1]-4.0),2),0.5)-1.0)/eps)'
salt = f'c_0*exp(-lam * (1 - (1-0.5*({func}))))'
iphil = f'(1-0.5*({func}))'
idata = Expression((iphil,"0",f'1-({iphil})',salt,"0","0","0","0","0"),degree=2,eps=eps,c_0=c_0,lam=lam,r0=r0)
old_q = interpolate(idata,Q)

# no newline in print
print('Init..', end='')
old_q,e,it,conv,dissi = evolve(old_q,1e-4)
print('..done.')
print('init:',it,conv)
times = []
energies = []
sols = []

list_dissi_ml = []
list_dissi_mc = []
list_dissi_mv = []
list_dissi_ms = []
list_dissi_eva = []
list_dissi_cry = []

# initial time stepping, later adaptive
t = 0
tau = 0.3e-2
n_steps = 4000
dt = Constant(tau)
f1 = File(f'data/phil{variant}.pvd')
f2 = File(f'data/phic{variant}.pvd')
f3 = File(f'data/phiv{variant}.pvd')
f4 = File(f'data/salt{variant}.pvd')
for i in tqdm(range(n_steps)):
  dt.assign(tau)
  
  q,e,it,conv,dissi = evolve(old_q, dt)
  if conv:
    t += tau
    old_q.assign(q)
    if (i % 2 == 0):
      times.append(t)
      energies.append(e)
      # sols.append(q)
      phi_l,phi_c,phi_v,s,_,_,_,_,_ = q.split()
      phi_l.rename("phi_l","phase field liquid")
      phi_c.rename("phi_c","phase field crystal")
      phi_v.rename("phi_v","phase field vapor")
      s.rename("s","salt concentration")
      f1 << (phi_l,t)
      f2 << (phi_c,t)
      f3 << (phi_v,t)
      f4 << (s,t)
      list_dissi_ml.append(dissi[0])
      list_dissi_mc.append(dissi[1])
      list_dissi_mv.append(dissi[2])
      list_dissi_ms.append(dissi[3])
      list_dissi_eva.append(dissi[4])
      list_dissi_cry.append(dissi[5])
    if it < 3:
      tau *= 1.1
    if it > 4:
      tau *= 0.9
  else:
    tau *= 0.5
  if t>T:
    break
