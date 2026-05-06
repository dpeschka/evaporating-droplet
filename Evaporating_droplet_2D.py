from dolfin import *
import logging
import numpy as np
from tqdm import tqdm
from ufl import ln


logging.getLogger("FFC").setLevel(logging.ERROR)
logging.getLogger("UFL_LEGACY").setLevel(logging.ERROR)
logging.getLogger("UFL").setLevel(logging.ERROR)
set_log_level(LogLevel.ERROR)

parameters['linear_algebra_backend'] = 'PETSc'
parameters['reorder_dofs_serial'] = True
parameters['form_compiler']['quadrature_degree'] = 3
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True

# =============================================================================
# Parameter sets (compact format)
# =============================================================================
P_basic = dict(
    # Surface tensions: [γ_l, γ_c, γ_v]
    γ = [1/8, 1/8, 2.0],
    # Energy: [ε, s_sat, Λ]
    FP = [0.2*2,0.3,100],
    # Mobilities: [[m_ll,m_lv,m_lc], [m_vl,m_vv,m_vc], [m_sl,m_sv,m_sc]]
    m = [[1.0, 1.0, 0.01], [0.01, 1.0, 0.01], [0.01, 1.0, 0.01]],
    # Rates: [evaporation, crystallization]
    h = [1.0, 1.0],
    # Energy: [λ, β,c0]
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
γ_l, γ_c, γ_v = PARS['γ']
(m_ll, m_lv, m_lc), (m_vl, m_vv, m_vc), (m_sl, m_sv, m_sc) = PARS['m']
he0, hc0 = PARS['h']
ε,s_sat,Λ = PARS['FP']
λ,β,c_0 = PARS['E']
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

def W(φ):
  W1 = conditional(lt(φ,0),Λ*φ**2,18*(φ*(1-φ))**2)
  W2 = conditional(gt(φ,1),Λ*(1-φ)**2,W1)
  return W2

# energy
def energy(q):
    φ_l,φ_c,φ_v,s,μ_l,μ_c,μ_v,μ_s,κ = split(q)
    
    E  = γ_l * ε/2 * inner(grad(φ_l),grad(φ_l))*r*dx
    E += γ_c * ε/2 * inner(grad(φ_c),grad(φ_c))*r*dx
    E += γ_v * ε/2 * inner(grad(φ_v),grad(φ_v))*r*dx
    E += γ_l/ε * W(φ_l)*r*dx
    E += γ_c/ε * W(φ_c)*r*dx
    E += γ_v/ε * W(φ_v)*r*dx
    # E += δ * inner(grad(s),grad(s))*dx

    E += (s*ln(s) + (1-s)*ln(1-s) + β*φ_c*(s-s_sat))*r*dx
    E += λ * φ_v * s *r*dx
    E += κ*(φ_l + φ_c + φ_v - 1)*r*dx
    # E += μ_inf * φ_v*r*dx
    return E


# single time step
def evolve(old_q, τ):
    # set up function spaces
    q,v = Function(Q),TestFunction(Q)
    φ_l,φ_c,φ_v,s,μ_l,μ_c,μ_v,μ_s,κ = split(q)
    vφ_l,vφ_c,vφ_v,vs,vμ_l,vμ_c,vμ_v,vμ_s,vκ = split(v)
    old_φ_l,old_φ_c,old_φ_v,old_s,old_μ_l,old_μ_c,old_μ_v,old_μ_s,old_κ = split(old_q)

    # define energy
    E = energy(q)
    m_s = m_sl*abs(old_φ_l) + m_sv*abs(old_φ_v) + m_sc*abs(old_φ_c)
    m_l = m_ll*abs(old_φ_l) + m_lv*abs(old_φ_v) + m_lc*abs(old_φ_c)
    m_v = m_vl*abs(old_φ_l) + m_vv*abs(old_φ_v) + m_vc*abs(old_φ_c)

    h_eva = he0*abs(old_φ_l)*abs(old_φ_v)
    h_cry = hc0*abs(old_φ_l) 

    # define weak form
    Res  = (μ_l*vφ_l + μ_c*vφ_c + μ_v*vφ_v + μ_s*vs)*r*dx - derivative(E, q, v)
    Res += m_l*τ*inner(grad(μ_l),grad(vμ_l))*r*dx + vμ_l*(φ_l-old_φ_l)*r*dx
    Res += vμ_c*(φ_c-old_φ_c)*r*dx
    Res += m_v*τ*inner(grad(μ_v),grad(vμ_v))*r*dx + vμ_v*(φ_v-old_φ_v)*r*dx
    
    Res += abs(old_s)*abs(1-old_s)*m_s*τ*inner(grad(μ_s),grad(vμ_s))*r*dx + vμ_s*(s-old_s)*r*dx

    Res +=  τ*(-h_eva*(μ_v-μ_l) - h_cry*(μ_c-μ_l) )*vμ_l*r*dx
    Res +=  τ*(                   h_cry*(μ_c-μ_l) )*vμ_c*r*dx
    Res +=  τ*( h_eva*(μ_v-μ_l)                   )*vμ_v*r*dx

    bc = []

    Jac     = derivative(Res, q)
    problem = NonlinearVariationalProblem(Res, q, bc, Jac)
    solver  = NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm['newton_solver']['error_on_nonconvergence'] = False
    prm['newton_solver']['report'] = False
    prm['newton_solver']['absolute_tolerance'] = 1e-6
    prm['newton_solver']['relative_tolerance'] = 1e-6

    q.assign(old_q)
    iterations, converged = solver.solve()
    
    dissi_ml = assemble(m_l*inner(grad(μ_l),grad(μ_l))*r*dx)
    dissi_mc = 0.0
    dissi_mv = assemble(m_v*inner(grad(μ_v),grad(μ_v))*r*dx)
    dissi_ms = assemble(abs(old_s)*abs(1-old_s)*m_s*inner(grad(μ_s),grad(μ_s))*r*dx)
    dissi_eva = assemble(h_eva*abs(μ_v-μ_l)**2*r*dx)
    dissi_cry = assemble(h_cry*abs(μ_c-μ_l)**2*r*dx)
    dissi = [dissi_ml,dissi_mc,dissi_mv,dissi_ms,dissi_eva,dissi_cry]
    e = assemble(E)

    return q,e,iterations,converged,dissi

# initial data
#func = '1+tanh(3*(x[0]-r0)/ε)'
func = '1+tanh(3*(pow(pow(x[0],2)+pow(x[1],2),0.5)-r0)/ε)'
func = 'tanh(3*(pow(pow(x[0],2)+pow(x[1],2),0.5)-r0)/ε) + tanh(3*(pow(pow(x[0],2)+pow(x[1]-4.2,2),0.5)-1.0)/ε)'
#func = '1+tanh(3*(pow(pow(x[0],2)+pow(x[1],2),0.5)-r0)/ε) + 1+tanh(3*(pow(pow(x[0],2)+pow((x[1]-4.0),2),0.5)-1.0)/ε)'
salt = f'c_0*exp(-λ * (1 - (1-0.5*({func}))))'
iphil = f'(1-0.5*({func}))'
idata = Expression((iphil,"0",f'1-({iphil})',salt,"0","0","0","0","0"),degree=2,ε=ε,c_0=c_0,λ=λ,r0=r0)
old_q = interpolate(idata,Q)

old_q,e,it,conv,dissi = evolve(old_q,1e-4)
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
τ = 0.3e-2
n_steps = 4000
dt = Constant(τ)
f1 = File(f'data/phil{variant}.pvd')
f2 = File(f'data/phic{variant}.pvd')
f3 = File(f'data/phiv{variant}.pvd')
f4 = File(f'data/salt{variant}.pvd')
for i in tqdm(range(n_steps)):
  dt.assign(τ)
  
  q,e,it,conv,dissi = evolve(old_q, dt)
  if conv:
    t += τ
    old_q.assign(q)
    if (i % 2 == 0):
      times.append(t)
      energies.append(e)
      # sols.append(q)
      φ_l,φ_c,φ_v,s,_,_,_,_,_ = q.split()
      φ_l.rename("φ_l","phase field liquid")
      φ_c.rename("φ_c","phase field crystal")
      φ_v.rename("φ_v","phase field vapor")
      s.rename("s","salt concentration")
      f1 << (φ_l,t)
      f2 << (φ_c,t)
      f3 << (φ_v,t)
      f4 << (s,t)
      list_dissi_ml.append(dissi[0])
      list_dissi_mc.append(dissi[1])
      list_dissi_mv.append(dissi[2])
      list_dissi_ms.append(dissi[3])
      list_dissi_eva.append(dissi[4])
      list_dissi_cry.append(dissi[5])
    if it < 3:
      τ *= 1.1
    if it > 4:
      τ *= 0.9
  else:
    τ *= 0.5
  if t>T:
    break