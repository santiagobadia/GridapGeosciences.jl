module ThetaMethodFullNewtonNSWECubedSphereTests

using Test
using Gridap
using GridapGeosciences
using Plots
using LinearAlgebra
using WriteVTK
using JLD
using LineSearches

# Solves the steady state Williamson2 test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a modified coriolis term that exactly balances
# the potential gradient term to achieve a steady state
# reference:
# D. L. Williamson, J. B. Drake, J. J.HackRüdiger Jakob, P. N.Swarztrauber, (1992)
# J Comp. Phys. 102 211-224

# Constants of the Williamson2 test case
const α  = π/4.0              # deviation of the coriolis term from zonal forcing
const U₀ = 38.61068276698372  # velocity scale
const H₀ = 2998.1154702758267 # mean fluid depth

# Modified coriolis term
function f₀(xyz)
   θϕr   = xyz2θϕr(xyz)
   θ,ϕ,r = θϕr
   2.0*Ωₑ*( -cos(θ)*cos(ϕ)*sin(α) + sin(ϕ)*cos(α) )
end

# Initial velocity (williamsom2)
function u₀(xyz)
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  u     = U₀*(cos(ϕ)*cos(α) + cos(θ)*sin(ϕ)*sin(α))
  v     = -U₀*sin(θ)*sin(α)
  spherical_to_cartesian_matrix(θϕr)⋅VectorValue(u,v,0)
end

# Initial fluid depth (williamsom2)
function h₀(xyz)
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  h  = -cos(θ)*cos(ϕ)*sin(α) + sin(ϕ)*cos(α)
  H₀ - (rₑ*Ωₑ*U₀ + 0.5*U₀*U₀)*h*h/g
end

# Topography
function topography(xyz)
  0.0
end

# Compute initial volume flux (williamsom2)
function F₀(u₀,h₀,U,V,dΩ)
  a(u,v) = ∫(v⋅u)dΩ
  b(v)   = ∫((v⋅(h₀*u₀)))dΩ
  solve(AffineFEOperator(a,b,U,V))
end

# Compute initial potential vorticity (williamsom2)
function q₀(u₀,h₀,f,R,S,n,dΩ)
  a(r,s) = ∫( s*(r*h₀) )dΩ
  b(s)   = ∫( s*f - ⟂(∇(s),n)⋅u₀ )dΩ
  solve(AffineFEOperator(a,b,R,S))
end

# Generate initial monolothic solution (williamsom2)
function uhqF₀(u₀,h₀,q₀,F₀,X,Y,dΩ)
  a((u,p,r,u2),(v,q,s,v2))=∫(v⋅u+q*p+s*r+v2⋅u2)dΩ
  b((v,q,s,v2))=∫( v⋅u₀+ q*h₀ + s*q₀ + v2⋅F₀ )dΩ
  solve(AffineFEOperator(a,b,X,Y))
end

function new_vtk_step(Ω,file,hn,un)
  createvtk(Ω,
            file,
            cellfields=["hn"=>hn, "un"=>un],
            nsubcells=4)
end

function generate_energy_plots(outdir,N,ke,pe,kin_to_pot,pot_to_kin)
  plot(1:N,
       [ke,pe],
       label=["Ek (kinetic)" "Ep (potential)"],
       xlabel="Step",ylabel="Energy")
  savefig(joinpath(outdir,"energy.png"))
  plot(1:N,
       [kin_to_pot,-pot_to_kin,kin_to_pot-pot_to_kin],
       label=["hⁿ∇⋅uⁿ" "uⁿ⋅∇(hⁿ)" "Balance"],
       xlabel="Step",ylabel="Kinetic to potential energy balance",legend = :outertopleft)
  savefig(joinpath(outdir,"kinetic_to_potential_energy_balance.png"))
end

"""
  Solves the nonlinear rotating shallow water equations
  T : [0,T] simulation interval
  N : number of time subintervals
  θ : Theta-method parameter [0,1)
"""
function solve_nswe_theta_method_full_newton(
      model,order,degree,θ,T,N;
      nlrtol=1.0e-08, # Newton solver relative residual tolerance
      write_results=false,
      out_dir="nswe_ncells_$(num_cells(model))_order_$(order)_theta_method_full_newton",
      out_period=N/10)

  RT = ReferenceFE(raviart_thomas,Float64,order)
  DG = ReferenceFE(lagrangian,Float64,order)
  CG = ReferenceFE(lagrangian,Float64,order+1)

  V  = FESpace(model,RT; conformity=:Hdiv) # Velocity and mass flux FE space
  Q  = FESpace(model,DG; conformity=:L2)   # Fluid depth FE space
  S  = FESpace(model,CG; conformity=:H1)   # Potential vorticity FE space
  U  = TrialFESpace(V)
  P  = TrialFESpace(Q)
  R  = TrialFESpace(S)

  Y = MultiFieldFESpace([V,Q,S,V])         # Monolithic FE space
  X = MultiFieldFESpace([U,P,R,U])

  Ω  = Triangulation(model)
  n  = get_normal_vector(model)
  dΩ = Measure(Ω,degree)
  dω = Measure(Ω,degree,ReferenceDomain())

  a1(u,v)=∫(v⋅u)dΩ
  l1(v)=∫(v⋅u₀)dΩ
  un=solve(AffineFEOperator(a1,l1,U,V)); unv=get_free_dof_values(un)

  a2(u,v)=∫(v*u)dΩ
  l2(v)=∫(v*h₀)dΩ
  hn=solve(AffineFEOperator(a2,l2,P,Q)); hnv=get_free_dof_values(hn)

  a3(u,v)=∫(v*u)dΩ
  l3(v)=∫(v*f₀)dΩ
  fn=solve(AffineFEOperator(a3,l3,R,S))

  b = interpolate_everywhere(topography,P)

  # Compute:
  #     - Initial potential vorticity (q₀)
  #     - Initial volume flux (F₀)
  #     - Initial full solution
  ΔuΔhqF=uhqF₀(un,hn,q₀(un,hn,fn,R,S,n,dΩ),F₀(un,hn,U,V,dΩ),X,Y,dΩ)
  Δu,Δh,_,_ = ΔuΔhqF
  function run_simulation(pvd=nothing)
    # Allocate work space vectors
    # if (write_results)
    #   ke  = Vector{Float64}(undef,N)
    #   pe  = Vector{Float64}(undef,N)
    #   kin_to_pot = Vector{Float64}(undef,N)
    #   pot_to_kin = Vector{Float64}(undef,N)
    #   mass = Vector{Float64}(undef,N)
    # end
    dt  = T/N
    τ   = dt/2 # APVM stabilization parameter
    hc  = CellField(h₀,Ω)
    uc  = CellField(u₀,Ω)
    for step=1:N
       # Williamsom2
       e = hn-h₀;err_h = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(hc⋅hc)*dΩ))
       e = un-u₀;err_u = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(uc⋅uc)*dΩ))

       println("step=", step, ",\terr_u: ", err_u, ",\terr_h: ", err_h,
               " ", norm(get_free_dof_values(Δu)), " ", norm(get_free_dof_values(Δh)))

       # ui(Δu,un)  = un       + (1-θ) * Δu
       # hi(Δh,hn)  = hn       + (1-θ) * Δh
       # hbi(Δh,hn,b) = hn + b   + (1-θ) * Δh

       function residual((u,h,qvort,F),(v,q,s,v2))
         uiΔu  = u
         hiΔh  = h
         hbiΔh = h
         ∫((1.0/dt)*v⋅(u-un)-(∇⋅(v))*(g*hbiΔh + 0.5*uiΔu⋅uiΔu)+
             (qvort-τ*(uiΔu⋅∇(qvort)))*(v⋅⟂(F,n)) +   # eq1
           (1.0/dt)*q*(h-hn))dΩ + ∫(q*(DIV(F)))dω +  # eq2
         ∫(s*qvort*hiΔh + ⟂(∇(s),n)⋅uiΔu - s*fn +   # eq3
             v2⋅(F-hiΔh*uiΔu))dΩ                      # eq4
       end

       function jacobian((u,h,qvort,F),(du,dh,dq,dF),(v,q,s,v2))
         uiΔu  = u
         uidu  = du
         hiΔh  = h
         hidh  = dh
         hbidh = dh
         ∫((1.0/dt)*v⋅du +  (dq    - τ*(uiΔu⋅∇(dq)+uidu⋅∇(qvort)))*(v⋅⟂(F ,n))
                         +  (qvort - τ*(           uiΔu⋅∇(qvort)))*(v⋅⟂(dF,n))
                         -  (∇⋅(v))*(g*hbidh +uiΔu⋅uidu)   +  # eq1
           (1.0/dt)*q*dh)dΩ + ∫(q*(DIV(dF)))dω             +  # eq2
           ∫(s*(qvort*hidh+dq*hiΔh) + ⟂(∇(s),n)⋅uidu       +  # eq3
             v2⋅(dF-hiΔh*uidu-hidh*uiΔu))dΩ                   # eq4
       end

       # Solve fully-coupled monolithic nonlinear problem
       # Use previous time-step solution, ΔuΔhqF, as initial guess
       # Overwrite solution into ΔuΔhqF
       # Adjust absolute tolerance ftol s.t. it actually becomes relative
       dY = get_fe_basis(Y)
       residualΔuΔhqF=residual(ΔuΔhqF,dY)
       @time r=assemble_vector(residualΔuΔhqF,Y)
       op=FEOperator(residual,jacobian,X,Y)
       nls=NLSolver(show_trace=true, method=:newton, ftol=nlrtol*norm(r,Inf), xtol=1.0e-02)
       solver=FESolver(nls)

       solve!(ΔuΔhqF,solver,op)

       # Update current solution
       unv .= get_free_dof_values(Δu)
       hnv .= get_free_dof_values(Δh)

       if (write_results)
        #  ke[step]=Eₖ(un,H,dΩ)
        #  pe[step]=Eₚ(hn,g,dΩ)
        #  kin_to_pot[step]=compute_kin_to_pot!(u1v,unv,divvh,hnv)
        #  pot_to_kin[step]=compute_pot_to_kin!(h1v,hnv,qdivu,unv)
        #  mass[step] = compute_mass(L2MM,hnv)
         if mod(step, out_period) == 0
           println(step)
           pvd[Float64(step)] = new_vtk_step(Ω,joinpath(out_dir,"n=$(step)"),hn,un)
         end
       end
    end
    if (write_results)
      pvd[Float64(N)] = new_vtk_step(Ω,joinpath(out_dir,"n=$(N)"),hn,un)
      vtk_save(pvd)
      # generate_energy_plots(out_dir,N,ke,pe,kin_to_pot,pot_to_kin)
      # # save global scalar snapshots
      # save(joinpath(out_dir,"wave_eq_geosciences_data.jld"), "hn_dot_div_un", kin_to_pot,
      #                                           "un_dot_grad_hn", pot_to_kin,
      #                                           "mass", mass,
      #                                           "kinetic", ke,
      #                                           "potential", pe)
    end
    #un,hn
  end
  if (write_results)
    rm(out_dir,force=true,recursive=true)
    mkdir(out_dir)
    pvdfile=joinpath(out_dir,
      "nswe_ncells_$(num_cells(model))_order_$(order)_theta_method_full_newton")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end

T=14580
model=CubedSphereDiscreteModel(8;radius=rₑ)
N=20
order=1
degree=4
θ=0.5
@time solve_nswe_theta_method_full_newton(model, order, degree, θ, T, N;
                                          write_results=false,out_period=10)


# RT = ReferenceFE(raviart_thomas,Float64,order)
# DG = ReferenceFE(lagrangian,Float64,order)
# CG = ReferenceFE(lagrangian,Float64,order+1)
# V  = FESpace(model,RT; conformity=:Hdiv) # Velocity and mass flux FE space
# Q  = FESpace(model,DG; conformity=:L2)   # Fluid depth FE space
# S  = FESpace(model,CG; conformity=:H1)   # Potential vorticity FE space
# U  = TrialFESpace(V)
# P  = TrialFESpace(Q)
# R  = TrialFESpace(S)

# Y = MultiFieldFESpace([V,Q,S,V])         # Monolithic FE space
# X = MultiFieldFESpace([U,P,R,U])

# Ω  = Triangulation(model)
# n  = get_normal_vector(model)
# dΩ = Measure(Ω,degree)
# dω = Measure(Ω,degree,ReferenceDomain())

# dt  = T/N
# τ   = dt/2


# #un = interpolate_everywhere(u₀_rognes,U)
# #hn = interpolate_everywhere(h₀_rognes,P)
# #b  = interpolate_everywhere(topography,P)

# a(u,v)=∫(v⋅u)dΩ
# l(v)=∫(v⋅u₀)dΩ
# un=solve(AffineFEOperator(a,l,U,V))

# a(u,v)=∫(v*u)dΩ
# l(v)=∫(v*h₀)dΩ
# hn=solve(AffineFEOperator(a,l,P,Q))

# a(u,v)=∫(v*u)dΩ
# l(v)=∫(v*f₀)dΩ
# fn=solve(AffineFEOperator(a,l,R,S))

# # Williamsom2

# function residual((Δu,Δh,qvort,F),(v,q,s,v2))
#   uiΔu  = un#+Δu #Operation(ui)(Δu,un)
#   hiΔh  = hn#+Δh #Operation(hi)(Δh,hn)
#   hbiΔh = hn#+Δh #Operation(hbi)(Δh,hn,b)
#   #∫((1.0/dt)*v⋅Δu
#   ∫(-(∇⋅(v))*(g*hbiΔh + 0.5*uiΔu⋅uiΔu)+
#        (qvort-τ*(uiΔu⋅∇(qvort)))*(v⋅⟂(F,n)))dΩ +   # eq1
#     #(1.0/dt)*q*Δh)dΩ
#      ∫(q*(divergence(F)))dΩ +  # eq2
#   ∫(s*qvort*hiΔh + ⟂(∇(s),n)⋅uiΔu - s*fn +   # eq3
#     v2⋅(F-hiΔh*uiΔu))dΩ                      # eq4
# end

# ΔuΔhqF=uhqF₀(q₀(un,hn,fn,R,S,n,dΩ),F₀(un,hn,U,V,dΩ),X,Y,dΩ)
# #Δu,Δh,q,F = ΔuΔhqF
# dY = get_fe_basis(Y)
# residualΔuΔhqF=residual(ΔuΔhqF,dY)
# @time r=assemble_vector(residualΔuΔhqF,Y)
# println(norm(r))

# rh=FEFunction(X,r)
# ruh,rhh,rqh,rFh=rh

# writevtk(Triangulation(model),"kk",cellfields=["un"=>un,"un_"=>u₀,
#                                                "hn"=>hn,"hn_"=>h₀,
#                                                "fn"=>fn,"fn_"=>f₀])
# writevtk(Triangulation(model),"rr",cellfields=["ruh"=>ruh,"rhh"=>rhh,"rqh"=>rqh,"rFh"=>rFh])


end # module
