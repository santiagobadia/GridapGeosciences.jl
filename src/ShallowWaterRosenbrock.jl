
function compute_velocity_depth_residual!(duh₁,dΩ,dω,Y,Bchol,u2,q,F,ϕ,n)
  bᵤ₁(v) = ∫(-1.0*(q₁ - τ*u₁⋅∇(q₁))*(v⋅⟂(F,n)))dΩ + ∫(DIV(v)*ϕ)dω
  bₕ₂(q) = ∫(-q*DIV(F))dω
  bₕᵤ₁(v,q) = bᵤ₁(v) + bₕ₁(q)
  Gridap.FESpaces.assemble_vector!(bₕᵤ₁, get_free_dof_values(duh₁), Y)
  ldiv!(Bchol, get_free_values(duh₁))
end

clone_fe_function(space,f)=FEFunction(space,copy(get_free_dof_values(f)))

function shallow_water_rosenbrock_time_step!(
     y₂, ϕ, F, q₁, q₂, duh₁, duh₂, H1h, H1hchol, y_wrk,  # in/out args
     model, dΩ, dω, Y, V, Q, R, S, f, g, y₁,             # in args
     RTMMchol, L2MMchol, Amat, Bchol, dt, τ, λ)          # more in args

  # energetically balanced second order rosenbrock shallow water solver
  # reference: eqns (24) and (39) of
  # https://github.com/BOM-Monash-Collaborations/articles/blob/main/energetically_balanced_time_integration/EnergeticallyBalancedTimeIntegration_SW.tex
  #
  # f          : coriolis force (field)
  # g          : gravity (constant)
  # h₁         : fluid depth at current time level
  # u₁         : fluid velocity at current time level
  # RTMM       : H(div) mass matrix, ∫β⋅βdΩ, ∀β∈ H(div,Ω)
  # L2MM       : L² mass matrix, ∫γγdΩ, ∀γ∈ L²(Ω)
  # dt         : time step
  # τ          : potential vorticity upwinding parameter
  # dΩ         : measure of the elements

  n = get_normal_vector(model)

  # multifield terms
  u₁, h₁ = y₁
  u₂, h₂ = y₂

  du₁,dh₁ = duh₁
  du₂,dh₂ = duh₂

  # yₚ  = clone_fe_function(Y,y₁)
  # uₚ,hₚ = yₚ

  # 1.1: the mass flux
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*h₁)
  # 1.2: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,u₁⋅u₁,h₁,g)
  # 1.3: the potential vorticity
  compute_potential_vorticity!(q₁,H1h,H1hchol,dΩ,R,S,h₁,u₁,f,n)
  # 1.4: assemble the velocity residual
  bᵤ₁(v) = ∫(-1.0*(q₁ - τ*u₁⋅∇(q₁))*(v⋅⟂(F,n)))dΩ + ∫(DIV(v)*ϕ)dω
  Gridap.FESpaces.assemble_vector!(bᵤ₁, get_free_dof_values(du₁), V)
  # 1.5: assemble the depth residual
  bₕ₁(q) = ∫(-q*DIV(F))dω
  Gridap.FESpaces.assemble_vector!(bₕ₁, get_free_dof_values(dh₁), Q)

  bₕᵤ₁((v,q)) = bᵤ₁(v) + bₕ₁(q)
  Gridap.FESpaces.assemble_vector!(bₕᵤ₁, get_free_dof_values(duh₁), Y)

  # Solve for du₁, dh₁ over a MultiFieldFESpace

  ldiv!(Bchol, get_free_dof_values(duh₁))

  # update
  get_free_dof_values(u₂) .= get_free_dof_values(u₁) .+ dt .* get_free_dof_values(du₁)
  get_free_dof_values(h₂) .= get_free_dof_values(h₁) .+ dt .* get_free_dof_values(dh₁)

  # 2.1: the mass flux
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*(2.0*h₁ + h₂)/6.0+u₂*(h₁ + 2.0*h₂)/6.0)
  # 2.2: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,(u₁⋅u₁ + u₁⋅u₂ + u₂⋅u₂)/3.0,0.5*(h₁ + h₂),g)
  # 2.3: the potential vorticity
  compute_potential_vorticity!(q₂,H1h,H1hchol,dΩ,R,S,h₂,u₂,f,n)
  # 2.4: assemble the velocity residual
  bᵤ₂(v) = ∫(-0.5*(q₁ - τ*u₁⋅∇(q₁) + q₂ - τ*u₂⋅∇(q₂))*(v⋅⟂(F,n)))dΩ + ∫(DIV(v)*ϕ)dω
  Gridap.FESpaces.assemble_vector!(bᵤ₂, get_free_dof_values(du₂), V)
  # 2.5: assemble the depth residual
  bₕ₂(q) = ∫(-q*DIV(F))dω
  Gridap.FESpaces.assemble_vector!(bₕ₂, get_free_dof_values(dh₂), Q)

  # add A*[du₁,dh₁] to the [du₂,dh₂] vector
  bₕᵤ₂((v,q)) = bᵤ₂(v) + bₕ₂(q)
  Gridap.FESpaces.assemble_vector!(bₕᵤ₂, get_free_dof_values(duh₂), Y)

  mul!(y_wrk, Amat, get_free_dof_values(duh₁))
  get_free_dof_values(duh₂) .= -dt.*λ.*y_wrk .+ get_free_dof_values(duh₂)

  # solve for du₂, dh₂
  ldiv!(Bchol, get_free_dof_values(duh₂))

  du₂, dh₂ = duh₂
  # update yⁿ⁺¹
  get_free_dof_values(u₂) .= get_free_dof_values(u₁) .+ dt .* get_free_dof_values(du₂)
  get_free_dof_values(h₂) .= get_free_dof_values(h₁) .+ dt .* get_free_dof_values(dh₂)
end

function new_vtk_step(Ω,file,hn,un,wn)
  createvtk(Ω,
            file,
            cellfields=["hn"=>hn, "un"=>un, "wn"=>wn],
            nsubcells=4)
end

function compute_mean_depth!(wrk, L2MM, h)
  # compute the mean depth over the sphere, for use in the approximate Jacobian
  mul!(wrk, L2MM, get_free_dof_values(h))
  h_int = sum(wrk)
  wrk  .= 1.0
  tmp   = L2MM*wrk # create a new vector, only doing this once during initialisation
  a_int = sum(tmp)
  h_avg = h_int/a_int
  println("mean depth: ", h_avg)
  h_avg
end

function shallow_water_rosenbrock_time_stepper(model, order, degree,
                        h₀, u₀, f₀, g,
                        dt, τ, N;
                        write_diagnostics=true,
                        write_diagnostics_freq=1,
                        dump_diagnostics_on_screen=true,
                        write_solution=false,
                        write_solution_freq=N/10,
                        output_dir="nswe_eq_ncells_$(num_cells(model))_order_$(order)_rosenbrock")

  # Forward integration of the shallow water equations
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  dω = Measure(Ω, degree, ReferenceDomain())

  # Setup the trial and test spaces
  reffe_rt  = ReferenceFE(raviart_thomas, Float64, order)
  V = FESpace(model, reffe_rt ; conformity=:HDiv)
  U = TrialFESpace(V)
  reffe_lgn = ReferenceFE(lagrangian, Float64, order)
  Q = FESpace(model, reffe_lgn; conformity=:L2)
  P = TrialFESpace(Q)
  reffe_lgn = ReferenceFE(lagrangian, Float64, order+1)
  S = FESpace(model, reffe_lgn; conformity=:H1)
  R = TrialFESpace(S)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  # assemble the mass matrices
  amm(a,b) = ∫(a⋅b)dΩ
  H1MM = assemble_matrix(amm, R, S)
  RTMM = assemble_matrix(amm, U, V)
  L2MM = assemble_matrix(amm, P, Q)
  H1MMchol = lu(H1MM)
  RTMMchol = lu(RTMM)
  L2MMchol = lu(L2MM)

  # Project the initial conditions onto the trial spaces
  b₁(q)   = ∫(q*h₀)dΩ
  rhs1    = assemble_vector(b₁, Q)
  hn      = FEFunction(Q, copy(rhs1))
  ldiv!(L2MMchol, get_free_dof_values(hn))

  b₂(v)   = ∫(v⋅u₀)dΩ
  rhs2    = assemble_vector(b₂, V)
  un      = FEFunction(V, copy(rhs2))
  ldiv!(RTMMchol, get_free_dof_values(un))

  b₃(s)   = ∫(s*f₀)*dΩ
  rhs3    = assemble_vector(b₃, S)
  f       = FEFunction(S, copy(rhs3))
  ldiv!(H1MMchol, get_free_dof_values(f))

  # work arrays
  h_tmp = copy(get_free_dof_values(hn))
  w_tmp = copy(get_free_dof_values(f))
  # build the potential vorticity lhs operator once just to initialise
  bmm(a,b) = ∫(a*hn*b)dΩ
  H1h      = assemble_matrix(bmm, R, S)
  H1hchol  = lu(H1h)

  # assemble the approximate MultiFieldFESpace Jacobian
  n = get_normal_vector(model)
  H₀ = compute_mean_depth!(h_tmp, L2MM, hn)
  λ = 0.5 # magnitude of the descent direction of the implicit solve (neutrally stable for 0.5)
  Amat((u,p),(v,q)) =  ∫(f₀*(v⋅⟂(u,n)))dΩ - ∫(g*(DIV(v)*p))dω + ∫(H₀*(q*DIV(u)))dω # this one does NOT contain the mass matrices in the diagonal blocks
  Mmat((u,p),(v,q)) =  ∫(u⋅v)dΩ + ∫(p*q)dΩ # block mass matrix
  A = assemble_matrix(Amat, X,Y)
  M = assemble_matrix(Mmat, X,Y)
  B = M-dt*λ*A
  Bchol = lu(B)

  yn = un,hn

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"nswe__rosenbrock_diagnostics.csv")

    clone_fe_function(space,f)=FEFunction(space,copy(get_free_dof_values(f)))

    hm1    = clone_fe_function(Q,hn)
    ϕ      = clone_fe_function(Q,hn)
    dh1    = clone_fe_function(Q,hn)
    dh2    = clone_fe_function(Q,hn)

    um1    = clone_fe_function(V,un)
    F      = clone_fe_function(V,un)
    du1    = clone_fe_function(V,un)
    du2    = clone_fe_function(V,un)

    wn     = clone_fe_function(S,f)
    q1     = clone_fe_function(S,f)
    q2     = clone_fe_function(S,f)

    # mulifield fe functions
    ym1     = clone_fe_function(Y,yn)
    duh1    = clone_fe_function(Y,yn)
    duh2    = clone_fe_function(Y,yn)
    y_wrk   = copy(get_free_dof_values(yn))

    um1, hm1 = ym1
    dh1, du1 = duh1
    dh2, du2 = duh2

    if (write_diagnostics)
      initialize_csv(diagnostics_file,"time", "mass", "vorticity", "kinetic", "potential", "power")
    end

    # time step iteration loop
    for istep in 1:N
      # hm1,hn = hn,hm1
      # um1,un = un,um1

      ym1 = yn

      shallow_water_rosenbrock_time_step!(yn, ϕ, F, q1, q2, duh1, duh2, H1h, H1hchol, y_wrk,
                                          model, dΩ, dω, Y, V, Q, R, S, f, g, ym1,
                                          RTMMchol, L2MMchol, A, Bchol, dt, τ, λ)

      # shallow_water_rosenbrock_time_step!(yn, ϕ, F, q1, q2, duh1, duh2, H1h, H1hchol,
      #                                     model, dΩ, dω, Y, R, S, f, g, ym1,
      #                                     RTMMchol, L2MMchol, Amat, Bchol, dt, τ)

      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(model))
        dump_diagnostics_shallow_water!(h_tmp, w_tmp,
                                        model, dΩ, dω, S, L2MM, H1MM,
                                        hn, un, wn, ϕ, F, g, istep, dt,
                                        diagnostics_file,
                                        dump_diagnostics_on_screen)
      end
      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(model))
        pvd[Float64(istep)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(istep)"),hn,un,wn)
      end
    end
    hn, un
  end
  if (write_diagnostics || write_solution)
    rm(output_dir,force=true,recursive=true)
    mkdir(output_dir)
  end
  if (write_solution)
    pvdfile=joinpath(output_dir,"nswe_eq_ncells_$(num_cells(model))_order_$(order)_rosenbrock")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end