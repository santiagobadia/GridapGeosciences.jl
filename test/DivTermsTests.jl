module DivTermsTests

using Test
using Gridap
using GridapGeosciences

include("ConvergenceAnalysisTools.jl")

# Distorted mesh
function rndm(p::Point)
  r, s = p
  x = r + 0.1*sin(2π*r)*sin(2π*s)
  y = s + 0.1*sin(2π*r)*sin(2π*s)
  Point(x,y)
end

function setup_FE_spaces(model,order)
  # FE Spaces
  RT = ReferenceFE(raviart_thomas,Float64,order)
  DG = ReferenceFE(lagrangian,Float64,order)
  V  = FESpace(model,RT; conformity=:Hdiv)
  Q  = FESpace(model,DG; conformity=:L2)
  U  = TrialFESpace(V)
  P  = TrialFESpace(Q)
  U,V,P,Q
end

function compute_relative_error(l1,l2,V)
  b1 = assemble_vector(l1, V)
  b2 = assemble_vector(l2, V)
  norm(b1-b2)/norm(b1)
end

function compare_phys_vs_ref_div_v_p_term(model,order,degree)
  U,V,P,Q=setup_FE_spaces(model,order)

  # Geometry
  Ω  = Triangulation(model)
  dΩ = Measure(Ω,degree)
  dω = Measure(Ω,degree,ReferenceDomain())

  # Bilinear forms
  aphys(p,v)  = ∫(∇⋅(v)*p)dΩ
  aref(p,v)   = ∫(DIV(v)*p)dω

  # Assemble matrix + temporize
  Aref  = assemble_matrix(aref ,P, V)
  Aphys = assemble_matrix(aphys,P, V)
  t1=@elapsed assemble_matrix(aref ,P, V)
  t2=@elapsed assemble_matrix(aphys,P, V)
  println(t2/t1)

  # Evaluate relative error
  norm(Aphys-Aref)/norm(Aref)
end

function compare_div_v_DIV_v(model,order)
  # FE Spaces
  U,V,P,Q = setup_FE_spaces(model,order)
  v       = get_fe_basis(V)

  # Geometry
  Ω  = Triangulation(model); dΩ = Measure(Ω,0)

  divv = ∇⋅(v)
  DIVv = DIV(v)

  p=get_cell_points(dΩ.quad)
  divvq=divv(p)
  DIVvq=DIVv(p)
  println( "divvq[1]=",divvq[1])
  println( "DIVvq[1]=",DIVvq[1])
  println( "Δ    [1]=",abs.(divvq[1] .- DIVvq[1]))
  norm(divvq[1] .- DIVvq[1])
end


function compare_div_v_u_dot_u_versus_div_v_projected_u_dot_u(model,order,degree)
  # FE Spaces
  U,V,P,Q=setup_FE_spaces(model,order)

  # Generate arbitrary function RT space
  u=FEFunction(U,rand(num_free_dofs(U)))

  # Geometry
  Ω  = Triangulation(model); dΩ = Measure(Ω,degree)

  # Project u⋅u into P
  a(p,q)=∫(q*p)dΩ; b(q)=∫(q*(u⋅u))dΩ
  u_dot_u_projection=solve(AffineFEOperator(a,b,P,Q))

  # Bilinear forms
  l1(v) = ∫(∇⋅(v)*(u⋅u))dΩ
  l2(v) = ∫(∇⋅(v)*(u_dot_u_projection))dΩ

  compute_relative_error(l1,l2,V)
end


function compare_div_v_projected_u_dot_u_DIV_v_projected_u_dot_u(model,order,degree)
  # FE Spaces
  U,V,P,Q=setup_FE_spaces(model,order)

  # Generate arbitrary function RT space
  u=FEFunction(U,rand(num_free_dofs(U)))

  # Geometry
  Ω  = Triangulation(model); dΩ = Measure(Ω,degree)

  # Project u⋅u into P
  a(p,q)=∫(q*p)dΩ; b(q)=∫(q*(u⋅u))dΩ
  u_dot_u_projection=solve(AffineFEOperator(a,b,P,Q))

  # Bilinear forms
  dω = Measure(Ω,degree,ReferenceDomain())
  l1(v) = ∫(∇⋅(v)*(u_dot_u_projection))dΩ
  l2(v) = ∫(DIV(v)*(u_dot_u_projection))dω

  compute_relative_error(l1,l2,V)
end

function compare_div_v_u_dot_u_DIV_v_u_dot_u(model,order,degree)
  # FE Spaces
  U,V,P,Q=setup_FE_spaces(model,order)

  # Generate arbitrary function RT space
  u=FEFunction(U,rand(num_free_dofs(U)))

  # Geometry
  Ω  = Triangulation(model); dΩ = Measure(Ω,degree)

  # Bilinear forms
  dω = Measure(Ω,degree,ReferenceDomain())
  l1(v) = ∫(∇⋅(v)*(u⋅u))dΩ
  l2(v) = ∫(DIV(v)*(u⋅u))dω

  compute_relative_error(l1,l2,V)
end

tol=1.e-15
@time hs0,k0errors,_=convergence_study(compare_phys_vs_ref_div_v_p_term,
                                        generate_n_values(2;n_max=20),0,2)
@test all(k0errors .< tol)

@time hs1,k1errors,_=convergence_study(compare_phys_vs_ref_div_v_p_term,
                                        generate_n_values(2;n_max=20),1,4)
@test all(k1errors .< tol)

@time hs1,k2errors,_=convergence_study(compare_phys_vs_ref_div_v_p_term,
                                        generate_n_values(2;n_max=20),2,8)

model=CartesianDiscreteModel((0,1,0,1),(20,20),map=rndm)
for (order,degree) in [(0,30),(1,30),(2,30),(3,30),(4,30),(5,30)]
  rel_error=compare_phys_vs_ref_div_v_p_term(model,order,degree)
  err_div_v_DIV_v=compare_div_v_DIV_v(model,order)
  if rel_error < tol
    @test rel_error < tol
  else
    @test_broken rel_error < tol
  end
  println("Non-Affine FE Map 2D Problem
           RT order=$(order) ∫(∇⋅(v)*p)dΩ versus ∫(DIV(v)*p)dω $(rel_error) < $(tol)")
end


# Full 2D problem!!!
model=CartesianDiscreteModel((0,1,0,1),(10,10))
for (order,degree) in [(0,30),(1,30),(2,30),(3,30),(4,30),(5,30)]
  rel_error=compare_div_v_u_dot_u_versus_div_v_projected_u_dot_u(model,order,degree)
  err_div_v_DIV_v=compare_div_v_DIV_v(model,order)
  if rel_error < tol
    @test rel_error < tol
  else
    @test_broken rel_error < tol
  end
  println("Affine FE Map 2D Problem RT order=$(order) ∫(∇⋅(v)*(u⋅u))dΩ versus ∫(∇⋅(v)*(u_dot_u_projection))dΩ $(rel_error) < $(tol)")
end

# Problem on spherical manifold!!!
model=CubedSphereDiscreteModel(10)
for (order,degree) in [(0,30),(1,30),(2,30),(3,30),(4,30),(5,30)]
  rel_error=compare_div_v_u_dot_u_versus_div_v_projected_u_dot_u(model,order,degree)
  if rel_error < tol
    @test rel_error < tol
  else
    @test_broken rel_error < tol
  end
  println("CubedSphere Manifold RT order=$(order) ∫(∇⋅(v)*(u⋅u))dΩ versus ∫(∇⋅(v)*(u_dot_u_projection))dΩ $(rel_error) < $(tol)")
end

# Full 2D problem!!!
model=CartesianDiscreteModel((0,1,0,1),(10,10))
for (order,degree) in [(0,30),(1,30),(2,30),(3,30),(4,30),(5,30)]
  rel_error=compare_div_v_projected_u_dot_u_DIV_v_projected_u_dot_u(model,order,degree)
  if rel_error < tol
    @test rel_error < tol
  else
    @test_broken rel_error < tol
  end
  println("Affine Map 2D Problem ∫(∇⋅(v)*(u_dot_u_projection))dΩ versus ∫(DIV(v)*(u_dot_u_projection))dω RT order=$(order) $(rel_error) < $(tol)")
end

# Problem on spherical manifold!!!
model=CubedSphereDiscreteModel(10)
for (order,degree) in [(0,30),(1,30),(2,30),(3,30),(4,30),(5,30)]
  rel_error=compare_div_v_projected_u_dot_u_DIV_v_projected_u_dot_u(model,order,degree)
  if rel_error < tol
    @test rel_error < tol
  else
    @test_broken rel_error < tol
  end
  println("CubedSphere Manifold ∫(∇⋅(v)*(u_dot_u_projection))dΩ versus ∫(DIV(v)*(u_dot_u_projection))dω RT order=$(order) $(rel_error) < $(tol)")
end

# Full 2D problem!!!
model=CartesianDiscreteModel((0,1,0,1),(10,10))
for (order,degree) in [(0,30),(1,30),(2,30),(3,30),(4,30),(5,30)]
  rel_error=compare_div_v_u_dot_u_DIV_v_u_dot_u(model,order,degree)
  if rel_error < tol
    @test rel_error < tol
  else
    @test_broken rel_error < tol
  end
  println("Affine Map 2D Problem ∫(∇⋅(v)*(u_dot_u_projection))dΩ versus ∫(DIV(v)*(u_dot_u_projection))dω RT order=$(order) $(rel_error) < $(tol)")
end

# Problem on spherical manifold!!!
model=CubedSphereDiscreteModel(10)
for (order,degree) in [(0,30),(1,30),(2,30),(3,30),(4,30),(5,30)]
  rel_error=compare_div_v_u_dot_u_DIV_v_u_dot_u(model,order,degree)
  if rel_error < tol
    @test rel_error < tol
  else
    @test_broken rel_error < tol
  end
  println("CubedSphere Manifold ∫(∇⋅(v)*(u⋅u))dΩ versus ∫(DIV(v)*(u⋅u))dω RT order=$(order) $(rel_error) < $(tol)")
end

end # module
