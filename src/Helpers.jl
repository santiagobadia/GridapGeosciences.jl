"""copy a FEFunction on a FE Space"""
clone_fe_function(space,f)=FEFunction(space,copy(get_free_dof_values(f)))

"""
 Given an arbitrary number of FEFunction arguments,
 returns a tuple with their corresponding free DoF Values
"""
function Gridap.get_free_dof_values(functions...)
  map(get_free_dof_values,functions)
end

function setup_mixed_spaces(model, order)
  reffe_rt  = ReferenceFE(raviart_thomas, Float64, order)
  V = FESpace(model, reffe_rt ; conformity=:HDiv)
  U = TrialFESpace(V)
  reffe_lgn = ReferenceFE(lagrangian, Float64, order)
  Q = FESpace(model, reffe_lgn; conformity=:L2)
  P = TrialFESpace(Q)
  reffe_lgn = ReferenceFE(lagrangian, Float64, order+1)
  S = FESpace(model, reffe_lgn; conformity=:H1)
  R = TrialFESpace(S)

  R, S, U, V, P, Q
end

function setup_and_factorize_mass_matrices(dΩ, R, S, U, V, P, Q)
  amm(a,b) = ∫(a⋅b)dΩ
  H1MM = assemble_matrix(amm, R, S)
  RTMM = assemble_matrix(amm, U, V)
  L2MM = assemble_matrix(amm, P, Q)
  H1MMchol = lu(H1MM)
  RTMMchol = lu(RTMM)
  L2MMchol = lu(L2MM)

  H1MM, RTMM, L2MM, H1MMchol, RTMMchol, L2MMchol
end

function new_vtk_step(Ω,file,hn,un,wn)
  createvtk(Ω,
            file,
            cellfields=["hn"=>hn, "un"=>un, "wn"=>wn],
            nsubcells=4)
end
