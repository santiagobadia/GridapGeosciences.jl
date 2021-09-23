module GalewskyShallowWaterIMEX

using Gridap
using GridapGeosciences

include("GalewskyInitialConditions.jl")

# Solves the Galewsky test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a shear flow instability of a zonal
# jet triggered by an initial gravity wave.
# reference:
#   Galewsky, Scott and Polvani (2004) Tellus, 56A 429-440

order  = 1
degree = 4

n      = 48
nstep  = 20*24*60 # 20 days
dt     = 60.0

model = CubedSphereDiscreteModel(n; radius=rₑ)

hf, uf = shallow_water_imex_time_stepper(model, order, degree,
                                    h₀, u₀, f, g,
                                    dt, 0.5*dt, nstep;
                                    write_solution=true,
                                    write_solution_freq=240,
                                    write_diagnostics=true,
                                    write_diagnostics_freq=1,
                                    dump_diagnostics_on_screen=true)

end
