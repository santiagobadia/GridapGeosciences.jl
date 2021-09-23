using GridapGeosciences
using Test

@testset "GridapGeosciences" begin
  @time @testset "CubedSphereDiscreteModelsTests" begin include("CubedSphereDiscreteModelsTests.jl") end
  @time @testset "DarcyCubedSphereTests" begin include("DarcyCubedSphereTests.jl") end
  @time @testset "WeakDivPerpTests" begin include("WeakDivPerpTests.jl") end
  @time @testset "LaplaceBeltramiCubedSphereTests" begin include("LaplaceBeltramiCubedSphereTests.jl") end
  @time @testset "WaveEquationCubedSphereTests" begin include("WaveEquationCubedSphereTests.jl") end
  @time @testset "Williamson2ShallowWaterExplicitTests" begin include("Williamson2ShallowWaterExplicitTests.jl") end
  @time @testset "Williamson2ShallowWaterRosenbrockTests" begin include("Williamson2ShallowWaterRosenbrockTests.jl") end
  @time @testset "Williamson2ShallowWaterIMEXTests" begin include("Williamson2ShallowWaterIMEXTests.jl") end
  
end
