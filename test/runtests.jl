# OPALSx test suite.
#
# Run with:
#   julia --project=. -e 'using Pkg; Pkg.test()'
# or directly:
#   julia --project=. test/runtests.jl
#
# The tests exercise the main functions of LevelSet.jl, Geometry.jl and
# Analysis.jl on a synthetic concentric-cylinder geometry (see synthetic.jl):
# a large cylinder for the cement line, a small one for the Haversian canal, and
# random osteocyte positions in between. The source modules are `include`d
# directly (rather than `using OPALSx`) so the tests stay light and do not pull
# in GLMakie.

using Test
using Statistics

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "LevelSet.jl"))    # LevelSet (no deps on the others)
include(joinpath(SRC, "Geometry.jl"))    # Geometry
include(joinpath(SRC, "Analysis.jl"))    # Analysis — uses ..LevelSet / ..Geometry
using .LevelSet, .Geometry, .Analysis

include("synthetic.jl")

@testset "OPALSx" begin
    include("test_levelset.jl")
    include("test_geometry.jl")
    include("test_analysis.jl")
end
