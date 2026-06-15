"""
    OPALSx

Osteon Profile And Level-Set analysis toolkit.

OPALSx reconstructs the geometry of remodelling osteons from segmented confocal
image stacks and analyses where and when osteocytes were embedded during bone
formation. The package is organised into five submodules, each of which is
included and re-exported here:

- [`OPALSx.Imaging`](@ref)  – load segmented stacks and build outer/inner masks.
- [`OPALSx.LevelSet`](@ref) – signed distance transforms and the time-dependent
  level-set field `ϕ` whose zero contour is the bone-formation front.
- [`OPALSx.Geometry`](@ref) – contour extraction, centroids, cutting planes and
  plane-to-plane projections.
- [`OPALSx.Analysis`](@ref) – curvature estimators (2-D and 3-D) and osteocyte
  formation-time / curvature analysis.
- [`OPALSx.Plotting`](@ref) – Makie helpers (GLMakie or CairoMakie) for contours,
  isosurfaces and osteocyte-distribution figures.

# Usage
The package defines a Julia environment (see `Project.toml`).
Analysis scripts under `scripts/` activate this environment and `include` the
submodules directly, e.g.

```julia
julia --project=. scripts/Osteon_Formation_Analysis/Multi_Osteon_Analysis.jl
```

You can also load the package as a whole with `using OPALSx`, which brings the
exported functions of every submodule into scope.
"""
module OPALSx

using FileIO, Images, ImageBinarization, ImageMorphology, ImageSegmentation, ImageFiltering, Statistics
using DistanceTransforms
using Makie                 # backend-agnostic; load GLMakie or CairoMakie to render
using CSV, DataFrames
import Contour as CTR

include("Imaging.jl")
include("LevelSet.jl")
include("Geometry.jl")
include("Analysis.jl")
include("Plotting.jl")

using .Imaging
using .LevelSet
using .Geometry
using .Analysis
using .Plotting

export
    # Imaging — segmented stacks → outer/inner masks
    generate_RG_img_from_data, extract_sample_name, build_outer_inner, circle_mask,
    # LevelSet — signed distance fields and the level-set field ϕ
    edt, edt_S, edt_aniso, edt_S_aniso, compute_EDT_S, ϕ_func, compute_ϕ_at_t, compute_ϕ_stack,
    compute_ϕ_at_t_3D, compute_ϕ_stack_3D, estimate_Ocy_formation_time, smooth_ϕ,
    # Geometry — contours, centroids, cutting planes, projections
    compute_zero_contour_xy_coords, Ω, compute_xy_center, Plane,
    compute_planes_and_intersections, proj_3D_onto_XZ,
    # Analysis — curvature and osteocyte analysis
    analysis_Tdelay_pairs, compute_curvature, compute_curvature_4th,
    compute_2D_curvature, curvature_at_point, estimate_osteocyte_curvature_3D,
    compute_curvature_near_osteocyte,
    # Plotting — Makie entry points (GLMakie or CairoMakie backend)
    plot_3d_contours!, plot_3d_contours_w_intersections!, plot_example_slices!,
    plot_α_β!, plot_3d_surfaces!, plot_osteocyte_distribution,
    plot_formation_time_density, plot_curvature_density, plot_tform_curvature_hexbin,
    plot_curvature_by_time_bracket, plot_formation_time_ecdf, pooled_kde

end # module OPALSx
