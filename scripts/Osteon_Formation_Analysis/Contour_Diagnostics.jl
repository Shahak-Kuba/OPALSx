# Per-osteocyte contour diagnostics (local / interactive).
#
# Two diagnostic figures for a single dataset:
#   1. plot_osteocyte_contour   — the 2-D contour a chosen osteocyte's curvature is
#      measured on, with the osteocyte marked and a grey reference circle of the
#      contour's mean curvature behind it (title = mean available curvature).
#   2. plot_smoothing_effect    — raw vs Gaussian-smoothed level-set field ϕ at a
#      formation time / slice, each as a grayscale heatmap with the zero contour
#      in red, to inspect how smoothing changes the contour.
#
# Run from the project root:
#   julia --project=. scripts/Osteon_Formation_Analysis/Contour_Diagnostics.jl
# Figures are saved to ./figures/ and (under GLMakie) shown interactively.

# ── Configuration ────────────────────────────────────────────────────────────
dataset       = "FM40-4-R2"
osteocyte_idx = 10            # which osteocyte to inspect (1 = earliest-forming; see note below)
dx = 0.379; dy = 0.379; dz = 0.358    # voxel spacings [µm]
σ_smooth      = 2.0          # Gaussian σ [µm] applied to ϕ before curvature
k_scale_um    = 60.0         # arc length [µm] over which curvature is measured

# ── Source modules ─────────────────────────────────────────────────────────────
include("../../src/Imaging.jl")
include("../../src/LevelSet.jl")
include("../../src/Geometry.jl")
include("../../src/Analysis.jl")
include("../../src/Plotting.jl")
using .Imaging, .LevelSet, .Geometry, .Analysis, .Plotting

# ── Plotting backend ─────────────────────────────────────────────────────────
const BACKEND = :gl     # :gl → GLMakie (interactive)   :cairo → CairoMakie (high-res files)
if BACKEND == :cairo
    using CairoMakie; CairoMakie.activate!(px_per_unit = 3)
else
    using GLMakie; GLMakie.activate!()
end

using CSV, DataFrames

const FIGDIR = "./figures"; mkpath(FIGDIR)
saveshow(name, fig) = (save(joinpath(FIGDIR, "$(dataset)_$(name)"), fig); BACKEND == :gl && display(fig); fig)

function load_osteocytes(csv_path; dx, dy, dz)
    data = DataFrame(permutedims(Matrix(CSV.read(csv_path, DataFrame))), :auto)
    Ocy_pos       = Tuple{Float64,Float64,Float64}[]
    Ocy_pos_voxel = Tuple{Int,Int,Int}[]
    for idx in eachindex(data.x3)[2:end]
        data.x2[idx] == "True" || continue
        x = parse(Float64, data.x3[idx]); y = parse(Float64, data.x4[idx]); z = parse(Float64, data.x5[idx])
        push!(Ocy_pos,       (x, y, z))
        push!(Ocy_pos_voxel, (round(Int, x/dx), round(Int, y/dy), round(Int, z/dz)))
    end
    return Ocy_pos, Ocy_pos_voxel
end

# ── Build masks / EDT / formation times ──────────────────────────────────────
println("Processing: $dataset")
img_paths              = readdir("./DATA/$dataset/Processed_Images/"; join=true)
Ocy_pos, Ocy_pos_voxel = load_osteocytes("./DATA/$dataset/cells_$(dataset)_.csv"; dx, dy, dz)
println("  Osteocytes (complete): $(length(Ocy_pos))")

a_outer, a_inner       = build_outer_inner(img_paths)
outer_dt_S, inner_dt_S = compute_EDT_S(a_outer, a_inner; dx, dy, dz)

t_form     = estimate_Ocy_formation_time(outer_dt_S, inner_dt_S, Ocy_pos_voxel)
idx_sort   = sortperm(t_form)
t_sorted   = t_form[idx_sort]                 # formation-time-ordered (matches compute_curvature_near_osteocyte)
pos_sorted = Ocy_pos_voxel[idx_sort]

@assert 1 ≤ osteocyte_idx ≤ length(t_sorted) "osteocyte_idx must be in 1:$(length(t_sorted))"
println("Inspecting osteocyte $osteocyte_idx  (t_form = $(round(t_sorted[osteocyte_idx]; sigdigits=3)))")

set_theme!(fontsize = 26, figure_padding = 20)

# ── Figure 1: the osteocyte's curvature contour ──────────────────────────────
f1 = plot_osteocyte_contour(osteocyte_idx, t_sorted, outer_dt_S, inner_dt_S,
                            pos_sorted, dx, dy, dz, σ_smooth; k_scale_um = k_scale_um)
saveshow("osteocyte_$(osteocyte_idx)_contour.png", f1)

# ── Figure 2: smoothing effect on the contour (same osteocyte's t and slice) ─
t_inspect = t_sorted[osteocyte_idx]
z_inspect = pos_sorted[osteocyte_idx][3]
f2 = plot_smoothing_effect(t_inspect, outer_dt_S, inner_dt_S, z_inspect, dx, dy, dz, σ_smooth)
saveshow("smoothing_effect_t$(round(t_inspect; sigdigits=2)).png", f2)

println("\nFinished. Figures saved to $FIGDIR")
