# Curvature-scale sweep on a SINGLE dataset — local / interactive version.
#
# Measures osteocyte curvature at several `k_scale_um` values for one dataset and
# overlays comparison figures across scales. The masks, anisotropic distance
# transforms and formation times are computed ONCE (they don't depend on the
# curvature scale); only the per-osteocyte curvature step is repeated per scale.
#
# Run from the project root:
#   julia --project=. scripts/Osteon_Formation_Analysis/kScale_Sweep.jl
#
# Edit `dataset` and `k_scale_um_array` below. Figures are saved to ./figures/
# and (under GLMakie) shown in interactive windows you can rotate/zoom.
# This is the local counterpart of kScale_Sweep_HPC.jl (which is headless and
# CLI-driven). For headless/HPC batch runs use that one instead.

# ── Configuration ────────────────────────────────────────────────────────────
dataset          = "FM40-1-R1"
k_scale_um_array = [20.0, 60.0, 100.0]    # curvature scales to compare [µm]
dx = 0.379; dy = 0.379; dz = 0.358        # voxel spacings [µm]
σ_smooth = 2.0                            # Gaussian σ [µm] applied to ϕ before curvature

# ── Source modules ─────────────────────────────────────────────────────────────
include("../../src/Imaging.jl")
include("../../src/LevelSet.jl")
include("../../src/Geometry.jl")
include("../../src/Analysis.jl")
include("../../src/Plotting.jl")
using .Imaging, .LevelSet, .Geometry, .Analysis, .Plotting

# ── Plotting backend ─────────────────────────────────────────────────────────
#   :gl    → GLMakie    — interactive windows + 3-D rotation (local use)
#   :cairo → CairoMakie — high-resolution static figures for posters/talks
const BACKEND = :gl
if BACKEND == :cairo
    using CairoMakie
    CairoMakie.activate!(px_per_unit = 3)
else
    using GLMakie
    GLMakie.activate!()
end

using CSV, DataFrames

const FIGDIR = "./figures"; mkpath(FIGDIR)
# Save (prefixed with the dataset) and, under GLMakie, also pop up a window.
saveshow(name, fig) = (save(joinpath(FIGDIR, "$(dataset)_$(name)"), fig); BACKEND == :gl && display(fig); fig)

"""
    load_osteocytes(csv_path; dx, dy, dz) -> (Ocy_pos, Ocy_pos_voxel)

Read osteocyte positions from a Napari `cells_<name>_.csv` export (stored
transposed). Only osteocytes flagged complete are kept; returns µm positions and
the matching integer voxel indices.
"""
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

# ── Compute the scale-independent quantities ONCE ────────────────────────────
println("\n" * "="^50)
println("  Processing: $dataset")
println("="^50)

img_paths              = readdir("./DATA/$dataset/Processed_Images/"; join=true)
Ocy_pos, Ocy_pos_voxel = load_osteocytes("./DATA/$dataset/cells_$(dataset)_.csv"; dx, dy, dz)
println("  Osteocytes (complete): $(length(Ocy_pos))")

t0 = time()
a_outer, a_inner = build_outer_inner(img_paths)
println("  Masks built from $(length(img_paths)) slices ($(round(time() - t0; digits=1)) s)")

t0 = time()
outer_dt_S, inner_dt_S = compute_EDT_S(a_outer, a_inner; dx, dy, dz)
println("  Anisotropic distance transforms done ($(round(time() - t0; digits=1)) s)")

t_form     = estimate_Ocy_formation_time(outer_dt_S, inner_dt_S, Ocy_pos_voxel)
idx_sort   = sortperm(t_form)
t_sorted   = t_form[idx_sort]
pos_sorted = Ocy_pos_voxel[idx_sort]

# ── Sweep the curvature scale (only this step depends on k) ──────────────────
k_labels   = String[]
κ_at_all   = Vector{Vector{Float64}}()
mean_κ_all = Vector{Vector{Float64}}()
for (ki, k) in enumerate(k_scale_um_array)
    println("\n  k_scale_um = $k µm   ($ki/$(length(k_scale_um_array)))")
    κ_at, mean_κ = compute_curvature_near_osteocyte(
        t_sorted, outer_dt_S, inner_dt_S, pos_sorted, dx, dy, dz, σ_smooth; k_scale_um=k)
    push!(k_labels,   "k=$(isinteger(k) ? Int(k) : k) µm")
    push!(κ_at_all,   κ_at)
    push!(mean_κ_all, mean_κ)
end
println("\nAll scales processed.")

# ── Numeric results + KDE curves (saved to ./figures) ────────────────────────
results = DataFrame(k_scale_um=Float64[], dataset=String[], t_form=Float64[], kappa_at=Float64[], mean_kappa=Float64[])
for ki in eachindex(k_scale_um_array), j in eachindex(t_sorted)
    push!(results, (k_scale_um_array[ki], dataset, t_sorted[j], κ_at_all[ki][j], mean_κ_all[ki][j]))
end
CSV.write(joinpath(FIGDIR, "$(dataset)_curvature_results.csv"), results)

function kde_table(variable, data_per_k, labels)
    df = DataFrame(variable=String[], group=String[], x=Float64[], density=Float64[], bandwidth=Float64[])
    for i in eachindex(labels)
        k = pooled_kde(data_per_k[i]); n = length(k.x)
        append!(df, DataFrame(variable=fill(variable,n), group=fill(string(labels[i]),n),
                              x=k.x, density=k.density, bandwidth=fill(k.bandwidth,n)))
    end
    return df
end
Δκ_all = [κ_at_all[i] .- mean_κ_all[i] for i in eachindex(κ_at_all)]
kde_df = vcat(kde_table("kappa", κ_at_all, k_labels), kde_table("kappa_minus_mean", Δκ_all, k_labels))
CSV.write(joinpath(FIGDIR, "$(dataset)_kde_curves.csv"), kde_df)

# ── Figures ──────────────────────────────────────────────────────────────────
set_theme!(fontsize=30, figure_padding=20)

# 1) curvature vs formation time, one series per scale (t_form is shared)
f1 = Figure(size=(1400, 650))
a1 = Axis(f1[1,1], xlabel="t_form", ylabel="κ  [µm⁻¹]",          title="Curvature at osteocyte — $dataset")
a2 = Axis(f1[1,2], xlabel="t_form", ylabel="κ − mean_κ  [µm⁻¹]", title="Curvature relative to contour mean")
for ki in eachindex(k_scale_um_array)
    scatter!(a1, t_sorted, κ_at_all[ki];                   markersize=12, label=k_labels[ki])
    scatter!(a2, t_sorted, κ_at_all[ki] .- mean_κ_all[ki]; markersize=12, label=k_labels[ki])
end
Legend(f1[1,3], a1, "scale")
saveshow("curvature_vs_tform_by_scale.png", f1)

# 2) curvature KDE overlaid by scale (absolute κ; sign = convex/concave)
f2 = plot_curvature_density(κ_at_all, mean_κ_all, k_labels; relative=false)
saveshow("curvature_density_by_scale.png", f2)

# 3) violin/boxplot of curvature per scale (how magnitude/spread change with k)
f3 = plot_curvature_by_scale(k_scale_um_array, κ_at_all, mean_κ_all; relative=false)
saveshow("curvature_by_scale.png", f3)

# 4) relative-curvature density overlaid by scale (Δκ = κ − mean)
f4 = plot_curvature_density(κ_at_all, mean_κ_all, k_labels; relative=true)
saveshow("curvature_rel_density_by_scale.png", f4)

# 5) formation-time distribution (scale-independent, for context)
f5 = plot_formation_time_density([t_sorted], [dataset])
saveshow("formation_time_density.png", f5)

println("\nFinished. Figures saved to $FIGDIR")
