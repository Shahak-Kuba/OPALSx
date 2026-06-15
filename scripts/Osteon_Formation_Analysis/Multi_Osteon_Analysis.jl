# Multi-dataset osteon formation analysis.
#
# For each dataset this script reconstructs the bone-formation front from a
# segmented image stack, estimates the formation time of every osteocyte from
# its position between the cement line and the Haversian canal, and measures the
# surface curvature at each osteocyte. Results for all datasets are overlaid in a
# pair of scatter plots (curvature vs. formation time).
#
# Edit the `datasets` list below, then run from the project root:
#   julia --project=. scripts/Osteon_Formation_Analysis/Multi_Osteon_Analysis.jl
#
# Each dataset must have the following layout under DATA/:
#   DATA/<name>/Processed_Images/   – stack of segmented (red/green) images
#   DATA/<name>/cells_<name>_.csv   – osteocyte positions exported from Napari

# ── Dataset list ───────────────────────────────────────────────────────────────
datasets = ["FM40-1-R1", "FM40-1-R2", "FM40-2-R2"]

# ── Parameters ─────────────────────────────────────────────────────────────────
# voxel spacings [µm]
dx = 0.379; dy = 0.379; dz = 0.358   # voxel spacings [µm]
# Gaussian σ [µm] applied to ϕ before curvature
σ_smooth = 2.0                     
# arc length [µm] over which curvature is measured (~osteocyte size)
k_scale_um = 60.0                

# ── Source modules ─────────────────────────────────────────────────────────────
include("../../src/Imaging.jl")
include("../../src/LevelSet.jl")
include("../../src/Geometry.jl")
include("../../src/Analysis.jl")
include("../../src/Plotting.jl")          # needed for plot_3d_surfaces! (GLMakie)
using .Imaging, .LevelSet, .Geometry, .Analysis, .Plotting

# ── Plotting backend ─────────────────────────────────────────────────────────
# All figures below are written with the backend-agnostic Makie API, so the same
# code renders in either backend. Choose one:
#   :gl    → GLMakie    — interactive windows + 3-D rotation/animation (local use)
#   :cairo → CairoMakie — high-resolution static figures for posters/talks
const BACKEND = :gl
if BACKEND == :cairo
    using CairoMakie
    CairoMakie.activate!(px_per_unit = 3)        # 3× pixel density for crisp exports
else
    using GLMakie
    GLMakie.activate!()
end

using CSV, DataFrames

# Output directory for saved figures (created if missing).
const FIGDIR = "./figures"; mkpath(FIGDIR)
# Convenience: save in the chosen backend, and only pop up a window under GLMakie.
saveshow(name, fig) = (save(joinpath(FIGDIR, name), fig); BACKEND == :gl && display(fig); fig)

"""
    load_osteocytes(csv_path; dx, dy, dz) -> (Ocy_pos, Ocy_pos_voxel)

Read osteocyte positions from a Napari `cells_<name>_.csv` export.

The export is stored transposed (rows = degree / complete / x / y / z), so it is
transposed back: column `x2` is the "complete" flag and `x3`/`x4`/`x5` are the
x/y/z positions in µm. Only osteocytes flagged complete are kept; their µm
positions are returned alongside the corresponding integer voxel indices.
"""
function load_osteocytes(csv_path; dx, dy, dz)
    data = DataFrame(permutedims(Matrix(CSV.read(csv_path, DataFrame))), :auto)

    Ocy_pos       = Tuple{Float64,Float64,Float64}[]
    Ocy_pos_voxel = Tuple{Int,Int,Int}[]
    for idx in eachindex(data.x3)[2:end]
        data.x2[idx] == "True" || continue
        x = parse(Float64, data.x3[idx])
        y = parse(Float64, data.x4[idx])
        z = parse(Float64, data.x5[idx])
        push!(Ocy_pos,       (x, y, z))
        push!(Ocy_pos_voxel, (round(Int, x/dx), round(Int, y/dy), round(Int, z/dz)))
    end
    return Ocy_pos, Ocy_pos_voxel
end

# ── Per-dataset result storage ─────────────────────────────────────────────────
dataset_labels       = String[]
t_form_all           = Vector{Vector{Float64}}()
κ_at_osteocyte_all   = Vector{Vector{Float64}}()
mean_available_κ_all = Vector{Vector{Float64}}()

# ── Main loop: one iteration per dataset ───────────────────────────────────────
for (di, name) in enumerate(datasets)
    println("\n" * "═"^50)
    println("  Processing: $name   (dataset $di/$(length(datasets)))")
    println("═"^50)

    img_paths              = readdir("./DATA/$name/Processed_Images/"; join=true)
    Ocy_pos, Ocy_pos_voxel = load_osteocytes("./DATA/$name/cells_$(name)_.csv"; dx, dy, dz)
    println("  Osteocytes (complete): $(length(Ocy_pos))")

    # Build outer/inner masks and their anisotropic signed distance fields.
    # These are each a single slow call, so we just time them; the per-osteocyte
    # curvature step below prints its own progress bar.
    t0 = time()
    a_outer, a_inner = build_outer_inner(img_paths)
    println("  Masks built from $(length(img_paths)) slices ($(round(time() - t0; digits=1)) s)")

    t0 = time()
    outer_dt_S, inner_dt_S = compute_EDT_S(a_outer, a_inner; dx, dy, dz)
    println("  Anisotropic distance transforms done ($(round(time() - t0; digits=1)) s)")

    # Estimate formation times and order osteocytes from earliest to latest.
    t_form   = estimate_Ocy_formation_time(outer_dt_S, inner_dt_S, Ocy_pos_voxel)
    idx_sort = sortperm(t_form)

    # Curvature of the formation front at each osteocyte (progress bar inside).
    κ_at_osteocyte, mean_available_κ = compute_curvature_near_osteocyte(
        t_form[idx_sort], outer_dt_S, inner_dt_S, Ocy_pos_voxel[idx_sort],
        dx, dy, dz, σ_smooth; k_scale_um=k_scale_um)

    push!(dataset_labels,       name)
    push!(t_form_all,           t_form[idx_sort])
    push!(κ_at_osteocyte_all,   κ_at_osteocyte)
    push!(mean_available_κ_all, mean_available_κ)
    println("  Done — $(length(κ_at_osteocyte)) curvature values computed.")
end

println("\nAll datasets processed.")

# ── Export the KDE curves to CSV ─────────────────────────────────────────────
# The same Gaussian / Silverman-bandwidth KDE the density figures draw, as
# numbers: one tidy file with per-dataset and pooled curves for each quantity.
"Tidy DataFrame of per-dataset + pooled KDE curves for `variable`."
function kde_table(variable, data_all, labels)
    df = DataFrame(variable=String[], group=String[], x=Float64[], density=Float64[], bandwidth=Float64[])
    for i in eachindex(labels)
        k = pooled_kde(data_all[i]); n = length(k.x)
        append!(df, DataFrame(variable=fill(variable,n), group=fill(string(labels[i]),n),
                              x=k.x, density=k.density, bandwidth=fill(k.bandwidth,n)))
    end
    kp = pooled_kde(data_all); n = length(kp.x)
    append!(df, DataFrame(variable=fill(variable,n), group=fill("pooled",n),
                          x=kp.x, density=kp.density, bandwidth=fill(kp.bandwidth,n)))
    return df
end

Δκ_all = [κ_at_osteocyte_all[i] .- mean_available_κ_all[i] for i in eachindex(κ_at_osteocyte_all)]
kde_df = vcat(kde_table("formation_time",   t_form_all,         dataset_labels),
              kde_table("kappa",            κ_at_osteocyte_all, dataset_labels),
              kde_table("kappa_minus_mean", Δκ_all,             dataset_labels))
CSV.write(joinpath(FIGDIR, "kde_curves.csv"), kde_df)
println("Saved $(joinpath(FIGDIR, "kde_curves.csv"))")

# ── Plots ──────────────────────────────────────────────────────────────────────
set_theme!(fontsize=30, figure_padding=20)   # padding keeps axis labels/ticks off the figure edge

# ── Figure 1: curvature vs formation time (scatter) ──────────────────────────
f1 = Figure(size=(1400, 650))
a1 = Axis(f1[1, 1], xlabel="t_form", ylabel="κ  [µm⁻¹]",          title="Curvature at osteocyte")
a2 = Axis(f1[1, 2], xlabel="t_form", ylabel="κ − mean_κ  [µm⁻¹]", title="Curvature relative to contour mean")

for i in eachindex(dataset_labels)
    scatter!(a1, t_form_all[i], κ_at_osteocyte_all[i];                           markersize=15, label=dataset_labels[i])
    scatter!(a2, t_form_all[i], κ_at_osteocyte_all[i] .- mean_available_κ_all[i]; markersize=15, label=dataset_labels[i])
end

Legend(f1[1, 3], a1, "Dataset")
display(f1)
#saveshow("curvature_vs_tform.png", f1)

# ── Figure 2: osteocyte distribution (formation time & curvature preference) ──
# Top histogram → preferred formation time; right histogram → convex/concave
# preference (relative to the mean front curvature). Pass relative=false to see
# the absolute κ whose sign separates convex (>0) from concave (<0) front.
f2 = plot_osteocyte_distribution(t_form_all, κ_at_osteocyte_all, mean_available_κ_all,
                                 dataset_labels; relative=true, bins=15)
#saveshow("osteocyte_distribution.png", f2)

# ── Figure 3 Formation-time KDE (smooth alternative to the histogram).
f3 = plot_formation_time_density(t_form_all, dataset_labels)
#saveshow("formation_time_density.png", f3)

# ── Figure 4 Curvature distribution per formation-time bracket (violin + boxplot):
#    does the curvature preference change over time?
f4 = Plotting.plot_curvature_by_time_bracket(t_form_all, κ_at_osteocyte_all, mean_available_κ_all;
                                    relative=false, nbrackets=4)
#saveshow("curvature_by_time_bracket.png", f4)

# ── Figure 5 ECDF of formation time vs the uniform reference (diagonal): a clean read on
#    whether osteocytes form earlier/later than uniform.
f5 = Plotting.plot_formation_time_ecdf(t_form_all, dataset_labels)
#saveshow("formation_time_ecdf.png", f5)

f6 = plot_curvature_density(κ_at_osteocyte_all, mean_available_κ_all, dataset_labels; relative=false)

# ── Figure 2: 3-D formation-front surface ────────────────────────────────────
dataset_idx        = 1
dataset            = datasets[dataset_idx]
surface_downsample = 4
surface_tvals      = collect(0:0.5:1)

img_paths = readdir("./DATA/$dataset/Processed_Images/"; join=true)
a_outer, a_inner = build_outer_inner(img_paths; downsample=surface_downsample)
Ocy_pos, Ocy_pos_voxel = load_osteocytes("./DATA/$dataset/cells_$(dataset)_.csv"; dx, dy, dz)
# Downsampling is in-plane only, so x/y spacings scale by the stride; z is unchanged.
sdx, sdy = dx * surface_downsample, dy * surface_downsample
outer_dt_S, inner_dt_S = compute_EDT_S(a_outer, a_inner; dx=sdx, dy=sdy, dz=dz)
ϕ = compute_ϕ_stack_3D(outer_dt_S, inner_dt_S, surface_tvals)

surface_cmap  = :jet
surface_alpha = 0.7          # 1.0 = fully solid (hides inner shells); ~0.6–0.8 shows nesting

f6 = Figure(size=(1000, 750))
ax = Axis3(f6[1, 1]; title = "Formation front — $dataset", aspect=:data, viewmode=:fit,
           xlabel="x [µm]", ylabel="y [µm]", zlabel="z [µm]")
plot_3d_surfaces!(ax, ϕ, surface_tvals;
                  dx=sdx, dy=sdy, dz=dz, σ_μm=σ_smooth,
                  colormap=surface_cmap, alpha=surface_alpha,
                  show_osteocytes=true, osteocytes=Ocy_pos,   # Ocy_pos is in µm
                  osteocyte_color=:red, osteocyte_markersize=10)
# Colorbar: low colour = outermost surface (t=0, cement line), high = innermost (t=1, canal).
Colorbar(f6[1, 2]; colormap=surface_cmap, limits=extrema(surface_tvals), label="formation time  t")
#saveshow("formation_front_$(dataset).png", f6)
