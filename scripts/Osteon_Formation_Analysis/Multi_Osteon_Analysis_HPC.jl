# =============================================================================
# Multi-osteon formation analysis — HPC / headless batch version
# =============================================================================
#
# Same computation as Multi_Osteon_Analysis.jl, but designed to run on a compute
# node with NO display and NO GPU:
#   • plots with CairoMakie (software renderer) instead of GLMakie;
#   • the 3-D formation-front surface is built with marching cubes and drawn as a
#     mesh (CairoMakie cannot render GLMakie's volumetric isosurface);
#   • never calls `display()` — figures are saved to OPALSx/output/;
#   • all paths are resolved relative to this file, so it runs from ANY working
#     directory (this avoids the "doubled path" error you hit).
#
# ── Running it ───────────────────────────────────────────────────────────────
#   julia /path/to/OPALSx/scripts/Osteon_Formation_Analysis/Multi_Osteon_Analysis_HPC.jl
# (the script activates the OPALSx project itself, so --project is optional).
#
# ── One-time setup on the HPC (do this on a LOGIN node with internet) ─────────
# Compute nodes usually have no internet, but Pkg needs it to fetch packages.
# Build/precompile everything once on the login node:
#   cd /path/to/OPALSx
#   julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
# The distance transform is now native Julia (no SciPy/Conda), so no conda
# environment is needed on the compute node.
# =============================================================================

import Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(PROJECT_ROOT)                       # use the OPALSx environment

# ── Headless plotting backend (NO GLMakie, NO display needed) ────────────────
using CairoMakie
CairoMakie.activate!()

# ── Core analysis modules ────────────────────────────────────────────────────
# None of these need Makie. Order matters: LevelSet and Geometry must be loaded
# before Analysis, which imports `smooth_ϕ` / `compute_zero_contour_xy_coords`
# from them via `using ..LevelSet` / `using ..Geometry`.
include(joinpath(PROJECT_ROOT, "src", "Imaging.jl"))
include(joinpath(PROJECT_ROOT, "src", "LevelSet.jl"))
include(joinpath(PROJECT_ROOT, "src", "Geometry.jl"))
include(joinpath(PROJECT_ROOT, "src", "Analysis.jl"))
include(joinpath(PROJECT_ROOT, "src", "Plotting.jl"))   # Makie-based; CairoMakie supplies the backend
using .Imaging, .LevelSet, .Geometry, .Analysis, .Plotting

using CSV, DataFrames
using Meshing, GeometryBasics            # headless 3-D isosurface → mesh

# ── Configuration ────────────────────────────────────────────────────────────
datasets = ["FM40-2-E5"]
dx = 0.379; dy = 0.379; dz = 0.4          # voxel spacings [µm]
σ_smooth = 2.0                             # Gaussian σ [µm] for the curvature step

SAVE_SURFACE_3D    = true                  # also save the 3-D formation-front figure
SURFACE_DATASET    = datasets[1]           # which dataset to render in 3-D
SURFACE_TVALS      = collect(0.0:0.5:1.0)  # formation times to show as isosurfaces
SURFACE_DOWNSAMPLE = 2                     # mesh stride (1 = full res; larger = lighter/faster)

const DATA_DIR = joinpath(PROJECT_ROOT, "DATA")
const OUT_DIR  = joinpath(PROJECT_ROOT, "output"); mkpath(OUT_DIR)

# ── Helpers ──────────────────────────────────────────────────────────────────
"""
    load_osteocytes(csv_path; dx, dy, dz) -> (Ocy_pos, Ocy_pos_voxel)

Read osteocyte positions from a Napari `cells_<name>_.csv` export (stored
transposed: rows = degree / complete / x / y / z). Only osteocytes flagged
complete are kept; returns µm positions and the matching integer voxel indices.
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

"""
    save_formation_surface(path, outer_dt_S, inner_dt_S, tvals; dx, dy, dz, σ_μm, downsample)

Headless replacement for `Plotting.plot_3d_surfaces!`. For each formation time in
`tvals`, build the level-set field, extract its zero-level isosurface with
marching cubes, and draw it as a `mesh!` (which CairoMakie *can* render, unlike
the volumetric `contour!`). `downsample` strides the field so the mesh stays
light enough for CairoMakie's CPU renderer. Saves a PNG to `path`.
"""
function save_formation_surface(path, outer_dt_S, inner_dt_S, tvals;
                                dx, dy, dz, σ_μm=0.0, downsample::Int=1)
    s  = downsample
    oa = @view outer_dt_S[1:s:end, 1:s:end, 1:s:end]
    ia = @view inner_dt_S[1:s:end, 1:s:end, 1:s:end]

    fig = Figure(size=(900, 800))
    ax  = Axis3(fig[1, 1]; title="Formation front", xlabel="x", ylabel="y", zlabel="z")
    cols = cgrad(:plasma, max(length(tvals), 2); categorical=true)

    for (i, t) in enumerate(tvals)
        ϕ = @. Float32((1 - t) * oa - t * ia)
        if σ_μm > 0                       # spacings scale with the stride
            ϕ = smooth_ϕ(ϕ; dx=dx*s, dy=dy*s, dz=dz*s, σ_μm=σ_μm)
        end
        verts, faces = isosurface(ϕ, MarchingCubes())
        isempty(verts) && continue
        pts  = [Point3f(v...)        for v in verts]
        tris = [TriangleFace{Int}(f...) for f in faces]
        mesh!(ax, GeometryBasics.Mesh(pts, tris); color=cols[i], transparency=true, alpha=0.45)
    end

    save(path, fig)
    return path
end

# ── Per-dataset result storage ───────────────────────────────────────────────
dataset_labels       = String[]
t_form_all           = Vector{Vector{Float64}}()
κ_at_osteocyte_all   = Vector{Vector{Float64}}()
mean_available_κ_all = Vector{Vector{Float64}}()

# ── Main loop: one iteration per dataset ─────────────────────────────────────
for (di, name) in enumerate(datasets)
    println("\n" * "="^50)
    println("  Processing: $name   (dataset $di/$(length(datasets)))")
    println("="^50)

    img_paths              = readdir(joinpath(DATA_DIR, name, "Processed_Images"); join=true)
    Ocy_pos, Ocy_pos_voxel = load_osteocytes(joinpath(DATA_DIR, name, "cells_$(name)_.csv"); dx, dy, dz)
    println("  Osteocytes (complete): $(length(Ocy_pos))")

    t0 = time()
    a_outer, a_inner = build_outer_inner(img_paths)
    println("  Masks built from $(length(img_paths)) slices ($(round(time() - t0; digits=1)) s)")

    t0 = time()
    outer_dt_S, inner_dt_S = compute_EDT_S(a_outer, a_inner; dx, dy, dz)
    println("  Anisotropic distance transforms done ($(round(time() - t0; digits=1)) s)")

    t_form   = estimate_Ocy_formation_time(outer_dt_S, inner_dt_S, Ocy_pos_voxel)
    idx_sort = sortperm(t_form)

    κ_at_osteocyte, mean_available_κ = compute_curvature_near_osteocyte(
        t_form[idx_sort], outer_dt_S, inner_dt_S, Ocy_pos_voxel[idx_sort],
        dx, dy, dz, σ_smooth)

    push!(dataset_labels,       name)
    push!(t_form_all,           t_form[idx_sort])
    push!(κ_at_osteocyte_all,   κ_at_osteocyte)
    push!(mean_available_κ_all, mean_available_κ)
    println("  Done — $(length(κ_at_osteocyte)) curvature values computed.")
end

println("\nAll datasets processed.")

# ── Save numeric results (so they can be re-plotted/analysed anywhere) ───────
results = DataFrame(dataset=String[], t_form=Float64[], kappa_at=Float64[], mean_kappa=Float64[])
for i in eachindex(dataset_labels), j in eachindex(t_form_all[i])
    push!(results, (dataset_labels[i], t_form_all[i][j], κ_at_osteocyte_all[i][j], mean_available_κ_all[i][j]))
end
CSV.write(joinpath(OUT_DIR, "curvature_results.csv"), results)
println("Saved $(joinpath(OUT_DIR, "curvature_results.csv"))")

# ── Curvature figure (2-D, CairoMakie) ───────────────────────────────────────
set_theme!(theme_black(), fontsize=30)

f1 = Figure(size=(1400, 650))
a1 = Axis(f1[1, 1], xlabel="t_form", ylabel="κ  [µm⁻¹]",          title="Curvature at osteocyte")
a2 = Axis(f1[1, 2], xlabel="t_form", ylabel="κ − mean_κ  [µm⁻¹]", title="Curvature relative to contour mean")

for i in eachindex(dataset_labels)
    scatter!(a1, t_form_all[i], κ_at_osteocyte_all[i];                           markersize=15, label=dataset_labels[i])
    scatter!(a2, t_form_all[i], κ_at_osteocyte_all[i] .- mean_available_κ_all[i]; markersize=15, label=dataset_labels[i])
end

Legend(f1[1, 3], a1, "Dataset")
save(joinpath(OUT_DIR, "curvature_vs_tform.png"), f1; px_per_unit=3)
println("Saved $(joinpath(OUT_DIR, "curvature_vs_tform.png"))")

# ── Osteocyte distribution (formation time & curvature preference) ───────────
f3 = plot_osteocyte_distribution(t_form_all, κ_at_osteocyte_all, mean_available_κ_all,
                                 dataset_labels; relative=true, bins=15)
save(joinpath(OUT_DIR, "osteocyte_distribution.png"), f3; px_per_unit=3)
println("Saved $(joinpath(OUT_DIR, "osteocyte_distribution.png"))")

# ── Formation-front surface (3-D, headless via marching cubes) ───────────────
if SAVE_SURFACE_3D
    println("\nBuilding 3-D formation-front surface for $SURFACE_DATASET …")
    img_paths = readdir(joinpath(DATA_DIR, SURFACE_DATASET, "Processed_Images"); join=true)
    a_outer, a_inner = build_outer_inner(img_paths)
    outer_dt_S, inner_dt_S = compute_EDT_S(a_outer, a_inner; dx, dy, dz)
    out = joinpath(OUT_DIR, "formation_front_$(SURFACE_DATASET).png")
    save_formation_surface(out, outer_dt_S, inner_dt_S, SURFACE_TVALS;
                           dx=dx, dy=dy, dz=dz, σ_μm=σ_smooth, downsample=SURFACE_DOWNSAMPLE)
    println("Saved $out")
end

println("\nFinished. Outputs in $OUT_DIR")
