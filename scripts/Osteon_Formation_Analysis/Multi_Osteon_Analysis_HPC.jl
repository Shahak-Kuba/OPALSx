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
# (the script activates the dedicated `hpc/` environment itself, so --project is
# optional).
#
# The datasets, curvature scale and run-folder name can be set from the terminal
# (no file edits):
#   julia .../Multi_Osteon_Analysis_HPC.jl --datasets=FM40-1-R1,FM40-2-R2 --k_scale_um=30 --run=FM40_k30
#   julia --project=hpc .../Multi_Osteon_Analysis_HPC.jl --k_scale_um 30
# `--datasets` is a comma-separated list (no spaces). Omitted flags use defaults.
#
# ── Outputs ──────────────────────────────────────────────────────────────────
# All outputs (figures, curvature_results.csv, run_info.txt) are written to ONE
# self-contained folder: output/<run>/  (a timestamp unless you pass --run=NAME),
# and that folder is also bundled into a single file output/<run>.zip.
# Pull it onto your laptop in one go (run this FROM your laptop):
#   scp user@hpc.address:/path/to/OPALSx/output/<run>.zip ~/Downloads/   # the .zip
#   scp -r user@hpc.address:/path/to/OPALSx/output/<run> ~/Downloads/    # or the folder
#
# This uses the OPALSx/hpc/ environment, which is the main project MINUS GLMakie.
# That way a headless compute node never installs or precompiles GLMakie (it needs
# system OpenGL/GLFW that such nodes lack) — plotting goes through CairoMakie only.
#
# ── One-time setup on the HPC (do this on a LOGIN node with internet) ─────────
# Compute nodes usually have no internet, but Pkg needs it to fetch packages.
# Build/precompile the HPC environment once on the login node:
#   cd /path/to/OPALSx
#   julia --project=hpc -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
# The distance transform is native Julia (no SciPy/Conda), so no conda
# environment is needed either.
# =============================================================================

import Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(PROJECT_ROOT, "hpc"))      # GLMakie-free environment

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
using ZipFile                            # bundle the run folder into one .zip

# ── Command-line arguments ───────────────────────────────────────────────────
"""Locate a CLI flag's value string (`--name=VALUE` or `--name VALUE`); else `nothing`."""
function cli_value(flag::AbstractString)
    for (i, a) in enumerate(ARGS)
        startswith(a, flag * "=")        && return split(a, "="; limit = 2)[2]
        (a == flag && i < length(ARGS))  && return ARGS[i + 1]
    end
    return nothing
end

"""Read a Float64 CLI flag (`--name=VALUE` or `--name VALUE`); fall back to `default`."""
function cli_float(flag::AbstractString, default::Real)
    valstr = cli_value(flag)
    valstr === nothing && return Float64(default)
    v = tryparse(Float64, valstr)
    v === nothing && error("Could not parse $flag value '$valstr' as a number.")
    return v
end

"""Read a comma-separated list CLI flag (`--name=A,B,C` or `--name A,B,C`); else `default`."""
function cli_strings(flag::AbstractString, default::Vector{String})
    valstr = cli_value(flag)
    valstr === nothing && return default
    items = filter(!isempty, strip.(split(valstr, ",")))
    isempty(items) && error("$flag was given but no names were parsed from '$valstr'.")
    return String.(items)
end

# ── Configuration ────────────────────────────────────────────────────────────
# Datasets to process. Override from the terminal (comma-separated, no spaces),
# e.g.:  julia --project=hpc <script> --datasets=FM40-1-R1,FM40-2-R2
datasets = cli_strings("--datasets", ["FM40-2-E5"])
println("Datasets: ", join(datasets, ", "))

dx = 0.379; dy = 0.379; dz = 0.358        # voxel spacings [µm]
σ_smooth = 2.0                             # Gaussian σ [µm] for the curvature step
# arc length [µm] over which curvature is measured (~osteocyte size).
# Override from the terminal, e.g.:  julia --project=hpc <script> --k_scale_um=30
k_scale_um = cli_float("--k_scale_um", 15.0)
println("Using k_scale_um = $k_scale_um µm")

SAVE_SURFACE_3D    = true                  # also save the 3-D formation-front figure
SURFACE_DATASET    = datasets[1]           # which dataset to render in 3-D
SURFACE_TVALS      = collect(0.0:0.5:1.0)  # formation times to show as isosurfaces
SURFACE_DOWNSAMPLE = 2                     # mesh stride (1 = full res; larger = lighter/faster)
MAKE_ZIP           = true                  # also bundle the run folder into output/<run>.zip

const DATA_DIR = joinpath(PROJECT_ROOT, "DATA")

# ── Self-contained run folder ────────────────────────────────────────────────
# Everything this run produces (figures, CSV, run_info.txt) goes into ONE folder
# under output/, so you can pull the whole thing off the HPC in one go. The name
# defaults to a timestamp; override with `--run`, e.g. `--run=FM40_k30`.
using Dates
run_name = something(cli_value("--run"), "run_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))
const OUT_DIR = joinpath(PROJECT_ROOT, "output", run_name); mkpath(OUT_DIR)
println("Output folder: $OUT_DIR")

# Write a small manifest of the run parameters so the folder is self-documenting.
open(joinpath(OUT_DIR, "run_info.txt"), "w") do io
    println(io, "OPALSx HPC run")
    println(io, "timestamp        : ", now())
    println(io, "datasets         : ", join(datasets, ", "))
    println(io, "k_scale_um  [µm] : ", k_scale_um)
    println(io, "σ_smooth    [µm] : ", σ_smooth)
    println(io, "dx, dy, dz  [µm] : ", (dx, dy, dz))
    println(io, "surface 3-D      : ", SAVE_SURFACE_3D, "  (dataset=", SURFACE_DATASET,
                ", downsample=", SURFACE_DOWNSAMPLE, ")")
    println(io, "julia version    : ", VERSION)
end

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

    # Physical-µm coordinate of each (strided) sample, so the surface is drawn to
    # scale in µm rather than in Meshing's normalised cube. The stride scales the
    # effective spacing on every axis (z is strided here too).
    sdx, sdy, sdz = dx * s, dy * s, dz * s
    H, W, D = size(oa)
    X = (1:H) .* sdx;  Y = (1:W) .* sdy;  Z = (1:D) .* sdz

    fig = Figure(size=(900, 800))
    ax  = Axis3(fig[1, 1]; title="Formation front", aspect=:data, viewmode=:fit,
                xlabel="x [µm]", ylabel="y [µm]", zlabel="z [µm]")
    cols = cgrad(:plasma, max(length(tvals), 2); categorical=true)

    for (i, t) in enumerate(tvals)
        ϕ = @. Float32((1 - t) * oa - t * ia)
        if σ_μm > 0                       # spacings scale with the stride
            ϕ = smooth_ϕ(ϕ; dx=sdx, dy=sdy, dz=sdz, σ_μm=σ_μm)
        end
        verts, faces = isosurface(ϕ, MarchingCubes(), X, Y, Z)   # vertices in µm
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
        dx, dy, dz, σ_smooth; k_scale_um=k_scale_um)

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
CSV.write(joinpath(OUT_DIR, "kde_curves.csv"), kde_df)
println("Saved $(joinpath(OUT_DIR, "kde_curves.csv"))")

# ── Curvature figure (2-D, CairoMakie) ───────────────────────────────────────
set_theme!(fontsize=30, figure_padding=20)   # padding keeps axis labels/ticks off the figure edge

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

# ── Formation-time KDE (smooth alternative to the histogram) ─────────────────
f4 = plot_formation_time_density(t_form_all, dataset_labels)
save(joinpath(OUT_DIR, "formation_time_density.png"), f4; px_per_unit=3)
println("Saved $(joinpath(OUT_DIR, "formation_time_density.png"))")

# ── Curvature KDE (relative=false → sign separates convex >0 / concave <0) ───
f5 = plot_curvature_density(κ_at_osteocyte_all, mean_available_κ_all, dataset_labels; relative=false)
save(joinpath(OUT_DIR, "curvature_density.png"), f5; px_per_unit=3)
println("Saved $(joinpath(OUT_DIR, "curvature_density.png"))")

# ── Curvature KDE (relative=true → κ_at_osteocyte - mean_available_κ) ───
f6 = plot_curvature_density(κ_at_osteocyte_all, mean_available_κ_all, dataset_labels; relative=true)
save(joinpath(OUT_DIR, "curvature_density_relative.png"), f6; px_per_unit=3)
println("Saved $(joinpath(OUT_DIR, "curvature_density_relative.png"))")

# ── Curvature per formation-time bracket (violin + boxplot) ──────────────────
f7 = plot_curvature_by_time_bracket(t_form_all, κ_at_osteocyte_all, mean_available_κ_all;
                                    relative=false, nbrackets=4)
save(joinpath(OUT_DIR, "curvature_by_time_bracket.png"), f7; px_per_unit=3)
println("Saved $(joinpath(OUT_DIR, "curvature_by_time_bracket.png"))")

# ── Formation-time ECDF vs the uniform reference (diagonal) ──────────────────
f8 = plot_formation_time_ecdf(t_form_all, dataset_labels)
save(joinpath(OUT_DIR, "formation_time_ecdf.png"), f8; px_per_unit=3)
println("Saved $(joinpath(OUT_DIR, "formation_time_ecdf.png"))")

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

# ── Bundle the whole run folder into a single .zip for easy transfer ──────────
"Zip the contents of `folder` into `zippath`, stored under the folder's own name."
function zip_folder(folder::AbstractString, zippath::AbstractString)
    base = dirname(folder)
    w = ZipFile.Writer(zippath)
    try
        for (root, _, files) in walkdir(folder)
            for fname in files
                full  = joinpath(root, fname)
                entry = ZipFile.addfile(w, relpath(full, base); method = ZipFile.Deflate)
                write(entry, read(full))
            end
        end
    finally
        close(w)
    end
    return zippath
end

if MAKE_ZIP
    zippath = OUT_DIR * ".zip"                       # output/<run>.zip (sits next to the folder)
    zip_folder(OUT_DIR, zippath)
    println("Bundled outputs → $zippath")
end

println("\nFinished. Outputs in $OUT_DIR")
MAKE_ZIP && println("Single-file bundle: $(OUT_DIR).zip")
