# =============================================================================
# Curvature-scale sweep on a SINGLE dataset — HPC / headless batch version
# =============================================================================
#
# Measures osteocyte curvature at several `k_scale_um` values for ONE dataset and
# produces comparison figures across scales. The masks, anisotropic distance
# transforms and formation times are computed ONCE (they don't depend on the
# curvature scale); only the per-osteocyte curvature step is repeated per k, so
# sweeping many scales is cheap.
#
# Headless: uses CairoMakie and the GLMakie-free OPALSx/hpc environment.
#
# ── Running it ───────────────────────────────────────────────────────────────
#   julia /path/to/OPALSx/scripts/Osteon_Formation_Analysis/kScale_Sweep_HPC.jl
# (activates the hpc/ environment itself, so --project is optional). Options:
#   --dataset=FM40-1-R1        single dataset to analyse
#   --k=20,60,100              comma-separated curvature scales [µm] (no spaces)
#   --mean_method=CCF          contour-mean method: CCF (default), CTF or ALF
#   --run=NAME                 output-folder name (default: a timestamp)
# e.g.:
#   julia .../kScale_Sweep_HPC.jl --dataset=FM40-1-R1 --k=20,60,100 --mean_method=CCF --run=FM40-1-R1_ksweep
#
# ── Outputs (bundled into output/<run>.zip; the run folder is deleted after
#    zipping with MAKE_ZIP=true, so only the .zip remains) ──────────────────────
#   curvature_results.csv          long format: k_scale_um, t_form, kappa_at, mean_kappa
#   kde_curves.csv                 per-scale KDE curves (kappa and kappa-mean)
#   curvature_vs_tform_by_scale.png
#   curvature_rel_vs_tform_by_scale.png
#   curvature_density_by_scale.png
#   curvature_by_scale.png
#   formation_time_density.png     (scale-independent, for reference)
#   run_info.txt
# Pull it to your laptop:  scp user@hpc:/path/to/OPALSx/output/<run>.zip ~/Downloads/
# =============================================================================

import Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(PROJECT_ROOT, "hpc"))      # GLMakie-free environment

using CairoMakie
CairoMakie.activate!()

include(joinpath(PROJECT_ROOT, "src", "Imaging.jl"))
include(joinpath(PROJECT_ROOT, "src", "LevelSet.jl"))
include(joinpath(PROJECT_ROOT, "src", "Geometry.jl"))
include(joinpath(PROJECT_ROOT, "src", "Analysis.jl"))
include(joinpath(PROJECT_ROOT, "src", "Plotting.jl"))
using .Imaging, .LevelSet, .Geometry, .Analysis, .Plotting

using CSV, DataFrames, Dates, ZipFile

# ── Command-line arguments ───────────────────────────────────────────────────
"""Locate a CLI flag's value string (`--name=VALUE` or `--name VALUE`); else `nothing`."""
function cli_value(flag::AbstractString)
    for (i, a) in enumerate(ARGS)
        startswith(a, flag * "=")       && return split(a, "="; limit = 2)[2]
        (a == flag && i < length(ARGS)) && return ARGS[i + 1]
    end
    return nothing
end

"""Read a comma-separated list of Float64 (`--name=20,60,100`); else `default`."""
function cli_floats(flag::AbstractString, default::Vector{Float64})
    valstr = cli_value(flag)
    valstr === nothing && return default
    items = filter(!isempty, strip.(split(valstr, ",")))
    isempty(items) && error("$flag was given but no numbers were parsed from '$valstr'.")
    return [(v = tryparse(Float64, s); v === nothing ? error("Bad number '$s' in $flag.") : v) for s in items]
end

"""Read the `--mean_method` flag (CCF/CTF/ALF, case-insensitive, optional `:`); else `default`."""
function cli_mean_method(default::Symbol = :CCF)
    valstr = cli_value("--mean_method")
    valstr === nothing && return default
    s = uppercase(strip(valstr)); startswith(s, ":") && (s = s[2:end])
    m = Symbol(s)
    m in (:ALF, :CCF, :CTF) || error("--mean_method must be one of ALF, CCF, CTF (got '$valstr').")
    return m
end

# ── Configuration ────────────────────────────────────────────────────────────
dataset          = something(cli_value("--dataset"), "FM40-1-R1")
k_scale_um_array = cli_floats("--k", [20.0, 60.0, 100.0])    # curvature scales [µm]
mean_method      = cli_mean_method()                          # :CCF (default), :CTF or :ALF
dx = 0.379; dy = 0.379; dz = 0.358                            # voxel spacings [µm]
σ_smooth = 2.0                                                # Gaussian σ [µm] before curvature
MAKE_ZIP = true
println("Dataset      : $dataset")
println("k_scale_um's : ", join(k_scale_um_array, ", "), " µm")
println("mean_method  : :$mean_method")

const DATA_DIR = joinpath(PROJECT_ROOT, "DATA")

# ── Self-contained run folder ────────────────────────────────────────────────
run_name = something(cli_value("--run"), "ksweep_$(dataset)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))
const OUT_DIR = joinpath(PROJECT_ROOT, "output", run_name); mkpath(OUT_DIR)
println("Output folder: $OUT_DIR")

open(joinpath(OUT_DIR, "run_info.txt"), "w") do io
    println(io, "OPALSx curvature-scale sweep")
    println(io, "timestamp        : ", now())
    println(io, "dataset          : ", dataset)
    println(io, "k_scale_um  [µm] : ", join(k_scale_um_array, ", "))
    println(io, "mean_method      : ", mean_method)
    println(io, "σ_smooth    [µm] : ", σ_smooth)
    println(io, "dx, dy, dz  [µm] : ", (dx, dy, dz))
    println(io, "julia version    : ", VERSION)
end

# ── Helper: read osteocyte positions ─────────────────────────────────────────
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

img_paths              = readdir(joinpath(DATA_DIR, dataset, "Processed_Images"); join=true)
Ocy_pos, Ocy_pos_voxel = load_osteocytes(joinpath(DATA_DIR, dataset, "cells_$(dataset)_.csv"); dx, dy, dz)
println("  Osteocytes (complete): $(length(Ocy_pos))")

t0 = time()
a_outer, a_inner = build_outer_inner(img_paths)
println("  Masks built from $(length(img_paths)) slices ($(round(time() - t0; digits=1)) s)")

t0 = time()
outer_dt_S, inner_dt_S = compute_EDT_S(a_outer, a_inner; dx, dy, dz)
println("  Anisotropic distance transforms done ($(round(time() - t0; digits=1)) s)")

t_form   = estimate_Ocy_formation_time(outer_dt_S, inner_dt_S, Ocy_pos_voxel)
idx_sort = sortperm(t_form)
t_sorted = t_form[idx_sort]
pos_sorted = Ocy_pos_voxel[idx_sort]

# ── Sweep the curvature scale (only this step depends on k) ──────────────────
k_labels   = String[]
κ_at_all   = Vector{Vector{Float64}}()
mean_κ_all = Vector{Vector{Float64}}()
for (ki, k) in enumerate(k_scale_um_array)
    println("\n  k_scale_um = $k µm   ($ki/$(length(k_scale_um_array)))")
    κ_at, mean_κ = compute_curvature_near_osteocyte(
        t_sorted, outer_dt_S, inner_dt_S, pos_sorted, dx, dy, dz, σ_smooth;
        k_scale_um=k, mean_method=mean_method)
    push!(k_labels,   "k=$(isinteger(k) ? Int(k) : k) µm")
    push!(κ_at_all,   κ_at)
    push!(mean_κ_all, mean_κ)
end
# t_form is the same for every scale (it does not depend on k).
t_form_all = [t_sorted for _ in k_scale_um_array]

println("\nAll scales processed.")

# ── Numeric results (long format) ────────────────────────────────────────────
results = DataFrame(k_scale_um=Float64[], dataset=String[], t_form=Float64[], kappa_at=Float64[], mean_kappa=Float64[])
for ki in eachindex(k_scale_um_array), j in eachindex(t_sorted)
    push!(results, (k_scale_um_array[ki], dataset, t_sorted[j], κ_at_all[ki][j], mean_κ_all[ki][j]))
end
CSV.write(joinpath(OUT_DIR, "curvature_results.csv"), results)
println("Saved $(joinpath(OUT_DIR, "curvature_results.csv"))")

# ── KDE curves per scale (the Gaussian/Silverman KDE the density plot draws) ──
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
kde_df = vcat(kde_table("kappa",            κ_at_all, k_labels),
              kde_table("kappa_minus_mean", Δκ_all,   k_labels))
CSV.write(joinpath(OUT_DIR, "kde_curves.csv"), kde_df)
println("Saved $(joinpath(OUT_DIR, "kde_curves.csv"))")

# ── Figures ──────────────────────────────────────────────────────────────────
set_theme!(fontsize=30, figure_padding=20)

# 1) curvature vs formation time, one series per scale (t_form is shared)
f1 = Figure(size=(1400, 650))
a1 = Axis(f1[1,1], xlabel="t_form", ylabel="κ  [µm⁻¹]",          title="Curvature at osteocyte — $dataset")
a2 = Axis(f1[1,2], xlabel="t_form", ylabel="κ − mean_κ  [µm⁻¹]", title="Curvature relative to contour mean")
for ki in eachindex(k_scale_um_array)
    scatter!(a1, t_sorted, κ_at_all[ki];               markersize=12, label=k_labels[ki])
    scatter!(a2, t_sorted, κ_at_all[ki] .- mean_κ_all[ki]; markersize=12, label=k_labels[ki])
end
Legend(f1[1,3], a1, "scale")
save(joinpath(OUT_DIR, "curvature_vs_tform_by_scale.png"), f1; px_per_unit=3)
println("Saved curvature_vs_tform_by_scale.png")

# 2) curvature KDE overlaid by scale (absolute κ; sign = convex/concave)
f2 = plot_curvature_density(κ_at_all, mean_κ_all, k_labels; relative=false)
save(joinpath(OUT_DIR, "curvature_density_by_scale.png"), f2; px_per_unit=3)
println("Saved curvature_density_by_scale.png")

# 3) violin/boxplot of curvature per scale (how magnitude/spread change with k)
f3 = plot_curvature_by_scale(k_scale_um_array, κ_at_all, mean_κ_all; relative=false)
save(joinpath(OUT_DIR, "curvature_by_scale.png"), f3; px_per_unit=3)
println("Saved curvature_by_scale.png")

# 4) relative-curvature density overlaid by scale (Δκ = κ − mean)
f4 = plot_curvature_density(κ_at_all, mean_κ_all, k_labels; relative=true)
save(joinpath(OUT_DIR, "curvature_rel_density_by_scale.png"), f4; px_per_unit=3)
println("Saved curvature_rel_density_by_scale.png")

# 5) formation-time distribution (scale-independent, for context)
f5 = plot_formation_time_density([t_sorted], [dataset])
save(joinpath(OUT_DIR, "formation_time_density.png"), f5; px_per_unit=3)
println("Saved formation_time_density.png")

# ── Bundle the run folder into one .zip ──────────────────────────────────────
function zip_folder(folder::AbstractString, zippath::AbstractString)
    base = dirname(folder); w = ZipFile.Writer(zippath)
    try
        for (root, _, files) in walkdir(folder), fname in files
            full = joinpath(root, fname)
            e = ZipFile.addfile(w, relpath(full, base); method = ZipFile.Deflate)
            write(e, read(full))
        end
    finally
        close(w)
    end
    return zippath
end
if MAKE_ZIP
    zippath = OUT_DIR * ".zip"
    zip_folder(OUT_DIR, zippath)
    println("Bundled outputs → $zippath")
    # Keep only the .zip: delete the run folder once the bundle is safely written.
    if isfile(zippath) && filesize(zippath) > 0
        rm(OUT_DIR; recursive = true)
        println("Removed folder $OUT_DIR (kept $zippath)")
    else
        @warn "Zip $zippath is missing or empty — keeping folder $OUT_DIR"
    end
end

println("\nFinished.")
println(MAKE_ZIP ? "Single-file bundle: $(OUT_DIR).zip" : "Outputs in $OUT_DIR")
