"""
    Plotting

Makie helpers for visualising osteon reconstructions.

The functions use the **backend-agnostic `Makie` API**, so every plot works with
either backend — activate one before plotting:

- `using GLMakie; GLMakie.activate!()` — interactive windows and 3-D animation;
- `using CairoMakie; CairoMakie.activate!(px_per_unit=3)` — high-resolution
  static figures for posters/papers (`save("fig.png"/"fig.pdf", fig)`).

Drawing functions take a caller-supplied axis (`Axis`/`Axis3`) and mutate it in
place (trailing `!`); figure-level functions return a `Figure`. `smooth_levelset`
applies the same anisotropic Gaussian blur used elsewhere so plotted surfaces
match the analysed ones.
"""
module Plotting

using Makie                   # backend-agnostic API (GLMakie or CairoMakie supplies the backend)
using ImageFiltering
using Meshing                 # marching-cubes isosurface extraction
using KernelDensity           # the same KDE engine `density!` uses (Gaussian kernel)
import GeometryBasics         # Mesh / TriangleFace (qualified to avoid clashing with Makie exports)
import Contour as CTR

# Cross-module helpers for the per-osteocyte diagnostics below. `..` resolves to
# the enclosing module (OPALSx as a package, or Main when scripts `include` the
# sources) — Plotting is always loaded after LevelSet/Geometry/Analysis.
using ..LevelSet:  smooth_ϕ
using ..Geometry:  compute_zero_contour_xy_coords
using ..Analysis:  compute_2D_curvature

export plot_3d_contours!, plot_3d_contours_w_intersections!, plot_example_slices!, plot_α_β!,
       plot_3d_surfaces!, plot_osteocyte_distribution,
       plot_formation_time_density, plot_curvature_density, plot_tform_curvature_hexbin,
       plot_curvature_by_time_bracket, plot_curvature_by_scale, plot_formation_time_ecdf, pooled_kde,
       plot_osteocyte_contour, plot_smoothing_effect

# ── LaTeX label helpers ───────────────────────────────────────────────────────
# Every user-facing string — axis labels, titles, legend entries and tick labels —
# is compiled as LaTeX (`L"..."`) for consistent typography across all figures.
#
# `_tex` wraps an arbitrary string (e.g. a dataset name) as upright LaTeX text.
# MathTeXEngine renders an ASCII "-" (U+002D) as a minus sign even inside `\text{}`,
# which would mangle hyphenated identifiers like "FM40-1-R1"; swapping each hyphen
# for U+2010 (the typographic HYPHEN) makes them render correctly.
_tex(s) = L"\text{%$(replace(string(s), '-' => '‐'))}"

# Shared curvature axis label.
_kappa_label(relative) = relative ? L"\kappa - \overline{\kappa}\ [\mathrm{µm}^{-1}]" :
                                    L"\kappa\ \text{at osteocyte}\ [\mathrm{µm}^{-1}]"

# Format numeric tick values as LaTeX (clean numbers: strip float noise and trailing
# zeros; integers shown without a decimal point; negatives keep a real minus sign).
function _fmt_num(v)
    r = round(float(v); digits = 10)
    isinteger(r) && return string(Int(r))
    return rstrip(rstrip(string(r), '0'), '.')
end
_latex_ticks(values) = [L"%$(_fmt_num(v))" for v in values]
const _LTICKS = (xtickformat = _latex_ticks, ytickformat = _latex_ticks)

"""
    plot_3d_contours!(ax, ϕ, Δz, tvals)

Draw the zero-level contours of the bottom (`z=0`, red) and top (`z=Δz`, blue)
slices of the 4-D level-set stack `ϕ` onto `Axis3` `ax`, one contour pair per
formation time in `tvals`. Mutates `ax`.
"""
function plot_3d_contours!(ax, ϕ, Δz, tvals)
    H,W,_,_ = size(ϕ)
    x = collect(1:H); y = collect(1:W)
    for (ti,t) in enumerate(tvals)
        ϕ_bottom = ϕ[:,:,1,ti]
        cset_bot = CTR.contours(x,y,ϕ_bottom, [0])
        line_bot = first(CTR.lines(first(CTR.levels(cset_bot))))
        x_bot, y_bot = CTR.coordinates(line_bot)

        ϕ_top = ϕ[:,:,2,ti]
        cset_top = CTR.contours(x,y,ϕ_top, [0])
        line_top = first(CTR.lines(first(CTR.levels(cset_top))))
        x_top, y_top = CTR.coordinates(line_top)

        lines!(ax, x_bot, y_bot, zeros(size(x_bot)), linewidth=3, color=:red)
        lines!(ax, x_top, y_top, Δz.*ones(size(x_top)), linewidth=3, color=:blue)
    end
end

"""
    smooth_levelset(ϕ3d; dx, dy, dz, σ_μm) -> Array{Float32, 3}

Apply an anisotropic 3-D Gaussian blur to the level-set volume `ϕ3d` to
remove staircase artefacts that arise from the discrete voxel grid.

The standard deviation `σ_μm` is given in **physical units (µm)** and is
converted to voxel units per axis using the supplied spacings `dx`, `dy`, `dz`.
This ensures the smoothing is spatially isotropic even when the voxel
dimensions differ (e.g. dx = dy = 0.379 µm, dz = 0.358 µm).

A σ of ~1 µm removes the staircase while preserving the macro-scale surface
shape; increase to ~2 µm for heavier smoothing.

Uses a **separable** kernel (`KernelFactors.gaussian`), applied as three 1-D
passes — equivalent to a dense 3-D Gaussian but with far lower memory use.
"""
function smooth_levelset(ϕ3d::AbstractArray{<:Real,3};
                          dx::Real=0.379, dy::Real=0.379, dz::Real=0.358,
                          σ_μm::Real=1.0)
    # Convert physical σ to per-axis voxel σ
    σ_vox = (σ_μm/dx, σ_μm/dy, σ_μm/dz)
    kernel = ImageFiltering.KernelFactors.gaussian(σ_vox)
    return Float32.(imfilter(ϕ3d, kernel))
end

"""
    plot_3d_surfaces!(ax, ϕ, tvals; dx, dy, dz, σ_μm, colormap, alpha) -> last mesh plot

Render the zero-level surface of the level-set field `ϕ` for each formation time
in `tvals` as a **solid triangle mesh** (extracted with marching cubes), one per
time, drawn into the GLMakie `Axis3` `ax`.

Why a mesh and not `contour!`? GLMakie's volumetric `contour!` uploads the whole
volume to the GPU as a 3-D texture and renders it with view-dependent
transparency, which (a) looks "see-through" from some angles and (b) is heavy on
GPU memory. A marching-cubes mesh is a genuine opaque surface — consistent from
every angle and far lighter on the GPU.

Each surface is coloured by its formation time through a single `colormap`:
`tvals` is mapped onto the colormap's full range, so the **outermost** surface
(smallest `t`, the cement line) gets the **lowest** colour and the **innermost**
surface (largest `t`, the canal wall) the **highest** colour. Add a matching
`Colorbar(fig[...]; colormap, limits=extrema(tvals))` for a legend.

The optional `σ_μm` applies an anisotropic Gaussian smooth before extraction to
remove voxel staircase artefacts (see [`smooth_levelset`](@ref)).

Everything is drawn in **physical µm**: the mesh vertices are placed at
`index · spacing` along each axis (using `dx`, `dy`, `dz`), rather than in
Meshing's normalised cube. Pass the spacings that match `ϕ` — if `ϕ` was built
from a downsampled volume, use the *downsampled* in-plane spacings (e.g.
`dx*stride`). The optional osteocyte overlay is then scattered directly at its µm
positions, so the two line up.

Arguments
- `ax`    : a GLMakie `Axis3`
- `ϕ`     : 4-D level-set field `(H, W, Z, T)`
- `tvals` : formation-time vector corresponding to the T dimension of `ϕ`

Keyword arguments
- `dx`, `dy`, `dz` : voxel spacings in µm — used both to scale the surface to µm
                     and (when σ_μm > 0) for smoothing
- `σ_μm`           : Gaussian smoothing radius in µm (0 = no smoothing)
- `colormap`       : any Makie colormap name (default `:plasma`)
- `alpha`          : surface opacity in (0, 1]. `1.0` = fully solid, but the
                     nested outer surfaces then hide the inner ones; values
                     around 0.6–0.8 look solid yet still reveal the inner shells.
- `show_osteocytes`      : scatter osteocyte centroids (default `false`)
- `osteocytes`           : vector of `(x, y, z)` centroids **in µm**
                           (the `Ocy_pos` returned by `load_osteocytes`)
- `osteocyte_color`      : marker colour (default `:red`)
- `osteocyte_markersize` : marker size (default `12`)
"""
function plot_3d_surfaces!(ax, ϕ::AbstractArray{<:Real,4},
                            tvals::AbstractVector;
                            dx::Real=0.379, dy::Real=0.379, dz::Real=0.358,
                            σ_μm::Real=1.0, colormap=:plasma, alpha::Real=0.7,
                            show_osteocytes::Bool=false,
                            osteocytes=nothing,
                            osteocyte_color=:red,
                            osteocyte_markersize::Real=12)
    lo, hi   = extrema(tvals)
    H, W, D  = size(ϕ, 1), size(ϕ, 2), size(ϕ, 3)
    # Physical-µm coordinate of each sample along each axis (index i → i·spacing),
    # so the surface is drawn in µm rather than Meshing's normalised cube.
    X = (1:H) .* dx
    Y = (1:W) .* dy
    Z = (1:D) .* dz
    plt = nothing

    for ti in eachindex(tvals)
        t     = tvals[ti]
        field = σ_μm > 0 ? smooth_levelset(ϕ[:,:,:,ti]; dx, dy, dz, σ_μm) :
                           Float32.(ϕ[:,:,:,ti])

        verts, faces = isosurface(field, MarchingCubes(), X, Y, Z)   # zero level set, in µm
        isempty(verts) && continue

        pts  = [Point3f(v...)                  for v in verts]
        tris = [GeometryBasics.TriangleFace{Int}(f...) for f in faces]

        # Colour the whole surface by its formation time t (mapped through the
        # colormap over the full tvals range). Per-vertex constant colour keeps
        # the plot Colorbar-compatible.
        plt = mesh!(ax, GeometryBasics.Mesh(pts, tris);
                    color       = fill(Float32(t), length(pts)),
                    colormap    = colormap,
                    colorrange  = (lo, hi),
                    alpha       = alpha,
                    transparency = alpha < 1)
    end

    # Optional osteocyte-centroid overlay, in the same physical µm coordinates.
    if show_osteocytes
        if osteocytes === nothing
            @warn "show_osteocytes=true but `osteocytes` was not provided; skipping scatter."
        else
            ocy_pts = [Point3f(p[1], p[2], p[3]) for p in osteocytes]
            scatter!(ax, ocy_pts; color=osteocyte_color, markersize=osteocyte_markersize)
        end
    end

    return plt
end

"""
    plot_3d_contours_w_intersections!(ax, ϕ, Δz, tvals, center_top, center_bot,
                                      intersecting_points_per_contour)

Like [`plot_3d_contours!`](@ref), but also marks the top/bottom centres and the
cutting-plane intersection points from
[`Geometry.compute_planes_and_intersections`](@ref), joining each matched pair
across consecutive contours with a line. Mutates `ax`.
"""
function plot_3d_contours_w_intersections!(ax, ϕ, Δz, tvals, center_top, center_bot, intersecting_points_per_contour)
    H,W,_,_ = size(ϕ)
    x = collect(1:H); y = collect(1:W)
    for (ti,t) in enumerate(tvals)
        ϕ_bottom = ϕ[:,:,1,ti]
        cset_bot = CTR.contours(x,y,ϕ_bottom, [0])
        line_bot = first(CTR.lines(first(CTR.levels(cset_bot))))
        x_bot, y_bot = CTR.coordinates(line_bot)

        ϕ_top = ϕ[:,:,2,ti]
        cset_top = CTR.contours(x,y,ϕ_top, [0])
        line_top = first(CTR.lines(first(CTR.levels(cset_top))))
        x_top, y_top = CTR.coordinates(line_top)

        lines!(ax, x_bot, y_bot, zeros(size(x_bot)), linewidth=3, color=:red)
        lines!(ax, x_top, y_top, Δz.*ones(size(x_top)), linewidth=3, color=:blue)
    end
    scatter!(ax, center_bot[1], center_bot[2], 0, markersize=30)
    scatter!(ax, center_top[1], center_top[2], Δz, markersize=30)
    # plotting intersections
    for ii in axes(intersecting_points_per_contour,1)[1:end-1]
        for jj in axes(intersecting_points_per_contour[1],1)
            scatter!(ax,intersecting_points_per_contour[ii][jj][1][1], intersecting_points_per_contour[ii][jj][1][2], intersecting_points_per_contour[ii][jj][1][3], markersize=25, color=:green)
            scatter!(ax,intersecting_points_per_contour[ii+1][jj][2][1], intersecting_points_per_contour[ii+1][jj][2][2], intersecting_points_per_contour[ii+1][jj][2][3], markersize=25, color=:cyan)
            lines!(ax, [intersecting_points_per_contour[ii][jj][1][1], intersecting_points_per_contour[ii+1][jj][2][1]], [intersecting_points_per_contour[ii][jj][1][2], intersecting_points_per_contour[ii+1][jj][2][2]], [intersecting_points_per_contour[ii][jj][1][3], intersecting_points_per_contour[ii+1][jj][2][3]], linewidth=3, color = :orange)
        end
    end
end

"""
    plot_example_slices!(axes, Tdelay_proj_points, line_∇)

Plot the projected formation-front segments on a vector of 2-D axes — one axis
per angle. Each segment is coloured by its gradient `line_∇` (jet colormap,
range ±10). Mutates the supplied `axes`.
"""
function plot_example_slices!(axes, Tdelay_proj_points, line_∇)
    for ang in eachindex(axes)
        ax = axes[ang]; # 1 axes is 1 angle
        for cont in eachindex(Tdelay_proj_points) 
            x = [Tdelay_proj_points[cont][ang][1][1], Tdelay_proj_points[cont][ang][2][1]]
            y = [Tdelay_proj_points[cont][ang][1][2], Tdelay_proj_points[cont][ang][2][2]]
            lines!(ax, x, y, linewidth=3,color=line_∇[cont][ang].*ones(size(x)), colormap=:jet, colorrange=(-10,10))
        end
    end
end

"""
    plot_α_β!(ax, α, β, tvals)

Plot the formation-front inclination angle `α` against the angular position `β`
(converted to degrees) for every other contour in `α`, labelling each line with
its formation time from `tvals`. Mutates `ax`.
"""
function plot_α_β!(ax, α, β, tvals)
    for jj in 1:2:size(α,1)
        scatter!(ax, rad2deg.(β), α[jj,:], markersize = 15)
        lines!(ax, rad2deg.(β), α[jj,:], linewidth = 3, label = L"T = %$(tvals[jj])")
    end
end

"""
    plot_osteocyte_distribution(t_form_all, κ_at_all, mean_κ_all, labels;
                                relative=true, bins=15, markersize=10) -> Figure

Joint figure for analysing **where, in formation time, and at what curvature**
osteocytes get embedded — with marginal histograms so two questions can be read
off a single plot:

- **Top histogram (formation time):** is there a *preferred time* for osteocytes
  to form? Peaks indicate over-represented formation times.
- **Right histogram (curvature):** do osteocytes favour *convex* or *concave*
  parts of the 2-D formation front? A dashed reference line marks the neutral
  value; mass to one side indicates a preference.

The central panel is the joint scatter (coloured by dataset), revealing whether
the curvature preference itself changes over formation time.

Arguments
- `t_form_all`  : vector (one per dataset) of osteocyte formation-time vectors
- `κ_at_all`    : matching curvature-at-osteocyte vectors  (µm⁻¹)
- `mean_κ_all`  : matching contour-mean curvature vectors  (µm⁻¹)
- `labels`      : dataset labels

Keyword arguments
- `relative` : if `true` (default) plot Δκ = κ_at − mean_κ (curvature *relative to
               the contour mean*; the line sits at 0 = the average front
               curvature). If `false`, plot the absolute κ_at, whose **sign**
               distinguishes convex from concave front (line at κ = 0).
- `bins`     : number of histogram bins
- `markersize`: scatter marker size

Backend-agnostic: activate GLMakie or CairoMakie before calling.
"""
function plot_osteocyte_distribution(t_form_all, κ_at_all, mean_κ_all, labels;
                                     relative::Bool = true, bins::Integer = 15, markersize = 10)
    @assert length(t_form_all) == length(κ_at_all) == length(mean_κ_all) == length(labels)

    yvals = relative ? [κ_at_all[i] .- mean_κ_all[i] for i in eachindex(κ_at_all)] :
                       [collect(κ_at_all[i])          for i in eachindex(κ_at_all)]
    ylab  = _kappa_label(relative)
    title = relative ? L"\text{Osteocyte distribution: formation time vs curvature relative to the mean}" :
                       L"\text{Osteocyte distribution: formation time vs curvature (sign = convex/concave)}"

    tpool = reduce(vcat, t_form_all)
    ypool = reduce(vcat, yvals)

    fig = Figure(size = (1000, 850))
    ax  = Axis(fig[2, 1]; _LTICKS..., xlabel = L"\text{formation time}\ t", ylabel = ylab)
    axt = Axis(fig[1, 1]; _LTICKS..., ylabel = L"\text{count}")   # top marginal — formation time
    axk = Axis(fig[2, 2]; _LTICKS..., xlabel = L"\text{count}")   # right marginal — curvature
    linkxaxes!(ax, axt)
    linkyaxes!(ax, axk)
    hidexdecorations!(axt; grid = false)
    hideydecorations!(axk; grid = false)
    rowsize!(fig.layout, 1, Relative(0.18))
    colsize!(fig.layout, 2, Relative(0.18))

    for i in eachindex(labels)
        scatter!(ax, t_form_all[i], yvals[i]; markersize = markersize, label = _tex(labels[i]))
    end
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash)

    hist!(axt, tpool; bins = bins, color = (:dodgerblue, 0.6))
    hist!(axk, ypool; bins = bins, direction = :x, color = (:dodgerblue, 0.6))
    hlines!(axk, [0.0]; color = :gray, linestyle = :dash)

    Legend(fig[1, 2], ax; framevisible = false)
    Label(fig[0, :], title; fontsize = 20, font = :bold)
    return fig
end

# Helper: stack the per-dataset curvature vectors into one pooled vector, either
# Δκ = κ_at − mean_κ (relative) or the absolute κ_at.
_pool_curvature(κ_at_all, mean_κ_all, relative) =
    relative ? reduce(vcat, [κ_at_all[i] .- mean_κ_all[i] for i in eachindex(κ_at_all)]) :
               reduce(vcat, [collect(κ_at_all[i])          for i in eachindex(κ_at_all)])

"""
    pooled_kde(data; npoints=200, bandwidth=nothing) -> (x, density, bandwidth)

Compute the **same kernel-density estimate that `density!` draws** for the pooled
data — i.e. the bold "all pooled" curve in the density figures — and return it as
numbers you can export, integrate or overlay.

`data` is either a flat vector or a vector-of-vectors (it is pooled with
`reduce(vcat, …)`, matching how the plots pool all datasets). The KDE uses a
**Normal (Gaussian) kernel** and, by default, **Silverman's rule-of-thumb
bandwidth** `0.9·min(σ, IQR/1.34)·n^(-1/5)` (`KernelDensity.default_bandwidth`),
exactly as Makie's `density!`. Pass `bandwidth` to override it, and `npoints` to
set the number of evaluation points (Makie's default is 200).

Returns a named tuple `(x, density, bandwidth)`: `x` the evaluation grid,
`density` the estimated pdf there (∫ density dx ≈ 1), and the bandwidth used.

```julia
k = pooled_kde(t_form_all)                 # the formation-time KDE
CSV.write("kde.csv", DataFrame(x=k.x, density=k.density))   # export it
```
"""
function pooled_kde(data; npoints::Integer = 200, bandwidth = nothing)
    pooled = data isa AbstractVector{<:Real} ? float.(collect(data)) : reduce(vcat, data)
    h = bandwidth === nothing ? KernelDensity.default_bandwidth(pooled) : float(bandwidth)
    U = KernelDensity.kde(pooled; npoints = npoints, bandwidth = h)
    return (x = collect(U.x), density = collect(U.density), bandwidth = h)
end

"""
    plot_formation_time_density(t_form_all, labels) -> Figure

Kernel-density estimate (smooth alternative to a histogram) of the osteocyte
formation-time distribution — one filled curve per dataset plus a bold pooled
curve. Peaks indicate over-represented formation times.

Note: raw density is biased by how much bone surface forms per unit time
(geometry), so read it as "where osteocytes land", not yet a normalised
preference. Backend-agnostic.
"""
function plot_formation_time_density(t_form_all, labels)
    cols = Makie.wong_colors()
    fig = Figure(size = (900, 550))
    ax  = Axis(fig[1, 1]; _LTICKS..., xlabel = L"\text{formation time}\ t", ylabel = L"\text{density}",
               title = L"\text{Formation time distribution (KDE)}")
    for i in eachindex(labels)
        c = cols[mod1(i, length(cols))]
        density!(ax, t_form_all[i]; color = (c, 0.25), strokecolor = c, strokewidth = 2,
                 label = _tex(labels[i]))
    end
    density!(ax, reduce(vcat, t_form_all); color = (:black, 0.0), strokecolor = :black,
             strokewidth = 3, label = _tex("all pooled"))
    xlims!(ax, 0, 1)
    axislegend(ax; position = :rt, framevisible = false)
    return fig
end

"""
    plot_curvature_density(κ_at_all, mean_κ_all, labels; relative=true) -> Figure

Kernel-density estimate of the osteocyte **curvature** distribution — one filled
curve per dataset plus a bold pooled curve (the smooth analogue of the curvature
histogram). A dashed reference line marks the neutral value; mass to one side of
it indicates a curvature preference.

- `relative=true` (default): plots Δκ = κ_at − mean_κ (curvature *relative to the
  contour mean*; the line at 0 is the average front curvature).
- `relative=false`: plots the absolute κ_at, whose **sign** separates convex
  (> 0) from concave (< 0) front (line at κ = 0).

Backend-agnostic — activate GLMakie or CairoMakie first.
"""
function plot_curvature_density(κ_at_all, mean_κ_all, labels; relative::Bool = true)
    @assert length(κ_at_all) == length(mean_κ_all) == length(labels)
    cols  = Makie.wong_colors()
    yvals = relative ? [κ_at_all[i] .- mean_κ_all[i] for i in eachindex(κ_at_all)] :
                       [collect(κ_at_all[i])          for i in eachindex(κ_at_all)]
    xlab  = _kappa_label(relative)

    fig = Figure(size = (900, 550))
    ax  = Axis(fig[1, 1]; _LTICKS..., xlabel = xlab, ylabel = L"\text{density}",
               title = L"\text{Curvature distribution (KDE)}")
    for i in eachindex(labels)
        c = cols[mod1(i, length(cols))]
        density!(ax, yvals[i]; color = (c, 0.25), strokecolor = c, strokewidth = 2,
                 label = _tex(labels[i]))
    end
    density!(ax, reduce(vcat, yvals); color = (:black, 0.0), strokecolor = :black,
             strokewidth = 3, label = _tex("all pooled"))
    vlines!(ax, [0.0]; color = :gray, linestyle = :dash)   # 0 = mean (relative) or convex/concave boundary
    axislegend(ax; position = :rt, framevisible = false)
    return fig
end

"""
    plot_curvature_by_time_bracket(t_form_all, κ_at_all, mean_κ_all; relative=true, nbrackets=4) -> Figure

Split osteocytes into `nbrackets` equal formation-time windows and show the
curvature distribution in each as a **violin + boxplot** — the clearest way to
see whether the curvature preference changes over formation time (e.g. early vs
late osteocytes favouring different curvatures). Dashed line = curvature
reference. Backend-agnostic.
"""
function plot_curvature_by_time_bracket(t_form_all, κ_at_all, mean_κ_all; relative::Bool = true, nbrackets::Integer = 4)
    tpool = reduce(vcat, t_form_all)
    ypool = _pool_curvature(κ_at_all, mean_κ_all, relative)
    ylab  = _kappa_label(relative)
    edges = range(0, 1; length = nbrackets + 1)
    cat   = clamp.(floor.(Int, tpool .* nbrackets) .+ 1, 1, nbrackets)   # bracket index per osteocyte
    ticks = [L"[%$(round(edges[b]; digits=2)),\ %$(round(edges[b+1]; digits=2))]" for b in 1:nbrackets]

    fig = Figure(size = (950, 600))
    ax  = Axis(fig[1, 1]; ytickformat = _latex_ticks, xlabel = L"\text{formation time bracket}", ylabel = ylab,
               xticks = (1:nbrackets, ticks), title = L"\text{Curvature by formation time bracket}")
    violin!(ax, cat, ypool; color = (:dodgerblue, 0.35), width = 0.85)
    boxplot!(ax, cat, ypool; width = 0.25, color = :dodgerblue, strokecolor = :black, markersize = 4)
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash)
    return fig
end

"""
    plot_formation_time_ecdf(t_form_all, labels) -> Figure

Empirical cumulative distribution of formation time, one curve per dataset plus a
pooled curve, against the **uniform reference** (the diagonal). Deviations above
the diagonal mean osteocytes form earlier than uniform, below means later — a
clean way to judge a time preference. Backend-agnostic.
"""
function plot_formation_time_ecdf(t_form_all, labels)
    fig = Figure(size = (800, 600))
    ax  = Axis(fig[1, 1]; _LTICKS..., xlabel = L"\text{formation time}\ t", ylabel = L"\text{cumulative fraction}",
               title = L"\text{Formation time ECDF}")
    lines!(ax, [0, 1], [0, 1]; color = :gray, linestyle = :dash, label = _tex("uniform"))
    for i in eachindex(labels)
        ecdfplot!(ax, t_form_all[i]; label = _tex(labels[i]))
    end
    ecdfplot!(ax, reduce(vcat, t_form_all); color = :black, linewidth = 3, label = _tex("all datasets"))
    xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    axislegend(ax; position = :lt, framevisible = false)
    return fig
end

"""
    plot_curvature_by_scale(k_values, κ_at_per_k, mean_κ_per_k; relative=true) -> Figure

Compare the osteocyte curvature distribution measured at several `k_scale_um`
values **on one dataset** — one violin + boxplot per scale, so you can see how
the magnitude and spread of curvature change with the measurement scale.

- `relative=true` (default): Δκ = κ − mean_κ (dashed line at 0 = the contour mean)
- `relative=false`: absolute κ (sign separates convex > 0 from concave < 0)

`κ_at_per_k` / `mean_κ_per_k` are vectors-of-vectors, one entry per `k_values`.
Backend-agnostic.
"""
function plot_curvature_by_scale(k_values, κ_at_per_k, mean_κ_per_k; relative::Bool = true)
    @assert length(k_values) == length(κ_at_per_k) == length(mean_κ_per_k)
    yvals = relative ? [κ_at_per_k[i] .- mean_κ_per_k[i] for i in eachindex(κ_at_per_k)] :
                       [collect(κ_at_per_k[i])           for i in eachindex(κ_at_per_k)]
    ylab  = _kappa_label(relative)

    cat = Int[]; vals = Float64[]
    for i in eachindex(k_values)
        append!(cat,  fill(i, length(yvals[i])))
        append!(vals, yvals[i])
    end
    ticks = [isinteger(k) ? L"%$(Int(k))" : L"%$(k)" for k in k_values]

    fig = Figure(size = (950, 600))
    ax  = Axis(fig[1, 1]; ytickformat = _latex_ticks, xlabel = L"k_{\text{scale}}\ [\mathrm{µm}]", ylabel = ylab,
               xticks = (1:length(k_values), ticks), title = L"\text{Curvature vs measurement scale}")
    violin!(ax, cat, vals; color = (:dodgerblue, 0.35), width = 0.8)
    boxplot!(ax, cat, vals; width = 0.25, color = :dodgerblue, strokecolor = :black, markersize = 4)
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash)
    return fig
end

# z half-width of the smoothing kernel (slab radius) for a given σ and spacings.
_kz_half(σ_μm, dx, dy, dz) = (length(KernelFactors.gaussian((σ_μm/dx, σ_μm/dy, σ_μm/dz))[3]) - 1) ÷ 2

"""
    plot_osteocyte_contour(idx, t_form_ordered, outer_dt_S, inner_dt_S,
                           Ocy_pos_voxel_ordered, dx, dy, dz, σ_μm; k_scale_um=15.0) -> Figure

Visualise the exact 2-D contour on which osteocyte `idx`'s curvature is measured.
`idx` indexes the **formation-time-ordered** vectors (same order as
[`Analysis.compute_curvature_near_osteocyte`](@ref)), so it matches that
function's outputs.

The figure (in physical µm, equal aspect) shows:
- the zero-level contour the curvature is computed along (black),
- a marker at the osteocyte's position (red),
- a light-grey reference circle of the **same curvature as the contour mean**
  (radius = 1/|mean κ|), centred on the contour centroid, drawn behind.

The title reports the mean available curvature for that contour.
"""
function plot_osteocyte_contour(idx, t_form_ordered, outer_dt_S, inner_dt_S,
                                Ocy_pos_voxel_ordered, dx, dy, dz, σ_μm; k_scale_um = 15.0)
    t       = t_form_ordered[idx]
    z_layer = Ocy_pos_voxel_ordered[idx][3]
    ox      = Ocy_pos_voxel_ordered[idx][1] * dx
    oy      = Ocy_pos_voxel_ordered[idx][2] * dy

    # Rebuild the smoothed z-slab exactly as compute_curvature_near_osteocyte does.
    kz = _kz_half(σ_μm, dx, dy, dz)
    Z  = size(outer_dt_S, 3)
    z0 = max(1, z_layer - kz); z1 = min(Z, z_layer + kz)
    oa = @view outer_dt_S[:, :, z0:z1]
    ia = @view inner_dt_S[:, :, z0:z1]
    ϕ_smooth = smooth_ϕ((@. Float32((1 - t) * oa - t * ia)); dx, dy, dz, σ_μm)

    X, Y = compute_zero_contour_xy_coords(ϕ_smooth, z_layer - z0 + 1, 1)
    Xµ, Yµ = X .* dx, Y .* dy
    κ = compute_2D_curvature(copy(Xµ), copy(Yµ); arclen = k_scale_um)
    mean_κ = sum(κ) / length(κ)
    n = length(κ)
    j = argmin(@. (Xµ[1:n] - ox)^2 + (Yµ[1:n] - oy)^2)   # nearest contour point to the osteocyte
    κ_at = κ[j]

    cx, cy = sum(Xµ) / length(Xµ), sum(Yµ) / length(Yµ)   # contour centroid

    fig = Figure(size = (1150, 820))
    ax  = Axis(fig[1, 1]; _LTICKS..., xlabel = L"x\ [\mathrm{µm}]", ylabel = L"y\ [\mathrm{µm}]", aspect = DataAspect(),
               title = L"\text{Osteocyte } %$idx:\ \overline{\kappa} = %$(round(mean_κ; sigdigits=3))\ \mathrm{µm}^{-1}")
    if mean_κ != 0                                          # grey reference circle of the mean curvature
        R = 1 / abs(mean_κ)
        θ = range(0, 2π; length = 200)
        lines!(ax, cx .+ R .* cos.(θ), cy .+ R .* sin.(θ);
               color = (:gray, 0.6), linewidth = 4, label = L"\text{circle of mean }\kappa\ (R = %$(round(R; sigdigits=3))\ \mathrm{µm})")
    end
    lines!(ax, Xµ, Yµ; color = :black, linewidth = 3, label = L"\text{contour}")
    scatter!(ax, [ox], [oy]; color = :red, markersize = 18,
             label = L"\text{osteocyte}\ (\kappa = %$(round(κ_at; sigdigits=3)))")
    Legend(fig[1, 2], ax; framevisible = false)   # legend in its own column → never over the contour
    return fig
end

"""
    plot_smoothing_effect(t, outer_dt_S, inner_dt_S, z_layer, dx, dy, dz, σ_μm;
                          inset_size_um=30.0, inset_center=nothing) -> Figure

Show how Gaussian smoothing changes the zero contour (which the curvature is
measured along) on slice `z_layer` at formation time `t`. Two side-by-side
grayscale heatmaps of the level-set field ϕ — left: raw ϕ, right: smoothed ϕ
(σ = `σ_μm` µm, the same 3-D slab smoothing used in the curvature step) — each
with its zero contour drawn in red. Axes are in physical µm with equal aspect.

Each panel also gets a **zoom inset in the bottom-right corner** showing the same
`inset_size_um × inset_size_um` µm window around the contour (the box is marked
on the main panel). The raw panel's inset reveals the jagged voxel-staircase
contour; the smoothed panel's inset shows it rounded off. `inset_center` is an
`(x, y)` µm point to zoom on (default: the rightmost point of the contour).
"""
function plot_smoothing_effect(t, outer_dt_S, inner_dt_S, z_layer, dx, dy, dz, σ_μm;
                               inset_size_um::Real = 30.0, inset_center = nothing)
    kz = _kz_half(σ_μm, dx, dy, dz)
    Z  = size(outer_dt_S, 3)
    z0 = max(1, z_layer - kz); z1 = min(Z, z_layer + kz)
    oa = @view outer_dt_S[:, :, z0:z1]
    ia = @view inner_dt_S[:, :, z0:z1]
    ϕ_raw_slab = @. Float32((1 - t) * oa - t * ia)
    ϕ_sm_slab  = smooth_ϕ(ϕ_raw_slab; dx, dy, dz, σ_μm)
    zc = z_layer - z0 + 1
    ϕ_raw = ϕ_raw_slab[:, :, zc]
    ϕ_sm  = ϕ_sm_slab[:, :, zc]

    H, W = size(ϕ_raw)
    xs = (1:H) .* dx; ys = (1:W) .* dy

    # zoom window (shared by both panels so raw vs smoothed are directly comparable):
    # centre on `inset_center`, else the rightmost point of the raw contour.
    Xr, Yr = compute_zero_contour_xy_coords(ϕ_raw_slab, zc, 1)
    Xrµ = Xr .* dx; Yrµ = Yr .* dy
    cxz, cyz = inset_center === nothing ? (let j = argmax(Xrµ); (Xrµ[j], Yrµ[j]) end) :
                                          (Float64(inset_center[1]), Float64(inset_center[2]))
    half = inset_size_um / 2
    xlo, xhi, ylo, yhi = cxz - half, cxz + half, cyz - half, cyz + half
    zoomclr = :dodgerblue

    # window indices for the zoom-panel local colour range
    ixlo, ixhi = clamp(round(Int, xlo / dx), 1, H), clamp(round(Int, xhi / dx), 1, H)
    iylo, iyhi = clamp(round(Int, ylo / dy), 1, W), clamp(round(Int, yhi / dy), 1, W)

    # LaTeX labels (shared)
    xlab    = L"x\ [\mathrm{µm}]"
    ylab    = L"y\ [\mathrm{µm}]"
    zoomlab = L"%$(Int(round(inset_size_um)))\,\mathrm{µm\ zoom}"

    cr = extrema(vcat(vec(ϕ_raw), vec(ϕ_sm)))     # shared colour scale → one meaningful colorbar
    fig = Figure(size = (1250, 1230))
    #Label(fig[0, :], L"t = %$(round(t; sigdigits=3)),\quad z = %$(z_layer)";
          #fontsize = 32, font = :bold)
    hm = nothing
    # rows: raw (top), smoothed (bottom).  cols: full contour (left), zoom (right).
    for (row, (ϕ, full, lab)) in enumerate(((ϕ_raw_slab, ϕ_raw, L"\text{raw }\phi"),
                                            (ϕ_sm_slab,  ϕ_sm,  L"\text{smoothed }\phi\ (\sigma = %$(σ_μm))")))
        X, Y = compute_zero_contour_xy_coords(ϕ, zc, 1)

        # left: full contour (drop the x-label on the top row to avoid crowding the row below)
        ax = Axis(fig[row, 1]; _LTICKS..., xlabel = row == 1 ? "" : xlab, ylabel = ylab,
                  aspect = DataAspect(), title = lab)
        hm = heatmap!(ax, xs, ys, full; colormap = :grays, colorrange = cr)
        lines!(ax, X .* dx, Y .* dy; color = :red, linewidth = 2)
        lines!(ax, [xlo, xhi, xhi, xlo, xlo], [ylo, ylo, yhi, yhi, ylo];   # mark zoom region
               color = zoomclr, linewidth = 2)

        # right: same contour zoomed to the window, with a local colour range so the
        # (nearly flat) ϕ there isn't washed out.
        cr_loc = extrema(@view full[ixlo:ixhi, iylo:iyhi])
        axz = Axis(fig[row, 2]; _LTICKS..., xlabel = row == 1 ? "" : xlab, ylabel = ylab, aspect = DataAspect(),
                   title = zoomlab, titlecolor = zoomclr, spinewidth = 2,
                   leftspinecolor = zoomclr, rightspinecolor = zoomclr,
                   topspinecolor = zoomclr, bottomspinecolor = zoomclr)
        heatmap!(axz, xs, ys, full; colormap = :grays, colorrange = cr_loc)
        lines!(axz, X .* dx, Y .* dy; color = :red, linewidth = 3)
        limits!(axz, xlo, xhi, ylo, yhi)
    end
    Colorbar(fig[1:2, 3], hm; label = L"\phi\ [\mathrm{µm}]", tickformat = _latex_ticks)   # far right, spanning both rows
    # equal square panels (DataAspect axes otherwise auto-size their columns unequally)
    colsize!(fig.layout, 1, Aspect(1, 1.0))
    colsize!(fig.layout, 2, Aspect(1, 1.0))
    return fig
end

end # end of module