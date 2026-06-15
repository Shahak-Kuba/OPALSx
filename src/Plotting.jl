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

export plot_3d_contours!, plot_3d_contours_w_intersections!, plot_example_slices!, plot_α_β!,
       plot_3d_surfaces!, plot_osteocyte_distribution,
       plot_formation_time_density, plot_curvature_density, plot_tform_curvature_hexbin,
       plot_curvature_by_time_bracket, plot_curvature_by_scale, plot_formation_time_ecdf, pooled_kde

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
        lines!(ax, rad2deg.(β), α[jj,:], linewidth = 3, label = "T = $(tvals[jj])")
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
    ylab  = relative ? "κ − mean κ   [µm⁻¹]" : "κ at osteocyte   [µm⁻¹]"
    title = relative ? "Osteocyte distribution: formation time vs curvature relative to the mean" :
                       "Osteocyte distribution: formation time vs curvature (sign = convex/concave)"

    tpool = reduce(vcat, t_form_all)
    ypool = reduce(vcat, yvals)

    fig = Figure(size = (1000, 850))
    ax  = Axis(fig[2, 1]; xlabel = "formation time  t", ylabel = ylab)
    axt = Axis(fig[1, 1]; ylabel = "count")                  # top marginal — formation time
    axk = Axis(fig[2, 2]; xlabel = "count")                  # right marginal — curvature
    linkxaxes!(ax, axt)
    linkyaxes!(ax, axk)
    hidexdecorations!(axt; grid = false)
    hideydecorations!(axk; grid = false)
    rowsize!(fig.layout, 1, Relative(0.18))
    colsize!(fig.layout, 2, Relative(0.18))

    for i in eachindex(labels)
        scatter!(ax, t_form_all[i], yvals[i]; markersize = markersize, label = string(labels[i]))
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
    ax  = Axis(fig[1, 1]; xlabel = "formation time  t", ylabel = "density",
               title = "Formation-time distribution (KDE)")
    for i in eachindex(labels)
        c = cols[mod1(i, length(cols))]
        density!(ax, t_form_all[i]; color = (c, 0.25), strokecolor = c, strokewidth = 2,
                 label = string(labels[i]))
    end
    density!(ax, reduce(vcat, t_form_all); color = (:black, 0.0), strokecolor = :black,
             strokewidth = 3, label = "all pooled")
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
    xlab  = relative ? "κ − mean κ   [µm⁻¹]" : "κ at osteocyte   [µm⁻¹]"

    fig = Figure(size = (900, 550))
    ax  = Axis(fig[1, 1]; xlabel = xlab, ylabel = "density",
               title = "Curvature distribution (KDE)")
    for i in eachindex(labels)
        c = cols[mod1(i, length(cols))]
        density!(ax, yvals[i]; color = (c, 0.25), strokecolor = c, strokewidth = 2,
                 label = string(labels[i]))
    end
    density!(ax, reduce(vcat, yvals); color = (:black, 0.0), strokecolor = :black,
             strokewidth = 3, label = "all pooled")
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
    ylab  = relative ? "κ − mean κ   [µm⁻¹]" : "κ at osteocyte   [µm⁻¹]"
    edges = range(0, 1; length = nbrackets + 1)
    cat   = clamp.(floor.(Int, tpool .* nbrackets) .+ 1, 1, nbrackets)   # bracket index per osteocyte
    ticks = ["[$(round(edges[b]; digits=2)), $(round(edges[b+1]; digits=2))]" for b in 1:nbrackets]

    fig = Figure(size = (950, 600))
    ax  = Axis(fig[1, 1]; xlabel = "formation-time bracket", ylabel = ylab,
               xticks = (1:nbrackets, ticks), title = "Curvature by formation-time bracket")
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
    ax  = Axis(fig[1, 1]; xlabel = "formation time  t", ylabel = "cumulative fraction",
               title = "Formation-time ECDF")
    lines!(ax, [0, 1], [0, 1]; color = :gray, linestyle = :dash, label = "uniform")
    for i in eachindex(labels)
        ecdfplot!(ax, t_form_all[i]; label = string(labels[i]))
    end
    ecdfplot!(ax, reduce(vcat, t_form_all); color = :black, linewidth = 3, label = "all datasets")
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
    ylab  = relative ? "κ − mean κ   [µm⁻¹]" : "κ at osteocyte   [µm⁻¹]"

    cat = Int[]; vals = Float64[]
    for i in eachindex(k_values)
        append!(cat,  fill(i, length(yvals[i])))
        append!(vals, yvals[i])
    end
    ticks = [isinteger(k) ? string(Int(k)) : string(k) for k in k_values]

    fig = Figure(size = (950, 600))
    ax  = Axis(fig[1, 1]; xlabel = "k_scale_um   [µm]", ylabel = ylab,
               xticks = (1:length(k_values), ticks), title = "Curvature vs measurement scale")
    violin!(ax, cat, vals; color = (:dodgerblue, 0.35), width = 0.8)
    boxplot!(ax, cat, vals; width = 0.25, color = :dodgerblue, strokecolor = :black, markersize = 4)
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash)
    return fig
end

end # end of module