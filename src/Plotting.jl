"""
    Plotting

GLMakie helpers for visualising osteon reconstructions.

Functions here draw onto a caller-supplied axis (`Axis`/`Axis3`) and mutate it
in place (hence the trailing `!`). They cover the per-slice zero-level contours
of the level-set field, the volumetric zero isosurface, the cutting-plane
intersection points, and the formation-front inclination plots. `smooth_levelset`
applies the same anisotropic Gaussian blur used elsewhere so that plotted
surfaces match the analysed ones.
"""
module Plotting

using GLMakie
using ImageFiltering
using Meshing                 # marching-cubes isosurface extraction
import GeometryBasics         # Mesh / TriangleFace (qualified to avoid clashing with GLMakie exports)
import Contour as CTR

export plot_3d_contours!, plot_3d_contours_w_intersections!, plot_example_slices!, plot_α_β!, plot_3d_surfaces!

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
dimensions differ (e.g. dx = dy = 0.379 µm, dz = 0.4 µm).

A σ of ~1 µm removes the staircase while preserving the macro-scale surface
shape; increase to ~2 µm for heavier smoothing.

Uses a **separable** kernel (`KernelFactors.gaussian`), applied as three 1-D
passes — equivalent to a dense 3-D Gaussian but with far lower memory use.
"""
function smooth_levelset(ϕ3d::AbstractArray{<:Real,3};
                          dx::Real=0.379, dy::Real=0.379, dz::Real=0.4,
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
                            dx::Real=0.379, dy::Real=0.379, dz::Real=0.4,
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

end # end of module