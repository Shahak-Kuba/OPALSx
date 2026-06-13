"""
    LevelSet

Signed distance transforms and the time-dependent level-set field `ϕ`.

Given the Boolean `outer`/`inner` masks from [`Imaging.build_outer_inner`](@ref),
this module forms the signed Euclidean distance transform (EDT) of each region
and combines them into

    ϕ(t) = (1 − t)·EDT_S(outer) − t·EDT_S(inner),   t ∈ [0, 1].

The zero level-set of `ϕ(t)` is the modelled bone-formation front: at `t = 0`
it coincides with the cement line (outer boundary) and at `t = 1` with the
Haversian-canal wall (inner boundary), so `t` acts as a normalised formation
time that increases inward as the osteon infills. The module provides 2-D
(slice-wise) and 3-D builders, isotropic ([`edt_S`](@ref)) and anisotropic
([`edt_S_aniso`](@ref)) signed distance transforms — the anisotropic one is a
native, exact, weighted Felzenszwalb–Huttenlocher transform (no Python) — an
osteocyte formation-time estimator, and Gaussian smoothing of `ϕ` prior to
curvature extraction.
"""
module LevelSet

export edt, edt_S, edt_aniso, edt_S_aniso, compute_EDT_S,
       ϕ_func, compute_ϕ_at_t, compute_ϕ_stack, compute_ϕ_at_t_3D, compute_ϕ_stack_3D,
       estimate_Ocy_formation_time, smooth_ϕ

using DistanceTransforms
using ImageFiltering

"""
    edt(mask) -> Array{Float32}

Unsigned Euclidean distance transform of a Boolean `mask`: each cell holds the
Euclidean distance (in voxel units) to the nearest `true` cell. Computed via
`DistanceTransforms.transform` on the squared-distance indicator, then
square-rooted.
"""
function edt(mask::BitArray)
    return sqrt.(DistanceTransforms.transform(boolean_indicator(mask)))
end

"""
    edt_S(mask) -> Array{Float32}

Signed Euclidean distance transform of a Boolean `mask`: positive outside the
`true` region and negative inside, defined as `edt(mask) - edt(.!mask)`. This is
the signed distance field whose zero crossing is the region boundary.
"""
function edt_S(mask::BitArray)
    return edt(mask) .- edt(.!mask)
end


"""
    ϕ_func(t, S_outer, S_inner)

Linear interpolation between the outer and inner signed distance fields:
`ϕ = (1 - t)·S_outer - t·S_inner`. Returns the level-set field at formation
time `t ∈ [0, 1]`; the zero level-set sweeps from the outer boundary / cement
line (`t=0`) to the inner boundary / canal wall (`t=1`).
"""
ϕ_func = (t,S_DTʰ,S_DTᶜ) -> (1-t) .* S_DTʰ - (t) .* S_DTᶜ

# --------------------------- 2D Implementation of level set function computation ------------------------------
"""
    compute_ϕ_at_t(outer, inner, tval) -> Array{Float32,3}

Slice-wise (2-D) build of the level-set field `ϕ` at a single formation time
`tval`. The signed EDT of `outer` and `inner` is computed **per z-slice**
(`edt_S` applied to each `(:, :, z)` plane independently), then combined with
[`ϕ_func`](@ref). Use this when distances should not couple across slices; see
[`compute_ϕ_at_t_3D`](@ref) for the fully 3-D variant.

Arguments
- `outer`, `inner` : Boolean volumes `(H, W, Z)` from [`Imaging.build_outer_inner`](@ref)
- `tval`           : formation time in `[0, 1]`
"""
function compute_ϕ_at_t(outer, inner, tval::Float64)
    H, W, Z = size(outer)
    ϕ = zeros(Float32, H, W, Z)
    outer_dt_S = similar(ϕ, Float32, H, W, Z)
    inner_dt_S = similar(ϕ, Float32, H, W, Z)
    for z in 1:Z
        outer_dt_S[:,:,z] .= edt_S(outer[:,:,z])
        inner_dt_S[:,:,z] .= edt_S(inner[:,:,z])
        ϕ[:,:,z] .= ϕ_func(tval, outer_dt_S[:,:,z], inner_dt_S[:,:,z])
    end
    return ϕ
end

"""
    compute_ϕ_stack(outer, inner, tvals) -> Array{Float32,4}

Slice-wise (2-D) build of the level-set field for a sequence of formation times
`tvals`. The per-slice signed EDTs are computed once and reused for every time,
returning a 4-D stack `(H, W, Z, length(tvals))` where `ϕ[:, :, :, k]` is the
field at `tvals[k]`. The batched counterpart of [`compute_ϕ_at_t`](@ref).
"""
function compute_ϕ_stack(outer, inner, tvals)
    H, W, Z = size(outer)
    ϕ = zeros(Float32, H, W, Z, length(tvals))

    outer_dt_S = similar(ϕ, Float32, H, W, Z)
    inner_dt_S = similar(ϕ, Float32, H, W, Z)

    for z in 1:Z
        outer_dt_S[:,:,z] .= edt_S(outer[:,:,z])
        inner_dt_S[:,:,z] .= edt_S(inner[:,:,z])
    end

    for (ti, t) in enumerate(tvals)
        for z in 1:Z
            ϕ[:,:,z,ti] .= ϕ_func(t, outer_dt_S[:,:,z], inner_dt_S[:,:,z])
        end
    end
    return ϕ
end

# --------------------------- 3D Implementation of level set function computation ------------------------------
"""
    compute_ϕ_at_t_3D(outer, inner, tval) -> Array{Float32,3}

Fully 3-D build of the level-set field `ϕ` at a single formation time `tval`.
Unlike [`compute_ϕ_at_t`](@ref), the signed EDT is computed over the whole
volume at once, so distances couple across z-slices (correct when the voxel grid
is genuinely volumetric).

Arguments
- `outer`, `inner` : Boolean volumes `(H, W, Z)`
- `tval`           : formation time in `[0, 1]`
"""
function compute_ϕ_at_t_3D(outer, inner, tval::Float64)
    outer_dt_S = edt_S(outer)
    inner_dt_S = edt_S(inner)
    return ϕ_func(tval, outer_dt_S, inner_dt_S)
end

"""
    compute_ϕ_stack_3D(outer, inner, tvals) -> Array{Float32,4}

Fully 3-D build of the level-set field for a sequence of formation times
`tvals`. The volumetric signed EDTs are computed once and reused for every time,
returning a 4-D stack `(H, W, Z, length(tvals))`. The batched counterpart of
[`compute_ϕ_at_t_3D`](@ref).
"""
function compute_ϕ_stack_3D(outer::BitArray{3}, inner::BitArray{3}, tvals::Vector{Float64})
    H, W, Z = size(outer)
    ϕ = zeros(Float32, H, W, Z, length(tvals))

    outer_dt_S = edt_S(outer)
    inner_dt_S = edt_S(inner)

    for (ti, t) in enumerate(tvals)
        ϕ[:,:,:,ti] .= ϕ_func(t, outer_dt_S, inner_dt_S)
    end
    return ϕ
end

"""
    compute_ϕ_stack_3D(outer, inner, tvals) -> Array{Float32,4}

Fully 3-D build of the level-set field for a sequence of formation times
`tvals`. The volumetric signed EDTs are computed once and reused for every time,
returning a 4-D stack `(H, W, Z, length(tvals))`. The batched counterpart of
[`compute_ϕ_at_t_3D`](@ref).
"""
function compute_ϕ_stack_3D(outer_dt_S::Array{Float32,3}, inner_dt_S::Array{Float32,3}, tvals::Vector{Float64})
    H, W, Z = size(outer_dt_S)
    ϕ = zeros(Float32, H, W, Z, length(tvals))

    for (ti, t) in enumerate(tvals)
        ϕ[:,:,:,ti] .= ϕ_func(t, outer_dt_S, inner_dt_S)
    end
    return ϕ
end

"""
    estimate_Ocy_formation_time(outer_dt_S, inner_dt_S, Ocy_pos_voxel) -> Vector{Float64}

Estimate the normalised formation time `t ∈ [0, 1]` at which each osteocyte was
embedded, from its position between the cement line and the Haversian canal.

`outer_dt_S` and `inner_dt_S` are the **precomputed signed distance fields** of
the outer (cement-line) and inner (canal) masks — e.g. from [`edt_S`](@ref)
(isotropic) or [`compute_EDT_S`](@ref) (anisotropic). For each voxel position
`(x, y, z)` in `Ocy_pos_voxel`,

    t = s_out / (s_out + s_in),

where `s_out = outer_dt_S[x,y,z]` and `s_in = inner_dt_S[x,y,z]`. This matches
the [`ϕ_func`](@ref) convention: `t → 0` near the cement line (formed early),
`t → 1` near the canal wall (formed late).

Arguments
- `outer_dt_S`, `inner_dt_S` : signed distance volumes `(H, W, Z)`
- `Ocy_pos_voxel`            : iterable of integer `(x, y, z)` voxel indices
"""
function estimate_Ocy_formation_time(outer_dt_S, inner_dt_S, Ocy_pos_voxel)
    t_form = zeros(length(Ocy_pos_voxel))
    for ii in eachindex(t_form)
        x, y, z = Ocy_pos_voxel[ii]
        s_out = outer_dt_S[x, y, z]
        s_in  = inner_dt_S[x, y, z]
        t_form[ii] = s_out / (s_out + s_in)
    end
    return t_form
end

# ─────────────────────────────────────────────────────────────────────────────
# Anisotropic signed distance transform (native Felzenszwalb–Huttenlocher)
#
# DistanceTransforms.jl assumes isotropic voxels, and ImageMorphology's weighted
# feature transform is not exact for anisotropic data. Confocal osteon stacks are
# anisotropic (dz ≠ dx, dy), so we use a native, exact, separable squared-EDT:
# the classic Felzenszwalb–Huttenlocher 1-D transform applied along each axis,
# with that axis's parabola weighted by its spacing² (w²·(p−q)²). This is O(N),
# pure Julia (no Python), and matches SciPy's anisotropic EDT to ~1e-13.
# ─────────────────────────────────────────────────────────────────────────────

# 1-D lower-envelope-of-parabolas transform: d[p] = minₚ f[q] + w2·(p−q)².
# `f` is the input cost vector, `d` the output, `v`/`z` are scratch buffers
# (parabola apex indices and intersection abscissae) sized ≥ length(f).
@inline function _edt1d!(d, f, w2, v, z)
    n = length(f)
    k = 1
    v[1] = 1; z[1] = -Inf; z[2] = Inf
    @inbounds for q in 2:n
        s = ((f[q] + w2*q*q) - (f[v[k]] + w2*v[k]*v[k])) / (2*w2*(q - v[k]))
        while s <= z[k]
            k -= 1
            s = ((f[q] + w2*q*q) - (f[v[k]] + w2*v[k]*v[k])) / (2*w2*(q - v[k]))
        end
        k += 1; v[k] = q; z[k] = s; z[k+1] = Inf
    end
    k = 1
    @inbounds for q in 1:n
        while z[k+1] < q
            k += 1
        end
        dq = q - v[k]
        d[q] = w2*dq*dq + f[v[k]]
    end
    return d
end

"""
    edt_aniso(seed; dx=0.379, dy=0.379, dz=0.4) -> Array{Float64,3}

Unsigned **anisotropic** Euclidean distance transform of a 3-D Boolean `seed`:
each cell holds the physical distance (µm) to the nearest `true` cell, using
per-axis voxel spacings `(dx, dy, dz)`. Exact, via a separable, weighted
Felzenszwalb–Huttenlocher squared-distance transform along each axis.
"""
function edt_aniso(seed::AbstractArray{Bool,3}; dx::Real=0.379, dy::Real=0.379, dz::Real=0.4)
    H, W, D = size(seed)
    f = Array{Float64,3}(undef, H, W, D)
    @inbounds for I in eachindex(seed)
        f[I] = seed[I] ? 0.0 : 1e20      # 0 at features, ~∞ elsewhere
    end

    nmax = max(H, W, D)
    v   = Vector{Int}(undef, nmax)
    z   = Vector{Float64}(undef, nmax + 1)
    col = Vector{Float64}(undef, nmax)
    out = Vector{Float64}(undef, nmax)

    # Pass along dim 1 (spacing dx), then dim 2 (dy), then dim 3 (dz).
    w2 = dx*dx
    @inbounds for k in 1:D, j in 1:W
        for i in 1:H; col[i] = f[i,j,k]; end
        _edt1d!(view(out,1:H), view(col,1:H), w2, v, z)
        for i in 1:H; f[i,j,k] = out[i]; end
    end
    w2 = dy*dy
    @inbounds for k in 1:D, i in 1:H
        for j in 1:W; col[j] = f[i,j,k]; end
        _edt1d!(view(out,1:W), view(col,1:W), w2, v, z)
        for j in 1:W; f[i,j,k] = out[j]; end
    end
    w2 = dz*dz
    @inbounds for j in 1:W, i in 1:H
        for k in 1:D; col[k] = f[i,j,k]; end
        _edt1d!(view(out,1:D), view(col,1:D), w2, v, z)
        for k in 1:D; f[i,j,k] = out[k]; end
    end

    @inbounds for I in eachindex(f); f[I] = sqrt(f[I]); end
    return f
end

"""
    edt_S_aniso(mask; dx=0.379, dy=0.379, dz=0.4) -> Array{Float64,3}

Signed **anisotropic** Euclidean distance transform of a 3-D Boolean `mask`:
positive outside the `true` region and negative inside, defined as
`edt_aniso(mask) - edt_aniso(.!mask)` with physical voxel spacings `(dx, dy, dz)`
in µm. The anisotropic counterpart of [`edt_S`](@ref); exact and pure Julia.
"""
function edt_S_aniso(mask::AbstractArray{Bool,3}; dx::Real=0.379, dy::Real=0.379, dz::Real=0.4)
    return edt_aniso(mask; dx, dy, dz) .- edt_aniso(.!mask; dx, dy, dz)
end

"""
    compute_EDT_S(outer, inner; dx=0.379, dy=0.379, dz=0.4) -> (outer_dt_S, inner_dt_S)

Signed anisotropic EDTs of the `outer` (cement-line) and `inner` (canal) Boolean
masks, computed natively with [`edt_S_aniso`](@ref) (no Python). The returned
fields are the inputs to [`ϕ_func`](@ref) / [`estimate_Ocy_formation_time`](@ref).
"""
function compute_EDT_S(outer, inner; dx::Real=0.379, dy::Real=0.379, dz::Real=0.4)
    outer_dt_S = Float32.(edt_S_aniso(Array{Bool,3}(outer); dx, dy, dz))
    inner_dt_S = Float32.(edt_S_aniso(Array{Bool,3}(inner); dx, dy, dz))
    return outer_dt_S, inner_dt_S
end

# ─────────────────────────────────────────────────────────────────────────────
# Gaussian smoothing of ϕ
# ─────────────────────────────────────────────────────────────────────────────

"""
    smooth_ϕ(ϕ; dx, dy, dz, σ_μm) -> Array{Float32, 3}

Apply an anisotropic 3-D Gaussian blur to the level-set volume `ϕ`.

The blur uses a **separable** kernel (`KernelFactors.gaussian`), applied as three
1-D passes. This is equivalent to a dense 3-D Gaussian but allocates far less
memory (~5× less here) — important because this is called once per osteocyte.

The standard deviation `σ_μm` is specified in **physical µm** and converted to
per-axis voxel units `(σ_μm/dx, σ_μm/dy, σ_μm/dz)`, so the blur is isotropic
in real space even when the voxel spacings differ (e.g. dx = dy = 0.379 µm,
dz = 0.4 µm).

Smoothing ϕ before extracting zero-level contours removes the staircase
artefacts that arise from the discrete, voxelised binary masks used to build
the signed distance transform.  A σ of 0.5–1.5 µm typically gives clean
contours without meaningfully shifting the zero-crossing location.

**Typical usage inside [`Analysis.compute_curvature_near_osteocyte`](@ref):**
```julia
ϕ       = ϕ_func(t_formed, outer_dt_S, inner_dt_S)
ϕ       = smooth_ϕ(ϕ; dx=dx, dy=dy, dz=dz, σ_μm=1.0)
X, Y    = compute_zero_contour_xy_coords(ϕ, z_layer, idx)
```

Arguments
- `ϕ`     : 3-D level-set field (H × W × Z)
- `dx, dy, dz` : voxel spacings in µm (default 0.379, 0.379, 0.4)
- `σ_μm`  : smoothing radius in physical µm
"""
function smooth_ϕ(ϕ::AbstractArray{<:Real,3};
                   dx::Real=0.379, dy::Real=0.379, dz::Real=0.4,
                   σ_μm::Real=1.0)
    kernel = KernelFactors.gaussian((σ_μm/dx, σ_μm/dy, σ_μm/dz))
    return Float32.(imfilter(ϕ, kernel))
end

"""
    smooth_ϕ(ϕ2d; dx, dy, σ_μm) -> Array{Float32, 2}

2-D overload: apply a Gaussian blur to a single z-slice of ϕ.

Useful when you have already extracted the slice you need
(e.g. `ϕ_slice = ϕ[:, :, z_layer]`) and want to smooth just that plane
before calling `compute_zero_contour_xy_coords`.

Arguments
- `ϕ2d`   : 2-D level-set slice (H × W)
- `dx, dy` : in-plane voxel spacings in µm
- `σ_μm`  : smoothing radius in physical µm
"""
function smooth_ϕ(ϕ::AbstractArray{<:Real,2};
                   dx::Real=0.379, dy::Real=0.379,
                   σ_μm::Real=1.0)
    kernel = KernelFactors.gaussian((σ_μm/dx, σ_μm/dy))
    return Float32.(imfilter(ϕ, kernel))
end


end # end of module