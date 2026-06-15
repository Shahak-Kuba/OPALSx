"""
    Analysis

Curvature estimation and osteocyte formation analysis.

This module provides the quantitative back-end of OPALSx:

- **Mean curvature of the level-set surface** — `compute_curvature` /
  `compute_curvature_4th` (whole-volume, 2nd/4th order) and `curvature_at_point`
  (a single voxel), all using the Osher & Fedkiw (2003) formula
  `κ = ∇·(∇ϕ/‖∇ϕ‖)`.
- **2-D contour curvature** — `compute_2D_curvature`, a local least-squares
  parabola fit along a planar contour.
- **Osteocyte curvature** — `estimate_osteocyte_curvature_3D`, which evaluates
  the surface curvature at each osteocyte's formation time and position.
- **T-delay pair analysis** — `analysis_Tdelay_pairs` and helpers, which measure
  the inclination angle `α` of formation-front segments across time.
"""
module Analysis

using LinearAlgebra
using Statistics
using ImageFiltering
using ProgressMeter

# Cross-module helpers used by `compute_curvature_near_osteocyte`. `..` resolves
# to the enclosing module (OPALSx when loaded as a package, Main when the source
# files are `include`d directly by a script), so this works in both cases —
# provided LevelSet and Geometry are loaded first (they are, by include order).
using ..LevelSet: smooth_ϕ
using ..Geometry: compute_zero_contour_xy_coords

export analysis_Tdelay_pairs, compute_curvature, compute_curvature_4th, compute_2D_curvature,
       curvature_at_point, estimate_osteocyte_curvature_3D, compute_curvature_near_osteocyte

"""
    generate_Tdelay_pairs(proj_points) -> Vector

Pair up projected formation-front points across consecutive contours. For each
angle, the *first* point of contour `c` is paired with the *second* point of
contour `c+1`, yielding the segments whose inclination is later measured. Helper
for [`analysis_Tdelay_pairs`](@ref).
"""
function generate_Tdelay_pairs(proj_points)
    Tdelay_proj_point_pairs = []
    for cont in eachindex(proj_points)[1:end-1]
        Tdelay_proj_point_pairs_per_cont = []
        for ang in eachindex(proj_points[1])
            push!(Tdelay_proj_point_pairs_per_cont, [proj_points[cont][ang][1], proj_points[cont+1][ang][2]])
        end
        push!(Tdelay_proj_point_pairs, Tdelay_proj_point_pairs_per_cont)
    end
    return Tdelay_proj_point_pairs
end

"""
    compute_Tdelay_gradients(Tdelay_proj_points) -> Vector

Slope `Δy/Δx` of each point pair produced by [`generate_Tdelay_pairs`](@ref).
Used to sign the inclination angle `α` in [`analysis_Tdelay_pairs`](@ref).
"""
function compute_Tdelay_gradients(Tdelay_proj_points)
    line_∇ = []
    for cont in eachindex(Tdelay_proj_points)
        line_∇_per_cont = []
        for ang in eachindex(Tdelay_proj_points[1])
            p1 = Tdelay_proj_points[cont][ang][1]
            p2 = Tdelay_proj_points[cont][ang][2]
            ∇ = (p2[2] - p1[2]) / (p2[1] - p1[1])
            push!(line_∇_per_cont, ∇)
        end
        push!(line_∇, line_∇_per_cont)
    end
    return line_∇
end

"""
    angle_between_vectors(v1, v2) -> Float64

Unsigned angle (radians) between vectors `v1` and `v2`, via the clamped
dot-product / norm ratio.
"""
function angle_between_vectors(v1, v2)
    cosθ = dot(v1, v2) / (norm(v1) * norm(v2))
    return acos(clamp(cosθ, -1.0, 1.0))
end

"""
    analysis_Tdelay_pairs(proj_points) -> (pairs, gradients, α)

Measure the inclination angle `α` of formation-front segments between
consecutive contours.

Point pairs are formed with [`generate_Tdelay_pairs`](@ref) and their slopes
with [`compute_Tdelay_gradients`](@ref). For each pair, `α` is the angle
(degrees) between the segment and the vertical, signed by the slope (negative
slope → negative `α`). Returns the pairs, their gradients, and the `α` matrix
(contours × angles).
"""
function analysis_Tdelay_pairs(proj_points)
    Tdelay_proj_point_pairs = generate_Tdelay_pairs(proj_points)
    Tdelay_line_∇ = compute_Tdelay_gradients(Tdelay_proj_point_pairs)
    # compute α here
    α = zeros(size(Tdelay_proj_point_pairs,1), size(Tdelay_proj_point_pairs[1],1))
    for cont in axes(α,1)
        for xy_ang in axes(α,2)
            v1 = Tdelay_proj_point_pairs[cont][xy_ang][1] .- Tdelay_proj_point_pairs[cont][xy_ang][2]
            v2 = [0.0, v1[2]]
            α_val = angle_between_vectors(v1, v2)
            if Tdelay_line_∇[cont][xy_ang] < 0.0
                α[cont,xy_ang] = -rad2deg(α_val)
            else
                α[cont,xy_ang] = rad2deg(α_val)
            end
        end
    end
    return Tdelay_proj_point_pairs, Tdelay_line_∇, α
end

"""
    kappa = compute_curvature(ϕ, dx, dy, dz; eps=1e-12)

Compute mean curvature κ of the level set ϕ(x,y,z) on a regular 3D grid,
using 2nd-order central differences.

κ = ( ϕx^2 ϕyy - 2 ϕx ϕy ϕxy + ϕy^2 ϕxx
    + ϕx^2 ϕzz - 2 ϕx ϕz ϕxz + ϕz^2 φxx
    + ϕy^2 ϕzz - 2 ϕy ϕz ϕyz + ϕz^2 φyy ) / |∇φ|^3

This method of calculating curvature comes from Oscher 2003 Eq(1.8). 
The spacing (dx,dy,dz) are the grid steps in x,y,z.
"""
function compute_curvature(ϕ::AbstractArray{<:Real,3},
                           dx::Real, dy::Real, dz::Real; eps=1e-12)

    nx, ny, nz = size(ϕ)
    kappa = fill!(similar(ϕ, Float64), 0.0)

    inv2dx = 1.0/(2*dx); inv2dy = 1.0/(2*dy); inv2dz = 1.0/(2*dz)
    invdx2 = 1.0/(dx*dx); invdy2 = 1.0/(dy*dy); invdz2 = 1.0/(dz*dz)
    inv4dxdy = 1.0/(4*dx*dy); inv4dxdz = 1.0/(4*dx*dz); inv4dydz = 1.0/(4*dy*dz)

    @inbounds for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
        ϕc = ϕ[i, j, k]

        # First derivatives (central)
        ϕx = (ϕ[i+1, j,   k  ] - ϕ[i-1, j,   k  ]) * inv2dx
        ϕy = (ϕ[i,   j+1, k  ] - ϕ[i,   j-1, k  ]) * inv2dy
        ϕz = (ϕ[i,   j,   k+1] - ϕ[i,   j,   k-1]) * inv2dz

        # Second derivatives (central)
        ϕxx = (ϕ[i+1, j,   k  ] - 2ϕc + ϕ[i-1, j,   k  ]) * invdx2
        ϕyy = (ϕ[i,   j+1, k  ] - 2ϕc + ϕ[i,   j-1, k  ]) * invdy2
        ϕzz = (ϕ[i,   j,   k+1] - 2ϕc + ϕ[i,   j,   k-1]) * invdz2

        # Mixed derivatives (central, 4-point stencils)
        ϕxy = (ϕ[i+1, j+1, k] - ϕ[i+1, j-1, k] - ϕ[i-1, j+1, k] + ϕ[i-1, j-1, k]) * inv4dxdy
        ϕxz = (ϕ[i+1, j, k+1] - ϕ[i+1, j, k-1] - ϕ[i-1, j, k+1] + ϕ[i-1, j, k-1]) * inv4dxdz
        ϕyz = (ϕ[i, j+1, k+1] - ϕ[i, j+1, k-1] - ϕ[i, j-1, k+1] + ϕ[i, j-1, k-1]) * inv4dydz


        # |∇ϕ|
        gradmag = sqrt(ϕx*ϕx + ϕy*ϕy + ϕz*ϕz) + eps  # eps avoids divide-by-zero
        denom = gradmag^3

        # Numerator
        num  =  (ϕx^2)*ϕyy - 2*ϕx*ϕy*ϕxy + (ϕy^2)*ϕxx
        num +=  (ϕx^2)*ϕzz - 2*ϕx*ϕz*ϕxz + (ϕz^2)*ϕxx
        num +=  (ϕy^2)*ϕzz - 2*ϕy*ϕz*ϕyz + (ϕz^2)*ϕyy

        kappa[i, j, k] = num / denom
    end

    return kappa
end


"""
    κ = compute_curvature_4th(ϕ, dx, dy, dz; eps=1e-12)

Compute the level-set mean curvature κ of ϕ(x,y,z) on a rectangular 3D grid with spacings dx, dy, and dz,
using 4th-order central differences on the interior
and 2nd-order central differences on a 2-cell boundary band.

From Oscher 2003, the formula is given by κ = ∇ϕ / ||∇ϕ|| which expands to:
κ = ( ϕx^2 ϕyy - 2 ϕx ϕy ϕxy + ϕy^2 ϕxx
    + ϕx^2 ϕzz - 2 ϕx ϕz ϕxz + ϕz^2 ϕxx
    + ϕy^2 ϕzz - 2 ϕy ϕz ϕyz + ϕz^2 ϕyy ) / |∇ϕ|^3

`eps` is added in to avoid division by zero.
"""
function compute_curvature_4th(ϕ::AbstractArray{<:Real,3},
                               dx::Real, dy::Real, dz::Real; eps=1e-12)
    nx, ny, nz = size(ϕ)
    kappa = fill!(similar(ϕ, Float64), 0.0)

    nx ≥ 5 && ny ≥ 5 && nz ≥ 5 || error("Need at least 5 points in each dim for 4th-order stencils.")

    inv12dx  = 1.0/(12*dx);  inv12dy  = 1.0/(12*dy);  inv12dz  = 1.0/(12*dz)
    inv12dx2 = 1.0/(12*dx*dx); inv12dy2 = 1.0/(12*dy*dy); inv12dz2 = 1.0/(12*dz*dz)
    inv2dx   = 1.0/(2*dx);   inv2dy   = 1.0/(2*dy);   inv2dz   = 1.0/(2*dz)
    invdx2   = 1.0/(dx*dx);  invdy2   = 1.0/(dy*dy);  invdz2   = 1.0/(dz*dz)
    inv4dxdy = 1.0/(4*dx*dy); inv4dxdz = 1.0/(4*dx*dz); inv4dydz = 1.0/(4*dy*dz)

    # ---- 1D 4th-order stencils at a single index (central, needs ±1,±2) ----
    Dx4(i,j,k)  = (-ϕ[i+2,j,k] + 8ϕ[i+1,j,k] - 8ϕ[i-1,j,k] + ϕ[i-2,j,k]) * inv12dx
    Dy4(i,j,k)  = (-ϕ[i,j+2,k] + 8ϕ[i,j+1,k] - 8ϕ[i,j-1,k] + ϕ[i,j-2,k]) * inv12dy
    Dz4(i,j,k)  = (-ϕ[i,j,k+2] + 8ϕ[i,j,k+1] - 8ϕ[i,j,k-1] + ϕ[i,j,k-2]) * inv12dz

    Dxx4(i,j,k) = (-ϕ[i+2,j,k] + 16ϕ[i+1,j,k] - 30ϕ[i,j,k] + 16ϕ[i-1,j,k] - ϕ[i-2,j,k]) * inv12dx2
    Dyy4(i,j,k) = (-ϕ[i,j+2,k] + 16ϕ[i,j+1,k] - 30ϕ[i,j,k] + 16ϕ[i,j-1,k] - ϕ[i,j-2,k]) * inv12dy2
    Dzz4(i,j,k) = (-ϕ[i,j,k+2] + 16ϕ[i,j,k+1] - 30ϕ[i,j,k] + 16ϕ[i,j,k-1] - ϕ[i,j,k-2]) * inv12dz2

    # Mixed derivatives via composition of 4th-order 1D operators (still 4th-order):
    # e.g. ϕ_xy(i,j,k) = D4x( Dy4(ϕ)(·,j,k) ) at i.
    function Dxy4(i,j,k)
        g_im2 = Dy4(i-2,j,k); g_im1 = Dy4(i-1,j,k); g_ip1 = Dy4(i+1,j,k); g_ip2 = Dy4(i+2,j,k)
        (-g_ip2 + 8g_ip1 - 8g_im1 + g_im2) * inv12dx
    end
    function Dxz4(i,j,k)
        g_im2 = Dz4(i-2,j,k); g_im1 = Dz4(i-1,j,k); g_ip1 = Dz4(i+1,j,k); g_ip2 = Dz4(i+2,j,k)
        (-g_ip2 + 8g_ip1 - 8g_im1 + g_im2) * inv12dx
    end
    function Dyz4(i,j,k)
        g_jm2 = Dz4(i,j-2,k); g_jm1 = Dz4(i,j-1,k); g_jp1 = Dz4(i,j+1,k); g_jp2 = Dz4(i,j+2,k)
        (-g_jp2 + 8g_jp1 - 8g_jm1 + g_jm2) * inv12dy
    end

    # ===================== 4th-order interior =====================
    @inbounds for k in 3:nz-2, j in 3:ny-2, i in 3:nx-2
        ϕx, ϕy, ϕz = Dx4(i,j,k), Dy4(i,j,k), Dz4(i,j,k)
        ϕxx, ϕyy, ϕzz = Dxx4(i,j,k), Dyy4(i,j,k), Dzz4(i,j,k)
        ϕxy, ϕxz, ϕyz = Dxy4(i,j,k), Dxz4(i,j,k), Dyz4(i,j,k)

        gradmag = sqrt(ϕx*ϕx + ϕy*ϕy + ϕz*ϕz) + eps
        denom = gradmag^3

        num  =  (ϕx^2)*ϕyy - 2*ϕx*ϕy*ϕxy + (ϕy^2)*ϕxx
        num +=  (ϕx^2)*ϕzz - 2*ϕx*ϕz*ϕxz + (ϕz^2)*ϕxx
        num +=  (ϕy^2)*ϕzz - 2*ϕy*ϕz*ϕyz + (ϕz^2)*ϕyy

        kappa[i,j,k] = num / denom
    end

    # ===================== 2nd-order boundary band =====================
    # Use your original 2nd-order stencils on i/j/k ∈ {2, nx-1} etc., and also first/last layer.
    @inbounds begin
        inv2dx = 1.0/(2*dx); inv2dy = 1.0/(2*dy); inv2dz = 1.0/(2*dz)
        invdx2 = 1.0/(dx*dx); invdy2 = 1.0/(dy*dy); invdz2 = 1.0/(dz*dz)
        inv4dxdy = 1.0/(4*dx*dy); inv4dxdz = 1.0/(4*dx*dz); inv4dydz = 1.0/(4*dy*dz)

        # Helper loop that guards against out-of-bounds and fills any index not done above
        function fill_second_order!(i,j,k)
            if 3 ≤ i ≤ nx-2 && 3 ≤ j ≤ ny-2 && 3 ≤ k ≤ nz-2
                return  # already 4th-order
            end
            2 ≤ i ≤ nx-1 && 2 ≤ j ≤ ny-1 && 2 ≤ k ≤ nz-1 || return  # need neighbors

            ϕc = ϕ[i,j,k]
            ϕx = (ϕ[i+1,j,k] - ϕ[i-1,j,k]) * inv2dx
            ϕy = (ϕ[i,j+1,k] - ϕ[i,j-1,k]) * inv2dy
            ϕz = (ϕ[i,j,k+1] - ϕ[i,j,k-1]) * inv2dz

            ϕxx = (ϕ[i+1,j,k] - 2ϕc + ϕ[i-1,j,k]) * invdx2
            ϕyy = (ϕ[i,j+1,k] - 2ϕc + ϕ[i,j-1,k]) * invdy2
            ϕzz = (ϕ[i,j,k+1] - 2ϕc + ϕ[i,j,k-1]) * invdz2

            ϕxy = (ϕ[i+1,j+1,k] - ϕ[i+1,j-1,k] - ϕ[i-1,j+1,k] + ϕ[i-1,j-1,k]) * inv4dxdy
            ϕxz = (ϕ[i+1,j,k+1] - ϕ[i+1,j,k-1] - ϕ[i-1,j,k+1] + ϕ[i-1,j,k-1]) * inv4dxdz
            ϕyz = (ϕ[i,j+1,k+1] - ϕ[i,j+1,k-1] - ϕ[i,j-1,k+1] + ϕ[i,j-1,k-1]) * inv4dydz

            gradmag = sqrt(ϕx*ϕx + ϕy*ϕy + ϕz*ϕz) + eps
            denom = gradmag^3

            num  =  (ϕx^2)*ϕyy - 2*ϕx*ϕy*ϕxy + (ϕy^2)*ϕxx
            num +=  (ϕx^2)*ϕzz - 2*ϕx*ϕz*ϕxz + (ϕz^2)*ϕxx
            num +=  (ϕy^2)*ϕzz - 2*ϕy*ϕz*ϕyz + (ϕz^2)*ϕyy

            kappa[i,j,k] = num / denom
        end

        # All points that have the 2nd-order neighborhood
        for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
            fill_second_order!(i,j,k)
        end
    end

    return kappa
end

"""
    ensure_ccw(X, Y)

Takes coordinate vectors `X` and `Y` representing a closed 2D curve and
returns `(X2, Y2)` oriented **counterclockwise** (anti-clockwise).

If the points are already CCW, they are returned unchanged.
If they are clockwise, they are reversed.

Returns
-------
`(X2, Y2, flipped)` where `flipped::Bool` is `true` if the order was reversed.
"""
function ensure_ccw(X::AbstractVector, Y::AbstractVector)
    @assert length(X) == length(Y) "X and Y must be same length"
    N = length(X)
    @assert N ≥ 3 "Need at least 3 points"

    # Compute signed area (shoelace formula)
    A2 = 0.0
    @inbounds for i in 1:N
        j = (i == N) ? 1 : i + 1
        A2 += X[i] * Y[j] - X[j] * Y[i]
    end

    if A2 > 0     # already CCW
        return (X, Y, false)
    else           # CW: reverse
        return (reverse(X), reverse(Y), true)
    end
end

"""
    compute_2D_curvature(x, y; k=3, eps=1e-10) -> Matrix{Float64}

Signed curvature along a closed planar contour `(x, y)` by local parabola
fitting.

The contour is first reoriented counterclockwise (`ensure_ccw`). At each point
a window of `±k` neighbours is taken, translated to the origin and rotated so
the local tangent aligns with the x-axis; a least-squares parabola `y = a x² +
b x + c` is fitted to the window and the curvature read off as `κ = 2a / (1 +
b²)^{3/2}`. `eps` guards the denominator. Returns one curvature value per
contour point (in the units of `x`, `y`); pass physical coordinates (e.g.
`X .* dx`) to obtain curvature in µm⁻¹.

The fitting window can be chosen in two ways:
- a fixed **point count** `±k` (the default), or
- a fixed **physical arc length** via `arclen` (in the units of `x`,`y`): each
  point's window then spans `±arclen/2` of cumulative arc length (at least `k_min`
  points per side). Use `arclen` to measure curvature at a constant physical
  scale (e.g. ≈ one osteocyte) regardless of how many points the contour has.

Arguments
- `x`, `y` : coordinate vectors of the closed contour (first point repeated)

Keyword arguments
- `k`      : half-width of the fitting window in points (used when `arclen===nothing`)
- `arclen` : if set, full arc length (same units as `x`,`y`) of the fitting window;
             overrides `k`
- `k_min`  : minimum points per side in arc-length mode (parabola-fit stability)
- `eps`    : small value added to the denominator for numerical stability
"""
function compute_2D_curvature(x,y; k=3, arclen=nothing, k_min::Int=3, eps=1e-10)
    @assert length(x) == length(y) "x and y must have same length"
    N = length(x)-1
    @assert N ≥ 5 "Need at least 5 points"
    # ensuring that the orientation from Marching Squares is anti clockwise
    X,Y = ensure_ccw(x, y)
    pop!(x); pop!(y)


    wrap(i) = mod1(i, N)

    function tangent(x,y,i)
        ip = wrap(i+1); im = wrap(i-1)
        tx = x[ip] - x[im]
        ty = y[ip] - y[im]
        n = hypot(tx, ty)
        return n == 0 ? (1.0, 0.0) : (tx/n, ty/n)
    end

    function rotate(p::Tuple{<:Real,<:Real}, θ::Real; about=(0.0, 0.0))
        x, y   = p
        cx, cy = about
        c = cos(θ); s = sin(θ)
        dx, dy = x - cx, y - cy
        return (cx + c*dx - s*dy,  cy + s*dx + c*dy)
    end

    # Arc-length window selection: gather points within ±`half` cumulative arc
    # length of `ii` on each side (at least `k_min`), without wrapping the whole
    # loop. Measures curvature at a fixed physical scale even though the number
    # of contour points varies with contour size.
    seglen(i) = (ip = wrap(i + 1); hypot(X[ip] - X[i], Y[ip] - Y[i]))
    function window_by_arclen(ii, half)
        right = Int[]; acc = 0.0; j = ii
        while length(right) < N - 2
            jn = wrap(j + 1); acc += seglen(j); push!(right, jn)
            (acc ≥ half && length(right) ≥ k_min) && break
            j = jn
        end
        left = Int[]; acc = 0.0; j = ii
        while length(left) < N - 2
            jp = wrap(j - 1); acc += seglen(jp); pushfirst!(left, jp)
            (acc ≥ half && length(left) ≥ k_min) && break
            j = jp
        end
        return vcat(left, ii, right)
    end

    half = arclen === nothing ? 0.0 : arclen / 2
    curvature = zeros(length(x),1)
    for ii in eachindex(x)
        # indices of the considered nodes on either side of point ii
        ii_considered = if arclen === nothing
            vcat([wrap(ii - s) for s in k:-1:1], ii, [wrap(ii + s) for s in 1:k])
        else
            window_by_arclen(ii, half)
        end
        x_central = X[ii]
        y_central = Y[ii]
        x_considered = X[ii_considered]
        y_considered = Y[ii_considered]

        # calculating tangent at ii
        tx, ty = tangent(X,Y,ii)

        rotation_θ = atan(ty,tx)
        # move the points so that the ii point is at the origin
        x_considered_center = x_considered .- x_central
        y_considered_center = y_considered .- y_central
        points = [(X,Y) for (X,Y) in zip(x_considered_center,y_considered_center)]

        # rotate the points
        rotated_points = []
        for point in points
            push!(rotated_points,rotate(point, -rotation_θ))
        end

        # calculating LS fit parabola: x = (A'A)¹A'b
        A = zeros(length(points), 3)
        for jj in eachindex(rotated_points)
            A[jj, :] .= [rotated_points[jj][1]^2, rotated_points[jj][1], 1]
        end

        y_array = [y for (x,y) in rotated_points]

        a,b,c = (A'*A)\(A'*y_array)

        num = 2*a
        # since I center the points about a central point (0,0) then 2ax = 0 
        # + eps is to avoid division by 0
        denom = ( 1 + b^2 )^(3/2) + eps 

        # estimating curvature from parabola
        curvature[ii] = num / denom
    end
    return curvature
end



# =============================================================================
# 3-D curvature at a single voxel
# =============================================================================

"""
    curvature_at_point(ϕ, ix, iy, iz; dx, dy, dz, order=2, eps=1e-12) -> Float64

Compute the 3-D mean curvature κ of the level-set surface at voxel (ix, iy, iz)
using the Osher & Fedkiw (2003) formula:

    κ = (ϕx² ϕyy − 2ϕx ϕy ϕxy + ϕy² ϕxx
       + ϕx² ϕzz − 2ϕx ϕz ϕxz + ϕz² ϕxx
       + ϕy² ϕzz − 2ϕy ϕz ϕyz + ϕz² ϕyy) / |∇ϕ|³

This is evaluated at a *single* voxel, so it is cheap to call per osteocyte.
The voxel should be close to (or on) the zero level-set of ϕ, as it is by
construction when ϕ is evaluated at the osteocyte's formation time t_form.

Pass a **pre-smoothed** ϕ (e.g. from `estimate_osteocyte_curvature_3D` with
σ_μm > 0, or from `smooth_levelset` in Plotting.jl) to suppress voxelization
staircase artefacts before differentiation.

Arguments
- `ϕ`          : 3-D level-set field (H × W × Z) at the formation time of interest
- `ix, iy, iz` : 1-based voxel indices (must be ≥ 2 from any edge for order=2,
                 ≥ 3 from any edge for order=4)
- `dx, dy, dz` : voxel spacings in µm (default 0.379, 0.379, 0.358)
- `order`      : finite-difference order — 2 (standard) or 4 (higher accuracy)
- `eps`        : floor added to |∇ϕ| to prevent division by zero at flat regions
"""
function curvature_at_point(ϕ::AbstractArray{<:Real,3},
                             ix::Int, iy::Int, iz::Int;
                             dx::Real=0.379, dy::Real=0.379, dz::Real=0.358,
                             order::Int=2, eps::Real=1e-12)

    nx, ny, nz = size(ϕ)
    margin = (order == 4) ? 2 : 1

    (margin < ix ≤ nx-margin && margin < iy ≤ ny-margin && margin < iz ≤ nz-margin) ||
        error("Voxel ($ix,$iy,$iz) is too close to the volume boundary. " *
              "Need ≥$margin voxels clearance for order=$order stencils " *
              "(volume is $nx×$ny×$nz).")

    i, j, k = ix, iy, iz

    if order == 4
        c12dx  = 1.0/(12dx);   c12dy  = 1.0/(12dy);   c12dz  = 1.0/(12dz)
        c12dx2 = 1.0/(12dx^2); c12dy2 = 1.0/(12dy^2); c12dz2 = 1.0/(12dz^2)

        # 4th-order first derivatives  (-f₊₂ + 8f₊₁ − 8f₋₁ + f₋₂) / 12h
        ϕx = (-ϕ[i+2,j,k] + 8ϕ[i+1,j,k] - 8ϕ[i-1,j,k] + ϕ[i-2,j,k]) * c12dx
        ϕy = (-ϕ[i,j+2,k] + 8ϕ[i,j+1,k] - 8ϕ[i,j-1,k] + ϕ[i,j-2,k]) * c12dy
        ϕz = (-ϕ[i,j,k+2] + 8ϕ[i,j,k+1] - 8ϕ[i,j,k-1] + ϕ[i,j,k-2]) * c12dz

        # 4th-order second derivatives  (-f₊₂ + 16f₊₁ − 30f₀ + 16f₋₁ − f₋₂) / 12h²
        ϕxx = (-ϕ[i+2,j,k] + 16ϕ[i+1,j,k] - 30ϕ[i,j,k] + 16ϕ[i-1,j,k] - ϕ[i-2,j,k]) * c12dx2
        ϕyy = (-ϕ[i,j+2,k] + 16ϕ[i,j+1,k] - 30ϕ[i,j,k] + 16ϕ[i,j-1,k] - ϕ[i,j-2,k]) * c12dy2
        ϕzz = (-ϕ[i,j,k+2] + 16ϕ[i,j,k+1] - 30ϕ[i,j,k] + 16ϕ[i,j,k-1] - ϕ[i,j,k-2]) * c12dz2

        # Mixed derivatives: apply the 4th-order 1-D operator twice (compositions)
        # ϕxy = D4x(D4y(ϕ)) at (i,j,k), etc.
        Dy4(ii,jj,kk) = (-ϕ[ii,jj+2,kk] + 8ϕ[ii,jj+1,kk] - 8ϕ[ii,jj-1,kk] + ϕ[ii,jj-2,kk]) * c12dy
        Dz4(ii,jj,kk) = (-ϕ[ii,jj,kk+2] + 8ϕ[ii,jj,kk+1] - 8ϕ[ii,jj,kk-1] + ϕ[ii,jj,kk-2]) * c12dz

        ϕxy = (-Dy4(i+2,j,k) + 8Dy4(i+1,j,k) - 8Dy4(i-1,j,k) + Dy4(i-2,j,k)) * c12dx
        ϕxz = (-Dz4(i+2,j,k) + 8Dz4(i+1,j,k) - 8Dz4(i-1,j,k) + Dz4(i-2,j,k)) * c12dx
        ϕyz = (-Dz4(i,j+2,k) + 8Dz4(i,j+1,k) - 8Dz4(i,j-1,k) + Dz4(i,j-2,k)) * c12dy

    else   # order == 2
        c2dx  = 1.0/(2dx);    c2dy  = 1.0/(2dy);    c2dz  = 1.0/(2dz)
        cdx2  = 1.0/dx^2;    cdy2  = 1.0/dy^2;    cdz2  = 1.0/dz^2
        c4dxy = 1.0/(4dx*dy); c4dxz = 1.0/(4dx*dz); c4dyz = 1.0/(4dy*dz)

        ϕc  = ϕ[i,j,k]
        ϕx  = (ϕ[i+1,j,k] - ϕ[i-1,j,k]) * c2dx
        ϕy  = (ϕ[i,j+1,k] - ϕ[i,j-1,k]) * c2dy
        ϕz  = (ϕ[i,j,k+1] - ϕ[i,j,k-1]) * c2dz

        ϕxx = (ϕ[i+1,j,k] - 2ϕc + ϕ[i-1,j,k]) * cdx2
        ϕyy = (ϕ[i,j+1,k] - 2ϕc + ϕ[i,j-1,k]) * cdy2
        ϕzz = (ϕ[i,j,k+1] - 2ϕc + ϕ[i,j,k-1]) * cdz2

        # 4-point cross stencil for mixed derivatives
        ϕxy = (ϕ[i+1,j+1,k] - ϕ[i+1,j-1,k] - ϕ[i-1,j+1,k] + ϕ[i-1,j-1,k]) * c4dxy
        ϕxz = (ϕ[i+1,j,k+1] - ϕ[i+1,j,k-1] - ϕ[i-1,j,k+1] + ϕ[i-1,j,k-1]) * c4dxz
        ϕyz = (ϕ[i,j+1,k+1] - ϕ[i,j+1,k-1] - ϕ[i,j-1,k+1] + ϕ[i,j-1,k-1]) * c4dyz
    end

    gmag = sqrt(ϕx^2 + ϕy^2 + ϕz^2) + eps
    num  = ϕx^2*ϕyy - 2ϕx*ϕy*ϕxy + ϕy^2*ϕxx
    num += ϕx^2*ϕzz - 2ϕx*ϕz*ϕxz + ϕz^2*ϕxx
    num += ϕy^2*ϕzz - 2ϕy*ϕz*ϕyz + ϕz^2*ϕyy

    return num / gmag^3
end


"""
    estimate_osteocyte_curvature_3D(outer_dt_S, inner_dt_S, t_form_ordered,
                                     Ocy_pos_vox_ordered;
                                     dx, dy, dz, σ_μm=1.0, order=2, eps=1e-12)
                                     -> Vector{Float64}

Estimate the 3-D mean curvature [µm⁻¹] of the osteon surface at the
formation time and voxel position of each osteocyte.

Workflow per osteocyte i
  1. Evaluate  ϕᵢ = (1 − tᵢ)·outer_dt_S − tᵢ·inner_dt_S
     (the level-set whose zero surface is the osteon wall at time tᵢ)
  2. Apply anisotropic 3-D Gaussian smoothing with σ_μm (physical µm) to
     remove the staircase artefacts inherent in a voxelised signed EDT.
     The smoothing kernel is the same one used by `plot_3d_surfaces!`, so
     the curvature reflects the same surface that is visualised.
  3. Evaluate `curvature_at_point` at the osteocyte's voxel.

Arguments
- `outer_dt_S`          : signed EDT of the outer (cement-line) mask (H×W×Z)
- `inner_dt_S`          : signed EDT of the inner (Haversian-canal) mask (H×W×Z)
- `t_form_ordered`      : vector of formation times tᵢ ∈ [0, 1]
- `Ocy_pos_vox_ordered` : matching vector of (ix, iy, iz) voxel index tuples

Keyword arguments
- `dx, dy, dz` : voxel spacings in µm (default 0.379, 0.379, 0.358)
- `σ_μm`       : Gaussian smoothing radius in physical µm before differentiation;
                 should match the σ_μm used in `plot_3d_surfaces!` for consistency.
                 Set to 0 to skip smoothing (not recommended — staircase artefacts
                 will bias the curvature estimate).
- `order`      : finite-difference order (2 or 4)
- `eps`        : floor added to |∇ϕ| to prevent division by zero
"""
function estimate_osteocyte_curvature_3D(outer_dt_S::AbstractArray{<:Real,3},
                                          inner_dt_S::AbstractArray{<:Real,3},
                                          t_form_ordered::AbstractVector,
                                          Ocy_pos_vox_ordered::AbstractVector;
                                          dx::Real=0.379, dy::Real=0.379, dz::Real=0.358,
                                          σ_μm::Real=1.0,
                                          order::Int=2,
                                          eps::Real=1e-12)
    n    = length(t_form_ordered)
    κ_3D = zeros(Float64, n)

    # Build the Gaussian kernel once — it is the same for every osteocyte
    # as long as σ_μm and the voxel spacings are constant.
    kernel = σ_μm > 0 ? ImageFiltering.Kernel.gaussian((σ_μm/dx, σ_μm/dy, σ_μm/dz)) : nothing

    for ii in 1:n
        t          = Float64(t_form_ordered[ii])
        ix, iy, iz = Ocy_pos_vox_ordered[ii]

        # Step 1 — level-set field at formation time t
        ϕ = @. Float32((1 - t)*outer_dt_S - t*inner_dt_S)

        # Step 2 — smooth to suppress voxelization staircase
        if kernel !== nothing
            ϕ = Float32.(imfilter(ϕ, kernel))
        end

        # Step 3 — 3-D mean curvature at the osteocyte voxel
        κ_3D[ii] = curvature_at_point(ϕ, ix, iy, iz; dx, dy, dz, order, eps)
    end

    return κ_3D
end


# =============================================================================
# 2-D contour curvature at each osteocyte
# =============================================================================

"""
    nearest_index(X, Y, x, y) -> Int

Index of the point in coordinate vectors `(X, Y)` closest (squared Euclidean
distance) to the query point `(x, y)`.
"""
function nearest_index(X, Y, x, y)
    @assert length(X) == length(Y) && !isempty(X)
    return argmin(@. (X - x)^2 + (Y - y)^2)
end

"""
    compute_curvature_near_osteocyte(t_form_ordered, outer_dt_S, inner_dt_S,
                                     Ocy_pos_voxel_ordered, dx, dy, dz, σ_μm)
        -> (κ_at_osteocyte, mean_available_κ)

Compute the 2-D contour curvature at each osteocyte's formation time and
position.

For osteocyte `i` the level-set field is built at its formation time `tᵢ`,
Gaussian-smoothed ([`LevelSet.smooth_ϕ`](@ref), radius `σ_μm` µm) to suppress
the voxelisation staircase, and its zero-level contour on the osteocyte's
z-slice is extracted ([`Geometry.compute_zero_contour_xy_coords`](@ref)),
converted to physical µm, and its curvature computed
([`compute_2D_curvature`](@ref)).

Only the osteocyte's own z-slice is needed, so the field is built and smoothed
over just a **z-slab** spanning `±kz_half` slices around it (`kz_half` = the
Gaussian's z half-width) and kept in `Float32`. Because the smoothing kernel
reaches at most `kz_half` slices, the smoothed value at the centre slice is
*identical* to smoothing the whole volume — but uses ~50–100× less memory and
time. Two values are recorded per osteocyte:

- `κ_at_osteocyte`   : curvature at the contour point nearest the osteocyte
- `mean_available_κ` : mean curvature over the whole contour at that time

Both are returned as vectors in the input order, in units of µm⁻¹.

The curvature is measured at a fixed **physical scale** `k_scale_um`: each
parabola is fitted over an arc of length `k_scale_um` µm of the contour
(`compute_2D_curvature(...; arclen=k_scale_um)`). This keeps the measurement
scale constant across formation times even though later (inner) contours are
smaller — set it to roughly the size of one (or a few) osteocytes.

Arguments
- `t_form_ordered`        : formation times `tᵢ ∈ [0, 1]`
- `outer_dt_S`, `inner_dt_S` : signed distance volumes (see [`LevelSet.compute_EDT_S`](@ref))
- `Ocy_pos_voxel_ordered` : matching `(x, y, z)` voxel indices
- `dx, dy, dz`            : voxel spacings in µm
- `σ_μm`                  : Gaussian smoothing radius in µm

Keyword arguments
- `k_scale_um`            : arc length (µm) of the curvature-fitting window — the
                            physical scale at which curvature is measured
                            (default `15.0`, ≈ one osteocyte).
- `show_progress`         : display a per-osteocyte progress bar (default `true`).
                            ProgressMeter throttles its own redraws (~10 Hz), so
                            the overhead is negligible next to the per-osteocyte
                            level-set smoothing.
"""
function compute_curvature_near_osteocyte(t_form_ordered, outer_dt_S, inner_dt_S,
                                          Ocy_pos_voxel_ordered, dx, dy, dz, σ_μm;
                                          k_scale_um::Real=15.0, show_progress::Bool=true)
    κ_at_osteocyte   = Float64[]
    mean_available_κ = Float64[]

    # z half-width of the smoothing kernel: the slab must extend this far on each
    # side of the osteocyte's slice for the smoothed centre slice to be exact.
    Z       = size(outer_dt_S, 3)
    kz_half = (length(KernelFactors.gaussian((σ_μm/dx, σ_μm/dy, σ_μm/dz))[3]) - 1) ÷ 2

    prog = Progress(length(t_form_ordered); enabled=show_progress,
                    desc="  Curvature: ", showspeed=true)

    for (idx, t_formed) in enumerate(t_form_ordered)
        z_layer     = Ocy_pos_voxel_ordered[idx][3]
        osteocyte_x = Ocy_pos_voxel_ordered[idx][1] * dx
        osteocyte_y = Ocy_pos_voxel_ordered[idx][2] * dy

        # Build ϕ over just the z-slab around this osteocyte, fused into Float32
        # (views avoid copying the slab out of the full volumes first).
        z0 = max(1, z_layer - kz_half)
        z1 = min(Z, z_layer + kz_half)
        oa = @view outer_dt_S[:, :, z0:z1]
        ia = @view inner_dt_S[:, :, z0:z1]
        ϕ_slab   = @. Float32((1 - t_formed) * oa - t_formed * ia)
        ϕ_smooth = smooth_ϕ(ϕ_slab; dx=dx, dy=dy, dz=dz, σ_μm=σ_μm)

        # Contour the osteocyte's slice (its local index within the slab).
        X, Y = compute_zero_contour_xy_coords(ϕ_smooth, z_layer - z0 + 1, idx)
        κ    = compute_2D_curvature(X .* dx, Y .* dy; arclen=k_scale_um)

        push!(mean_available_κ, mean(κ))
        push!(κ_at_osteocyte,   κ[nearest_index(X .* dx, Y .* dy, osteocyte_x, osteocyte_y)])

        next!(prog)
    end

    return κ_at_osteocyte, mean_available_κ
end

end # end of module