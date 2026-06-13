# ── Synthetic concentric-cylinder geometry for the test suite ────────────────
#
# Two cylinders running along z, centred on the grid axis:
#   • a LARGE cylinder (radius R_OUT_PX)  → synthetic cement line
#   • a SMALL cylinder (radius R_IN_PX)   → synthetic Haversian canal
#
# Masks follow the level-set convention used throughout OPALSx (signed EDTs that
# are positive in the matrix between the two walls):
#   outer mask = EXTERIOR of the cement line (r > R_OUT_PX) → S_outer = 0 at cement line
#   inner mask = canal LUMEN                 (r < R_IN_PX)  → S_inner = 0 at canal wall
#
# With this polarity the analytic results are clean and easy to assert:
#   • zero-contour radius at time t :  r(t) = (1-t)·R_OUT + t·R_IN
#   • osteocyte formation time      :  t    = (R_OUT - r) / (R_OUT - R_IN)
#   • surface mean curvature        :  κ    = 1 / r           (a cylinder)
#
# The geometry is constant along z, so the 3-D EDT reduces to the in-plane radial
# distance on the interior slices (the dz spacing is still exercised by the
# dedicated single-seed anisotropy test in test_levelset.jl).

using Random

const DX, DY, DZ = 0.379, 0.379, 0.4          # voxel spacings [µm]
const GRID_H, GRID_W, GRID_Z = 181, 181, 11    # grid (odd → axis sits on a voxel)
const CX, CY = (GRID_H + 1) / 2, (GRID_W + 1) / 2
const ZMID   = (GRID_Z + 1) ÷ 2
const R_OUT_PX = 66.0                           # cement-line radius [px]
const R_IN_PX  = 26.0                           # canal radius [px]
const R_OUT_UM = R_OUT_PX * DX                  # …in µm
const R_IN_UM  = R_IN_PX * DX

"In-plane radial distance of voxel `(i, j)` from the cylinder axis, in pixels."
radial_px(i, j) = sqrt((i - CX)^2 + (j - CY)^2)

"In-plane radial distance of voxel `(i, j)` from the cylinder axis, in µm."
radial_um(i, j) = radial_px(i, j) * DX

"Build the `(outer, inner)` Boolean cylinder masks, each `(GRID_H, GRID_W, GRID_Z)`."
function make_cylinders()
    outer = falses(GRID_H, GRID_W, GRID_Z)
    inner = falses(GRID_H, GRID_W, GRID_Z)
    @inbounds for k in 1:GRID_Z, j in 1:GRID_W, i in 1:GRID_H
        r = radial_px(i, j)
        outer[i, j, k] = r > R_OUT_PX     # exterior of the cement line
        inner[i, j, k] = r < R_IN_PX      # canal lumen
    end
    return outer, inner
end

"""
    make_osteocytes(n; rng) -> (pos_um, pos_voxel)

`n` random synthetic osteocyte positions strictly between the two cylinders,
on the middle z-slices. Returns µm positions and the matching integer voxel
indices (the two outputs of `load_osteocytes`).
"""
function make_osteocytes(n; rng = MersenneTwister(42))
    pos_um    = Tuple{Float64,Float64,Float64}[]
    pos_voxel = Tuple{Int,Int,Int}[]
    margin = 5.0
    while length(pos_voxel) < n
        rpx = (R_IN_PX + margin) + rand(rng) * ((R_OUT_PX - margin) - (R_IN_PX + margin))
        θ   = 2π * rand(rng)
        i   = round(Int, CX + rpx * cos(θ))
        j   = round(Int, CY + rpx * sin(θ))
        k   = rand(rng, (ZMID - 1):(ZMID + 1))
        (1 ≤ i ≤ GRID_H && 1 ≤ j ≤ GRID_W) || continue
        push!(pos_voxel, (i, j, k))
        push!(pos_um,    (i * DX, j * DY, k * DZ))
    end
    return pos_um, pos_voxel
end
