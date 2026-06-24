@testset "Analysis" begin
    outer, inner = make_cylinders()
    od, idd = compute_EDT_S(outer, inner; dx = DX, dy = DY, dz = DZ)
    _, pos = make_osteocytes(20)

    # signed-distance field of a single cylinder of radius `r0` µm (κ = 1/r exactly)
    cyl_sdf(r0) = Float64[radial_um(i, j) - r0 for i in 1:GRID_H, j in 1:GRID_W, k in 1:GRID_Z]

    @testset "ensure_ccw — orientation handling" begin
        X = [0.0, 1, 1, 0, 0]; Y = [0.0, 0, 1, 1, 0]      # CCW square
        _, _, flipped = Analysis.ensure_ccw(X, Y)
        @test !flipped
        _, _, flipped_cw = Analysis.ensure_ccw(reverse(X), reverse(Y))   # CW
        @test flipped_cw
    end

    @testset "compute_2D_curvature — circle has κ = 1/R" begin
        R = 30.0; M = 240
        θ = collect(range(0, 2π; length = M + 1))          # closed (last == first)
        x = R .* cos.(θ); y = R .* sin.(θ)
        κ = compute_2D_curvature(copy(x), copy(y); k = 10)
        @test isapprox(abs(sum(κ) / length(κ)), 1 / R; rtol = 0.05)
        @test std(κ) < 0.1 * abs(sum(κ) / length(κ))       # ~constant around the circle
    end

    @testset "contour_mean_curvature — methods agree on a circle (κ = 1/R)" begin
        R = 30.0; M = 240
        θ = collect(range(0, 2π; length = M + 1))
        x = R .* cos.(θ); y = R .* sin.(θ)
        @test isapprox(circle_fit_curvature(copy(x), copy(y)),  1 / R; rtol = 1e-3)  # :CCF
        @test isapprox(turning_fit_curvature(copy(x), copy(y)), 1 / R; rtol = 1e-3)  # :CTF
        for m in (:CCF, :CTF, :ALF)
            @test isapprox(contour_mean_curvature(copy(x), copy(y); method = m, arclen = 20.0),
                           1 / R; rtol = 0.05)
        end
        @test contour_mean_curvature(copy(x), copy(y)) == circle_fit_curvature(copy(x), copy(y))  # default :CCF
        @test_throws ArgumentError contour_mean_curvature(copy(x), copy(y); method = :nope)
    end

    @testset "circle_fit_curvature — free centre robust to uneven sampling" begin
        R = 40.0
        θ = vcat(range(-0.5, 0.5; length = 160), range(0.6, 2π - 0.6; length = 40))  # clustered on +x side
        x = R .* cos.(θ); y = R .* sin.(θ); push!(x, x[1]); push!(y, y[1])           # closed
        @test isapprox(circle_fit_curvature(x, y; center = :free), 1 / R; rtol = 1e-3)  # recovers 1/R
        @test circle_fit_curvature(x, y; center = :centroid) > 1.5 / R                  # centroid badly biased
        @test_throws ArgumentError circle_fit_curvature(x, y; center = :nope)
    end

    @testset "curvature_at_point — cylinder surface has κ ≈ 1/r" begin
        ϕ = Float32.(cyl_sdf(20.0))
        for rpx in (35.0, 50.0)
            i = round(Int, CX + rpx); j = Int(CY); k = ZMID
            r_um = radial_um(i, j)
            @test isapprox(abs(curvature_at_point(ϕ, i, j, k; dx = DX, dy = DY, dz = DZ, order = 2)), 1 / r_um; rtol = 0.1)
            @test isapprox(abs(curvature_at_point(ϕ, i, j, k; dx = DX, dy = DY, dz = DZ, order = 4)), 1 / r_um; rtol = 0.1)
        end
    end

    @testset "compute_curvature / compute_curvature_4th — volume κ ≈ 1/r" begin
        ϕ = cyl_sdf(20.0)
        κ2 = compute_curvature(ϕ, DX, DY, DZ)
        κ4 = compute_curvature_4th(ϕ, DX, DY, DZ)
        for θ in collect(range(0, 2π; length = 9))[1:end-1]   # ring of off-axis points
            i = round(Int, CX + 45cos(θ)); j = round(Int, CY + 45sin(θ)); k = ZMID
            r_um = radial_um(i, j)
            @test isapprox(abs(κ2[i, j, k]), 1 / r_um; rtol = 0.15)
            @test isapprox(abs(κ4[i, j, k]), 1 / r_um; rtol = 0.15)
        end
    end

    # The curvature at an osteocyte is estimated from the *discretized* signed
    # EDT, so per-voxel values carry staircase noise; smoothing (σ_μm > 0) is
    # required and the estimate is reliable in aggregate. We therefore check the
    # mean tightly and individual values loosely (but they must still track 1/r).
    @testset "estimate_osteocyte_curvature_3D — κ ≈ 1/r at each osteocyte" begin
        t = estimate_Ocy_formation_time(od, idd, pos)
        κ3 = estimate_osteocyte_curvature_3D(od, idd, t, pos; dx = DX, dy = DY, dz = DZ, σ_μm = 2.0)
        rinv = [1 / radial_um(i, j) for (i, j, k) in pos]
        @test length(κ3) == length(pos)
        @test isapprox(sum(abs, κ3) / length(κ3), sum(rinv) / length(rinv); rtol = 0.15)  # mean
        for (κi, ri) in zip(κ3, rinv)
            @test isapprox(abs(κi), ri; rtol = 0.4)                                        # per-point
        end
    end

    @testset "compute_curvature_near_osteocyte — 2-D contour κ ≈ 1/r" begin
        t = estimate_Ocy_formation_time(od, idd, pos)
        p = sortperm(t)
        κ_at, mean_κ = compute_curvature_near_osteocyte(t[p], od, idd, pos[p], DX, DY, DZ, 2.0; show_progress = false)
        rinv = [1 / radial_um(i, j) for (i, j, k) in pos[p]]
        @test length(κ_at) == length(pos)
        @test length(mean_κ) == length(pos)
        @test isapprox(sum(abs, κ_at) / length(κ_at), sum(rinv) / length(rinv); rtol = 0.15)  # mean
        for (κi, ri) in zip(κ_at, rinv)
            @test isapprox(abs(κi), ri; rtol = 0.4)                                            # per-point
        end
    end
end
