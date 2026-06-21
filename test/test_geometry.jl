@testset "Geometry" begin
    outer, inner = make_cylinders()
    od, idd = compute_EDT_S(outer, inner; dx = DX, dy = DY, dz = DZ)

    @testset "compute_zero_contour_xy_coords — traces a circle of known radius" begin
        for t in (0.0, 0.5, 1.0)
            ϕ = ϕ_func(t, od, idd)
            X, Y = compute_zero_contour_xy_coords(ϕ, ZMID, 1)
            @test length(X) == length(Y)
            @test length(X) > 20
            rexp = (1 - t) * R_OUT_PX + t * R_IN_PX        # contour is in pixel coords
            radii = @. sqrt((X - CX)^2 + (Y - CY)^2)
            @test sum(radii) / length(radii) ≈ rexp rtol = 0.05
            @test maximum(radii) - minimum(radii) < 4.0    # close to circular
        end
    end

    @testset "resample_closed_contour — uniform spacing, shape preserved" begin
        θ = collect(range(0, 2π; length = 137)); x = 40 .* cos.(θ); y = 40 .* sin.(θ)  # closed circle
        Xr, Yr = resample_closed_contour(x, y; spacing = 1.0)
        @test Xr[1] == Xr[end] && Yr[1] == Yr[end]                 # returned closed
        N = length(Xr) - 1
        seg = [hypot(Xr[i+1]-Xr[i], Yr[i+1]-Yr[i]) for i in 1:N]
        @test std(seg) / mean(seg) < 0.01                          # uniform spacing (tiny corner effect)
        @test sum(seg) ≈ 2π * 40 rtol = 0.01                       # perimeter preserved
        @test all(@. isapprox(sqrt(Xr[1:N]^2 + Yr[1:N]^2), 40; atol = 0.05))  # still on the circle
    end

    @testset "Ω — polygon area" begin
        @test Ω([0.0, 1, 1, 0], [0.0, 0, 1, 1]) ≈ 1.0       # unit square
        @test Ω([0.0, 2, 2, 0], [0.0, 0, 3, 3]) ≈ 6.0       # 2×3 rectangle
        # area enclosed by the cement-line contour ≈ π R²
        X, Y = compute_zero_contour_xy_coords(ϕ_func(0.0, od, idd), ZMID, 1)
        @test Ω(X, Y) ≈ π * R_OUT_PX^2 rtol = 0.05
    end

    @testset "compute_xy_center — centroid sits on the cylinder axis" begin
        for t in (0.0, 0.5)
            cx, cy = compute_xy_center(ϕ_func(t, od, idd), ZMID, 1)
            @test cx ≈ CX atol = 1.5
            @test cy ≈ CY atol = 1.5
        end
    end

    @testset "Plane — struct construction" begin
        p = Plane{Float64}((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))
        @test p.p0 == (1.0, 2.0, 3.0)
        @test p.n  == (0.0, 0.0, 1.0)
    end

    # Note: the cutting-plane functions (compute_planes_and_intersections,
    # proj_3D_onto_XZ) belong to the T-delay workflow and are not covered here,
    # consistent with skipping analysis_Tdelay_pairs.
end
