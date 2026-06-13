@testset "LevelSet" begin
    outer, inner = make_cylinders()

    @testset "edt — isotropic distance to nearest true (single seed)" begin
        seed = falses(7, 7, 3); seed[4, 4, 2] = true
        d = edt(seed)
        @test d[4, 4, 2] ≈ 0 atol = 1e-6
        @test d[1, 4, 2] ≈ 3.0 atol = 1e-6              # 3 px along dim 1
        @test d[4, 4, 1] ≈ 1.0 atol = 1e-6              # 1 px along dim 3
        @test d[1, 1, 2] ≈ sqrt(9 + 9) atol = 1e-6      # diagonal
    end

    @testset "edt_S — isotropic signed (negative inside the true region)" begin
        S = edt_S(outer)                                 # outer true = exterior
        # a matrix/lumen voxel is OUTSIDE the exterior region → positive
        @test S[Int(CX), Int(CY), ZMID] > 0
        # a voxel beyond the cement line is INSIDE the exterior region → negative
        i_ext = round(Int, CX + R_OUT_PX + 6)
        @test S[i_ext, Int(CY), ZMID] < 0
    end

    @testset "edt_aniso — anisotropic weighted distances (single seed)" begin
        seed = falses(7, 7, 5); seed[4, 4, 3] = true
        d = edt_aniso(seed; dx = DX, dy = DY, dz = DZ)
        @test d[4, 4, 3] ≈ 0 atol = 1e-9
        @test d[1, 4, 3] ≈ 3 * DX atol = 1e-6
        @test d[4, 1, 3] ≈ 3 * DY atol = 1e-6
        @test d[4, 4, 1] ≈ 2 * DZ atol = 1e-6                       # dz exercised here
        @test d[1, 4, 1] ≈ sqrt((3DX)^2 + (2DZ)^2) atol = 1e-6      # anisotropic diagonal
    end

    @testset "edt_S_aniso / compute_EDT_S — signed µm distances match geometry" begin
        od, idd = compute_EDT_S(outer, inner; dx = DX, dy = DY, dz = DZ)
        @test size(od) == size(outer)
        @test eltype(od) == Float32                       # feeds the Float32 ϕ-stack method
        # matrix voxel at a known radius
        i = round(Int, CX + 45); j = Int(CY); k = ZMID
        r_um = radial_um(i, j)
        @test od[i, j, k]  ≈ (R_OUT_UM - r_um) rtol = 0.05   # +distance to cement line
        @test idd[i, j, k] ≈ (r_um - R_IN_UM)  rtol = 0.05   # +distance to canal
        @test od ≈ Float32.(edt_S_aniso(outer; dx = DX, dy = DY, dz = DZ))
    end

    @testset "ϕ_func — linear interpolation of the two fields" begin
        a = rand(4, 4, 2); b = rand(4, 4, 2)
        @test ϕ_func(0.0, a, b) ≈ a
        @test ϕ_func(1.0, a, b) ≈ -b
        @test ϕ_func(0.3, a, b) ≈ 0.7 .* a .- 0.3 .* b
    end

    @testset "compute_ϕ_* — zero-contour radius interpolates from R_out to R_in" begin
        od, idd = compute_EDT_S(outer, inner; dx = DX, dy = DY, dz = DZ)
        for t in (0.0, 0.5, 1.0)
            ϕ = ϕ_func(t, od, idd)
            rexp = (1 - t) * R_OUT_UM + t * R_IN_UM
            # radius (µm) of the zero crossing scanning +x from the axis
            rcross = NaN
            for i in Int(CX):(GRID_H - 1)
                if sign(ϕ[i, Int(CY), ZMID]) != sign(ϕ[i + 1, Int(CY), ZMID])
                    rcross = radial_um(i, Int(CY)); break
                end
            end
            @test rcross ≈ rexp rtol = 0.06
        end
        # builder shapes
        @test size(compute_ϕ_stack_3D(od, idd, collect(0.0:0.5:1.0))) == (GRID_H, GRID_W, GRID_Z, 3)
        @test size(compute_ϕ_at_t_3D(outer, inner, 0.0)) == size(outer)
        @test size(compute_ϕ_at_t(outer, inner, 0.5))    == size(outer)
        @test size(compute_ϕ_stack(outer, inner, [0.0, 1.0])) == (GRID_H, GRID_W, GRID_Z, 2)
    end

    @testset "estimate_Ocy_formation_time — t = (R_out - r)/(R_out - R_in)" begin
        od, idd = compute_EDT_S(outer, inner; dx = DX, dy = DY, dz = DZ)
        _, pos = make_osteocytes(30)
        t = estimate_Ocy_formation_time(od, idd, pos)
        @test all(0 .≤ t .≤ 1)
        for (ti, (i, j, k)) in zip(t, pos)
            texp = (R_OUT_PX - radial_px(i, j)) / (R_OUT_PX - R_IN_PX)
            @test ti ≈ texp atol = 0.05
        end
    end

    @testset "smooth_ϕ — shape/type preserved, constant unchanged, noise reduced" begin
        ϕ = rand(Float32, 24, 24, 8)
        s3 = smooth_ϕ(ϕ; dx = DX, dy = DY, dz = DZ, σ_μm = 1.0)
        @test size(s3) == size(ϕ)
        @test eltype(s3) == Float32
        @test var(vec(s3)) < var(vec(ϕ))                       # smoothing reduces variance
        c = fill(2.0f0, 16, 16, 6)
        @test smooth_ϕ(c; dx = DX, dy = DY, dz = DZ, σ_μm = 1.0) ≈ c rtol = 1e-3
        @test size(smooth_ϕ(rand(Float32, 20, 20); dx = DX, dy = DY, σ_μm = 1.0)) == (20, 20)
    end
end
