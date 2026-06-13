# OPALSx

**O**steon **P**rofile **A**nd **L**evel-**S**et analysis.

OPALSx reconstructs the geometry of remodelling osteons from segmented confocal
image stacks and analyses where and when osteocytes were embedded during bone
formation. A time-dependent level-set field is built from the signed distance
transforms of the Haversian-canal and cement-line masks; its zero contour models
the bone-formation front sweeping inward from the cement line (`t = 0`) to the
canal wall (`t = 1`) over normalised formation time `t ∈ [0, 1]`.

## Layout

```
OPALSx/
├── Project.toml          Julia environment (package deps, incl. GLMakie)
├── Manifest.toml         pinned dependency versions
├── hpc/                  headless environment (same deps minus GLMakie)
├── src/
│   ├── OPALSx.jl         top-level module — includes & re-exports the submodules
│   ├── Imaging.jl        load segmented stacks → outer/inner Boolean masks
│   ├── LevelSet.jl       signed distance transforms and the level-set field ϕ
│   ├── Geometry.jl       contour extraction, centroids, cutting planes
│   ├── Analysis.jl       2-D/3-D curvature and osteocyte formation analysis
│   └── Plotting.jl       GLMakie helpers for contours and isosurfaces
├── scripts/
│   └── Osteon_Formation_Analysis/
│       └── Multi_Osteon_Analysis.jl
└── DATA/
    └── <dataset>/
        ├── Processed_Images/    segmented red/green PNG stack (one per z-slice)
        └── cells_<dataset>_.csv osteocyte positions exported from Napari
```

### Segmentation colour convention

Processed images use two channels: **red** = mineralised matrix outside the
Haversian canal, **green** = overlap of canal and osteonal labels. The *outer*
region (cement line) is `red ∪ green`; the *inner* region (canal) is the
complement of `green`.

## Setup

Requires Julia ≥ 1.9 (developed on 1.10). From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs the Julia dependencies. There is **no Python dependency** — the
anisotropic distance transform is implemented natively in Julia.

## Running the multi-osteon analysis

```bash
julia --project=. scripts/Osteon_Formation_Analysis/Multi_Osteon_Analysis.jl
```

The script processes every dataset listed in its `datasets` array
(default `["FM40-1-R1", "FM40-2-R2"]`). For each one it:

1. loads the `Processed_Images` stack and builds the outer/inner masks;
2. computes anisotropic signed distance transforms natively (voxel spacing
   `dx = dy = 0.379 µm`, `dz = 0.4 µm`);
3. estimates each osteocyte's formation time from its position between the canal
   wall and cement line;
4. measures the bone-surface curvature at each osteocyte's formation time.

It produces three figures: the curvature-vs-formation-time scatter, the 3-D
formation-front surface (with optional osteocyte overlay), and an
osteocyte-distribution figure (formation-time and curvature histograms) for
reading off a preferred formation time and a convex/concave curvature
preference. Figures are saved to `figures/`.

### Plotting backends

Plotting uses the backend-agnostic Makie API, so figures render in either
backend. Set `BACKEND` at the top of the script:

- `:gl` → **GLMakie** — interactive windows and 3-D rotation/animation (local use);
- `:cairo` → **CairoMakie** — high-resolution static figures for posters/talks
  (`CairoMakie.activate!(px_per_unit=3)`; `save` also supports `.pdf`/`.svg`).

### HPC / headless runs

`Multi_Osteon_Analysis_HPC.jl` is the headless CairoMakie variant for compute
nodes. It uses a **separate environment, `hpc/`, which is the main project minus
GLMakie**, so a headless node never installs or precompiles GLMakie (GLMakie
needs system OpenGL/GLFW that such nodes lack). The script activates `hpc/`
itself; on a login node (with internet) build it once:

```bash
julia --project=hpc -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=hpc scripts/Osteon_Formation_Analysis/Multi_Osteon_Analysis_HPC.jl
```

To analyse other samples, drop their `Processed_Images/` folder and
`cells_<name>_.csv` under `DATA/<name>/` and add `<name>` to the `datasets`
list.

## Tests

The test suite (`test/`) verifies the main functions of `LevelSet.jl`,
`Geometry.jl` and `Analysis.jl` against a synthetic concentric-cylinder geometry
(a large cylinder for the cement line, a small one for the Haversian canal, and
random osteocyte positions in between), where the distance fields, contours,
formation times and curvatures all have known analytic values. Run with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

