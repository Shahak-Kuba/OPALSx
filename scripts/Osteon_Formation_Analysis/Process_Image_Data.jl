
include("../../src/Imaging.jl")
include("../../src/LevelSet.jl")
include("../../src/Geometry.jl")
include("../../src/Analysis.jl")
using .Imaging, .LevelSet, .Geometry, .Analysis

# choose which dataset to process 
dataset = "FM40-4-E2"

base_dir = joinpath("DATA", dataset)
paths_HCa = readdir(joinpath(base_dir, "HCa"); join=true)
paths_On  = readdir(joinpath(base_dir, "On");  join=true)
output_dir = joinpath(base_dir, "Processed_Images")
mkpath(output_dir)

for (path_to_HCa, path_to_On) in zip(paths_HCa, paths_On)
    sample_name = Imaging.extract_sample_name(path_to_HCa)
    output_path = joinpath(output_dir, sample_name * ".png")
    Imaging.generate_RG_img_from_data(path_to_HCa, path_to_On, output_path)
end