using ACE1x
using Test

@testset "ACE1x.jl" begin
    @testset "pure2b" begin include("test_pure2b.jl"); end 
    @testset "pure2b-multi" begin include("test_pure2b_multi.jl"); end 
    @testset "defaults" begin include("test_default.jl"); end
    @testset "Purify-single" begin include("test_purify.jl"); end
    @testset "Purify-multi" begin include("test_purify_multi.jl"); end
    @testset "acemodel" begin include("test_acemodel.jl"); end
    @testset "Weird Bugs" begin include("test_bugs.jl"); end
end
