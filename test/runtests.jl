using ACE1x
using Test

@testset "ACE1x.jl" begin
    @testset "pure2b" begin include("test_pure2b.jl"); end 
    @testset "pure2b-multi" begin include("test_pure2b_multi.jl"); end 
end
