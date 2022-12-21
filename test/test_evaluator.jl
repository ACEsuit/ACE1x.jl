

##

using ACE1, ACE1.Testing, ACEcore, BenchmarkTools
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing, Random
using JuLIP: evaluate, evaluate_d, evaluate_ed
using ACE1: combine

using ACE1x
using Polynomials4ML

##

@info("Basic test of PIPotential construction and evaluation")
maxdeg = 10
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = ACE1.SparsePSHDegree()
P1 = ACE1.BasicPSH1pBasis(Pr; species = :X, D = D)
basis = ACE1.PIBasis(P1, 3, D, maxdeg)
c = ACE1.Random.randcoeffs(basis)
V = combine(basis, c)

Nat = 15
Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, :X)

##

@info("compare full model")
V_new = ACE1x.Evaluator.NewEvaluator(V)

val_new = ACE1.evaluate(V_new, Rs, Zs, z0)
val_old = ACE1.evaluate(V, Rs, Zs, z0)

@show val_new ≈ val_old

##

Nat = 50
Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, :X)

@btime ACE1.evaluate($V, $Rs, $Zs, $z0)

@btime ACE1.evaluate($V_new, $Rs, $Zs, $z0)

##

# @profview let V = V_new, Rs = Rs, Zs = Zs, z0 = z0
#    for _ = 1:20_000 
#       ACE1.evaluate(V, Rs, Zs, z0)
#    end
# end

# ##

# bA = V_new.bA
# cY = randn(ComplexF64, 50, length(V_new.bY))
# rY = randn(50, length(V_new.bY))
# Rn = randn(50, length(V_new.bR))

# cA = zeros(ComplexF64, length(bA))
# rA = zeros(length(bA))

# @info("complex evalpool")
# @btime ACEcore.evalpool!($cA, $bA, ($Rn, $cY))
# @info("real evalpool")
# @btime ACEcore.evalpool!($rA, $bA, ($Rn, $rY))

# ##

# rbY = Polynomials4ML.RYlmBasis(5)
# cbY = Polynomials4ML.CYlmBasis(5)
# @info("Complex Ylm")
# @btime (Y = Polynomials4ML.evaluate($cbY, $Rs); 
#         Polynomials4ML.release!(Y))
# @info("Real Ylm")
# @btime (Y = Polynomials4ML.evaluate($rbY, $Rs); 
#         Polynomials4ML.release!(Y))

##

dEs2 = zeros(JVecF, length(Rs))
dEs2 = ACE1.evaluate_d!(dEs2, nothing, V_new, Rs, Zs, z0)
dEs1 = ACE1.evaluate_d(V, Rs, Zs, z0)
@show dEs1 ≈ dEs2 

##

dEs1 = zeros(JVecF, length(Rs))
tmp_d = JuLIP.alloc_temp_d(V, length(Rs))
JuLIP.evaluate_d!(dEs1, tmp_d, V, Rs, Zs, z0)

dEs2 = zeros(JVecF, length(Rs))
dEs2 = ACE1.evaluate_d!(dEs2, nothing, V_new, Rs, Zs, z0)

@info("Profile site gradients")
@info("old")
@btime JuLIP.evaluate_d!($dEs1, $tmp_d, $V, $Rs, $Zs, $z0)
@info("new")
@btime ACE1.evaluate_d!($dEs2, nothing, $V_new, $Rs, $Zs, $z0)


##

@profview let dEs2 = dEs2, V_new = V_new, Rs = Rs, Zs = Zs, z0 = z0
   for _ = 1:30_000
      ACE1.evaluate_d!(dEs2, nothing, V_new, Rs, Zs, z0)
   end
end


