

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

##

bA = V_new.bA
cY = randn(ComplexF64, 50, length(V_new.bY))
rY = randn(50, length(V_new.bY))
Rn = randn(50, length(V_new.bR))

cA = zeros(ComplexF64, length(bA))
rA = zeros(length(bA))

@info("complex evalpool")
@btime ACEcore.evalpool!($cA, $bA, ($Rn, $cY))
@info("real evalpool")
@btime ACEcore.evalpool!($rA, $bA, ($Rn, $rY))

##

rbY = Polynomials4ML.RYlmBasis(5)
cbY = Polynomials4ML.CYlmBasis(5)
@info("Complex Ylm")
@btime (Y = Polynomials4ML.evaluate($cbY, $Rs); 
        Polynomials4ML.release!(Y))
@info("Real Ylm")
@btime (Y = Polynomials4ML.evaluate($rbY, $Rs); 
        Polynomials4ML.release!(Y))

##

dEs2 = zeros(JVecF, length(Rs))
dEs2 = ACE1.evaluate_d!(dEs2, nothing, V_new, Rs, Zs, z0)

dEs1 = ACE1.evaluate_d(V, Rs, Zs, z0)

dEs1 ≈ dEs2 

##
using JuLIP: JVec
Rn = randn(5, length(V_new.bR))
Ylm = randn(ComplexF64, 5, length(V_new.bY))

val, ∂Rn, ∂Ylm = ACE1x.Evaluator.evaluate_d_inner(V_new, Rn, Ylm)

@info("∂R")
U = randn(size(Rn))
size(∂Rn) == size(Rn)
for h in (0.1).^(2:10)
   val_h, _, _ = ACE1x.Evaluator.evaluate_d_inner(V_new, Rn + h * U, Ylm)
   val_dh = (val_h - val) / h 
   @printf(" %.2e  |  %.2e \n", h, norm(val_dh - dot(U, ∂Rn), Inf))
end

@info("∂Y")
U = randn(ComplexF64, size(Ylm))
__dot(U, ∂Ylm) = dot(real.(U), real.(∂Ylm)) - dot(imag.(U), imag.(∂Ylm))

size(∂Ylm) == size(Ylm)
for h in (0.1).^(2:10)
   val_h, _, _ = ACE1x.Evaluator.evaluate_d_inner(V_new, Rn, Ylm + h * U)
   val_dh = (val_h - val) / h 
   @printf(" %.2e  |  %.2e \n", h, norm(val_dh - __dot(U, ∂Ylm), Inf))
end


##


# using Printf 

# rr =  norm.(Rs)
# Rn, Rn_d = ACE1x.Evaluator.evaluate_ed(V_new.bR, rr)
# for h in (0.1).^(2:10)
#    Rn_h = ACE1x.Evaluator.evaluate(V_new.bR, rr .+ h)
#    Rn_dh = (Rn_h - Rn) / h 
#    @printf(" %.2e  |  %.2e \n", h, norm(Rn_d - Rn_dh, Inf))   
# end