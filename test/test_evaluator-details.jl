

##

using ACE1, ACE1.Testing, ACEcore
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

@info("check the translation of the radial basis")
rr = norm.(Rs)
xx = Pr.trans.(rr)
P_old  = collect(hcat(evaluate.(Ref(Pr.J), xx)...)')
Pr_new = ACE1x.Evaluator.TransformedRadial(Pr)
P_new = ACE1x.Evaluator.evaluate(Pr_new, rr)
@show P_old ≈ P_new

##

@info("A basis")
bY = Polynomials4ML.CYlmBasis(V.pibasis.basis1p.SH.maxL)
bR = ACE1x.Evaluator.TransformedRadial(V.pibasis.basis1p.J)

A_spec = V.pibasis.basis1p.spec
A_spec_new = Tuple{Int,Int}[] 
for b in A_spec 
   n = b.n 
   iY = Polynomials4ML.index_y(b.l, b.m)
   push!(A_spec_new, (n, iY))
end
bA = ACEcore.PooledSparseProduct(A_spec_new)

# new method 
Rn = ACE1x.Evaluator.evaluate(bR, rr)
Ylm = Polynomials4ML.evaluate(bY, Rs)
A_new = ACEcore.evalpool(bA, (Rn, Ylm))

# old method 
A_old = ACE1.evaluate(V.pibasis.basis1p, Rs, Zs, z0)

@show A_new ≈ A_old

##

@info("AA basis")

iAA2iA = V.pibasis.inner[1].iAA2iA 
orders = V.pibasis.inner[1].orders
new_spec = [ iAA2iA[i, 1:orders[i]][:] for i = 1:length(orders) ]
bAA = ACEcore.SparseSymmProdDAG(new_spec)

AA_new = ACEcore.evaluate(bAA, A_new)
AA_old = ACE1.evaluate(V.pibasis, Rs, Zs, z0)

@show AA_new ≈ AA_old

##

@info("full model")
V_new = ACE1x.Evaluator.NewEvaluator(V)

val_new = ACE1.evaluate(V_new, Rs, Zs, z0)
val_old = ACE1.evaluate(V, Rs, Zs, z0)

@show val_new ≈ val_old

##
# print("     energy: ")
# println(@test energy(V, at) ≈ naive_energy(V, at) )
# print("site-energy: ")
# println(@test energy(V, at) ≈ sum( site_energy(V, at, n)
#                                       for n = 1:length(at) ) )
# print("     forces: ")
# println(@test JuLIP.Testing.fdtest(V, at; verbose=false))
# print("site-forces: ")
# println(@test JuLIP.Testing.fdtest( x -> site_energy(V, set_dofs!(at, x), 3),
#                                     x -> mat(site_energy_d(V, set_dofs!(at, x), 3))[:],
#                                     dofs(at); verbose=false ) )
