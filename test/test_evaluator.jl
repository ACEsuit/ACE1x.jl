

##

using ACE1, ACE1.Testing, ACEcore, BenchmarkTools
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing, Random
using JuLIP: evaluate, evaluate_d, evaluate_ed
using ACE1: combine

using ACE1x
using Polynomials4ML

##

@info("Basic test of PIPotential construction and evaluation")
maxdeg = 16
order = 3
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = ACE1.SparsePSHDegree()
P1 = ACE1.BasicPSH1pBasis(Pr; species = :X, D = D)
basis = ACE1.PIBasis(P1, order, D, maxdeg)
c = ACE1.Random.randcoeffs(basis)
V = combine(basis, c)

@show length(basis)

Nat = 30
Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, :X)

## 

# construct a symmetry-reduced model

spec = ACE1.get_basis_spec(V.pibasis, 1)

inv_spec = Dict{Any, Int}() 
for (i, bb) in enumerate(spec)
   inv_spec[bb] = i
end

_refl(b::ACE1.RPI.PSH1pBasisFcn) = ACE1.RPI.PSH1pBasisFcn(b.n, b.l, -b.m, b.z)
_refl(bb::ACE1.PIBasisFcn) = ACE1.PIBasisFcn(bb.z0, _refl.(bb.oneps), bb.top)

refl_spec = _refl.(spec) 


mirrors = [ haskey(inv_spec, bb) ? inv_spec[bb] : missing for bb in refl_spec ]
nmiss = count(ismissing, mirrors) + count([ i === mirrors[i] for i = 1:length(spec) ])

inner = V.pibasis.inner[1] 

new_spec = [] 
new_coeffs = Float64[] 

for (i, (bb, refl_bb)) in enumerate(zip(spec, refl_spec))
   if (bb == refl_bb) || !haskey(inv_spec, refl_bb)
      push!(new_spec, bb)
      push!(new_coeffs, V.coeffs[1][i])
   else
      i_refl = inv_spec[refl_bb]
      if i_refl > i
         push!(new_spec, bb)
         t = (-1)^(sum(b.m for b in bb.oneps))
         push!(new_coeffs, V.coeffs[1][i] + t * V.coeffs[1][i_refl])
      end
   end
end

length(spec) == (length(new_spec) - nmiss) * 2 + nmiss

red_basis = ACE1.pibasis_from_specs(V.pibasis.basis1p, (new_spec,))
V_red = ACE1.combine(red_basis, new_coeffs)

##

@info("compare full model")
V_new = ACE1x.Evaluator.NewEvaluator(V)
V_new_red = ACE1x.Evaluator.NewEvaluator(V_red)

val_new = ACE1.evaluate(V_new, Rs, Zs, z0)
val_old = ACE1.evaluate(V, Rs, Zs, z0)
val_red = ACE1.evaluate(V_red, Rs, Zs, z0)
val_new_red = ACE1.evaluate(V_new_red, Rs, Zs, z0)

@show val_new ≈ val_old ≈ val_red ≈ val_new_red

##

# Nat = 50
# Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, :X)

@info("compare evaluation of single site")
print(" original: "); @btime ACE1.evaluate($V, $Rs, $Zs, $z0)
print("  reduced: "); @btime ACE1.evaluate($V_red, $Rs, $Zs, $z0)
print("      new: "); @btime ACE1.evaluate($V_new, $Rs, $Zs, $z0)
print("  new_red: "); @btime ACE1.evaluate($V_new_red, $Rs, $Zs, $z0)

##

display(@benchmark ACE1.evaluate($V_new_red, $Rs, $Zs, $z0))

##

@profview let V = V_new, Rs = Rs, Zs = Zs, z0 = z0
   for _ = 1:20_000 
      ACE1.evaluate(V, Rs, Zs, z0)
   end
end

##

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

##

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

# @profview let dEs2 = dEs2, V_new = V_new, Rs = Rs, Zs = Zs, z0 = z0
#    for _ = 1:30_000
#       ACE1.evaluate_d!(dEs2, nothing, V_new, Rs, Zs, z0)
#    end
# end

##

# Nat = length(Rs)
# Nbatch = 32
# Rs_ = [ deepcopy(Rs) for _=1:Nbatch ]
# Zs_ = [ deepcopy(Zs) for _=1:Nbatch ]
# z0_ = [ copy(z0) for _=1:Nbatch ]
# Rs_b = repeat(Rs, Nbatch)
# Zs_b = repeat(Zs, Nbatch)
# Z0s = fill(z0, Nbatch * Nat)
# I0s = vcat( [fill(i, Nat) for i = 1:Nbatch]... )

# val = ACE1.evaluate(V_new, Rs, Zs, z0)
# vals = ACE1x.Evaluator.eval_batch!(nothing, V_new, Rs_b, Zs_b, Z0s, I0s)

# @show all(vals .≈ val)

##

# print("  original potential: ")
# @btime begin; e = 0.0; for i=1:$Nbatch; e += ACE1.evaluate($V, $Rs_[i], $Zs_[i], $z0_[i]); end; e; end 
# print("         new kernels: ")
# @btime begin; e = 0.0; for i=1:$Nbatch; e += ACE1.evaluate($V_new, $Rs_[i], $Zs_[i], $z0_[i]); end; e; end 
# print(" new kernels batched: ")
# @btime sum(ACE1x.Evaluator.eval_batch!(nothing, $V_new_red, $Rs_b, $Zs_b, $Z0s, $I0s))


##

# @profview let V_new = V_new, Rs_b = Rs_b, Zs_b = Zs_b, Z0s = Z0s, I0s = I0s
#    for _ = 1:3000
#       ACE1x.Evaluator.eval_batch!(nothing, V_new_red, Rs_b, Zs_b, Z0s, I0s)
#    end
# end

##

# length(V_new.bA)
# length(V_new.bAA)

# A = randn(ComplexF64, Nbatch, length(V_new.bA))
# AA = zeros(ComplexF64, Nbatch, length(V_new.bAA))
# c = randn(size(AA, 2))
# vals = zeros(Nbatch)

# # @btime ACEcore.evaluate!($AA,$(V_new.bAA), $A)
# # @btime ACEcore.evaluate_dot!($vals, $AA, $(V_new.bAA), $A, $c, real)

# rAA = real.(AA)
# rA = real.(A)

# # @btime mul!($vals, $rAA, $c)
# rAA1 = rAA[1,:]
# # @btime dot($rAA1, $c)

# @btime ACEcore.evaluate!($AA,$(V_new.bAA), $A)
# @btime ACEcore.evaluate!($rAA,$(V_new.bAA), $rA)

# # ACEcore.evaluate!(AA,(V_new.bAA), A)

##

# @profview let AA=rAA, bAA = V_new.bAA,  A = rA
#    for _ = 1:6000
#       ACEcore.evaluate!(AA, bAA, A)
#    end
# end


##

# V_new


##

