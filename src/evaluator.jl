
module Evaluator

import ACE1, ObjectPools, ACEcore, Polynomials4ML

using ACEcore, JuLIP, ForwardDiff

using ACEcore: SparseSymmProdDAG, PooledSparseProduct
using LinearAlgebra: dot, norm 
import ACE1: PIPotential
import Polynomials4ML: CYlmBasis, OrthPolyBasis1D3T

# This envelope interface needs to be redesigned 
# at the moment, because we have two inputs, r anx x, 
# one needs to return 2 derivatives, the r and the x derivative. 
# this makes it a bit awkward to use. 
struct XEnvelope{T}
   xl::T 
   xr::T 
   pl::Int
   pr::Int
end

evaluate(env::XEnvelope, r, x) = 
      (x - env.xl)^(env.pl) * (env.xr - x)^(env.pr)

evaluate_d(env::XEnvelope, r, x) = 
      ForwardDiff.derivative(x -> evaluate(env, r, x), x)

struct TransformedRadial{TB, TT, TE}
   basis::TB 
   trans::TT 
   envelope::TE 
   # ------- temps 
end

function TransformedRadial(Pr::ACE1.OrthPolys.TransformedPolys)
   # radial basis after transform 
   # a small correction to account for a small difference in the implementation
   Pr_new = Polynomials4ML.OrthPolyBasis1D3T(
               copy(Pr.J.A), copy(Pr.J.B), copy(Pr.J.C))
   Pr_new.A[2] *= Pr_new.A[1]
   Pr_new.B[2] *= Pr_new.A[1]

   # transform 
   trans = Pr.trans

   # envelope : this makes some assumptions right now that we need to fix and 
   # check 
   env = XEnvelope(Pr.J.tl, Pr.J.tr, Pr.J.pl, Pr.J.pr)

   return TransformedRadial(Pr_new, trans, env)
end

Base.length(t::TransformedRadial) = length(t.basis)

function evaluate(basis::TransformedRadial, rr::AbstractVector{<: Real})
   RR = zeros(length(rr), length(basis))
   evaluate!(RR, basis, rr)
   return RR
end

function evaluate!(RR, basis::TransformedRadial, rr::AbstractVector{<: Real})
   xx = ACE1.Transforms.transform.(Ref(basis.trans), rr)
   ee = evaluate.(Ref(basis.envelope), rr, xx)
   Polynomials4ML.evaluate!(RR, basis.basis, xx)
   RR .= RR .* ee
   return nothing 
end

import Polynomials4ML: evaluate_ed

function evaluate_ed(basis::TransformedRadial, rr::AbstractVector{<: Real})
   RR = zeros(length(rr), length(basis))
   dRR = zeros(length(rr), length(basis))
   evaluate_ed!(RR, dRR, basis, rr)
   return RR, dRR 
end

function evaluate_ed!(RR, dRR, basis::TransformedRadial, rr::AbstractVector{<: Real})
   xx = ACE1.Transforms.transform.(Ref(basis.trans), rr)
   xx_d = ACE1.Transforms.transform_d.(Ref(basis.trans), rr)
   ee = evaluate.(Ref(basis.envelope), rr, xx)
   ee_d = evaluate_d.(Ref(basis.envelope), rr, xx)
   Polynomials4ML.evaluate_ed!(RR, dRR, basis.basis, xx)
   dRR .= (RR .* ee_d .+ dRR .* ee) .* xx_d
   RR .= RR .* ee
   return nothing 
end



struct NewEvaluator{T, TBR}
   bR::TBR
   bY::CYlmBasis{T}
   bA::PooledSparseProduct{2}
   bAA::SparseSymmProdDAG{Complex{T}}
   params::Vector{T}
end

function NewEvaluator(V::PIPotential)
   bY = Polynomials4ML.CYlmBasis(V.pibasis.basis1p.SH.maxL)
   bR = TransformedRadial(V.pibasis.basis1p.J)
   
   A_spec = V.pibasis.basis1p.spec
   A_spec_new = Tuple{Int,Int}[] 
   for b in A_spec 
      n = b.n 
      iY = Polynomials4ML.index_y(b.l, b.m)
      push!(A_spec_new, (n, iY))
   end
   bA = ACEcore.PooledSparseProduct(A_spec_new)
         
   iAA2iA = V.pibasis.inner[1].iAA2iA 
   orders = V.pibasis.inner[1].orders
   new_spec = [ iAA2iA[i, 1:orders[i]][:] for i = 1:length(orders) ]
   bAA = ACEcore.SparseSymmProdDAG(new_spec, T = ComplexF64)

   return NewEvaluator(bR, bY, bA, bAA, copy(V.coeffs[1]))
end

ACE1.evaluate(V::NewEvaluator, Rs, Zs, z0) = 
      ACE1.evaluate!(nothing, V::NewEvaluator, Rs, Zs, z0)


function real_dot(a::AbstractVector{T}, b::AbstractVector) where {T <: Real}
   val = zero(T) 
   @assert length(a) == length(b) 
   @simd ivdep for i = 1:length(a)
      @inbounds val += a[i] * real(b[i])
   end
   return val 
end

function ACE1.evaluate!(tmp, V::NewEvaluator, Rs, Zs, z0)
   Rn = evaluate(V.bR, norm.(Rs))
   Ylm = Polynomials4ML.evaluate(V.bY, Rs)
   A = ACEcore.evalpool(V.bA, (Rn, Ylm))
   AA = ACEcore.evaluate(V.bAA, A)
   val = real_dot(V.params, AA)
   
   Polynomials4ML.release!(Ylm)
   ACEcore.release!(A)
   ACEcore.release!(AA)

   return val 
end


function ACE1.evaluate_d!(dEs, tmp, V::NewEvaluator, Rs, Zs, z0)
   # the embeddings are differentiated in forward-mode 
   Rn, Rn_d = evaluate_ed(V.bR, norm.(Rs))
   Ylm, Ylm_d = Polynomials4ML.evaluate_ed(V.bY, Rs)

   # the rest in backward mode
   # first the forward pass 
   A = ACEcore.evalpool(V.bA, (Rn, Ylm))
   AA = ACEcore.evaluate(V.bAA, A)
   val = real_dot(V.params, AA)

   # and the backward pass 
   # [∂AA] := ∂V / ∂AA = params 
   ∂AA = V.params    
   # ∂V / ∂A = [∂AA] * ∂AA / ∂A
   T∂A = promote_type(eltype(∂AA), eltype(A))
   ∂A = zeros(T∂A, length(A))
   ACEcore.pullback_arg!(∂A, ∂AA, V.bAA, AA)
   # ∂V / ∂P = [∂A] * ∂A / ∂P
   ∂Rn, ∂Ylm = ACEcore._pullback_evalpool(∂A, V.bA, (Rn, Ylm))

   # now we have the pullback all the way to the embeddings and 
   # can now combine this with the forward-mode derivatives 
   @assert length(dEs) >= length(Rs)
   # TODO: need more asserts once we move this to @inbounds 

   fill!(dEs, zero(eltype(dEs)))
   for n = 1:size(Rn, 2)
      for i = 1:length(Rs)
         rr = Rs[i]
         r = norm(rr)
         dEs[i] += (Rn_d[i, n] * real(∂Rn[i, n]) / r) * rr
      end
   end
   for lm = 1:size(Ylm, 2)
      for i = 1:length(Rs)
         dEs[i] += real(Ylm_d[i, lm] * ∂Ylm[i, lm])
      end
   end

   Polynomials4ML.release!(Ylm)
   Polynomials4ML.release!(Ylm_d)
   ACEcore.release!(A)
   ACEcore.release!(AA)
   ACEcore.release!(∂Rn)
   ACEcore.release!(∂Ylm)

   return dEs
end



end
