
module Evaluator

import ACE1
using ACEcore, JuLIP, Polynomials4ML

using ACEcore: SparseSymmProdDAG, PooledSparseProduct
using LinearAlgebra: dot, norm 
import ACE1: PIPotential
import Polynomials4ML: CYlmBasis, OrthPolyBasis1D3T

struct XEnvelope{T}
   xl::T 
   xr::T 
   pl::Int
   pr::Int
end

evaluate(env::XEnvelope, r, x) = 
      (x - env.xl)^(env.pl) * (env.xr - x)^(env.pr)

struct TransformedRadial{TB, TT, TE}
   basis::TB 
   trans::TT 
   envelope::TE 
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

struct NewEvaluator{T, TBR}
   bR::TBR
   bY::CYlmBasis{T}
   bA::PooledSparseProduct{2}
   bAA::SparseSymmProdDAG{T}
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
   bAA = ACEcore.SparseSymmProdDAG(new_spec)

   return NewEvaluator(bR, bY, bA, bAA, copy(V.coeffs[1]))
end

ACE1.evaluate(V::NewEvaluator, Rs, Zs, z0) = 
      ACE1.evaluate!(nothing, V::NewEvaluator, Rs, Zs, z0)

function ACE1.evaluate!(tmp, V::NewEvaluator, Rs, Zs, z0)
   Rn = evaluate(V.bR, norm.(Rs))
   Ylm = Polynomials4ML.evaluate(V.bY, Rs)
   A = ACEcore.evalpool(V.bA, (Rn, Ylm))
   AA = ACEcore.evaluate(V.bAA, A)
   val = dot(V.params, real.(AA))
   
   Polynomials4ML.release!(Ylm)
   ACEcore.release!(A)
   ACEcore.release!(AA)

   return val 
end

end
