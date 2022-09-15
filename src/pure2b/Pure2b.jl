
module Pure2b

using ACE1 

using ACE1: SparsePSHDegree, BasicPSH1pBasis, PIBasis, get_basis_spec, 
            PIBasisFcn, pibasis_from_specs, degree, 
            rand_radial, order, evaluate
using ACE1.RPI: RPIBasis, RPIFilter   

using LinearAlgebra: qr, norm
using SparseArrays: SparseVector, sparse

const NLMZ = ACE1.RPI.PSH1pBasisFcn

function Rn_prod_coeffs(Rn, NN; tol=1e-12)
   rr = [ rand_radial(Rn) for _=1:max(10*length(Rn), length(Rn)^2) ]
   RR = zeros(length(rr), length(Rn))
   for i = 1:length(rr) 
      RR[i, :] = evaluate(Rn, rr[i])
   end
   qrF = qr(RR)
   
   Pnn = Dict{Vector{Int}, SparseVector{Float64, Int64}}()
   
   for nn in NN
      Rnn = RR[:, nn[1]]
      for t = 2:length(nn) 
         Rnn = Rnn .* RR[:, nn[t]]
      end
      p_nn = map(p -> (abs(p) < tol ? 0.0 : p), qrF \ Rnn)
      @assert norm(RR * p_nn - Rnn, Inf) < tol
      Pnn[nn] = sparse(p_nn)
   end
   
   return Pnn 
end

function get_NN(pibasis)
   NN = Vector{Int}[] 
   spec_x = get_basis_spec(pibasis, 1)
   N00Z = Dict{NLMZ, Int}()
   
   for (ibb, bb) in enumerate(spec_x)
      if (order(bb) > 1) && # we don't need to do anything about 1-correlations
            all(b.l == 0 for b in bb.oneps) &&  # l ≂̸ becomes pure after symmetrisation
            all(b.z == bb.oneps[1].z for b in bb.oneps)  # all z must be the same
         nn = [ b.n for b in bb.oneps ]      
         push!(NN, nn)
      end
   
      if order(bb) == 1 
         b = bb.oneps[1] 
         if b.l == 0 
            N00Z[b] = ibb 
         end
      end
   end

   return NN, N00Z 
end 

function extended_rpibasis(species, Rn, D, maxdeg, order, 
                           constants::Bool)
   # compute a first PIBasis                         
   B1p = BasicPSH1pBasis(Rn; species = species, D = D)
   pibasis = PIBasis(B1p, order, D, maxdeg; filter = RPIFilter(constants))
   # compute the largest n occuring in the standard PI basis
   spec = get_basis_spec(pibasis, 1)
   maxn = maximum( maximum(b.n for b in bb.oneps) for bb in spec )
   # check that the maxn in Rn is ok 
   maxn_x = maxn + (Rn.J.pl + Rn.J.pr) * (order - 1)
   @assert length(Rn) >= maxn_x 

   # get the basis spec and extend it as required for the pure symmetrisation 
   spec = get_basis_spec(pibasis, 1) 
   orders = [ length(bb.oneps) for bb in spec ]
   N1 = maximum(findall(orders .== 1))
   @assert all(isequal(1), orders[1:N1])
   new_spec = [ PIBasisFcn(z0, (NLMZ(n, 0, 0, AtomicNumber(:X)), ) )
                for z0 in species for n = maxn+1:maxn_x ]
   spec_x = [ spec[1:N1]; new_spec; spec[N1+1:end] ]

   # TODO: fix this part of the code for multiple species 
   #       and hence multiple possibly different specs 

   # generate the new pi basis with the extended 2b contribution 
   spec1_x = filter( b -> (degree(D, b) <= maxdeg) || 
                         (b.n <= maxn_x && b.l == 0), B1p.spec )
   B1p_x = BasicPSH1pBasis(Rn, B1p.zlist, spec1_x)
   pibasis_x = pibasis_from_specs(B1p_x, (spec_x,) )
   for iz = 1:length(species)
      @assert get_basis_spec(pibasis_x, iz) == spec_x
   end

   return pibasis_x
end

function correct_coupling_coeffs!(rpibasis)

   Rn = rpibasis.pibasis.basis1p.J
   NN, N00Z = get_NN(rpibasis.pibasis)
   Pnn = Rn_prod_coeffs(Rn, NN)
   spec_x = get_basis_spec(rpibasis.pibasis, 1)

   CC = rpibasis.A2Bmaps[1]
   I2b = Int[] 
   
   for idx = 1:size(CC, 1)
      iAA = findfirst(CC[idx, :] .!= 0) 
      if isnothing(iAA); continue; end 
      nn = [b.n for b in spec_x[iAA].oneps] 
      zz = [b.z for b in spec_x[iAA].oneps] 
      ll = [b.l for b in spec_x[iAA].oneps] 
      if all(ll .== 0) && all(zz .== zz[1]) && length(nn) > 1
         # add the entries P^nn_n1 t0 CC in the column (z, n1, 0, 0)
         z = zz[1] 
         p_nn = Pnn[nn]
         for (n1, p_nn_n1) in zip(p_nn.nzind, p_nn.nzval)
            b = NLMZ(n1, 0, 0, z)
            iAA_b = N00Z[b]
            CC[iAA_b] = p_nn_n1 
         end
      end
      if length(nn) == 1 
         push!(I2b, idx)
         # push!(spec_2b, (nn[1], ll[1], zz[1]))
      end
   end   
   return I2b 
end

function pure2b_basis(; species = nothing, Rn = nothing,
                        D = nothing, maxdeg = nothing, order = nothing, 
                        constants = false, 
                        delete2b = false)

   # construct the extended PI Basis    
   pibasis_x = extended_rpibasis(species, Rn, D, maxdeg, order, 
                                 constants)
   
   # symmetrize it 
   rpibasis = RPIBasis(pibasis_x)
   
   # correct the coupling coefficient matrix / A2B map 
   I2b = correct_coupling_coeffs!(rpibasis)

   return rpibasis
end


end