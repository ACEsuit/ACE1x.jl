
module Pure2b

using ACE1 

using ACE1: SparsePSHDegree, BasicPSH1pBasis, PIBasis, get_basis_spec, 
            PIBasisFcn, pibasis_from_specs, degree, 
            rand_radial, order, evaluate, i2z, z2i, numz
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
   N00Z = [ Dict{NLMZ, Int}() for iz0 = 1:length(pibasis.inner) ]
   
   for iz0 = 1:length(pibasis.inner)
      spec_x = get_basis_spec(pibasis, iz0)
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
               N00Z[iz0][b] = ibb 
            end
         end
      end
   end 

   return unique(NN), N00Z
end 

function extended_rpibasis(species, Rn, D, maxdeg, order, 
                           constants::Bool)
   # compute a first PIBasis                         
   B1p = BasicPSH1pBasis(Rn; species = species, D = D)
   pibasis = PIBasis(B1p, order, D, maxdeg; filter = RPIFilter(constants))
   _i2z(i::Integer) = i2z(pibasis, i)

   spec_x = Dict{Any, Any}([z => nothing for z in species]...)
   maxn_x_total = 0 

   for iz0 = 1:length(pibasis.inner)
      z0 = _i2z(iz0)
      # compute the largest n occuring in the standard PI basis
      spec = get_basis_spec(pibasis, iz0)
      maxn = Dict([z => 0 for z in species]...)
      for bb in spec, b in bb.oneps 
         maxn[b.z] = max(maxn[b.z], b.n)
      end
      # check that the maxn in Rn is ok 
      ninc = (Rn.J.pl + Rn.J.pr) * (order - 1)
      maxn_x = Dict([z => maxn[z] + ninc for z in species]...)
      max_maxn_x = maximum( maxn_x[z] for z in species )
      if !(length(Rn) >= max_maxn_x)
         @info("""Rn is too short: length(Rn) = $(length(Rn)), 
                  need at least maxn = $(maxn_x). (cf z0 = $z0)""")
         error("length(Rn) < maxn + (pl + pr) * (order - 1)")
      end
      maxn_x_total = max(maxn_x_total, maxn_x[z0])

      # get the basis spec and extend it as required for the pure symmetrisation 
      orders = [ length(bb.oneps) for bb in spec ]
      N1 = maximum(findall(orders .== 1)) 
      # check the assumption that the order-1 basis functions all come before N1
      @assert all(isequal(1), orders[1:N1])   
      new_spec = [ PIBasisFcn(z0, (NLMZ(n, 0, 0, _i2z(iz)), ))
                   for iz in 1:length(species) for n = 1:maxn_x[_i2z(iz)] ]
      spec_x[z0] = [ new_spec; spec[N1+1:end] ]
   end 

   # generate the new pi basis with the extended 2b contribution 
   @assert B1p isa BasicPSH1pBasis
   spec1_x = filter( b -> (degree(D, b, nothing) <= maxdeg) || 
                         (b.n <= maxn_x_total && b.l == 0), 
                     B1p.spec )
   B1p_x = BasicPSH1pBasis(Rn, B1p.zlist, spec1_x)
   spec_x_list = ntuple(iz -> spec_x[_i2z(iz)], length(species))
   pibasis_x = pibasis_from_specs(B1p_x, spec_x_list )
   for iz = 1:length(species)
      @assert get_basis_spec(pibasis_x, iz) == spec_x[_i2z(iz)]
   end

   return pibasis_x
end


function correct_coupling_coeffs!(rpibasis)

   Rn = rpibasis.pibasis.basis1p.J
   NN, N00Z = get_NN(rpibasis.pibasis)
   Pnn = Rn_prod_coeffs(Rn.J, NN)
   
   NZ = numz(rpibasis)
   _i2z(i) = i2z(rpibasis, i)
   _z2i(z) = z2i(rpibasis, z)
   
   for iz0 in 1:NZ
      z0 = _i2z(iz0) 
      spec_x = get_basis_spec(rpibasis.pibasis, iz0)

      # to get the nnll_factors we need to evaluate the basis before messing 
      # with the coupling coefficients 
      Iz0 = rpibasis.Bz0inds[iz0]
      # rr = [ rand_radial(Rn, z, z0) for _=1:3 ]
      len_rr = 3 
      rr = Dict([ z => [ rand_radial(Rn, z, z0) for _=1:len_rr ] 
                  for z in _i2z.(1:NZ) ]... )
      rr_B =  [ evaluate(rpibasis, [JVecF(rr[z][ir], 0, 0),],  [z,], z0)[Iz0]
                for ir = 1:len_rr, z in _i2z.(1:NZ) ]
      # rr_Rn = [ evaluate(Rn, r) for r in rr ]
      rr_Rn = [ evaluate(Rn, rr[z][ir], z, z0) 
                for ir = 1:len_rr, z in _i2z.(1:NZ) ]

      CC = rpibasis.A2Bmaps[iz0]
   
      for idx = 1:size(CC, 1)  # loop over all BB basis functions for z0 
         iAA = findfirst(CC[idx, :] .!= 0) 
         if isnothing(iAA); continue; end   # empty basis function ?!?!
         nn = [b.n for b in spec_x[iAA].oneps]
         zz = [b.z for b in spec_x[iAA].oneps] 
         ll = [b.l for b in spec_x[iAA].oneps] 

         if all(zz .== zz[1]) && length(nn) > 1
            z = zz[1] 
            iz = _z2i(z)

            # to get the multiplicative factor from the Ylms and from the 
            # rescaled coupling coefficients we need the product of Rn: 
            # we compute the normalization constant several times and check 
            # that it is the same every time. 
            prod_Rn = [ prod(rr_Rn[ir, iz][nn[a]] for a = 1:length(nn))
                        for ir = 1:len_rr ]
            Fac = [ rr_B[ir, iz][idx] / prod_Rn[ir] * (2*sqrt(pi))
                    for ir = 1:len_rr ]
            if !all(f ≈ Fac[1] for f in Fac) 
               @show Fac 
               error("Ylm factor inconsistent - something is wrong.")
            end
            c_nnll = Fac[1]

            # add the entries P^nn_n1 t0 CC in the column (z, n1, 0, 0)
            # TODO: I am making an assumptions here: 
            #       The (z n 0 0) take the form Rn(r) Y00(rhat) 
            #       with Y00 = 1 / sqrt(4 pi) (cf factor above)
            # This needs to be checked and fixed somehow. 

            p_nn = Pnn[nn]
            for (n1, p_nn_n1) in zip(p_nn.nzind, p_nn.nzval)
               b = NLMZ(n1, 0, 0, z)
               iAA_b = N00Z[iz0][b]
               CC[idx, iAA_b] -= c_nnll * p_nn_n1 
            end
         end

      end   
   end 

   return nothing 
end

function remove_2b!(rpibasis, delete2b, maxdeg)

   # decide whether to delete all 2b, or just the ones with 
   # too high a degree (note they will remain in the AA basis 
   # and only get deleted from the B basis)
   for iz0 = 1:numz(rpibasis)
      CC = rpibasis.A2Bmaps[iz0]
      for idx = 1:size(CC, 1)
         iAA = findlast(CC[idx, :] .!= 0)
         spec_iAA = get_basis_spec(rpibasis.pibasis, iz0, iAA).oneps
         if length(spec_iAA) == 1
            delete_iAA = delete2b || (spec_iAA[1].n > maxdeg)
            if delete_iAA
               CC[idx, :] .= 0.0
            end
         end
      end
   end

   return nothing 
end


function pure2b_basis(; species = nothing, Rn = nothing,
                        D = nothing, maxdeg = nothing, order = nothing, 
                        constants = false, 
                        delete2b = false)
   # @assert D.chc == 0 
   # @assert D.csp == 1 

   # construct the extended PI Basis    
   pibasis_x = extended_rpibasis(species, Rn, D, maxdeg, order, 
                                 constants)
                              
   # symmetrize it 
   rpibasis = RPIBasis(pibasis_x)
   
   # correct the coupling coefficient matrix / A2B map 
   correct_coupling_coeffs!(rpibasis)

   # remove the 2b 
   #      if delete2b == true then all will be removed. 
   #      if delete2b == false then only the ones with too high degree.
   remove_2b!(rpibasis, delete2b, maxdeg)

   # remove all zero-basis functions that we might have accidentally created. 
   rpibasis = ACE1.RPI.remove_zeros(rpibasis)
   # and finally cleanup the rest of the basis 
   rpibasis = ACE1._cleanup(rpibasis)

   return rpibasis
end


end