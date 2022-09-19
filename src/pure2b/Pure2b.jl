
module Pure2b

using ACE1 

using ACE1: SparsePSHDegree, BasicPSH1pBasis, PIBasis, get_basis_spec, 
            PIBasisFcn, pibasis_from_specs, degree, 
            rand_radial, order, evaluate, i2z, z2i
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
   _i2z(i::Integer) = i2z(pibasis, i)

   spec_x = Dict{Any, Any}([z => nothing for z in species]...)
   maxn_x_total = Dict{Any, Int}([z => 0 for z in species]...)

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
      maxn_x_total[z0] = max(maxn_x_total[z0], maxn_x[z0])

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
   spec1_x = filter( b -> (degree(D, b) <= maxdeg) || 
                         (b.n <= maxn_x_total[b.z] && b.l == 0), B1p.spec )
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
   Pnn = Rn_prod_coeffs(Rn, NN)
   spec_x = get_basis_spec(rpibasis.pibasis, 1)

   # to get the nnll_factors we need to evaluate the basis before messing 
   # with the coupling coefficients 
   zX = AtomicNumber(:X)
   rr = [ rand_radial(Rn) for _=1:3 ] 
   rr_B = [ evaluate(rpibasis, [JVecF(r, 0, 0),],  [zX,], zX) for r in rr ]
   rr_Rn = [ evaluate(Rn, r) for r in rr ] 

   CC = rpibasis.A2Bmaps[1]
   I2b = Int[] 
   
   for idx = 1:size(CC, 1)
      iAA = findfirst(CC[idx, :] .!= 0) 
      if isnothing(iAA); continue; end 
      nn = [b.n for b in spec_x[iAA].oneps]
      zz = [b.z for b in spec_x[iAA].oneps] 
      ll = [b.l for b in spec_x[iAA].oneps] 

      if all(zz .== zz[1]) && length(nn) > 1
         # to get the multiplicative factor from the Ylms and from the 
         # rescaled coupling coefficients we need the product of Rn: 
         # we compute the normalization constant several times and check 
         # that it is the same every time. 
         prod_Rn = [ prod(rr_Rn[i][nn[a]] for a = 1:length(nn))
                     for i = 1:length(rr) ]
         Fac = [ rr_B[i][idx] / prod_Rn[i] * (2*sqrt(pi))
                 for i = 1:length(rr) ]
         @assert all(f ≈ Fac[1] for f in Fac)
         c_nnll = Fac[1] 

         # add the entries P^nn_n1 t0 CC in the column (z, n1, 0, 0)
         # TODO: I am making two assumptions here: 
         #   (i) The (z n 0 0) take the form Rn(r) Y00(rhat) 
         #        with Y00 = 1 / sqrt(4 pi) (cf factor above)
         #   (ii) secondly, I am assuming that the AA basis spec 
         #        starts with the (z n 0 0) basis functions. 
         # Both of these need to be checked and fixed somehow. 
         z = zz[1] 
         p_nn = Pnn[nn]
         for (n1, p_nn_n1) in zip(p_nn.nzind, p_nn.nzval)
            b = NLMZ(n1, 0, 0, z)
            iAA_b = N00Z[b]
            CC[idx, iAA_b] -= c_nnll * p_nn_n1 
         end
      end
      if length(nn) == 1 
         push!(I2b, idx)
         # push!(spec_2b, (nn[1], ll[1], zz[1]))
      end

   end   
   return I2b 
end

function remove_2b!(rpibasis, I2b, delete2b, maxdeg)
   # decide whether to delete all 2b, or just the ones with 
   # too high a degree (note they will remain in the AA basis 
   # and only get deleted from the B basis)
   CC = rpibasis.A2Bmaps[1]

   if delete2b 
      Idel = I2b 
   else 
      degrees = zeros(Int, length(I2b))
      for i = 1:length(I2b)
         idx = I2b[i]
         iAA = findfirst(CC[idx, :] .!= 0)
         n = get_basis_spec(rpibasis.pibasis, 1, iAA).oneps[1].n
         degrees[i] = n 
      end
      Idel = I2b[ degrees .<= maxdeg ]
   end

   CC[Idel, :] .= 0.0 
   return nothing 
end


function pure2b_basis(; species = nothing, Rn = nothing,
                        D = nothing, maxdeg = nothing, order = nothing, 
                        constants = false, 
                        delete2b = false)

   @assert D.chc == 0 
   @assert D.csp == 1 

   # construct the extended PI Basis    
   pibasis_x = extended_rpibasis(species, Rn, D, maxdeg, order, 
                                 constants)
   
   # symmetrize it 
   rpibasis = RPIBasis(pibasis_x)
   
   # correct the coupling coefficient matrix / A2B map 
   I2b = correct_coupling_coeffs!(rpibasis)

   # remove the 2b 
   #      if delete2b == true then all will be removed. 
   #      if delete2b == false then only the ones with too high degree.
   remove_2b!(rpibasis, I2b, delete2b, maxdeg)

   return rpibasis
end


end