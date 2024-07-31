module Purify

using ACE1
using ACE1x

using ACE1x: getspeclenlist, spec2col, insertspect!

using SparseArrays: sparse, spzeros, SparseVector, dropzeros
using RepLieGroups.O3: ClebschGordan

const NLMZ = ACE1.RPI.PSH1pBasisFcn

function pureRPIBasis(basis::ACE1.RPIBasis; remove = 0)

   corord = maximum(length.(get_nl(basis)))
   @assert corord >= 2

   # clean zeros for sanity and prevent useless basis being purified
   basis = ACE1.RPI.remove_zeros(basis)
   basis = ACE1._cleanup(basis)

   pin, pcut = basis.pibasis.basis1p.J.J.pl, basis.pibasis.basis1p.J.J.pr
   ninc = (pin + pcut) * (corord - 1)
   zList = basis.pibasis.zlist.list
   species = sort([list.z for list in zList])

   # get the original C_sym for each atom
   C_sym_list = deepcopy(basis.A2Bmaps)
   # remove zeros that generated for some reason...
   C_sym_list = ntuple(i -> dropzeros(C_sym_list[i]), length(C_sym_list))


   # setting correpsonding basis to 0 if we want to remove some basis
   for C_sym in C_sym_list
       for idx = 1:size(C_sym, 1)
           iAA = findlast(C_sym[idx, :] .!= 0)
           spec_iAA = ACE1.get_basis_spec(basis.pibasis, 1, iAA).oneps
           if length(spec_iAA) <= remove
               C_sym[idx, :] .= 0.0
           end
       end 
   end
   
   # get spec for each atom
   spec_list = [ACE1.get_basis_spec(basis.pibasis, i) for i in eachindex(zList)]
   maxn_list = []
   maxl_list = []
   for spec in spec_list
       push!(maxn_list, maximum( maximum(b.n for b in bb.oneps) for bb in spec))
       push!(maxl_list, maximum( sum(b.l for b in bb.oneps) for bb in spec))
   end
   maxn = maximum(maxn_list)
   maxl = maximum(maxl_list)
   Ydim = length(ACE1.SphericalHarmonics.SHBasis(maxl))


   # === get extended b1p, spec1p, Rn_coef and purification operator that map extended impure basis to pure basis
   # get extended polynomial basis
   Rn = basis.pibasis.basis1p.J

   if Rn.trans isa PolyTransform
      Rn_x = ACE1.OrthPolys.transformed_jacobi(maxn+ninc, Rn.trans, Rn.ru, Rn.rl; pcut = pcut, pin = pin)
   elseif Rn.trans isa Transforms.MultiTransform
      Rn_x = ACE1.transformed_jacobi(maxn+ninc, Rn.trans; pin = pin, pcut = pcut)
   else
      @error("Please make sure Rn.trans is a MultiTransform or PolyTransform")
   end

   # get NN2b for Rn_prod_coeffs by using the extended basis
   NN2b = ACE1x.getNN2b(maxn, ninc, pin, pcut)
   Rn_coef = ACE1x.Rn_prod_coeffs(Rn_x.J, NN2b)

   # === 
   # Define extended 1 particle basis in ACEcore, and convert it to ACE1 format
   spec1p_x_core = [(z, i, y) for z in [list.z for list in zList] for i = 1:(maxn + ninc) for y = 1:Ydim]
   spec1p_x = ACE1x.getACE1Spec1p_multi(spec1p_x_core)

   # TODO: make this construction minimal
   b1p_x = BasicPSH1pBasis(Rn_x, basis.pibasis.basis1p.zlist, spec1p_x)
   
   # we only need spec here, spec_x is constructed in generalImpure2PureMap3D_env_test since we only know what we need to evaluate extra in that function
   # getting spec_core in terms of index in extended spec1p_x_core, here we assume multi species so we have a spec_core for each 
   # of the species

   # A2Bmaps <- C_sym * P * C_pure are constructed individually for each atom
   spec_core_list = ACE1x.getACEcoreSpec_1px_multi(basis, spec1p_x_core, zList)
   
   # now for each of the species, we get the transformation, and also the extended basis
   newA2Bmap_list = []
   spec_x_list = []
   cg = ClebschGordan()
   for i in eachindex(zList)
      # get spec and spec_core respectively, they should be equivalent
      spec = ACE1.get_basis_spec(basis.pibasis, i)
      spec_core = spec_core_list[i]

      spec3b_core = spec_core[length.(spec_core) .== 2]
      Cnn_all = Dict{Vector{Int64}, SparseVector{Float64, Int64}}()
      updateCnn_all!(Cnn_all, Rn_coef, spec1p_x_core, spec3b_core, cg)
      C_pure, spec_x_core, pure_spec = getPurifyOpCCS(Cnn_all, Rn_coef, spec_core, spec1p_x_core, corord; cg = cg)
      # get spec_x in terms of ACE1 spec
      spec_x = ACE1x.getACE1Spec_multi(spec_x_core, spec1p_x_core, species[i])
      # a spec_x_list, each elements inside a tuple correpsonding to a species
      push!(spec_x_list, spec_x)
      
      # get the projection map
      inv_spec_x = Dict([ key => idx for (idx, key) in enumerate(spec_x) ]...)

      P = spzeros(length(spec), length(spec_x))
      for i in eachindex(spec)
          P[i, inv_spec_x[spec[i]]] = 1.0
      end
      # construct newA2Bmap, C_pure : impure_x -> pure_x, P: pure_x -> pure, C_sym: pure -> pure_sym
      newA2Bmap = C_sym_list[i] * P * C_pure
      # newA2Bmap = P * C_pure
      push!(newA2Bmap_list, newA2Bmap)
   end
  
   # get pibasis from spec
   pibasis_x = ACE1.pibasis_from_specs(b1p_x, Tuple(spec_x_list))

   # symmetrize it
   pure_rpibasis = RPIBasis(pibasis_x, Tuple(newA2Bmap_list), basis.Bz0inds)

   # delete zero basis functions
   pure_rpibasis = ACE1.RPI.remove_zeros(pure_rpibasis)

   # and finally clean up
   pure_rpibasis = ACE1._cleanup(pure_rpibasis; tol = 1e-12)

   return pure_rpibasis
end


"""
param: Cnn_all :: Dict{Vector{Int}, SparseVector{Float64, Int64}}(), coefficient of order ≤ 2 basis in fitting
param: Pnn_all :: Dict{Vector{Int}, SparseVector{Float64, Int64}}(), coefficient of radial basis for expanding with CG coeffs
param: spec_core :: Vector{Vector{Int}}}, specification of ACE basis in ACEcore style
param: spec1p :: Vector{Tuple{Int}}}, specification of 1 particle basis
param: Remove :: Integer, all basis of order ≤ Remove will be purified

Return: order_C :: Matrix, a transformation matrix for the purification
Return: spec_x_order :: Vector{Vector{Int}}, extended specification in ACEcore style
Return: pure_spec :: Vector{Vector{Int}}, pure basis that has to be evaluated
"""
function getPurifyOpCCS(Cnn_all::Dict, Pnn_all::Dict, spec_core::Vector{Vector{Int}}, spec1p::Vector{Tuple{Int16, Int, Int}}, Remove::Integer; cg = ClebschGordan())

   old_spec = deepcopy(spec_core)
   spec = deepcopy(spec_core)
   spec_len_list = getspeclenlist(old_spec)
   # for each order
   for Remove_ν = 3:Remove
      for Recursive_ν in reverse(3:Remove_ν)
        # we first have to remove the current order
         i = sum(spec_len_list[1:Recursive_ν-1]) + 1
         while (i <= sum(spec_len_list[1:Recursive_ν]))
            # adjusting coefficent for the term \matcal{A}_{k1...kN} * A_{k_{N+1}}
            # first we get the coefficient corresponding to purified basis of order ν - 1
            _target = spec[i][1:end - 1]
            if spec2col(_target, spec) == -1
               insertspect!(spec, _target)
               i += 1
               # update info
               spec_len_list[length(_target)] += 1
               if length(_target) >= 2
                  spec3b = spec[length.(spec) .== 2]
                  updateCnn_all!(Cnn_all, Pnn_all, spec1p, spec3b, cg)
               end
            end
            
            # adjusting coefficent for terms Σ^{ν - 1}_{β = 1} P^{κ} \mathcal{A}
            # update Cnn_all if they are not in dict, but we don't need to purify that since we only need the coefficients and its nnz
            # to add basis in the next step
            for j = 1:Recursive_ν - 1
               _target = Int64[spec[i][j], spec[i][Recursive_ν]]
               if spec2col(_target, spec) == -1
                  spec3b = spec[length.(spec) .== 2]
                  updateCnn_all!(Cnn_all, Pnn_all, spec1p, vcat(spec3b, [_target]), cg)
               end
            end
            # now this is the actual pure basis that we need to expand 
            P_κ_list  = [Cnn_all[[spec[i][j], spec[i][Recursive_ν]]] for j = 1:Recursive_ν - 1]
            for (idx, P_κ) in enumerate(P_κ_list)
               for k = 1:length(P_κ.nzind) # for each kappa, P_κ.nzind[k] == κ in the sum
                  # first we get the coefficient corresponding to the \mathcal{A}
                  pureA_spec = Int64[spec[i][r] for r = 1:Recursive_ν-1 if r != idx] # r!=idx since κ runs through the 'idx' the coordinate
                  push!(pureA_spec, P_κ.nzind[k]) # add κ into the sum
                  sort!(pureA_spec)
                  if spec2col(pureA_spec, spec) == -1
                     # add an extra basis to evalute, that also require purification
                     insertspect!(spec, pureA_spec)
                     i += 1
                     # update info
                     spec_len_list[length(pureA_spec)] += 1
                     if length(pureA_spec) == 2
                        spec3b = spec[length.(spec) .== 2]
                        updateCnn_all!(Cnn_all, Pnn_all, spec1p, spec3b, cg)
                     end
                  end
               end
            end
            # update info, simply for sanity and this is not necessary
            spec_len_list = getspeclenlist(spec)
            spec3b = spec[length.(spec) .== 2]
            updateCnn_all!(Cnn_all, Pnn_all, spec1p, spec3b, cg)
            i += 1
         end
      end
   end
   
   # update spec len list
   pure_spec = deepcopy(spec)

   S = length(spec)
   hmap = Dict{Int, Vector{Tuple{Int, Float64}}}(i => [] for i = 1:S)
   # from here no more extra pure basis should be inserted into the spec
   # println("Number of basis needed to be purified: ", spec_len_list)

   # corresponding to 2 and 3 body basis
   for i = 1:sum(spec_len_list[1:2])
      push!(hmap[i], (i, 1.0))
   end

   # Base case, adjust coefficient for 3 body (ν = 2)
   for i = spec_len_list[1] + 1:sum(spec_len_list[1:2])
      pnn = Cnn_all[spec[i]]
      for k = 1:length(pnn.nzind)
         if !([pnn.nzind[k]] in spec)
            # add an extra basis to evalute, but it does not require purification
            push!(spec, [pnn.nzind[k]])
         end
         push!(hmap[i], (spec2col([pnn.nzind[k]], spec), -pnn.nzval[k]))
      end
   end
   
   # for each order
   for ν = 3:Remove
      # for each of basis of order ν
      @inbounds begin
         for i = sum(spec_len_list[1:ν-1]) + 1:sum(spec_len_list[1:ν])
            # adjusting coefficent for the term \matcal{A}_{k1...kN} * A_{N+1}
            # first we get the coefficient corresponding to purified basis of order ν - 1
            _target = spec2col(spec[i][1:end - 1], spec)
            last_ip2pmap = hmap[_target]
            
            for k in eachindex(last_ip2pmap) # for each of the coefficient in last_ip2pmap, might be repeated
               # first we get the corresponing specification correpsonding to (spec of last_ip2pmap[i], spec[end])
               # target_spec = [t for t in spec[last_ip2pmap[1][k]]]
               target_spec = [t for t in spec[last_ip2pmap[k][1]]]
               push!(target_spec, spec[i][end])
               sort!(target_spec)
               if !(target_spec in spec)
                  # add an extra basis to evalute, but it does not require purification
                  push!(spec, target_spec)
               end
               push!(hmap[i], (spec2col(target_spec, spec), last_ip2pmap[k][2]))
            end

            # adjusting coefficent for terms Σ^{ν -1}_{β = 1} P^{κ} \mathcal{A}
            P_κ_list  = [Cnn_all[[spec[i][j], spec[i][ν]]] for j = 1:ν - 1]
            for (idx, P_κ) in enumerate(P_κ_list)
               for k = 1:length(P_κ.nzind) # for each kappa, P_κ.nzind[k] == κ in the sum
                  # first we get the coefficient corresponding to the \mathcal{A}
                  pureA_spec = [spec[i][r] for r = 1:ν-1 if r != idx] # r!=idx since κ runs through the 'idx' the coordinate
                  push!(pureA_spec, P_κ.nzind[k]) # add κ into the sum
                  sort!(pureA_spec)
                  _target = spec2col(pureA_spec, spec)
                  last_ip2pmap = hmap[_target]
                  for m in eachindex(last_ip2pmap)
                     push!(hmap[i], (last_ip2pmap[m][1], -P_κ.nzval[k] * last_ip2pmap[m][2]))
                  end
               end
            end
         end
         # println("Done order = ", ν)
      end
   end

   spec_x_order = deepcopy(pure_spec)
   for i = length(pure_spec) + 1 : length(spec)
      insertspect!(spec_x_order, spec[i])
   end

   inv_spec_x_ordered = Dict{Vector{Int64}, Int64}([ key => idx for (idx, key) in enumerate(spec_x_order) ]...)
   inv_spec = Dict{Vector{Int64}, Int64}([ key => idx for (idx, key) in enumerate(spec) ]...)

   newhmap = Dict{Int, Vector{Tuple{Int, Float64}}}()

   for i in eachindex(spec_x_order)
      if spec_x_order[i] in pure_spec
         prev_row = inv_spec[spec_x_order[i]]
         newhmap[i] = [ (inv_spec_x_ordered[spec[j]], val) for (j, val) in hmap[prev_row]]
      else
         newhmap[i] = [(i, 1.0)]
      end
   end

   Irow, Jcol, vals = Int[], Int[], Float64[]
   for i in keys(newhmap)
      for j in newhmap[i]
         # push!.( (Irow, Jcol, vals), (i, j[1], j[2]) )
         push!(Irow, i)
         push!(Jcol, j[1])
         push!(vals, j[2])
      end
   end
   Nk = length(spec)
   C = sparse(Irow, Jcol, vals, Nk, Nk)
   return C, spec_x_order, pure_spec
end




"""
Add elements in Cnn_all if needed, which a Dict containing the coefficients of product of spec3b. This function modifies Cnn_all directly.

param: Cnn_all :: Dict{Vector{Int}, SparseVector{Float64, Int64}}(), coefficient of order ≤ 2 basis in fitting
param: Pnn_all :: Dict{Vector{Int}, SparseVector{Float64, Int64}}(), coefficient of radial basis for expanding with CG coeffs
param: spec1p :: Vector{Tuple{Int}}}, specification of 1 particle basis
param: spec3b :: Vector{Tuple{Int}}}, specification of 3 body basis

"""
function updateCnn_all!(Cnn_all::Dict, Pnn_all::Dict, spec1p, spec3b, cg)
   for nlm in spec3b
      if !(nlm in keys(Cnn_all))
         niceidx1, niceidx2 = spec1p[nlm[1]], spec1p[nlm[2]]
         z1, z2 = niceidx1[1], niceidx2[1]
         if z1 == z2
            # get the coefficient Pκ from Pnn_all
            Pκk1k2 = Pnn_all[[niceidx1[2], niceidx2[2]]]      
            (l1, m1), (l2, m2) = ACE1x.idx2lm(niceidx1[3]), ACE1x.idx2lm(niceidx2[3])
            
            M = m1 + m2
            # expand as new coefficients, C_nn should be a vector of length = number of 1p, each index correpsonding to one (κ, c, γ) in spec1p
            C_nn = zeros(length(spec1p))
            for κ = 1:length(Pκk1k2.nzind)
               for L = abs(M):(l1 + l2)
                     C_nn[ACE1x.LinearSearch((z1, Pκk1k2.nzind[κ], ACE1x.index_y(L, M)), spec1p)] = Pκk1k2.nzval[κ] * sqrt( (2*l1+1)*(2*l2+1) / (4 * π * (2*L+1)) ) *
                     cg(l1,  0, l2,  0, L, 0) *
                     cg(l1, m1, l2, m2, L, M)
               end
            end
            # assign to store as sparse matrix in a Dictionary
            Cnn_all[nlm] = sparse(C_nn)
         else
            Cnn_all[nlm] = spzeros(length(spec1p))
         end
      end
   end
end

end # end of module Purify