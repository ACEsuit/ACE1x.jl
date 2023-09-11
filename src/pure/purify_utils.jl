using RepLieGroups.O3: ClebschGordan
using ACE1: gensparse, rand_radial, PIBasisFcn
using LinearAlgebra: qr, norm
using SparseArrays: SparseVector, sparse


const NLMZ = ACE1.RPI.PSH1pBasisFcn

# ========================================== utils from Polynomials4ML ========================================
"""
`index_y(l,m):`
Return the index into a flat array of real spherical harmonics `Y_lm`
for the given indices `(l,m)`. `Y_lm` are stored in l-major order i.e.
```
	[Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
```
"""
index_y(l::Integer, m::Integer) = m + l + (l*l) + 1

"""
Inverse of `index_y`: given an index into a vector of Ylm values, return the 
`l, m` indices.
"""
function idx2lm(i::Integer) 
	l = floor(Int, sqrt(i-1) + 1e-10)
	m = i - (l + (l*l) + 1)
	return l, m 
end 

##

# ========================================== utils for converting between ACE1 and ACEcore interface ===========================

"""
Get a list of spec in ACEcore style from ACE1 pibasis in spec1p_x_core index corresponding to different Atoms
We assume that the basis are grouping together according to their Atom
"""
function getACEcoreSpec_1px_multi(rpiB, spec1p_x_core, zList)

    old_pibasis = rpiB.pibasis
    old_spec_list = [ACE1.get_basis_spec(old_pibasis, i) for i in eachindex(zList)]
    ac_spec_list = Vector{Vector{Vector{Int64}}}()
    last_zL = 0
    for i in eachindex(zList)
        ac_spec = Vector{Vector{Int64}}()
        old_spec = old_spec_list[i]
        for t in old_spec
            push!(ac_spec, sort([spec2col((znlm.z, znlm.n, index_y(znlm.l, znlm.m)), spec1p_x_core) for znlm in t.oneps]))
        end
        ac_spec = unique(ac_spec)
        # assert that the spec getting from above has the some length as expected
        # @assert length(ac_spec) == (rpiB.Bz0inds[i][end] - rpiB.Bz0inds[i][1] + 1)
        push!(ac_spec_list, ac_spec)
        last_zL += length(old_spec)
    end
    return ac_spec_list
end


"""
Given a target spect (which is a list of index of spec1p) return a list of (n, l, m) correpsonding to index in spect
"""
function spect2nlmz(spect, spec1p)
    n_list = [spec1p[i][2] for i in spect]
    y_list = [spec1p[i][3] for i in spect]
    l_list = [idx2lm(y)[1] for y in y_list]
    m_list = [idx2lm(y)[2] for y in y_list]
    z_list = [spec1p[i][1] for i in spect]
    return [(z_list[k], n_list[k], l_list[k], m_list[k]) for k in eachindex(m_list)]
end


"""
Get spec in ACE1 format for multi species
"""
function getACE1Spec_multi(ac_spec, spec1p_core, center_species)
    zX = AtomicNumber(center_species)
    ace1_spec = Vector{PIBasisFcn{N, ACE1.RPI.PSH1pBasisFcn} where N}()

    for t in ac_spec
        target_nlm = spect2nlmz(t, spec1p_core)
        push!(ace1_spec, PIBasisFcn(zX, Tuple([NLMZ(znlm[2], znlm[3], znlm[4], znlm[1]) for znlm in target_nlm])))
    end

    return ace1_spec
end


"""
Get spec1p in ACE1 format for multi species, this just ignore the species channel.
"""
function getACE1Spec1p_multi(ac_spec1p)
    ace1_spec1p = Vector{ACE1.RPI.PSH1pBasisFcn}()
    for t in ac_spec1p
        n, (l, m) = t[2], idx2lm(t[3])
        push!(ace1_spec1p, NLMZ(n, l, m, 0))
    end
    unique!(ace1_spec1p)
    return ace1_spec1p
end

##


# ========================================== utils for purification ===========================
"""
For evaluating coefficients of product of polynomials
"""
function Rn_prod_coeffs(Rn, NN; tol=1e-12)
    NN23b = NN[length.(NN) .<= 2]
    rr = [ rand_radial(Rn) for _=1:max(10*length(Rn), length(Rn)^2) ]

    RR = zeros(length(rr), length(Rn))
    for i = 1:length(rr) 
       RR[i, :] = ACE1.evaluate(Rn, rr[i])
    end
    
    qrF = qr(RR)
    
    Pnn = Dict{Vector{Int}, SparseVector{Float64, Int64}}()
    
    for nn in NN23b
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


"""
Getting NN2b for the polynomial
"""
function getNN2b(maxn, ninc, pin, pcut)
    spec1p_poly = [i for i = 1:(maxn + ninc)]
    tup2b = vv -> [ spec1p_poly[v] for v in vv[vv .> 0]  ]
    admissible = bb -> (length(bb) == 0) || (sum(b[1] - 1 for b in bb ) < maxn + ninc - pin - pcut)

    # only product basis up to order = 2, use to get Pnn_all
    # NN2b = gensparse(; NU = 2, tup2b = tup2b, admissible = admissible, minvv = fill(0, 2), maxvv = fill(length(spec1p_poly), 2), ordered = true)
    NN2b = gensparse(2; admissible = admissible, tup2b = tup2b, ordered = true, maxν = length(spec1p_poly))
    NN2b = [ vv[vv .> 0] for vv in NN2b if !(isempty(vv[vv .> 0]))]
    return NN2b
end


# ============================ general utils =====================================

"""
Linear search, NN can be spec or spec1p
"""
function spec2col(NNi, NN)
   return LinearSearch(NNi, NN)::Int64
end

"""
Linear search, for getting index for CG coefficient expansion
"""
function _getκylmIdx(spec1p, κ, idxy)
   return LinearSearch((κ, idxy), spec1p)::Int64
end


function LinearSearch(NNi, NN)
   for k in eachindex(NN)
      if NN[k] == NNi
         return k
      end
   end
   return -1
end


""" 
Return true if the specification s1 < s2, else return false.
"""
function checkord(s1, s2)
   if length(s1) < length(s2)
      return true
   end
   if length(s1) > length(s2)
      return false
   end   
   for l in eachindex(s1)
      if s1[l] < s2[l]
         return true
      elseif s1[l] > s2[l]
         return false
      end
   end
   return true
end

"""
Insert the spect (target spec) to spec.
"""
function insertspect!(spec, spect)
   pos = 1
   while(checkord(spec[pos], spect) && pos < length(spec))
      pos += 1
   end
   if pos < length(spec)
      insert!(spec, pos, spect)   
   elseif pos == length(spec)
      push!(spec, spect)
   else
      @error("There is some problem with the function insertspect!, please check")
   end
end

"""
Return spec_len_list, which is an array with array[i] being number of basis of i th order
"""
function getspeclenlist(spec::Vector{Vector{Int}})
   max_ord = maximum(length.(spec))
   spec_len_list = zeros(Int64, max_ord)
   for t in spec
      spec_len_list[length(t)] += 1
   end
   return spec_len_list
end