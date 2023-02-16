
using ACE1
using ACE1.Testing: print_tf, println_slim
using ACE1.Transforms: AffineT, agnesi_transform, multitransform
using ACE1: transformed_jacobi, transformed_jacobi_env
using ACE1x
using JuLIP
using LinearAlgebra: Diagonal
using Statistics: mean
using Test 

## ------- basic parameters 

elements = [:Cs, :Pb, :Br] 

# the following is a dumb way to pick the r0 values. The right way is to 
# look at the bond-lengths and at the RDFs and come up with a sensible heuristic 
# from those. 

# atomic radii according to Google
# r0_el = Dict(:Cs => 3.43, :Pb => 1.8, :Br => 1.85)
# I'm correcting Cs since rnn(Cs) ≈ 1.5 * r0(Cs), but this is pure handwaving ... 
r0_el = Dict( [s => ACE1x.get_bond_len(s)/2 for s in elements ]... )
r0 = Dict( [(s1, s2) => (r0_el[s1] + r0_el[s2]) 
            for s1 in elements, s2 in elements]... )

pin = 2
pcut = 2
cor_order = 3
maxdeg = 9
maxdeg_pair = maxdeg 
r_cut_ACE = 8.0
r_cut_pair = r_cut_ACE

## ------ Many-Body Basis 
# transform for many-body basis. Note that rin = 0.0. If this 
# doesn't work, then I need to make some changes to ACE1.jl
# we should try with other parameters, but for now most combinations won't 
# work due to a gap in the implementation of the Agnesi(p, q) transform.
agnesi_p, agnesi_q = 2, 4
transforms = Dict([ (s1, s2) => agnesi_transform(r0[s1, s2], agnesi_p, agnesi_q) 
                    for s1 in elements, s2 in elements]... )
trans_ace = multitransform(transforms; rin = 0.0, rcut = r_cut_ACE)
ninc = (pcut + pin) * (cor_order-1)
maxn = maxdeg + ninc 
Pr_ace = transformed_jacobi(maxn, trans_ace; pcut = pin, pin = pin)

rpibasis = ACE1x.Pure2b.pure2b_basis(species = AtomicNumber.(elements),
                           Rn=Pr_ace, 
                           D=ACE1.RPI.SparsePSHDegree(),
                           maxdeg=maxdeg, 
                           order=cor_order, 
                           delete2b = true)

## -------- Pair Basis 

# the transform for the radial basis should be ok with (1, 4)
ag_p_pair = 1; ag_q_pair = 3
function _r_basis(s1, s2) 
   _env = ACE1.PolyEnvelope(2, r0[(s1, s2)], r_cut_pair)
   _trans = agnesi_transform(r0[s1, s2], ag_p_pair, ag_q_pair) 
   _Pr = transformed_jacobi_env(maxdeg_pair, _trans, _env, r_cut_pair)
   return _Pr 
end

Pr_pair = [ _r_basis(s1, s2) for s1 in elements, s2 in elements ]

pair = PolyPairBasis(Pr_pair, elements)

B = JuLIP.MLIPs.IPSuperBasis([pair, rpibasis]);
basis_length = length(B)

## ------- Automated Version, preliminary tests 
@info("preliminary consistency tests")
kwargs = (elements = elements, 
          order = cor_order, 
          totaldegree = maxdeg, 
          rcut = r_cut_ACE, )

kwargs = ACE1x._clean_args(kwargs)

println_slim(@test ACE1x._get_order(kwargs) == 3)

# read all r0s. 
_r0 = ACE1x._get_all_r0(kwargs)
println_slim(@test _r0 == r0)

ACE1x._get_all_rcut(kwargs)

_trans = ACE1x._transform(kwargs)
println_slim(@test _trans == trans_ace)

_rbasis = ACE1x._radial_basis(kwargs)
println_slim(@test _rbasis == Pr_ace)

_rpibasis = ACE1x.mb_ace_basis(kwargs)
println_slim(@test _rpibasis == rpibasis)

_pairbasis = ACE1x._pair_basis(kwargs)
println_slim(@test _pairbasis == pair)

@info("consistency of final basis to manual construction")
basis = ACE1x.ace_basis(; kwargs...)
println_slim(@test basis == B)
