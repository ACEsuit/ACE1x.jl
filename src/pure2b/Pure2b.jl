
module Pure2b

using ACE1 

using ACE1: SparsePSHDegree, BasicPSH1pBasis, PIBasis, get_basis_spec, 
            PIBasisFcn, pibasis_from_specs, degree 
using ACE1.RPI: RPIBasis            

const NLMZ = ACE1.RPI.PSH1pBasisFcn

function Rn_prod_coeffs()

end

function extended_rpibasis(species, Rn, D, maxdeg, order)
   # compute a first PIBasis                         
   B1p = BasicPSH1pBasis(Rn; species = species, D = D)
   pibasis = PIBasis(B1p, order, D, maxdeg)
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


function pure2b_basis(; species = nothing, Rn = nothing,
                        D = nothing, maxdeg = nothing, order = nothing) 

   # construct the extended PI Basis    
   pibasis_x = extended_rpibasis(species, Rn, D, maxdeg, order)
   
   # symmetrize it 
   rpibasis = RPIBasis(pibasis_x)
   
   # construct the product-Rn coefficients 


   # correct the coupling coefficient matrix / A2B map 


   # return the corrected basis 
   return rpibasis_x 
end


end