
import YAML, ACE1 
using NamedTupleTools: namedtuple
using JuLIP: AtomicNumber, rnn 
using ACE1.Transforms: agnesi_transform, multitransform
using ACE1.OrthPolys: transformed_jacobi, transformed_jacobi_env

# -------------- Bond-length heuristics 

_lengthscales_path = joinpath(@__DIR__, "..", "data", 
                              "length_scales_VASP_auto_length_scales.yaml")
_lengthscales = YAML.load_file(_lengthscales_path)

get_bond_len(s::Symbol) = get_bond_len(AtomicNumber(s))
get_bond_len(z::AtomicNumber) = get_bond_len(convert(Int, z)) 

function get_bond_len(z::Integer) 
   if haskey(_lengthscales, z)
      return _lengthscales[z]["bond_len"][1]
   elseif rnn(AtomicNumber(z)) > 0 
      return rnn(AtomicNumber(z)) 
   end 
   error("No typical bond length for atomic number $z is known. Please specify manually.")
end

get_r0(z1, z2) = (get_bond_len(z1) + get_bond_len(z2)) / 2



# -------------- ACE basis with good defaults 

const _kw_defaults = Dict(:elements => nothing, 
                          :order => nothing, 
                          :totaldegree => nothing, 
                          :wL => 1.5, 
                          :r0 => :bondlen, 
                          :rcut => (:bondlen, 2.5),
                          :transform => (:agnesi, 2, 4),
                          :envelope => (:x, 2, 2),
                          :rbasis => :legendre, 
                          :pure2b => true, 
                          :delete2b => true, 
                          #
                          :pair_rcut => :pair, 
                          :pair_degree => :totaldegree,
                          :pair_transform => (:agnesi, 1, 3), 
                          :pair_basis => :legendre, 
                          # 
                          )

const _kw_aliases = Dict( :N => :order, 
                          :species => :elements, 
                          :trans => :transform, 
                         )

function _clean_args(kwargs) 
   dargs = Dict{Symbol, Any}() 
   for key in keys(kwargs)
      if haskey(ACE1x._kw_aliases, key) 
         dargs[ACE1x._kw_aliases[key]] = kwargs[key]
      else 
         dargs[key] = kwargs[key]
      end
   end 
   for key in keys(_kw_defaults) 
      if !haskey(dargs, key) 
         dargs[key] = _kw_defaults[key]
      end
   end
   return namedtuple(dargs)
end

function _get_order(kwargs) 
   if haskey(kwargs, :order) 
      return kwargs[:order] 
   elseif haskey(kwargs, :bodyorder) 
      return kwargs[:bodyorder] - 1 
   end 
   error("Cannot determine correlation order or body order of ACE basis from the arguments provided.")
end

function _get_degrees(kwargs) 

   if haskey(kwargs, :totaldegree)
      if kwargs[:totaldegree] isa Union{Number, Dict} 
         cor_order = _get_order(kwargs)
         Deg, maxdeg = ACE1.Utils._auto_degrees(cor_order, kwargs[:totaldegree], 
                                                kwargs[:wL], nothing)
         maxn = maxdeg 
         return Deg, maxdeg, maxn 
      end
   end 
   error("Cannot determine total degree of ACE basis from the arguments provided.")
end

function _get_r0(kwargs, z1, z2)
   if kwargs[:r0] == :bondlen
      return get_r0(z1, z2)
   elseif kwargs[:r0] isa Number
      return kwargs[:r0]
   elseif kwargs[:r0] isa Dict
      return kwargs[:r0][(z1, z2)]
   end
   error("Unable to determine r0($z1, $z2) from the arguments provided.")
end

function _get_all_r0(kwargs)
   elements = kwargs[:elements]
   r0 = Dict( [ (s1, s2) => _get_r0(kwargs, s1, s2) 
                   for s1 in elements, s2 in elements]... )
end

function _get_rcut(kwargs, s1, s2) 
   _rcut = kwargs[:rcut]
   if _rcut isa Tuple 
      if _rcut[1] == :bondlen
         return _rcut[2] * get_r0(s1, s2)
      end
   elseif _rcut isa Number 
      return _rcut 
   elseif _rcut isa Dict 
      return _rcut[(s1, s2)]
   end
   error("Unable to determine rcut($s1, $s2) from the arguments provided.")
end

function _get_all_rcut(kwargs) 
   if kwargs[:rcut] isa Number 
      return kwargs[:rcut]
   end
   elements = kwargs[:elements]
   rcut = Dict( [ (s1, s2) => _get_rcut(kwargs, s1, s2) 
                   for s1 in elements, s2 in elements]... )
   return rcut 
end

function _transform(kwargs)
   elements = kwargs[:elements]
   transform = kwargs[:transform]

   if transform isa Tuple
      if transform[1] == :agnesi 
         p = transform[2]
         q = transform[3]
         r0 = _get_all_r0(kwargs)
         rcut = _get_all_rcut(kwargs)
         transforms = Dict([ (s1, s2) => agnesi_transform(r0[(s1, s2)], p, q)
                             for s1 in elements, s2 in elements]... )
         trans_ace = multitransform(transforms; rin = 0.0, rcut = 8.0)
         return trans_ace 
      end
   
   elseif transform isa ACE1.Transforms.DistanceTransform 
      return transform 
   end

   error("Unable to determine transform from the arguments provided.")
end


function _radial_basis(kwargs) 
   rbasis = kwargs[:rbasis]

   if rbasis == :legendre 
      Deg, maxn, maxl = _get_degrees(kwargs)      
      if kwargs[:pure2b] 
         cor_order = _get_order(kwargs)
         envelope = kwargs[:envelope] 
         if envelope isa Tuple 
            if envelope[1] == :x 
               pin = envelope[2]
               pcut = envelope[3]
               maxn += (pin + pcut) * (cor_order-1)
            end
         else 
            error("I can't construct the radial basis automatically without knowing the envelope.")
         end
      end 
      trans_ace = _transform(kwargs)
      Rn_basis = transformed_jacobi(maxn, trans_ace; pcut = pcut, pin = pin)
      return Rn_basis

   elseif rbasis isa ACE1.ScalarBasis
      return rbasis 
   end

   error("Unable to determine the radial basis from the arguments provided.")
end




function ace_basis(; kwargs...)
   kwargs = _clean_args(kwargs)
   elements = kwargs[:elements]
   cor_order = _get_order(kwargs)
   Deg, maxn, maxdeg = _get_degrees(kwargs)
   rbasis = _radial_basis(kwargs)

   if kwargs[:pure2b] 
      rpibasis = Pure2b.pure2b_basis(species = AtomicNumber.(elements),
                              Rn=rbasis, 
                              D=Deg,
                              maxdeg=maxdeg, 
                              order=cor_order, 
                              delete2b = kwargs[:delete2b])
   else
      error("todo - implement standard ACE basis")
   end 

   return rpibasis
end