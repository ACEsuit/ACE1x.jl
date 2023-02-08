
import YAML, ACE1, JuLIP 
using NamedTupleTools: namedtuple
using JuLIP: AtomicNumber, rnn, z2i 
using ACE1.Transforms: agnesi_transform, multitransform
using ACE1.PairPotentials: PolyPairBasis
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
                          :pair_rcut => :rcut, 
                          :pair_degree => :totaldegree,
                          :pair_transform => (:agnesi, 2, 4), 
                          :pair_basis => :legendre,  
                          :pair_envelope => (:r, 2), 
                          # 
                          :Eref => missing 
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

   if dargs[:pair_rcut] == :rcut 
      dargs[:pair_rcut] = kwargs[:rcut]
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

function _get_rcut(kwargs, s1, s2; _rcut = kwargs[:rcut])
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

function _get_all_rcut(kwargs; _rcut = kwargs[:rcut]) 
   if _rcut isa Number 
      return _rcut
   end
   elements = kwargs[:elements]
   rcut = Dict( [ (s1, s2) => _get_rcut(kwargs, s1, s2; _rcut = _rcut) 
                   for s1 in elements, s2 in elements]... )
   return rcut 
end

function _transform(kwargs; transform = kwargs[:transform])
   elements = kwargs[:elements]

   if transform isa Tuple
      if transform[1] == :agnesi 
         p = transform[2]
         q = transform[3]
         r0 = _get_all_r0(kwargs)
         rcut = _get_all_rcut(kwargs)
         transforms = Dict([ (s1, s2) => agnesi_transform(r0[(s1, s2)], p, q)
                             for s1 in elements, s2 in elements]... )
         trans_ace = multitransform(transforms; rin = 0.0, rcut = rcut)
         return trans_ace 
      end
   
   elseif transform isa ACE1.Transforms.DistanceTransform 
      return transform 
   end

   error("Unable to determine transform from the arguments provided.")
end


function _radial_basis(kwargs) 
   rbasis = kwargs[:rbasis]

   if rbasis isa ACE1.ScalarBasis
      return rbasis 

   elseif rbasis == :legendre 
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
   end 

   error("Unable to determine the radial basis from the arguments provided.")
end




function _pair_basis(kwargs)
   rbasis = kwargs[:pair_basis]
   elements = kwargs[:elements]

   if rbasis isa ACE1.ScalarBasis
      return rbasis

   elseif rbasis == :legendre 
      if kwargs[:pair_degree] == :totaldegree 
         Deg, maxn, maxl = _get_degrees(kwargs)       
      elseif kwargs[:pair_degree] isa Integer 
         maxn = kwargs[:pair_degree]
      else
         error("Cannot determine `maxn` for pair basis from information provided.")
      end 

      allrcut = _get_all_rcut(kwargs; _rcut = kwargs[:pair_rcut])
      if allrcut isa Number 
         allrcut = Dict([(s1, s2) => allrcut for s1 in elements, s2 in elements]...)
      end

      trans_pair = _transform(kwargs, transform = kwargs[:pair_transform])
      _s2i(s) = z2i(trans_pair.zlist, AtomicNumber(s))
      alltrans = Dict([(s1, s2) => trans_pair.transforms[_s2i(s1), _s2i(s2)].t 
                       for s1 in elements, s2 in elements]...)

      allr0 = _get_all_r0(kwargs)

      function _r_basis(s1, s2, penv) 
         _env = ACE1.PolyEnvelope(penv, allr0[(s1, s2)], allrcut[(s1, s2)] )
         return transformed_jacobi_env(maxn, alltrans[(s1, s2)], _env, allrcut[(s1, s2)])
      end
      
      _x_basis(s1, s2, pin, pcut)  = transformed_jacobi(maxn, alltrans[(s1, s2)], allrcut[(s1, s2)]; 
                                             pcut = pcut, pin = pin)
   
      envelope = kwargs[:pair_envelope]
      if envelope isa Tuple 
         if envelope[1] == :x 
            pin = envelope[2]
            pcut = envelope[3]
            rbases = [ _x_basis(s1, s2, pin, pcut) for s1 in elements, s2 in elements ]
         elseif envelope[1] == :r  
            penv = envelope[2]
            rbases = [ _r_basis(s1, s2, penv) for s1 in elements, s2 in elements ]
         end
      end
   end

   return PolyPairBasis(rbases, elements)
end



function mb_ace_basis(kwargs)
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

function ace_basis(; kwargs...)
   kwargs = _clean_args(kwargs)
   rpiB = mb_ace_basis(kwargs) 
   pairB = _pair_basis(kwargs)
   return JuLIP.MLIPs.IPSuperBasis([pairB, rpiB]);
end

