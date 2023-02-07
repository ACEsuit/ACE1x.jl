
import YAML
using JuLIP: AtomicNumber

# -------------- Bond-length heuristics 

_lengthscales_path = joinpath(@__DIR__, "..", "data", 
                              "length_scales_VASP_auto_length_scales.yaml")
_lengthscales = YAML.load_file(lengthscales_path)

get_bond_len(z::AtomicNumber) = get_bond_len(convert(Int, z)) 

function get_bond_len(z::Integer) 
   if haskey(_lengthscales, z)
      return _lengthscales[z]["bond_len"]
   end
   error("No typical bond length for atomic number $z is known. Please specify manually.")
end

get_r0(z1, z2) = get_bond_len(z1) + get_bond_len(z2)



# -------------- ACE basis with good defaults 

