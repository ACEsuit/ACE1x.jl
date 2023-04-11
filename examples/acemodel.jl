using ACE1x

## === configs ===

pin = 2
pcut = 2
ord = 3
totaldegree = 8
species = [:Ti, :Al]
delete2b = true
remove = delete2b ? 1 : 0

##

## === examples on creating acemodel ===

# 1. standard ACE model, formed by a pairbasis and acebasis
@info("standard model")
standard_model = ACE1x.acemodel(; elements = species, 
                         order = ord, 
                         totaldegree = totaldegree,
                         pure = false,
                         pure2b = false,
                         delete2b = false,
                         pin = pin,
                         pcut = pcut,
                         )
npair = length(standard_model.basis.BB[1])
nbasis = length(standard_model.basis.BB[2])
println("number of paired basis: $npair")
println("number of ace basis: $nbasis")

##

# 2. pure2b ACE model, formed by a pairbasis and 2b purified acebasis
@info("pure2b model")
pure2b_model = ACE1x.acemodel(; elements = species, 
                         order = ord, 
                         totaldegree = totaldegree,
                         pure = false,
                         pure2b = true,
                         delete2b = false,
                         pin = pin,
                         pcut = pcut,
                         )

npair = length(pure2b_model.basis.BB[1])
nbasis = length(pure2b_model.basis.BB[2])
println("number of paired basis: $npair")
println("number of ace basis: $nbasis")

##

# 3. pure ACE model, formed by a pairbasis and purified acebasis
@info("pure ACE model")
pure_model = ACE1x.acemodel(; elements = species, 
                         order = ord, 
                         totaldegree = totaldegree,
                         pure = true,
                         pure2b = true, # pure overrides pure2b
                         delete2b = false,
                         pin = pin,
                         pcut = pcut,
                         )
npair = length(pure_model.basis.BB[1])
nbasis = length(pure_model.basis.BB[2])
println("number of paired basis: $npair")
println("number of ace basis: $nbasis")

##

# 4. pure2b ACE + remove 2b model, formed by a pairbasis and 2b purified acebasis with 2b basis removed
@info("pure2b + remove 2b model")
pure2b_model_remove2b = ACE1x.acemodel(; elements = species, 
                         order = ord, 
                         totaldegree = totaldegree,
                         pure = false,
                         pure2b = true,
                         detele2b = true,
                         pin = pin,
                         pcut = pcut,
                         )
npair = length(pure2b_model_remove2b.basis.BB[1])
nbasis = length(pure2b_model_remove2b.basis.BB[2])
println("number of paired basis: $npair")
println("number of ace basis: $nbasis")

##

# 5. pure + remove 2b ACE model, formed by a pairbasis and purified acebasis with 2b basis removed
@info("pure + remove 2b model")
pure_model_remove2b = ACE1x.acemodel(; elements = species, 
                         order = ord, 
                         totaldegree = totaldegree,
                         pure = true,
                         pure2b = true, # pure overrides pure2b
                         delete2b = true,
                         pin = pin,
                         pcut = pcut,
                         )
npair = length(pure_model_remove2b.basis.BB[1])
nbasis = length(pure_model_remove2b.basis.BB[2])
println("number of paired basis: $npair")
println("number of ace basis: $nbasis")
                         





