using ACE1x
using Test
using Printf
using ACE1.Testing: print_tf 
using LinearAlgebra

## === configs ===

pin = 2
pcut = 2
ord = 3
totaldegree = maxdeg = 8
Deg = SparsePSHDegree()
species = elements = [:Ti, :Al]

Nat = 10 
ZZ = AtomicNumber.(species)

@info("Check pure overrides pure2b")

model = ACE1x.acemodel(; elements = species, 
                            order = ord, 
                            totaldegree = totaldegree,
                            pure = true,
                            pure2b = true, # pure overrides pure2b
                            delete2b = true,
                            pin = pin,
                            pcut = pcut,
                            )
Pr = model.basis.BB[2].pibasis.basis1p.J

for del2b in [true, false]
    @info("delete2b = $del2b")
    pure_model = ACE1x.acemodel(; elements = species, 
                            order = ord, 
                            totaldegree = totaldegree,
                            pure = true,
                            pure2b = true, # pure overrides pure2b
                            delete2b = del2b,
                            pin = pin,
                            pcut = pcut,
                            )

    _pure_model = ACE1x.acemodel(; elements = species, 
                            order = ord, 
                            totaldegree = totaldegree,
                            pure = true,
                            pure2b = false, # pure overrides pure2b
                            delete2b = del2b,
                            pin = pin,
                            pcut = pcut,
                            )

    print_tf(@test pure_model.basis == _pure_model.basis)
    print_tf(@test pure_model.meta == _pure_model.meta)
    print_tf(@test pure_model.Vref == _pure_model.Vref)

    coeffs = randn(length(pure_model.params))
    ACE1x._set_params!(pure_model, coeffs)
    ACE1x._set_params!(_pure_model, coeffs)    

    for ntest = 1:30
        z0 = rand(ZZ)
        Zs = [rand(ZZ) for _ = 1:Nat]
        Rs = [ ACE1.Random.rand_sphere() * (2.7 + 2 * rand()) for i = 1:Nat]
        print_tf(@test ACE1.evaluate(pure_model.potential.components[1], Rs, Zs, z0) ≈ ACE1.evaluate(_pure_model.potential.components[1], Rs, Zs, z0))
        print_tf(@test ACE1.evaluate(pure_model.potential.components[2], Rs, Zs, z0) ≈ ACE1.evaluate(_pure_model.potential.components[2], Rs, Zs, z0))
    end
    println()
end


@info("Check pure2b is the same as pure when ν = 2")
for del2b in [true, false]
    @info("delete2b = $del2b")
    pure2b_model = ACE1x.acemodel(; elements = species, 
                            order = 2, 
                            totaldegree = totaldegree,
                            pure = true,
                            pure2b = false,
                            delete2b = del2b,
                            pin = pin,
                            pcut = pcut,
                            )

    pure_model = ACE1x.acemodel(; elements = species, 
                            order = 2, 
                            totaldegree = totaldegree,
                            pure = false,
                            pure2b = true,
                            delete2b = del2b,
                            pin = pin,
                            pcut = pcut,
                            )

    coeffs = randn(length(pure2b_model.params))
    ACE1x._set_params!(pure_model, coeffs)
    ACE1x._set_params!(pure2b_model, coeffs)
    

    for ntest = 1:30
        z0 = rand(ZZ)
        Zs = [rand(ZZ) for _ = 1:Nat]
        Rs = [ ACE1.Random.rand_sphere() * (2.7 + 2 * rand()) for i = 1:Nat]
        print_tf(@test ACE1.evaluate(pure_model.potential.components[1], Rs, Zs, z0) ≈ ACE1.evaluate(pure2b_model.potential.components[1], Rs, Zs, z0))
        print_tf(@test ACE1.evaluate(pure_model.potential.components[2], Rs, Zs, z0) ≈ ACE1.evaluate(pure2b_model.potential.components[2], Rs, Zs, z0))
    end
    println()
end

