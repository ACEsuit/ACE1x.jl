using ACE1x, ACE1, Test 

using ACE1: PolyTransform, transformed_jacobi, SparsePSHDegree, BasicPSH1pBasis, evaluate, rand_radial, rand_nhd
using ACE1.Random: rand_vec
using ACE1.Testing: print_tf 
using LinearAlgebra
using LinearAlgebra: qr, norm, Diagonal, I, cond
using SparseArrays
using JuLIP 
using ACE1.Transforms: multitransform

using Test
using Printf


## === testing configs ===
ord = 4
maxdeg = 8
r0 = 2.8 
rin = 0.5 * r0
rcut = 5.5
pcut = 2 
pin = 2
D = SparsePSHDegree()
transforms = Dict(
   [ (s1, s2) => PolyTransform(2, (rnn(s1)+rnn(s2))/2)
     for (s1, s2) in [(:Ti, :Ti), (:Ti, :Al), (:Al, :Al) ]] ...)

trans = multitransform(transforms; rin = 0.0, rcut = 6.0)

ninc = (pcut + pin) * (ord-1)
maxn = maxdeg + ninc 

elements = [:Ti, :Al]
species = [AtomicNumber(x) for x in elements]
zX = species[1]
Pr = transformed_jacobi(maxn, trans; pcut = 2)

##

## === Define impure and pure basis ===
ACE_B = ACE1.Utils.rpi_basis(species=species, rbasis = Pr, D=D, 
                             maxdeg=maxdeg, N=ord)
pure_rpibasis = ACE1x.Purify.pureRPIBasis(ACE_B; remove = 0, elements = elements )

# get extended radial basis for testing
spec = ACE1.get_basis_spec(ACE_B.pibasis, 1)
maxn = maximum( maximum(b.n for b in bb.oneps) for bb in spec )
Rn = ACE_B.pibasis.basis1p.J
Rn_x = ACE1.transformed_jacobi(maxn+ninc, Rn.trans)

##

# === testings === 
@info("Basis construction and evaluation checks")
Nat = 15
for ntest = 1:30
    Rs, Zs, z0 = rand_nhd(Nat, Rn_x.J, elements)
    B = ACE1.evaluate(pure_rpibasis, Rs, Zs, z0)
    print_tf(@test(length(pure_rpibasis) == length(B)))
end
println()


@info("isometry and permutation invariance")
for ntest = 1:30
    Rs, Zs, z0 = rand_nhd(Nat, Rn_x.J, elements)
    Rsp, Zsp = ACE1.rand_sym(Rs, Zs)
    print_tf(@test(ACE1.evaluate(pure_rpibasis, Rs, Zs, z0) ≈
                    ACE1.evaluate(pure_rpibasis, Rsp, Zsp, z0)))
end
println()


@info("purify checks")
for (ord, remove) in zip([2, 3, 4], [1, 2, 3])
    
    local ACE_B = ACE1.Utils.rpi_basis(species=species, rbasis=Pr, D=D, 
                                 maxdeg=maxdeg, N=ord)
    local pure_rpibasis = ACE1x.Purify.pureRPIBasis(ACE_B; remove = remove, elements = elements )

    # @profview pureRPIBasis(ACE_B; species = species)

    if ord == 2 && remove == 1
        @info("Test evaluate of dimer = 0")
        for ntest = 1:30 
            z = rand(species)
            z0 = rand(species)
            r = ACE1.rand_radial(Pr, z, z0)
            Rs, Zs = [ JVecF(r, 0, 0), ], [ z, ]
            B = ACE1.evaluate(pure_rpibasis, Rs, Zs, z0)
            print_tf(@test( norm(B, Inf) < 1e-12 ))
        end
        println()
    end

    if ord == 3 && remove == 2
        @info("Test evaluate of trimer = 0")
        for ntest = 1:30 
            z0 = rand(species)

            z1 = rand(species)
            r1 = ACE1.rand_radial(Pr, z1, z0)

            z2 = rand(species)
            r2 = ACE1.rand_radial(Pr, z2, z0)

            Rs, Zs = [ JVecF(r1, 0, 0), JVecF(r2, 0, 0), ], [ z1, z2, ]
            B = ACE1.evaluate(pure_rpibasis, Rs, Zs, z0)
            print_tf(@test( norm(B, Inf) < 1e-12 ))
        end
        println()
    end

    if ord == 4 && remove == 3
        @info("Test evaluate of quadmer = 0")
        for ntest = 1:30 
            z0 = rand(species)

            z1 = rand(species)
            r1 = ACE1.rand_radial(Pr, z1, z0)

            z2 = rand(species)
            r2 = ACE1.rand_radial(Pr, z2, z0)

            z3 = rand(species)
            r3 = ACE1.rand_radial(Pr, z3, z0)
    
            Rs, Zs = [ JVecF(r1, 0, 0), JVecF(r2, 0, 0), JVecF(r3, 0, 0)], [ z1, z2, z3]
            B = ACE1.evaluate(pure_rpibasis, Rs, Zs, z0)
            print_tf(@test( norm(B, Inf) < 1e-12 ))
        end
        println()
    end
end