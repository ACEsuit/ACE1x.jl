
using ACE1x, ACE1, Test 

using ACE1: PolyTransform, transformed_jacobi, SparsePSHDegree, BasicPSH1pBasis, evaluate 
using ACE1.Random: rand_vec 
using ACE1.Testing: print_tf 
using LinearAlgebra: qr, norm, Diagonal, I
using SparseArrays
using JuLIP 
using ACE1.Transforms: multitransform

@info(" ============ ACE1x - Pure2b - Multi Species ============ ")
##

ord = 3
maxdeg = 8
r0 = 2.8 
rin = 0.5 * r0
rcut = 5.5
zTi = AtomicNumber(:Ti)
zAl = AtomicNumber(:Al)
pcut = 2 
pin = 2
D = SparsePSHDegree()

transforms = Dict(
   [ (s1, s2) => PolyTransform(2, (rnn(s1)+rnn(s2))/2)
     for (s1, s2) in [(:Ti, :Ti), (:Ti, :Al), (:Al, :Al) ]] ...)

trans = multitransform(transforms; rin = 0.0, rcut = 6.0)

ninc = (pcut + pin) * (ord-1)
maxn = maxdeg + ninc 
Pr = transformed_jacobi(maxn, trans; pcut = 2, pin = 2)
species = [ zTi, zAl ]

##

rpibasis1 = ACE1.Utils.rpi_basis(species=species, rbasis=Pr, D=D, 
                                 maxdeg=maxdeg, N=ord)

spec1 = ACE1.get_nl(rpibasis1)   
spec1_AA1 = ACE1.get_basis_spec(rpibasis1.pibasis, 1)

## 

rpibasis = ACE1x.Pure2b.pure2b_basis(species = species, 
                                       Rn=Pr, 
                                       D=D, 
                                       maxdeg=maxdeg, 
                                       order=ord, 
                                       delete2b = true)


spec = ACE1.get_nl(rpibasis)

##

tol = 1e-12 # this seems crude but is needed because of roundoff errors
            # in the larger degree basis functions 

@info("Test evaluate of dimer = 0")
for ntest = 1:30 
   z = rand(species)
   z0 = rand(species)
   r = ACE1.rand_radial(Pr, z, z0)
   Rs, Zs = [ JVecF(r, 0, 0), ], [ z, ]
   B = ACE1.evaluate(rpibasis, Rs, Zs, z0)
   print_tf(@test( norm(B, Inf) < tol )) 
end
println()


## 

@info("Test energy of dimer = 0")
for ntest = 1:30 
   z = rand(species)
   z0 = rand(species)
   r = ACE1.rand_radial(Pr, z, z0)
   at = Atoms(X = [ JVecF(0, 0, 0), JVecF(r, 0, 0) ], 
            Z = [z, z0], 
            cell = [5.0 0 0; 0 5.0 0; 0 0.0 5.0], 
            pbc = false)
   B = energy(rpibasis, at)
   print_tf(@test( norm(B, Inf) < tol )) 
end
println() 

## 

@info("Confirm that invariance is preserved")
for ntest = 1:30 
   nat = 10 
   Zs = rand(species, nat)
   z0 = rand(species)
   Rs = [ ACE1.rand_radial(Pr, z, z0) * ACE1.Random.rand_sphere() for z in Zs]
   Rs1, Zs1 = ACE1.Random.rand_sym(Rs, Zs)
   
   print_tf(@test( evaluate(rpibasis, Rs,  Zs,  z0) ≈ 
                   evaluate(rpibasis, Rs1, Zs1, z0) ))
end
println() 

