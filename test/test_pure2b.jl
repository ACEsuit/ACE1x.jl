
using ACE1x, ACE1

using ACE1: PolyTransform, transformed_jacobi, SparsePSHDegree, BasicPSH1pBasis, evaluate 
using ACE1.Random: rand_vec 
using LinearAlgebra: qr, norm, Diagonal, I
using SparseArrays
using ProgressMeter 

##

ord = 2
maxdeg = 8
r0 = 1.0
rin = 0.5
rcut = 3.0
zX = AtomicNumber(:X)
pcut = 2 
pin = 2
D = SparsePSHDegree()

trans = PolyTransform(1, r0)

ninc = (pcut + pin) * (ord-1)
maxn = maxdeg + ninc 
Pr = transformed_jacobi(maxn, trans, rcut, rin; pcut = pcut, pin = pin)
species = [AtomicNumber(:X),]

## 

rpibasis = ACE1x.Pure2b.pure2b_basis(species = species, 
                                       Rn=Pr, 
                                       D=D, 
                                       maxdeg=maxdeg, 
                                       order=ord, 
                                       delete2b = true)

rpibasis1 = ACE1.Utils.rpi_basis(; species=species, N = ord, maxdeg=maxdeg, 
                                 rbasis=Pr, D=D)

spec = ACE1.get_nl(rpibasis)
spec1 = ACE1.get_nl(rpibasis1)
spec[13:end] == spec1[9:end]

##

ii = 13:19 
ii1 = 9:15

r = ACE1.rand_radial(Pr)
Rs, Zs, z0 = [ JVecF(r, 0, 0), ], [ zX, ], zX 
B = ACE1.evaluate(rpibasis, Rs, Zs, z0)
B1 = ACE1.evaluate(rpibasis1, Rs, Zs, z0)

@info("Basis functions 16:20")
display(spec[ii])
@info("Corresponding basis function values")
display([ B[ii] B1[ii1] ] ) 

Br = B[ii] - B1[ii1]
Br ./ B1[ii1]

# ## 

# using JuLIP 
# zX = AtomicNumber(:X)

# # for ntest = 1:30 

# r = ACE1.rand_radial(Pr)
# at = Atoms(X = [ JVecF(0, 0, 0), JVecF(r, 0, 0) ], 
#            Z = [ zX, zX ], 
#            cell = [5.0 0 0; 0 5.0 0; 0 0.0 5.0], 
#            pbc = false)
# B = energy(rpibasis, at)

