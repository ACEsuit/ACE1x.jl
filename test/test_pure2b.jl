
using ACE1x, ACE1

using ACE1: PolyTransform, transformed_jacobi, SparsePSHDegree, BasicPSH1pBasis, evaluate 
using ACE1.Random: rand_vec 
using LinearAlgebra: qr, norm, Diagonal, I
using SparseArrays
using ProgressMeter 

##

ord = 4
maxdeg = 16
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

## 

rpibasis = ACE1x.Pure2b.pure2b_basis(species = [AtomicNumber(:X),], 
                                       Rn=Pr, 
                                       D=D, 
                                       maxdeg=maxdeg, 
                                       order=4, 
                                       delete2b = true)


## 

using JuLIP 
zX = AtomicNumber(:X)

# for ntest = 1:30 

r = ACE1.rand_radial(Pr)
at = Atoms(X = [ JVecF(0, 0, 0), JVecF(r, 0, 0) ], 
           Z = [ zX, zX ], 
           cell = [5.0 0 0; 0 5.0 0; 0 0.0 5.0], 
           pbc = false)
B = energy(rpibasis, at)

ACE1.get_nl(rpibasis,

# end