
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
rcut = 3.0
zX = AtomicNumber(:X)
pcut = 2 
pin = 2
D = SparsePSHDegree()

trans = PolyTransform(1, r0)

ninc = (pcut + pin) * (ord-1)
maxn = maxdeg + ninc 
Pr = transformed_jacobi(maxn, trans, rcut; pcut = pcut, pin = pin)


##

function make_B1p(maxdeg, ord) 
   ninc = (pcut + pin) * (ord-1)
   maxn = maxdeg + ninc 
   trans = PolyTransform(1, r0)
   Pr = transformed_jacobi(maxn, trans, rcut; pcut = pcut, pin = pin)
   B1p = BasicPSH1pBasis(Pr; species = :X, D = D)
   spec1 = filter( b -> (ACE1.degree(D, b) <= maxdeg) || 
                       (b.n <= maxn && b.l == 0), B1p.spec )
   
   return BasicPSH1pBasis(B1p.J, B1p.zlist, spec1)
end 



##

B1p = BasicPSH1pBasis(Pr; species = :X, D = D)
B1p_x = make_B1p(maxdeg, ord)

length(B1p.spec)
length(B1p_x.spec)

basis = ACE1.PIBasis(B1p, ord, D, maxdeg)
@show length(basis)

basis_x = ACE1.PIBasis(B1p_x, ord, D, maxdeg)
@show length(basis_x)

length(basis.basis1p.spec)
length(basis_x.basis1p.spec)

##

maxn_x = maximum(b.n for b in B1p_x.spec)

spec = ACE1.get_basis_spec(basis, 1)

maxn = maximum( maximum(b.n for b in bb.oneps) for bb in spec )

const NLMZ = ACE1.RPI.PSH1pBasisFcn

new_spec = [ ACE1.PIBasisFcn(AtomicNumber(:X), 
                             (NLMZ(n, 0, 0, AtomicNumber(:X)), ) )
               for n = maxn+1:maxn_x ] 
orders = [ length(bb.oneps) for bb in spec ]
N1 = maximum(findall(orders.==1))
@assert all(isequal(1), orders[1:N1])
spec_x = [ spec[1:N1]; new_spec; spec[N1+1:end] ]

basis_x = ACE1.pibasis_from_specs(B1p_x, [spec_x,] )
length(basis_x)

ACE1.get_basis_spec(basis_x, 1) == spec_x

## 

pibasis_x = ACE1x.Pure2b.pure2b_basis(species = [AtomicNumber(:X),], 
                                      Rn=Pr, 
                                      D=D, 
                                      maxdeg=maxdeg, 
                                      order=4)

rpibasis = ACE1.RPI.RPIBasis(pibasis_x)

