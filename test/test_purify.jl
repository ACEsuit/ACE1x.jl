using ACE1, ACE1x
using ACE1: rand_radial, evaluate, PIBasisFcn
using ACE1.Testing: println_slim, print_tf
using SparseArrays: sparse, spzeros, SparseVector
using LinearAlgebra
using Test
using Printf


# === test configs ===
elements = [:X]
species = [AtomicNumber(x) for x in elements]
zX = species[1]

ord = 4
maxdeg = 12
pin = 2
pcut = 2
ninc = (pin + pcut) * (ord - 1)
maxn = maxdeg + ninc 

D = SparsePSHDegree()
r0 = 1.0
rin = 0.5
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxn, trans, rcut, rin; pcut = pcut, pin = pin)
## 

# === define impure and pure basis === 
ACE_B = ACE1.Utils.rpi_basis(species=species, rbasis=Pr, D=D, 
                             maxdeg=maxdeg, N=ord)

pure_rpibasis = ACE1x.Purify.pureRPIBasis(ACE_B; remove = 0, elements = elements )

# construct Rn_x for Testing
spec = ACE1.get_basis_spec(ACE_B.pibasis, 1)
maxn = maximum( maximum(b.n for b in bb.oneps) for bb in spec )
Rn = ACE_B.pibasis.basis1p.J
Rn_x = ACE1.OrthPolys.transformed_jacobi(maxn+ninc, Rn.trans, Rn.ru, Rn.rl; pcut = pcut, pin = pin)

##

## === testings ===
@info("Basis construction and evaluation checks")
@info("check single species")
Nat = 15
for ntest = 1:30
    Rs, Zs, z0 = rand_nhd(Nat, Rn_x, elements)
    B = ACE1.evaluate(pure_rpibasis, Rs, Zs, z0)
    print_tf(@test(length(pure_rpibasis) == length(B)))
end
println()


@info("isometry and permutation invariance")
for ntest = 1:30
   Rs, Zs, z0 = ACE1.rand_nhd(Nat, Rn_x, elements)
   Rsp, Zsp = ACE1.rand_sym(Rs, Zs)
   print_tf(@test(ACE1.evaluate(pure_rpibasis, Rs, Zs, z0) ≈
                  ACE1.evaluate(pure_rpibasis, Rsp, Zsp, z0)))
end
println()


@info("check orthogonality within each body order")
sam = 10000
AA = zeros(sam, length(pure_rpibasis))
AA_ip = zeros(sam, length(ACE_B))
ord_list = length.(get_nl(ACE_B))
cut_list = [maximum(findall(ord_list .== i)) for i = 1:ord]

for Nat in 2:ord
    for i = 1:sam
        local Rs, Zs, z0 = ACE1.rand_nhd(Nat, Rn_x, elements)
        AA[i, :] = ACE1.evaluate(pure_rpibasis, Rs, Zs, z0)
        AA_ip[i, :] = ACE1.evaluate(ACE_B, Rs, Zs, z0)
    end
    AA_nb = AA[:, cut_list[Nat - 1] + 1:cut_list[Nat]]
    AA_ip_nb = AA_ip[:, cut_list[Nat - 1] + 1:cut_list[Nat]]
    println("conditional number of grammian for ν = $Nat")
    G_pure, G_impure = AA_nb' * AA_nb, AA_ip_nb' * AA_ip_nb
    println("pure basis :", cond(G_pure))
    println("impure basis :", cond(G_impure))

    # scaling w.r.t. diagonal to make sure cond(G) -> 1 if basis is orthonormal
    D_pure, D_impure = diagm(1 ./ sqrt.(diag(G_pure))), diagm(1 ./ sqrt.(diag(G_impure)))

    println("pure basis scaled:", cond(D_pure * G_pure * D_pure))
    println("impure basis scaled:", cond(D_impure * G_impure * D_impure))
    println("===")
end



@info("purify checks")
for (ord, remove) in zip([2, 3, 4], [1, 2, 3])
    
    local ACE_B = ACE1.Utils.rpi_basis(species= species, rbasis=Pr, D=D, 
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