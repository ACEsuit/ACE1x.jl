using ACE1, ACE1x

using ACE1: rand_radial, evaluate, PIBasisFcn
using ACE1.Testing: println_slim, print_tf

using SparseArrays: sparse, spzeros, SparseVector
using LinearAlgebra
using Test
using Printf


## === test configs ===
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

pure_rpibasis = ACE1x.Purify.pureRPIBasis(ACE_B; remove = 0)


##

## === testings ===
@info("Basis construction and evaluation checks")
@info("check single species")
Nat = 15
for ntest = 1:30
    local B 
    Rs, Zs, z0 = rand_nhd(Nat, Pr, elements)
    B = ACE1.evaluate(pure_rpibasis, Rs, Zs, z0)
    print_tf(@test(length(pure_rpibasis) == length(B)))
end
println()


@info("isometry and permutation invariance")
for ntest = 1:30
   Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, elements)
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
        local Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, elements)
        AA[i, :] = ACE1.evaluate(pure_rpibasis, Rs, Zs, z0)
        AA_ip[i, :] = ACE1.evaluate(ACE_B, Rs, Zs, z0)
    end
    AA_nb = AA[:, cut_list[Nat - 1] + 1:cut_list[Nat]]
    AA_ip_nb = AA_ip[:, cut_list[Nat - 1] + 1:cut_list[Nat]]
    println("condition number of gramian for ν = $Nat")
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
    local pure_rpibasis = ACE1x.Purify.pureRPIBasis(ACE_B; remove = remove)

    # @profview pureRPIBasis(ACE_B; species = species)    
    if ord == 2 && remove == 1
        @info("Test evaluate of dimer = 0")
    elseif ord == 3 && remove == 2
        @info("Test evaluate of trimer = 0")
    elseif ord == 4 && remove == 3
        @info("Test evaluate of quadmer = 0")
    end

    for ntest = 1:30
        local B 
        z0 = rand(species)

        Zs = [rand(species) for _ = 1:ord - 1]
        rL = [ACE1.rand_radial(Pr, Zs[i], z0) for i = 1:ord - 1]
        
        Rs = [ JVecF(rL[i], 0, 0) for i = 1:ord - 1 ]
        B = ACE1.evaluate(pure_rpibasis, Rs, Zs, z0)
        print_tf(@test( norm(B, Inf) < 1e-12 ))
    end
    println()

    if ord == 2 && remove == 1
        @info("Test energy of dimer = 0")
        for ntest = 1:30 
            local B 
            z = rand(species)
            z0 = rand(species)
            r = ACE1.rand_radial(Pr, z, z0)
            local at = Atoms(X = [ JVecF(0, 0, 0), JVecF(r, 0, 0) ], 
                        Z = [z, z0], 
                        cell = [5.0 0 0; 0 5.0 0; 0 0.0 5.0], 
                        pbc = false)
            B = energy(pure_rpibasis, at)
            print_tf(@test( norm(B, Inf) < 1e-12 )) 
        end
        println()
    end
end


@info("Check span")
Nat = 15
for i = 1:sam
    local Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, :X)
    AA[i, :] = ACE1.evaluate(pure_rpibasis, Rs, Zs, z0)
    AA_ip[i, :] = ACE1.evaluate(ACE_B, Rs, Zs, z0)
end
Q1_, R1_ = qr(AA)
Q2_, R2_ = qr(AA_ip)
Q1 = Matrix(Q1_)
Q2 = Matrix(Q2_)
R1 = Matrix(R1_)
R2 = Matrix(R2_)
@show rank(R1)
@show rank(R2)
@show norm(Q1 * (Q1' * AA_ip) - AA_ip)
@show norm(Q2 * (Q2' * AA) - AA)