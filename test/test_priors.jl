
using ACE1x

model = acemodel(; elements = [:Al, :Ti], 
                   order = 2, totaldegree = 8 )

basis = model.basis
spec = ACE1x._get_nnll(basis)

palg = algebraic_smoothness_prior(basis; p = 2).diag
pexp = exp_smoothness_prior(basis; al = 1.0, an = 1.5).diag
pgauss = gaussian_smoothness_prior(basis; σl = 2.2, σn = 1.5).diag

