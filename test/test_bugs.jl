
using ACE1x, Test

model1 = acemodel(elements = [:Si,], 
                        order = 3, 
                        totaldegree = 6, 
                        rcut = 5.5,  )

model2 = acemodel(elements = (:Si,),  
                        order = 3, 
                        totaldegree = 6, 
                        rcut = 5.5,  )                        

@test model1.basis == model2.basis