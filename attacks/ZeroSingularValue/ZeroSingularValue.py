#!/usr/bin/env python3

import torch
import torch.nn
import numpy as np

# make reproducible
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(1)

def ZeroSingularValue(A):
    u, s, vh = torch.linalg.svd(A, full_matrices=False)
    s[:,-1] = 0
    s_p = torch.diag_embed(s)
    A_prime = u @ s_p @ vh
    return A_prime

# generate 1000 100x100 matrices with values between -0.5 to 0.5
random_matrices = torch.rand((1000, 100, 100))

condA = torch.linalg.cond(random_matrices)
detA = torch.linalg.det(random_matrices)

# theoretically each matrice should have a det of 0
random_A_primes = ZeroSingularValue(random_matrices)

# condition number and determinants
condA_p = torch.linalg.cond(random_A_primes)
detA_p = torch.linalg.det(random_A_primes)

# save as csv file
np.savetxt(f"ZeroSingularValue_condA.csv", condA.numpy())
np.savetxt(f"ZeroSingularValue_condA_p.csv", condA_p.numpy())
np.savetxt(f"ZeroSingularValue_detA.csv", detA.numpy())
np.savetxt(f"ZeroSingularValue_detA_p.csv", detA_p.numpy())