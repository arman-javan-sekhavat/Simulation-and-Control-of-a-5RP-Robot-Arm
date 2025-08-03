#*****************************************************************************************************************
#=========================================  Author: Arman Javan Sekhavat =========================================
#*****************************************************************************************************************


import numpy as np
from math import sqrt, atan2

def Circular(A, B, K1, K2):
    
    K_squared = K1*K1 + K2*K2
    d = A*A - B*B

    psi_1 = (K_squared + d)/(2*A)
    psi_2 = (K_squared - d)/(2*B)

    sigma_1 = sqrt(K_squared - psi_1*psi_1)
    sigma_2 = sqrt(K_squared - psi_2*psi_2)

    M = np.array([[K2, K1], [-K1, K2]])/K_squared

    # solution 1
    v_1 = M@np.array([[-sigma_1], [psi_1]])
    v_2 = M@np.array([[+sigma_2], [psi_2]])
    s_1_1 = atan2(v_1[1], v_1[0])
    s_1_2 = atan2(v_2[1], v_2[0])

    # solution 2
    v_1 = M@np.array([[+sigma_1], [psi_1]])
    v_2 = M@np.array([[-sigma_2], [psi_2]])
    s_2_1 = atan2(v_1[1], v_1[0])
    s_2_2 = atan2(v_2[1], v_2[0])

    return (s_1_1, s_1_2, s_2_1, s_2_2)