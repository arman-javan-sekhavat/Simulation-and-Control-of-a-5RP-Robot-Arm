#*****************************************************************************************************************
#=========================================  Author: Arman Javan Sekhavat =========================================
#*****************************************************************************************************************



import numpy as np
import jax
from jax import numpy as jnp
from jax import jit

def Homogeneous(q):

    # Constant length parameters
    L1 = 0.170
    L2 = 0.200
    L3 = 0.200
    d5 = 0.390

    # Auxiliary variables
    psi_1 = L2*np.cos(q[1]) + L3*np.cos(q[1] + q[2]) - (d5 + q[5])*np.sin(q[1] + q[2] + q[3])
    psi_2 = L2*np.sin(q[1]) + L3*np.sin(q[1] + q[2]) + (d5 + q[5])*np.cos(q[1] + q[2] + q[3])

    # Row 1 of rotation matrix
    r11 = -np.sin(q[0])*np.sin(q[4]) + np.cos(q[0])*np.cos(q[4])*np.cos(q[1] + q[2] + q[3])
    r12 = -np.sin(q[0])*np.cos(q[4]) - np.cos(q[0])*np.sin(q[4])*np.cos(q[1] + q[2] + q[3])
    r13 = +np.cos(q[0])*np.sin(q[1] + q[2] + q[3])

    # Row 2 of rotation matrix
    r21 = +np.cos(q[0])*np.sin(q[4]) + np.sin(q[0])*np.cos(q[4])*np.cos(q[1] + q[2] + q[3])
    r22 = +np.cos(q[0])*np.cos(q[4]) - np.sin(q[0])*np.sin(q[4])*np.cos(q[1] + q[2] + q[3])
    r23 = +np.sin(q[0])*np.sin(q[1] + q[2] + q[3])

    # Row 3 of rotation matrix
    r31 = -np.cos(q[4])*np.sin(q[1] + q[2] + q[3])
    r32 = +np.sin(q[4])*np.sin(q[1] + q[2] + q[3])
    r33 = +np.cos(q[1] + q[2] + q[3])

    # End-effector position
    x = -np.cos(q[0])*psi_1
    y = -np.sin(q[0])*psi_1
    z = L1 + psi_2


    # Forming and returning the homogeneous transform matrix T
    T = np.array([[r11, r12, r13, x], [r21, r22, r23, y], [r31, r32, r33, z], [0, 0, 0, 1]])
    return T


def EEpos(q):

    # Constant length parameters
    L1 = 0.170
    L2 = 0.200
    L3 = 0.200
    d5 = 0.390

    # Auxiliary variables
    psi_1 = L2*jnp.cos(q[1]) + L3*jnp.cos(q[1] + q[2]) - (d5 + q[5])*jnp.sin(q[1] + q[2] + q[3])
    psi_2 = L2*jnp.sin(q[1]) + L3*jnp.sin(q[1] + q[2]) + (d5 + q[5])*jnp.cos(q[1] + q[2] + q[3])

    # End-effector position
    x = -jnp.cos(q[0])*psi_1
    y = -jnp.sin(q[0])*psi_1
    z = L1 + psi_2

    return jnp.array([x, y, z])

EEpos_jit = jit(EEpos)



def Quaternion(q):

    A = (q[0] - q[4])/2
    B = (q[0] + q[4])/2
    C = (q[1] + q[2] + q[3])/2


    q1 = +np.cos(B)*np.cos(C)
    q2 = -np.sin(A)*np.sin(C)
    q3 = +np.cos(A)*np.sin(C)
    q4 = +np.sin(B)*np.cos(C)

    q = np.array([[q1], [q2], [q3], [q4]])
    return q


def AngleAxis(q):

    c234 = np.cos(q[1] + q[2] + q[3])
    s234 = np.sin(q[1] + q[2] + q[3])

    theta = np.arccos(0.5*(1 + np.cos(q[0] + q[4]))*(1 + c234) - 1.0)
    K = np.array([(np.sin(q[4]) - np.sin(q[0]))*s234,
                  (np.cos(q[4]) + np.cos(q[0]))*s234,
                  (1.0 + c234)*np.sin(q[0] + q[4])])/(2*np.sin(theta))

    return (theta*K)


def Angles(q):

    r11 = -np.sin(q[0])*np.sin(q[4]) + np.cos(q[0])*np.cos(q[4])*np.cos(q[1] + q[2] + q[3])
    r21 = +np.cos(q[0])*np.sin(q[4]) + np.sin(q[0])*np.cos(q[4])*np.cos(q[1] + q[2] + q[3])
    r31 = -np.cos(q[4])*np.sin(q[1] + q[2] + q[3])
    r32 = +np.sin(q[4])*np.sin(q[1] + q[2] + q[3])
    r33 = +np.cos(q[1] + q[2] + q[3])

    sigma = np.sqrt(r11*r11 + r21*r21)

    # X-Y-Z fixed-angles or Z-Y-X Euler-angles
    alpha = np.arctan2(r21/sigma, r11/sigma)
    beta  = np.arctan2(-r31, sigma)
    gamma = np.arctan2(r32/sigma, r33/sigma)

    return (alpha, beta, gamma)



def Jacobian(q):

    # Constant length parameters
    L1 = 0.170
    L2 = 0.200
    L3 = 0.200
    d5 = 0.390

    # Auxiliary variables
    c1 = jnp.cos(q[0])
    s1 = jnp.sin(q[0])

    c2 = jnp.cos(q[1])
    s2 = jnp.sin(q[1])

    c234 = jnp.cos(q[1] + q[2] + q[3])
    s234 = jnp.sin(q[1] + q[2] + q[3])

    psi_1 = L2*c2 + L3*jnp.cos(q[1] + q[2]) - (d5 + q[5])*s234
    psi_2 = L2*s2 + L3*jnp.sin(q[1] + q[2]) + (d5 + q[5])*c234


    j = jnp.zeros(shape = (6, 6), dtype = np.float32)

    # ----------------------------------- qv
    j = j.at[0, 0].set(+s1*psi_1)
    j = j.at[0, 1].set(+c1*psi_2)
    j = j.at[0, 2].set(+c1*(psi_2 - L2*s2))
    j = j.at[0, 3].set(c1*c234*(d5 + q[5]))
    j = j.at[0, 5].set(+c1*s234)

    j = j.at[1, 0].set(-c1*psi_1)
    j = j.at[1, 1].set(+s1*psi_2)
    j = j.at[1, 2].set(+s1*(psi_2 - L2*s2))
    j = j.at[1, 3].set(+s1*c234*(d5 + q[5]))
    j = j.at[1, 5].set(+s1*s234)

    j = j.at[2, 1].set(psi_1)
    j = j.at[2, 2].set(psi_1 - L2*c2)
    j = j.at[2, 3].set(-s234*(d5 + q[5]))
    j = j.at[2, 5].set(c234)

    # ----------------------------------- qw
    j = j.at[3, 1].set(-s1)
    j = j.at[3, 2].set(-s1)
    j = j.at[3, 3].set(-s1)
    j = j.at[3, 4].set(+c1*s234)

    j = j.at[4, 1].set(+c1)
    j = j.at[4, 2].set(+c1)
    j = j.at[4, 3].set(+c1)
    j = j.at[4, 4].set(+s1*s234)

    j = j.at[5, 0].set(+1.0)
    j = j.at[5, 4].set(+c234)

    return j

Jacobian_jit = jit(Jacobian)